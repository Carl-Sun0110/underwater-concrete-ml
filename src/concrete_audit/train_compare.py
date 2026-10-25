"""仅训练部分的嵌套分组模型比较；未确认材料列不进入X，测试集不评估。"""
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from .baselines import grouped_folds,regression_metrics,digest,write_json
from .feature_plan import validate_features
from .scope_gate import check_training_request, load_training_gate, select_training_scope


class CoreRepresentation(BaseEstimator,TransformerMixin):
    """只由Water/Cem两个输入构造比值；不读取目标或原表比例字段。"""
    def __init__(self,add_ratio=False):self.add_ratio=add_ratio
    def fit(self,X,y=None):
        self._validate(X);self.n_features_in_=2;return self
    def _validate(self,X):
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=2 or not np.isfinite(X).all() or np.any(X<=0):
            raise ValueError('需要正的Water/Cem两列；不得填补或引入其他字段')
        return X
    def transform(self,X):
        X=self._validate(X)
        return np.column_stack([X,X[:,0]/X[:,1]]) if self.add_ratio else X.copy()


def select_train_records(labels,split_rows):
    """严格按固定样本ID提取训练部分；标签文件顺序不决定划分。"""
    by_id={r['sample_id']:r for r in labels}
    if len(by_id)!=len(labels):raise ValueError('标签样本ID重复')
    ids=[r['sample_id'] for r in split_rows]
    if len(set(ids))!=len(ids):raise ValueError('划分样本ID重复')
    if any(r['split'] not in ('train','test') for r in split_rows):raise ValueError('划分状态无效')
    train=[r for r in split_rows if r['split']=='train'];test=[r for r in split_rows if r['split']=='test']
    for field in ('leakage_group','research_group','pdf_filename'):
        if {r[field] for r in train}&{r[field] for r in test}:raise ValueError('固定划分存在来源重叠')
    if not train or not test:raise ValueError('训练或封存测试为空')
    rows=[]
    for metadata in train:
        r=by_id.get(metadata['sample_id'])
        if r is None or r['source_row']!=metadata['source_row']:raise ValueError('训练身份映射失败')
        if (r['original_values']['Research Group'],r['original_values']['PDF filename'])!=(metadata['research_group'],metadata['pdf_filename']):
            raise ValueError('训练来源映射失败')
        rows.append(r)
    return rows,np.array([r['leakage_group'] for r in train]),{r['sample_id'] for r in test}


def make_estimator(name,feature_set,config):
    """各折内独立构造表示、缩放及模型；RF不需要标准化。"""
    if feature_set not in ('core2','core_plus_computed_ratio'):raise ValueError('未知表示')
    representation=CoreRepresentation(add_ratio=feature_set=='core_plus_computed_ratio')
    if name=='mean':return Pipeline([('representation',representation),('model',DummyRegressor(strategy='mean'))]),{}
    if name=='ridge':
        model=Ridge();grid={'model__alpha':config['ridge_alpha']}
    elif name=='svr':
        model=SVR(kernel='rbf');grid={'model__C':config['svr_C'],'model__gamma':config['svr_gamma'],'model__epsilon':config['svr_epsilon']}
    elif name=='rf':
        model=RandomForestRegressor(n_estimators=config['rf_n_estimators'],random_state=config['seed'],n_jobs=1)
        grid={'model__max_depth':config['rf_max_depth'],'model__min_samples_leaf':config['rf_min_samples_leaf']}
    elif name=='xgb':
        from xgboost import XGBRegressor
        model=XGBRegressor(objective='reg:squarederror',tree_method='hist',device='cpu',
             learning_rate=config['xgb_learning_rate'],min_child_weight=3,subsample=.8,
             colsample_bytree=1.,reg_lambda=1.,max_bin=64,n_jobs=1,random_state=config['seed'])
        grid={'model__n_estimators':config['xgb_n_estimators'],'model__max_depth':config['xgb_max_depth']}
    elif name=='mlp':
        from sklearn.compose import TransformedTargetRegressor
        from .advanced_models import TorchDropoutRegressor
        model=TransformedTargetRegressor(regressor=TorchDropoutRegressor(
              hidden_sizes=tuple(config['mlp_hidden_sizes']),epochs=config['mlp_epochs'],
              batch_size=config['mlp_batch_size'],random_state=config['seed']),
              transformer=StandardScaler())
        grid={'model__regressor__dropout':config['mlp_dropout']}
    else:raise ValueError('未知模型')
    steps=[('representation',representation)]
    if name in ('ridge','svr','mlp'):steps.append(('scale',StandardScaler()))
    steps.append(('model',model));return Pipeline(steps),grid


def run(labels_path,split_path,config_path,output):
    labels_path,split_path,config_path,output=map(Path,(labels_path,split_path,config_path,output))
    config=json.loads(config_path.read_text())
    scope_request=check_training_request(config)
    if config.get('training_execution_authorized') is False:
        raise ValueError('训练预案尚未启用；本轮只做范围审查，不允许拟合模型')
    if config['test_evaluation_allowed'] or config['material_imputation_allowed']:raise ValueError('本轮禁止测试评估和未知材料填补')
    if config['base_features']!=['Water','Cem']:raise ValueError('本轮仅允许既有两基础输入')
    validate_features(config['base_features'])
    protected={str(p.resolve()):digest(p) for p in (labels_path,split_path,config_path)}
    # 上游原始文件、可信状态档案同样只读保护。
    upstream=json.loads((labels_path.parent/'manifest.json').read_text())
    for path,h in upstream['inputs_sha256'].items():
        if digest(path)!=h:raise ValueError('上游数据已改变，禁止直接续跑')
        protected[path]=h
    labels=json.loads(labels_path.read_text());split=json.loads(split_path.read_text())
    rows,groups,test_ids=select_train_records(labels,split['rows'])
    scope_excluded=[]
    if scope_request is not None:
        scope_records,scope_protected=load_training_gate(scope_request,labels_path,split_path)
        rows,group_values,scope_excluded=select_training_scope(rows,groups,scope_records,scope_request['mode'])
        groups=np.asarray(group_values)
        protected.update(scope_protected)
    X=np.array([[r['original_values'][f] for f in config['base_features']] for r in rows],dtype=float)
    y=np.array([r[config['target']] for r in rows],dtype=float)
    if not np.isfinite(y).all() or np.any(y<=0):raise ValueError('无效目标，禁止填补')
    folds=grouped_folds(X,y,groups,config['outer_folds'])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'training_scope.json',{'train_sample_ids':[r['sample_id'] for r in rows],
               'protected_test_ids':sorted(test_ids),'test_predictions_generated':False,
               'scope_mode':scope_request['mode'] if scope_request else 'explicit_historical_reproduction',
               'excluded_by_scope_gate':scope_excluded,
               'input_features':config['base_features'],'derived_feature':'Water/Cem only'})
    models=config.get('models',['mean','ridge','svr','rf'])
    if not models or len(models)!=len(set(models)) or not set(models)<={'mean','ridge','svr','rf','xgb','mlp'}:raise ValueError('模型列表无效')
    reports=[];predictions=[]
    for feature_set in config['feature_sets']:
        for model in models:
            predicted=np.full(len(rows),np.nan);folds_output=[]
            for fold,(tr,va) in enumerate(folds,1):
                estimator,param_grid=make_estimator(model,feature_set,config)
                if param_grid:
                    inner=grouped_folds(X[tr],y[tr],groups[tr],config['inner_folds'])
                    fitted=GridSearchCV(estimator,param_grid,scoring=config['scoring'],cv=inner,
                                        n_jobs=config['n_jobs'],error_score='raise',refit=True).fit(X[tr],y[tr])
                    best=fitted.best_params_
                    inner_mae=float(-fitted.best_score_)
                else:fitted=estimator.fit(X[tr],y[tr]);best={};inner_mae=None
                pred=fitted.predict(X[va]);predicted[va]=pred
                folds_output.append({'fold':fold,'train_n':len(tr),'validation_n':len(va),
                    'train_groups':sorted(set(groups[tr])),'validation_groups':sorted(set(groups[va])),
                    'validation_sample_ids':[rows[i]['sample_id'] for i in va],
                    'best_params':best,'inner_selection_MAE':inner_mae,'metrics':regression_metrics(y[va],pred)})
            if not np.isfinite(predicted).all():raise RuntimeError('折外预测未覆盖全部训练记录')
            report={'feature_set':feature_set,'model':model,'pooled_nested_oof':regression_metrics(y,predicted),'folds':folds_output}
            reports.append(report)
            for i,r in enumerate(rows):
                predictions.append({'sample_id':r['sample_id'],'source_row':r['source_row'],'research_group':r['original_values']['Research Group'],
                                    'feature_set':feature_set,'model':model,'actual_provisional_MPa':float(y[i]),'oof_prediction_MPa':float(predicted[i])})
            print(feature_set,model,report['pooled_nested_oof'],flush=True)
    assert not(test_ids&{r['sample_id'] for r in predictions})
    summary={'training_samples':len(rows),'sealed_test_samples':len(test_ids),'training_groups':len(set(groups)),
             'experiments':len(reports),'outer_folds_per_experiment':len(folds),
             'test_set_evaluated':False,'input_or_reliability_changed':not all(digest(k)==h for k,h in protected.items()),
             'new_trusted_samples':0,'formal_model_claimed':False,
             'scope_mode':scope_request['mode'] if scope_request else 'explicit_historical_reproduction',
             'selection_warning':'比较多个配置的结果用于探索；后续按这些结果选择后的最优CV值不等于独立最终泛化成绩。'}
    if summary['input_or_reliability_changed']:raise RuntimeError('输入改变')
    for name,obj in [('nested_cv_results.json',reports),('training_oof_predictions.json',predictions),('summary.json',summary),
                     ('config_snapshot.json',config),('manifest.json',{'inputs_sha256':protected,'code_sha256':digest(Path(__file__)),
                                             'created_at_utc':datetime.now(timezone.utc).isoformat(),
                                             'advanced_model_code_sha256':digest(Path(__file__).with_name('advanced_models.py')) if set(models)&{'xgb','mlp'} else None})]:write_json(output/name,obj)
    cache=Path(tempfile.gettempdir())/'underwater-concrete-plot-cache';cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    names=[f'{r["feature_set"]}\n{r["model"]}' for r in reports]
    ax.bar(range(len(reports)),[r['pooled_nested_oof']['MAE_MPa'] for r in reports])
    ax.set_xticks(range(len(reports)),names,rotation=30,ha='right');ax.set_ylabel('Nested grouped OOF MAE (MPa)')
    fig.savefig(output/'training_cv_mae.png',dpi=180);plt.close(fig)
    return summary
