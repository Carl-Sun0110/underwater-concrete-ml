"""探索性均值/岭回归：来源隔离、嵌套分组验证和独立测试。"""
from collections import defaultdict
from datetime import datetime,timezone
import csv,hashlib,json
from pathlib import Path
import platform
import os
import tempfile
import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupShuffleSplit,GroupKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .feature_plan import validate_features
from .reliability import number


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def leakage_groups(rows,signature_fields):
    """共享Research Group/PDF或相同全配比签名的记录保守合组；完全不用目标。"""
    validate_features([f for f in signature_fields if f in ('Water','Cem')])
    banned=('CS_','ST_','FS_','DT_','Conversion','Converted','Strength')
    if any(any(t.lower() in f.lower() for t in banned) for f in signature_fields):
        raise ValueError('分组签名不能使用目标及强度派生信息')
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    seen={};signatures=defaultdict(list)
    for i,row in enumerate(rows):
        raw=row['original_values']
        for kind,name in [('research',raw.get('Research Group')),('pdf',raw.get('PDF filename'))]:
            if not name:raise ValueError('来源身份缺失，禁止猜测分组')
            key=(kind,name)
            if key in seen:union(i,seen[key])
            else:seen[key]=i
        values=[number(raw.get(f)) for f in signature_fields]
        if all(v is not None for v in values):signatures[tuple(values)].append(i)
    links=[]
    for signature,indices in signatures.items():
        sources={rows[i]['original_values']['Research Group'] for i in indices}
        if len(sources)>1:
            for j in indices[1:]:union(indices[0],j)
            links.append({'source_rows':[rows[i]['source_row'] for i in indices],
                          'research_groups':sorted(sources),'reason':'原表配比签名相同；保守防泄漏合组，不判定为已证实重复'})
    root_ids={root:f'G{j:03}' for j,root in enumerate(sorted({find(i) for i in range(len(rows))}),1)}
    return np.array([root_ids[find(i)] for i in range(len(rows))]),links


def ensure_isolated(groups,train,validation):
    """任何来源组重叠立即报错，不跳过隔离检查。"""
    if set(groups[train])&set(groups[validation]):raise ValueError('来源分组泄漏')
    if set(train)&set(validation):raise ValueError('样本索引泄漏')


def regression_metrics(y,pred):
    """误差单位MPa；MAPE以百分数报告，R²在单样本/常量目标时不伪造。"""
    y,pred=np.asarray(y,dtype=float),np.asarray(pred,dtype=float)
    if len(y)==0 or not np.isfinite(y).all() or not np.isfinite(pred).all():raise ValueError('无效评估值')
    mse=float(mean_squared_error(y,pred))
    return {'n':len(y),'R2':float(r2_score(y,pred)) if len(y)>1 and np.ptp(y)>0 else None,
            'MSE_MPa2':mse,'RMSE_MPa':float(np.sqrt(mse)),
            'MAE_MPa':float(mean_absolute_error(y,pred)),
            'MAPE_percent':float(np.mean(np.abs((y-pred)/y))*100) if np.all(y>0) else None}


def ridge_search(folds,config):
    """缩放器放入Pipeline，每个训练折独立拟合；仅按MAE选择alpha。"""
    return GridSearchCV(Pipeline([('scale',StandardScaler()),('ridge',Ridge())]),
                        {'ridge__alpha':config['ridge_alphas']},cv=folds,
                        scoring=config['tuning_metric'],n_jobs=config['n_jobs'],error_score='raise',refit=True)


def grouped_folds(X,y,groups,n):
    if len(set(groups))<n:raise ValueError('独立来源组不足，不能执行配置的折数')
    folds=list(GroupKFold(n_splits=n).split(X,y,groups))
    for t,v in folds:ensure_isolated(groups,t,v)
    return folds


def write_json(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def run(labels_path,config_path,output):
    labels_path,config_path,output=map(Path,(labels_path,config_path,output))
    config=json.loads(config_path.read_text())
    if config.get('historical_scope_reproduction_only') is not True:
        raise ValueError('旧基线入口仅用于显式历史范围复现；新范围请使用带scope_gate的run_train_comparison入口')
    input_rows=json.loads(labels_path.read_text())
    if not config.get('allow_exploratory_training'):raise ValueError('未授权探索性训练')
    if config['features']!=['Water','Cem']:raise ValueError('本次只授权Water/Cem两输入基线')
    validate_features(config['features'])
    if config['target']!='Provisional_Converted_CS_28d':raise ValueError('探索目标不匹配')
    # 所有原始附件、可靠档案、标签配置都沿上游清单保护。
    protected={str(labels_path.resolve()):digest(labels_path),str(config_path.resolve()):digest(config_path)}
    upstream=labels_path.parent/'manifest.json'
    if not upstream.exists():raise ValueError('缺少上游标签来源清单')
    protected[str(upstream.resolve())]=digest(upstream)
    for p,expected in json.loads(upstream.read_text())['inputs_sha256'].items():
        if digest(p)!=expected:raise ValueError('上游输入已改变，需先重建标签')
        protected[p]=expected
    rows=[];excluded=[]
    for r in input_rows:
        if not r['candidate_in_original_527'] or r['original_values'].get('CastingEnv')!='Underwater':
            raise ValueError('探索标签集范围不一致')
        values=[number(r['original_values'].get(f)) for f in config['features']]
        target=number(r.get(config['target']))
        if target is None or target<=0:raise ValueError('不允许填补或使用无效目标')
        if any(v is None or v<=0 for v in values):excluded.append(r['sample_id'])
        else:rows.append(r)
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('样本ID不唯一')
    X=np.array([[r['original_values'][f] for f in config['features']] for r in rows],dtype=float)
    y=np.array([r[config['target']] for r in rows],dtype=float)
    groups,links=leakage_groups(rows,config['group_signature_fields'])
    # 固定seed一次划分；不按目标分布或结果挑选测试集。
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=config['test_group_fraction'],random_state=config['seed']).split(X,y,groups))
    ensure_isolated(groups,train,test)
    output.mkdir(parents=True,exist_ok=False)
    split_records=[]
    for i,r in enumerate(rows):
        split_records.append({'sample_id':r['sample_id'],'source_row':r['source_row'],'split':'train' if i in set(train) else 'test',
                              'leakage_group':groups[i],'research_group':r['original_values']['Research Group'],
                              'pdf_filename':r['original_values']['PDF filename']})
    # 先持久化测试划分，再调参；测试目标不参与以下训练和验证。
    write_json(output/'split_manifest.json',{'seed':config['seed'],'rows':split_records,'group_links':links,
                 'excluded_core_unknown_ids':excluded,'features':config['features'],'target':config['target']})
    Xt,yt,gt=X[train],y[train],groups[train]
    outer=grouped_folds(Xt,yt,gt,config['outer_cv_folds'])
    oof={'mean':np.full(len(train),np.nan),'ridge':np.full(len(train),np.nan)}
    fold_reports=[]
    for fold,(tr,va) in enumerate(outer,1):
        inner=grouped_folds(Xt[tr],yt[tr],gt[tr],config['inner_cv_folds'])
        tuned=ridge_search(inner,config);tuned.fit(Xt[tr],yt[tr])
        mean=DummyRegressor(strategy='mean').fit(Xt[tr],yt[tr])
        predictions={'mean':mean.predict(Xt[va]),'ridge':tuned.predict(Xt[va])}
        for name,pred in predictions.items():oof[name][va]=pred
        fold_reports.append({'fold':fold,'train_n':len(tr),'validation_n':len(va),
          'training_groups':sorted(set(gt[tr])),'validation_groups':sorted(set(gt[va])),
          'best_alpha':float(tuned.best_params_['ridge__alpha']),
          'metrics':{k:regression_metrics(yt[va],v) for k,v in predictions.items()},
          'inner_tuning':[{'alpha':float(a['ridge__alpha']),'mean_CV_MAE':float(-score)} for a,score in zip(tuned.cv_results_['params'],tuned.cv_results_['mean_test_score'])]})
    if any(not np.isfinite(pred).all() for pred in oof.values()):raise RuntimeError('OOF预测不完整')
    # 最终调参仅在全部训练部分做分组CV，之后各模型只评估一次测试集。
    final_cv=grouped_folds(Xt,yt,gt,config['final_tuning_folds'])
    final_ridge=ridge_search(final_cv,config).fit(Xt,yt)
    final_mean=DummyRegressor(strategy='mean').fit(Xt,yt)
    test_predictions={'mean':final_mean.predict(X[test]),'ridge':final_ridge.predict(X[test])}
    metrics={
      'nested_group_cv_pooled_oof':{k:regression_metrics(yt,pred) for k,pred in oof.items()},
      'held_out_test':{k:regression_metrics(y[test],pred) for k,pred in test_predictions.items()},
      'best_final_alpha':float(final_ridge.best_params_['ridge__alpha']),
      'final_tuning':[{'alpha':float(a['ridge__alpha']),'mean_CV_MAE':float(-score)} for a,score in zip(final_ridge.cv_results_['params'],final_ridge.cv_results_['mean_test_score'])]}
    predicted=[]
    for part,indices,preds in [('train_nested_oof',train,oof),('test',test,test_predictions)]:
        for local,i in enumerate(indices):
            predicted.append({'sample_id':rows[i]['sample_id'],'source_row':rows[i]['source_row'],'split':part,
                  'research_group':rows[i]['original_values']['Research Group'],'leakage_group':groups[i],
                  'actual_provisional_MPa':float(y[i]),'mean_prediction_MPa':float(preds['mean'][local]),
                  'ridge_prediction_MPa':float(preds['ridge'][local])})
    by_source=[]
    for group in sorted({rows[i]['original_values']['Research Group'] for i in test}):
        idx=np.array([j for j,i in enumerate(test) if rows[i]['original_values']['Research Group']==group])
        by_source.append({'research_group':group,**{name:regression_metrics(y[test][idx],pred[idx]) for name,pred in test_predictions.items()}})
    summary={'scope':'探索性两特征基线；非最终可信研究模型','input_labels':len(input_rows),'selected_samples':len(rows),
      'excluded_core_unknown':len(excluded),'research_groups':len({r['original_values']['Research Group'] for r in rows}),
      'leakage_groups':len(set(groups)),'train_samples':len(train),'test_samples':len(test),
      'train_groups':len(set(gt)),'test_groups':len(set(groups[test])),
      'nested_outer_folds_executed':len(outer),'models':['mean','ridge'],
      'no_group_overlap':not bool(set(gt)&set(groups[test])),'new_trusted_samples':0,
      'inputs_unchanged':all(digest(p)==h for p,h in protected.items())}
    if not summary['inputs_unchanged']:raise RuntimeError('输入或可靠档案被修改')
    for filename,obj in [('metrics.json',metrics),('summary.json',summary),('cv_folds.json',fold_reports),
                         ('predictions.json',predicted),('test_source_metrics.json',by_source),('config_snapshot.json',config)]:write_json(output/filename,obj)
    with (output/'predictions.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predicted[0]));writer.writeheader();writer.writerows(predicted)
    joblib.dump(final_ridge.best_estimator_,output/'ridge_pipeline.joblib')
    joblib.dump(final_mean,output/'mean_model.joblib')
    # 科研图使用标准静态绘图库，输出独立图像，不启动Web服务。
    cache_root=Path(tempfile.gettempdir())/'underwater-concrete-plot-cache'
    cache_root.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache_root/'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(cache_root))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    lo=min(float(y[test].min()),*(float(p.min()) for p in test_predictions.values()))
    hi=max(float(y[test].max()),*(float(p.max()) for p in test_predictions.values()))
    for ax,(name,pred) in zip(axes,test_predictions.items()):
        ax.scatter(y[test],pred,alpha=.7,s=25);ax.plot([lo,hi],[lo,hi],'k--',linewidth=1)
        ax.set(xlabel='Provisional observed strength (MPa)',ylabel='Predicted strength (MPa)',title=name)
    fig.savefig(output/'test_predictions.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for name,pred in test_predictions.items():ax.scatter(y[test],pred-y[test],s=22,alpha=.65,label=name)
    ax.axhline(0,color='black',linestyle='--');ax.set(xlabel='Provisional observed strength (MPa)',ylabel='Prediction - observed (MPa)');ax.legend()
    fig.savefig(output/'test_residuals.png',dpi=180);plt.close(fig)
    import sklearn
    write_json(output/'manifest.json',{'inputs_sha256':protected,'source_code_sha256':digest(Path(__file__)),
                'created_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
                'numpy':np.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__,
                'training_authorization':config['authorization']})
    return summary,metrics
