"""诊断性对照：折内移除严格恒定列，避免未经学习的输入触发MLP任意响应。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .advanced_models import TorchDropoutRegressor
from .baselines import grouped_folds, regression_metrics
from .nested_exploration import derive_ratios, load_development
from .parallel_exploration import digest, feature_matrix, write_json


def filtered_search(X,y,groups,model,config):
    """恒定列判定不接触验证值和目标；每个内层折独立重拟合整个流水线。"""
    regressor=(SVR(kernel='rbf',epsilon=.1) if model=='svr' else
        TorchDropoutRegressor(epochs=config['mlp_epochs'],dropout=config['mlp_dropout'],
            learning_rate=.003,random_state=config['seed'],torch_threads=1))
    pipeline=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
        ('variance',VarianceThreshold(threshold=0.)),('scale',StandardScaler()),
        ('model',TransformedTargetRegressor(regressor=regressor,transformer=StandardScaler()))])
    folds=grouped_folds(X,y,groups,config['inner_folds'])
    search=GridSearchCV(pipeline,config['grids'][model],cv=folds,scoring=config['selection_scoring'],
                        n_jobs=1,refit=True,error_score='raise')
    search.fit(X,y)
    return search,folds


def run(config_path,output):
    config_path,output=Path(config_path),Path(output);config=json.loads(config_path.read_text())
    previous,meta,rows,groups,y,folds=load_development(config['baseline_run']);output.mkdir(exist_ok=False)
    preceding=Path(config['preceding_run']);old_manifest=json.loads((preceding/'manifest.json').read_text())
    for path,expected in old_manifest['input_hashes_after'].items():
        if digest(path)!=expected:raise ValueError(f'前次对照输入变更：{path}')
    paths=[config_path,Path(__file__),preceding/'manifest.json',preceding/'results.json',preceding/'oof_predictions.json']
    paths += [Path(p) for p in old_manifest['input_hashes_after']]
    before={str(p.resolve()):digest(p) for p in paths}
    fields=previous['config']['feature_sets']['materials']
    matrices={'materials':(feature_matrix(rows,fields),fields),'materials_plus_ratios':derive_ratios(rows,fields,config)}
    results=[];predictions=[];traces=[]
    for fs in config['feature_sets']:
        X,names=matrices[fs]
        for model in config['models']:
            experiment=f'{model}__{fs}__nested_filtered';pred=np.full(len(y),np.nan);scores=[]
            for k,t,v in folds:
                search,inner=filtered_search(X[t],y[t],groups[t],model,config)
                pred[v]=search.predict(X[v]);score=regression_metrics(y[v],pred[v]);scores.append({'outer_fold':k,**score})
                pipe=search.best_estimator_;all_names=pipe['impute'].get_feature_names_out(names)
                traces.append({'experiment':experiment,'outer_fold':k,'best_params':search.best_params_,
                    'removed_constant_columns':all_names[~pipe['variance'].get_support()].tolist(),
                    'outer_train_groups':sorted(set(groups[t])),'outer_validation_groups':sorted(set(groups[v])),
                    'inner_folds':[{'train_groups':sorted(set(groups[t][it])),
                                    'validation_groups':sorted(set(groups[t][iv]))} for it,iv in inner],
                    'candidates':[{'params':p,'mean_inner_MAE_MPa':float(-s)} for p,s in zip(
                        search.cv_results_['params'],search.cv_results_['mean_test_score'])]})
                print(f'{experiment} fold={k} MAE={score["MAE_MPa"]:.4f}',flush=True)
            gs=[{'group':str(g),**regression_metrics(y[groups==g],pred[groups==g])} for g in sorted(set(groups))]
            result={'experiment':experiment,'model':model,'feature_set':fs,**regression_metrics(y,pred),
                'macro_group_MAE_MPa':float(np.mean([r['MAE_MPa'] for r in gs])),
                'negative_predictions':int((pred<0).sum()),'groups':gs,'folds':scores}
            results.append(result)
            for i,r in enumerate(rows):predictions.append({'experiment':experiment,'sample_id':r['sample_id'],
                'source_row':r['source_row'],'paper':meta[i]['paper'],'group':str(groups[i]),'grade':meta[i]['grade'],
                'outer_fold':next(k for k,_,v in folds if i in v),'observed':float(y[i]),
                'predicted':float(pred[i]),'absolute_error':float(abs(y[i]-pred[i]))})
            print(f'{experiment} ALL MAE={result["MAE_MPa"]:.4f} RMSE={result["RMSE_MPa"]:.4f} R2={result["R2"]:.4f}',flush=True)
            write_json(output/'partial_results.json',results)
    write_json(output/'results.json',results);write_json(output/'oof_predictions.json',predictions)
    write_json(output/'inner_search_traces.json',traces)
    after={p:digest(p) for p in before}
    if after!=before:raise RuntimeError('输入版本变化')
    write_json(output/'manifest.json',{'experiment_id':config['experiment_id'],'created_utc':datetime.now(timezone.utc).isoformat(),
        'config':config,'data_version':previous['data_version'],'n':len(y),'input_hashes_before':before,'input_hashes_after':after,
        'reserved_test_predictions':0,'diagnostic_design':'看到前一轮外层结果后提出的流水线修正，仍在相同外层折评估，存在适应性选择偏差，非预注册独立验证',
        'no_label_or_sample_changes':True,'prediction_clipping':False})
