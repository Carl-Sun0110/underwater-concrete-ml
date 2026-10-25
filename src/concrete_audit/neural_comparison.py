"""SVR/带Dropout的MLP，同一AB队列、外层折及内层来源隔离。"""
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .advanced_models import TorchDropoutRegressor
from .baselines import grouped_folds, regression_metrics
from .nested_exploration import derive_ratios, load_development
from .parallel_exploration import digest, feature_matrix, write_json


def search_model(X,y,groups,model,config):
    """输入缩放和目标缩放均限每个内层训练折；输出自动还原到MPa。"""
    if model=='svr':
        estimator=SVR(kernel='rbf',epsilon=.1)
    elif model=='mlp':
        estimator=TorchDropoutRegressor(epochs=config['mlp_epochs'],dropout=config['mlp_dropout'],
            learning_rate=.003,random_state=config['seed'],torch_threads=1)
    else:raise ValueError('只接受SVR或MLP')
    target=TransformedTargetRegressor(regressor=estimator,transformer=StandardScaler())
    pipe=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                   ('scale',StandardScaler()),('model',target)])
    folds=grouped_folds(X,y,groups,config['inner_folds'])
    search=GridSearchCV(pipe,config['grids'][model],cv=folds,scoring=config['selection_scoring'],
                        n_jobs=1,error_score='raise',refit=True)
    search.fit(X,y)
    return search,folds


def run(config_path,output):
    config_path,output=Path(config_path),Path(output);config=json.loads(config_path.read_text())
    previous,meta,rows,groups,y,folds=load_development(config['baseline_run'])
    output.mkdir(parents=True,exist_ok=False)
    baseline=Path(config['baseline_run']);comparison=Path(config['comparison_run'])
    paths=[config_path,Path(__file__),Path('src/concrete_audit/advanced_models.py'),
        Path('src/concrete_audit/nested_exploration.py'),Path('src/concrete_audit/baselines.py'),
        baseline/'manifest.json',baseline/'development_folds.json',baseline/'quality_snapshot.json',
        comparison/'manifest.json',comparison/'results.json',comparison/'oof_predictions.json']
    paths += [Path(p) for p in previous['input_hashes_after']]
    before={str(p.resolve()):digest(p) for p in paths}
    # 比较之前也核对上轮来源输入版本，禁止不知情地拿不同数据版本作算法比较。
    previous_comparison=json.loads((comparison/'manifest.json').read_text())
    for path,expected in previous_comparison['input_hashes_after'].items():
        if digest(path)!=expected:raise ValueError(f'上一轮输入变更：{path}')
    fields=previous['config']['feature_sets']['materials']
    matrices={'materials':(feature_matrix(rows,fields),fields),'materials_plus_ratios':derive_ratios(rows,fields,config)}
    write_json(output/'cohort_and_folds.json',[{**r,'outer_fold':next(k for k,_,v in folds if i in v)} for i,r in enumerate(meta)])
    results=[];predictions=[];traces=[]
    for fs in config['feature_sets']:
        X,names=matrices[fs]
        for model in config['models']:
            name=f'{model}__{fs}__nested';pred=np.full(len(y),np.nan);fold_results=[]
            for k,t,v in folds:
                search,inner=search_model(X[t],y[t],groups[t],model,config)
                pred[v]=search.predict(X[v]);score=regression_metrics(y[v],pred[v])
                fold_results.append({'outer_fold':k,**score})
                traces.append({'experiment':name,'outer_fold':k,'best_params':search.best_params_,
                    'best_inner_mean_fold_MAE_MPa':float(-search.best_score_),
                    'outer_train_groups':sorted(set(groups[t])),'outer_validation_groups':sorted(set(groups[v])),
                    'input_scaler_training_mean':search.best_estimator_['scale'].mean_.tolist(),
                    'target_scaler_training_mean':search.best_estimator_['model'].transformer_.mean_.tolist(),
                    'inner_folds':[{'train_groups':sorted(set(groups[t][it])),
                                    'validation_groups':sorted(set(groups[t][iv]))} for it,iv in inner],
                    'candidates':[{'params':p,'mean_inner_MAE_MPa':float(-s)} for p,s in zip(
                        search.cv_results_['params'],search.cv_results_['mean_test_score'])]})
                print(f'{name} fold={k} MAE={score["MAE_MPa"]:.4f} params={search.best_params_}',flush=True)
            if not np.isfinite(pred).all():raise ValueError('折外预测不完整/非有限')
            group_scores=[{'group':str(g),**regression_metrics(y[groups==g],pred[groups==g])} for g in sorted(set(groups))]
            result={'experiment':name,'model':model,'feature_set':fs,**regression_metrics(y,pred),
                'macro_group_MAE_MPa':float(np.mean([g['MAE_MPa'] for g in group_scores])),
                'negative_predictions':int((pred<0).sum()),'groups':group_scores,'folds':fold_results}
            results.append(result)
            for i,r in enumerate(rows):
                predictions.append({'experiment':name,'sample_id':r['sample_id'],'source_row':r['source_row'],
                    'paper':meta[i]['paper'],'group':str(groups[i]),'grade':meta[i]['grade'],
                    'outer_fold':next(k for k,_,v in folds if i in v),'observed':float(y[i]),
                    'predicted':float(pred[i]),'absolute_error':float(abs(y[i]-pred[i]))})
            write_json(output/'partial_results.json',results)
            print(f'{name} ALL MAE={result["MAE_MPa"]:.4f} RMSE={result["RMSE_MPa"]:.4f} R2={result["R2"]:.4f}',flush=True)
    write_json(output/'results.json',results);write_json(output/'oof_predictions.json',predictions)
    write_json(output/'inner_search_traces.json',traces)
    # 保存本批最优候选作为可复现演示，不覆盖先前参考模型。
    best=min(results,key=lambda r:r['MAE_MPa']);X,names=matrices[best['feature_set']]
    search,_=search_model(X,y,groups,best['model'],config)
    joblib.dump({'pipeline':search.best_estimator_,'base_features':fields,'features':names,
                'feature_set':best['feature_set'],'ratio_config':config,'exploratory_only':True},output/'exploratory_model.joblib')
    after={p:digest(p) for p in before}
    if before!=after:raise RuntimeError('输入哈希变化')
    import torch
    write_json(output/'manifest.json',{'experiment_id':config['experiment_id'],'created_utc':datetime.now(timezone.utc).isoformat(),
        'data_version':previous['data_version'],'config':config,'n':len(y),'groups':len(set(groups)),
        'input_hashes_before':before,'input_hashes_after':after,'sklearn':sklearn.__version__,'torch':torch.__version__,
        'numpy':np.__version__,'best_batch_experiment':best['experiment'],'final_inner_params':search.best_params_,
        'target_scaling':'训练折内TransformedTargetRegressor+StandardScaler；预测还原MPa',
        'MLP':'CPU固定150轮（以配置为准），Dropout与AdamW；无使用外层验证集的early stopping',
        'clipping':'不裁剪负值或大误差预测，完整报告','reserved_test_predictions':0,
        'limitations':['多轮探索开发集仍有适应性偏差，非独立最终测试','材料比值仅在单位可比假设下解释',
                       'B级暂定标签不等于正式已核实；原始及可靠主档未变']})
    lines=['# SVR/MLP同折嵌套对照','',f'相同AB开发{len(y)}条，五个外层论文组折，内层三折。',
        '每模型/特征四组参数。输入插补、输入标准化和目标标准化仅训练折拟合；MLP包含Dropout，固定轮数，不使用外层验证目标早停。','',
        '|模型|特征|MAE MPa|RMSE MPa|R²|来源组平均MAE|负预测数|','|---|---|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f'|{r["model"]}|{r["feature_set"]}|{r["MAE_MPa"]:.3f}|{r["RMSE_MPa"]:.3f}|{r["R2"]:.3f}|{r["macro_group_MAE_MPa"]:.3f}|{r["negative_predictions"]}|')
    lines+=['','原始Excel、标签及145条历史留出均未参与修改或新评估。所有预测原样保留，负预测不截成0。',
        'search_model建立训练折内插补、输入/目标缩放和内层调参；run锁定旧样本/折并保存折外预测、选参证据与输入哈希。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
