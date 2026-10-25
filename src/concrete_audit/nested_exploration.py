"""同一AB队列的嵌套分组调参：外层只评估，内层拟合预处理和选择参数。"""
import csv
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .baselines import grouped_folds, regression_metrics
from .parallel_exploration import ALLOWED, digest, feature_matrix, write_json


class SklearnCatBoost(RegressorMixin, CatBoostRegressor, BaseEstimator):
    """补充sklearn 1.8要求的回归器标签；训练/预测仍由原生CatBoost执行。"""


def derive_ratios(rows, fields, config):
    """仅对已观察到的正分子/分母作比；缺失与未知零值不先补齐再作比。"""
    numerators=config['ratio_numerators'];denominator=config['ratio_denominator']
    if denominator!='Cem' or not set(numerators)<=ALLOWED:
        raise ValueError('配比派生必须来自白名单材料字段，以水泥为分母')
    base=feature_matrix(rows,fields)
    ratio_inputs=feature_matrix(rows,numerators+[denominator])
    ratios=np.full((len(rows),len(numerators)),np.nan)
    for j in range(len(numerators)):
        known=np.isfinite(ratio_inputs[:,j])&np.isfinite(ratio_inputs[:,-1])
        ratios[known,j]=ratio_inputs[known,j]/ratio_inputs[known,-1]
    return np.column_stack([base,ratios]),fields+[f'{f}_over_{denominator}' for f in numerators]


def estimator(model, seed):
    """限制树深、固定随机种子与单线程；CatBoost不向当前目录写训练日志。"""
    if model=='rf':
        return RandomForestRegressor(random_state=seed,n_jobs=1)
    if model=='xgb':
        return XGBRegressor(learning_rate=.04,subsample=.85,colsample_bytree=.85,reg_lambda=5,
                            objective='reg:squarederror',random_state=seed,n_jobs=1)
    if model=='catboost':
        return SklearnCatBoost(learning_rate=.04,loss_function='RMSE',random_seed=seed,
                              thread_count=1,verbose=False,allow_writing_files=False)
    raise ValueError('未知模型')


def tune(X,y,groups,model,config):
    """GridSearch只接收当前外层训练数据；插补器也在每个内层训练折重拟合。"""
    folds=grouped_folds(X,y,groups,config['inner_folds'])
    pipe=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                   ('model',estimator(model,config['seed']))])
    search=GridSearchCV(pipe,config['grids'][model],cv=folds,scoring=config['selection_scoring'],
                        n_jobs=config['n_jobs'],error_score='raise',refit=True,return_train_score=False)
    search.fit(X,y)
    return search,folds


def load_development(baseline):
    """从前次快照读取同一批AB和相同外层折，保留C的档案，不重新抽样。"""
    baseline=Path(baseline)
    manifest=json.loads((baseline/'manifest.json').read_text())
    for path,expected in manifest['input_hashes_after'].items():
        if digest(path)!=expected:raise ValueError(f'原实验输入版本已变化：{path}')
    all_rows={r['sample_id']:r for r in json.loads(Path(manifest['config']['labels']).read_text())}
    snapshot=json.loads((baseline/'quality_snapshot.json').read_text())
    dev=[r for r in snapshot if r['split']=='development' and r['grade'] in {'A','B'}]
    fmap={r['sample_id']:r for r in json.loads((baseline/'development_folds.json').read_text())}
    rows=[all_rows[r['sample_id']] for r in dev]
    groups=np.array([r['group'] for r in dev]);fold_ids=np.array([fmap[r['sample_id']]['fold'] for r in dev])
    y=np.array([r['Provisional_Converted_CS_28d'] for r in rows],dtype=float)
    if not np.isfinite(y).all() or (y<=0).any():raise ValueError('标签无效，禁止插补')
    folds=[]
    for k in sorted(set(fold_ids)):
        train,val=np.flatnonzero(fold_ids!=k),np.flatnonzero(fold_ids==k)
        if set(groups[train])&set(groups[val]):raise ValueError('外层论文组泄漏')
        folds.append((int(k),train,val))
    return manifest,dev,rows,groups,y,folds


def run(config_path, output):
    config_path,output=Path(config_path),Path(output)
    config=json.loads(config_path.read_text());baseline=Path(config['baseline_run'])
    previous,meta,rows,groups,y,folds=load_development(baseline)
    output.mkdir(parents=True,exist_ok=False)
    paths=[config_path,Path(__file__),Path('src/concrete_audit/baselines.py'),
           Path('src/concrete_audit/parallel_exploration.py'),baseline/'manifest.json',
           baseline/'quality_snapshot.json',baseline/'development_folds.json',baseline/'results.json',baseline/'oof_predictions.json']
    paths += [Path(p) for p in previous['input_hashes_after']]
    before={str(p.resolve()):digest(p) for p in paths}
    fields=previous['config']['feature_sets']['materials']
    matrices={'materials':(feature_matrix(rows,fields),fields),
              'materials_plus_ratios':derive_ratios(rows,fields,config)}
    write_json(output/'feature_definitions.json',{'base_fields':fields,'ratio_numerators':config['ratio_numerators'],
        'denominator':'Cem','zero_policy':'0含义未明，视为未知；不补标签；仅训练折内补输入',
        'unit_caveat':'原表配比比值，材料质量单位一致时才可视为无量纲；Water可能以L而非kg计，未统一证据的比值仍是探索代理量。',
        'binder_policy':'未将缺失矿物掺合料填0求胶凝总量，未伪造w/b',
        'availability':{name:dict(zip(names,np.isfinite(X).sum(axis=0).tolist())) for name,(X,names) in matrices.items()}})
    write_json(output/'cohort_and_folds.json',[{**r,'outer_fold':next(k for k,_,v in folds if i in v)} for i,r in enumerate(meta)])
    results=[];predictions=[];traces=[]
    for fs in config['feature_sets']:
        X,names=matrices[fs]
        for model in config['models']:
            experiment=f'{model}__{fs}__nested'
            pred=np.full(len(rows),np.nan);fold_results=[]
            for k,train,val in folds:
                search,inner=tune(X[train],y[train],groups[train],model,config)
                pred[val]=search.predict(X[val])
                detail={'experiment':experiment,'outer_fold':k,'best_params':search.best_params_,
                        'best_inner_mean_fold_MAE_MPa':float(-search.best_score_),
                        'outer_train_groups':sorted(set(groups[train])),'outer_validation_groups':sorted(set(groups[val])),
                        'inner_folds':[{'train_groups':sorted(set(groups[train][t])),
                                        'validation_groups':sorted(set(groups[train][v]))} for t,v in inner],
                        'candidates':[{'params':p,'mean_inner_MAE_MPa':float(-score)} for p,score in zip(
                            search.cv_results_['params'],search.cv_results_['mean_test_score'])]}
                traces.append(detail)
                scores=regression_metrics(y[val],pred[val]);fold_results.append({'outer_fold':k,**scores})
                print(f'{experiment} fold={k} MAE={scores["MAE_MPa"]:.4f} params={search.best_params_}',flush=True)
            if not np.isfinite(pred).all():raise ValueError('折外预测不完整')
            gs=[{'group':str(g),**regression_metrics(y[groups==g],pred[groups==g])} for g in sorted(set(groups))]
            result={'experiment':experiment,'model':model,'feature_set':fs,**regression_metrics(y,pred),
                    'macro_group_MAE_MPa':float(np.mean([r['MAE_MPa'] for r in gs])),
                    'groups':gs,'folds':fold_results}
            results.append(result)
            for i,r in enumerate(rows):
                predictions.append({'experiment':experiment,'sample_id':r['sample_id'],'source_row':r['source_row'],
                    'paper':meta[i]['paper'],'group':str(groups[i]),'grade':meta[i]['grade'],
                    'outer_fold':next(k for k,_,v in folds if i in v),'observed':float(y[i]),
                    'predicted':float(pred[i]),'absolute_error':float(abs(y[i]-pred[i]))})
            print(f'{experiment} ALL MAE={result["MAE_MPa"]:.4f} RMSE={result["RMSE_MPa"]:.4f} R2={result["R2"]:.4f}',flush=True)
            write_json(output/'partial_results.json',results)
    write_json(output/'results.json',results);write_json(output/'oof_predictions.json',predictions)
    write_json(output/'inner_search_traces.json',traces)
    old=json.loads((baseline/'results.json').read_text())
    comparison=[]
    for r in results:
        same=next((s for s in old if s['experiment']==f'AB__materials__median__{r["model"]}__unknown'),None)
        comparison.append({'experiment':r['experiment'],'n':r['n'],'MAE_MPa':r['MAE_MPa'],
            'RMSE_MPa':r['RMSE_MPa'],'R2':r['R2'],'macro_group_MAE_MPa':r['macro_group_MAE_MPa'],
            'prior_fixed_materials_MAE_MPa':same['MAE_MPa'] if same else None,
            'MAE_difference_from_prior_fixed_materials':r['MAE_MPa']-same['MAE_MPa'] if same else None})
    write_json(output/'paired_baseline_comparison.json',comparison)
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    best=min(results,key=lambda r:r['MAE_MPa'])
    X,names=matrices[best['feature_set']]
    final,final_inner=tune(X,y,groups,best['model'],config)
    joblib.dump({'pipeline':final.best_estimator_,'base_features':fields,'features':names,
                 'feature_set':best['feature_set'],'ratio_config':config,'experiment':best['experiment'],
                 'exploratory_only':True},output/'exploratory_model.joblib')
    pp=[p for p in predictions if p['experiment']==best['experiment']]
    bypaper=defaultdict(list)
    for p in pp:bypaper[p['paper']].append(p)
    priority=[]
    for paper,rr in bypaper.items():
        priority.append({'paper':paper,'n':len(rr),'MAE_MPa':float(np.mean([r['absolute_error'] for r in rr])),
                         'bias_MPa':float(np.mean([r['predicted']-r['observed'] for r in rr])),
                         'source_rows':[r['source_row'] for r in rr],
                         'review_focus':['测量标签和换算适用性','材料单位及绝对/相对用量','体系差异与训练覆盖范围'],
                         'action':'核查线索，不按误差自动删除或更改标签'})
    priority.sort(key=lambda p:p['MAE_MPa'],reverse=True);write_json(output/'paper_review_priorities.json',priority)
    after={p:digest(p) for p in before}
    if before!=after:raise RuntimeError('输入发生变化')
    manifest={'experiment_id':config['experiment_id'],'created_utc':datetime.now(timezone.utc).isoformat(),
        'config':config,'data_version':previous['data_version'],'n':len(rows),'groups':len(set(groups)),
        'outer_folds_reused':True,'n_outer_folds':len(folds),'reserved_test_predictions':0,
        'input_hashes_before':before,'input_hashes_after':after,'best_experiment':best['experiment'],
        'final_development_inner_params':final.best_params_,
        'final_fit_note':'外层评估完成后，在全部AB开发数据中用内层分组CV选择演示模型参数；不影响已有折外预测',
        'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,
        'catboost':__import__('catboost').__version__,'xgboost':__import__('xgboost').__version__,
        'limitations':['探索性暂定标签；未升级正式可靠状态','同一开发数据已被多轮探索使用，嵌套CV不能消除跨轮适应性选模偏差',
                      '本轮未评估历史留出数据；选出六组最优者的分数仍可能有选择偏差',
                      '配比比值的严格无量纲解释取决于来源单位证据；不把缺失矿物成分强行补零求胶凝总量']}
    write_json(output/'manifest.json',manifest)
    lines=['# 第二轮：嵌套分组调参与配比特征对照','',
           f'数据版本{previous["data_version"]}；同一AB开发队列{len(rows)}条、{len(set(groups))}来源连通组，复用上轮五个外层折。',
           '每个外层训练集内部做三折来源分组调参，每模型四组参数；中位数插补和缺失指示器同样只在训练折拟合。',
           '不读历史测试目标拟合、调参或评估；没有删样本或更改标签。原始材料比值仅是单位可比假设下的探索特征。','',
           '|模型|特征|MAE MPa|RMSE MPa|R²|来源组平均MAE|','|---|---|---:|---:|---:|---:|']
    for r in results:lines.append(f'|{r["model"]}|{r["feature_set"]}|{r["MAE_MPa"]:.3f}|{r["RMSE_MPa"]:.3f}|{r["R2"]:.3f}|{r["macro_group_MAE_MPa"]:.3f}|')
    lines+=['',f'本轮最低合并样本MAE：{best["experiment"]}。',
            '对照paired_baseline_comparison.json中的上轮固定参数结果，保留所有不改善的实验。不能预设调参或比值特征必然有收益。',
            '下一步依据相同验证样本的来源误差安排小批文献核查与体系覆盖改进，保留可信高强度样本。',
            '函数说明：derive_ratios只对已观察正数作比；tune只用当前训练来源选参；load_development锁定旧队列/外层折；run输出完整嵌套证据、折外预测和对照。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return manifest
