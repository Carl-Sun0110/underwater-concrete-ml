"""独立探索轨道：保留核查状态，缺失输入在训练折内处理，标签绝不插补。"""
import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .baselines import grouped_folds, leakage_groups, regression_metrics

ALLOWED = {'Water','Cem','SF','FA','GGBS','MK','NP','AH','RS','SS','CA','SCA','SP','AWA'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def numeric(value):
    """只读取有限数字；斜线、文本、空值均为未知。"""
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def feature_matrix(rows, fields, zero_policy='unknown'):
    """不使用强度或试件派生信息；可选材料零值的两种解释作为对照。"""
    if not fields or not set(fields) <= ALLOWED:
        raise ValueError('输入必须属于预先指定的配合比白名单，禁止标签和派生字段')
    if zero_policy not in {'unknown', 'recorded'}:
        raise ValueError('未知零值策略')
    result = np.full((len(rows), len(fields)), np.nan)
    for i, row in enumerate(rows):
        for j, field in enumerate(fields):
            value = numeric(row['original_values'].get(field))
            if value is not None and value >= 0:
                # 水/水泥零值始终不是有效配合比；可选材料零值不擅自认定真实未掺。
                if value > 0 or (zero_policy == 'recorded' and field not in {'Water','Cem'}):
                    result[i,j] = value
    return result


def quality(row, scope, override=None):
    """研究分层独立于正式可靠状态；B表示可探索，不等于已核实。"""
    raw = row['original_values']
    y = numeric(row.get('Provisional_Converted_CS_28d'))
    measured = numeric(raw.get('CS_28d'))
    factor = numeric(raw.get('Conversion Factor'))
    reasons = ['换算系数为用户许可暂定值', '密度假设及部分单位/零值语义仍待核查']
    if not raw.get('PDF filename') or not raw.get('Research Group') or y is None or y <= 0 or measured is None or measured <= 0:
        grade, why = 'D', '缺少明确来源身份或有效测量标签'
    elif factor not in {.95,.8,.76} or not np.isclose(y, measured*factor, rtol=0, atol=1e-8):
        grade, why = 'D', '当前暂定换算规则不支持或计算存在矛盾'
    elif str(scope.get('scope_state','')).startswith('非直接'):
        grade, why = 'C', '冲洗/渗透后重塑试件；目标来源可用，但研究范围解释存在条件'
    else:
        grade, why = 'B', '原表明确列出论文身份和28天测量值，尚未完成关键依据逐项核实'
    if override:
        if override['grade'] not in 'ABCD' or not override.get('reason') or not override.get('evidence'):
            raise ValueError('人工分层变更须有等级、原因和证据记录')
        # 人工分层也不能让不存在/不合法的监督标签变成可训练标签。
        if grade == 'D' and override['grade'] != 'D':
            raise ValueError('先修正有来源的标签版本，不能仅靠等级变更放行D级标签')
        grade, why = override['grade'], override['reason']
    return {'grade':grade,'reason':why,'unresolved':reasons,
            'prior_reliability_status':row.get('prior_reliability_status','未知'),
            'formal_reliability_promoted':False,'manual_override':override}


def allocate(rows, historical, signature_fields):
    """全候选集重建来源/配比连通组；历史测试所在组全部留出，不读取测试目标。"""
    groups, links = leakage_groups(rows, signature_fields)
    old = {r['sample_id']:r for r in historical['rows']}
    reserved = {g for r,g in zip(rows,groups) if old.get(r['sample_id'],{}).get('split') == 'test'}
    splits = ['reserved_historical_test' if g in reserved else 'development' for g in groups]
    return groups, splits, links


def make_pipeline(model, imputation, config):
    """标准化、插补、缺失指示器均只在当前训练折拟合。"""
    imputer = (SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
               if imputation == 'median' else
               KNNImputer(n_neighbors=5, weights='distance', add_indicator=True, keep_empty_features=True))
    if model == 'mean':
        estimator = DummyRegressor(strategy='mean')
    elif model == 'rf':
        estimator = RandomForestRegressor(n_estimators=config['trees'], min_samples_leaf=3,
                                          max_features=1.0, random_state=config['seed'], n_jobs=1)
    elif model == 'xgb':
        estimator = XGBRegressor(n_estimators=config['trees'], max_depth=3, learning_rate=.04,
                                 subsample=.85, colsample_bytree=.85, reg_lambda=5,
                                 objective='reg:squarederror', random_state=config['seed'], n_jobs=1)
    else:
        raise ValueError('未知模型')
    return Pipeline([('scale',StandardScaler()), ('impute',imputer), ('model',estimator)])


def profile(rows, fields):
    """保留原始零值计数，同时报告在两种语义下的可用量，不设统一删除阈值。"""
    result = {}
    for field in fields:
        values = [numeric(r['original_values'].get(field)) for r in rows]
        result[field] = {'n':len(rows),'positive':sum(v is not None and v>0 for v in values),
                         'zero':sum(v==0 for v in values), 'unknown_or_negative':sum(v is None or v<0 for v in values)}
    return result


def run(config_path, output):
    """生成版本快照，按统一分组折运行固定参数对照；不读取留出集标签做评估。"""
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    paths = [config_path,Path(__file__),*[Path(config[k]) for k in
             ('labels','historical_split','scope_registry','raw_inputs','signature_config')]]
    raw_inputs = json.loads(Path(config['raw_inputs']).read_text())
    paths += [Path(p) for p in raw_inputs.values() if isinstance(p,str) and Path(p).is_file()]
    before = {str(p.resolve()):digest(p) for p in paths}
    rows = json.loads(Path(config['labels']).read_text())
    scope = {r['sample_id']:r for r in json.loads(Path(config['scope_registry']).read_text())['records']}
    overrides = {r['sample_id']:r for r in config['quality_overrides']}
    if set(overrides)-{r['sample_id'] for r in rows}:
        raise ValueError('人工变更记录未对应当前数据版本')
    grades = [quality(r,scope.get(r['sample_id'],{}),overrides.get(r['sample_id'])) for r in rows]
    historical = json.loads(Path(config['historical_split']).read_text())
    signatures = json.loads(Path(config['signature_config']).read_text())['group_signature_fields']
    groups, splits, links = allocate(rows,historical,signatures)
    snapshot = [{**{k:r[k] for k in ['sample_id','source_row']},'paper':r['original_values']['PDF filename'],
                 'research_group':r['original_values']['Research Group'], 'group':str(g),'split':s,
                 **q, 'label_basis':r.get('strength_basis_status','未知')}
                for r,g,s,q in zip(rows,groups,splits,grades)]
    write_json(output/'quality_snapshot.json',snapshot)
    dev_indices = [i for i,(s,q) in enumerate(zip(splits,grades)) if s=='development' and q['grade']!='D']
    dev = [rows[i] for i in dev_indices]
    dev_groups = groups[dev_indices]
    dev_grades = np.array([grades[i]['grade'] for i in dev_indices])
    y = np.array([r['Provisional_Converted_CS_28d'] for r in dev],dtype=float)
    if not np.isfinite(y).all() or (y<=0).any():
        raise ValueError('监督标签缺失或无效，绝不插补标签')
    folds = grouped_folds(np.zeros((len(dev),1)),y,dev_groups,config['folds'])
    fold_ids = np.full(len(dev),-1)
    for k,(_,val) in enumerate(folds): fold_ids[val] = k
    write_json(output/'development_folds.json',[{'sample_id':r['sample_id'],'group':str(g),'fold':int(f)}
               for r,g,f in zip(dev,dev_groups,fold_ids)])
    fields = list(dict.fromkeys(f for fs in config['feature_sets'].values() for f in fs))
    write_json(output/'feature_availability.json',{'all_candidates':profile(rows,fields),'development':profile(dev,fields)})
    specs = []
    for cohort in ['AB','ABC']:
        specs.append((cohort,'core','median','mean','unknown'))
        for fs in config['feature_sets']:
            for imp in ['median','knn']:
                for model in ['rf','xgb']:
                    specs.append((cohort,fs,imp,model,'unknown'))
        # 零值照录仅是语义敏感性实验，不声称所有零值都是真实未掺。
        specs.append((cohort,'admixtures_sensitivity','median','rf','recorded'))
    results, predictions = [], []
    for cohort,fs,imp,model,zero in specs:
        name = '__'.join([cohort,fs,imp,model,zero])
        mask = np.isin(dev_grades,list(cohort))
        X = feature_matrix(dev,config['feature_sets'][fs],zero)
        pred = np.full(len(dev),np.nan)
        fold_scores = []
        for k,(train,val) in enumerate(folds):
            train, val = train[mask[train]], val[mask[val]]
            if len(train)==0 or len(val)==0: continue
            pipe = make_pipeline(model,imp,config)
            pipe.fit(X[train],y[train])
            pred[val] = pipe.predict(X[val])
            fold_scores.append({'fold':k,**regression_metrics(y[val],pred[val]),
                                'train_n':len(train),'train_groups':sorted(set(dev_groups[train])),
                                'validation_groups':sorted(set(dev_groups[val]))})
        if not np.isfinite(pred[mask]).all(): raise ValueError('交叉验证预测未覆盖所有候选')
        group_scores = [{'group':str(g),**regression_metrics(y[mask&(dev_groups==g)],pred[mask&(dev_groups==g)])}
                        for g in sorted(set(dev_groups[mask]))]
        shared = mask & np.isin(dev_grades,['A','B'])
        result = {'experiment':name,'cohort':cohort,'feature_set':fs,'imputation':imp,'model':model,
                  'zero_policy':zero,**regression_metrics(y[mask],pred[mask]),
                  'shared_AB_metrics':regression_metrics(y[shared],pred[shared]),
                  'macro_group_MAE_MPa':float(np.mean([s['MAE_MPa'] for s in group_scores])),
                  'groups':group_scores,'folds':fold_scores}
        results.append(result)
        for i in np.flatnonzero(mask):
            predictions.append({'experiment':name,'sample_id':dev[i]['sample_id'], 'source_row':dev[i]['source_row'],
                                'paper':dev[i]['original_values']['PDF filename'],'group':str(dev_groups[i]),
                                'grade':str(dev_grades[i]),'fold':int(fold_ids[i]),
                                'observed':float(y[i]),'predicted':float(pred[i]),'absolute_error':float(abs(y[i]-pred[i]))})
        print(f"{name}: n={result['n']} MAE={result['MAE_MPa']:.4f} RMSE={result['RMSE_MPa']:.4f} R2={result['R2']:.4f}",flush=True)
    write_json(output/'results.json',results)
    write_json(output/'oof_predictions.json',predictions)
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        keys=['experiment','n','MAE_MPa','RMSE_MPa','R2','macro_group_MAE_MPa']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    # 预先规定主要候选：AB+材料质量字段+未知零值；混合单位外加剂仅作敏感性分析。
    primary = [r for r in results if r['cohort']=='AB' and r['feature_set'] in {'core','materials'} and r['model']!='mean']
    best = min(primary,key=lambda r:r['MAE_MPa'])
    mask = np.isin(dev_grades,['A','B'])
    X = feature_matrix(dev,config['feature_sets'][best['feature_set']])
    importance = []
    for k,(train,val) in enumerate(folds):
        train,val=train[mask[train]],val[mask[val]]
        if not len(val):continue
        pipe=make_pipeline(best['model'],best['imputation'],config);pipe.fit(X[train],y[train])
        perm=permutation_importance(pipe,X[val],y[val],scoring='neg_mean_absolute_error',
                                    n_repeats=5,random_state=config['seed'],n_jobs=1)
        importance.append({'fold':k,'validation_n':len(val),'MAE_increase_by_feature':dict(zip(
            config['feature_sets'][best['feature_set']],perm.importances_mean.tolist()))})
    write_json(output/'validation_permutation_importance.json',importance)
    final=make_pipeline(best['model'],best['imputation'],config);final.fit(X[mask],y[mask])
    joblib.dump({'pipeline':final,'features':config['feature_sets'][best['feature_set']],
                 'experiment':best['experiment'],'zero_policy':'unknown','exploratory_only':True},output/'exploratory_model.joblib')
    worst=sorted([p for p in predictions if p['experiment']==best['experiment']],key=lambda p:p['absolute_error'],reverse=True)
    write_json(output/'review_priorities.json',worst[:30])
    after={p:digest(p) for p in before}
    if after!=before:raise RuntimeError('受保护输入发生变化')
    manifest={'experiment_id':config['experiment_id'],'created_utc':datetime.now(timezone.utc).isoformat(),
              'data_version':before[str(Path(config['labels']).resolve())], 'input_hashes_before':before,
              'input_hashes_after':after,'config':config,'python':platform.python_version(),'sklearn':sklearn.__version__,
              'numpy':np.__version__,'xgboost':__import__('xgboost').__version__,
              'candidate_n':len(rows),'development_n':len(dev),'reserved_n':splits.count('reserved_historical_test'),
              'candidate_grades':dict(Counter(q['grade'] for q in grades)),
              'development_grades':dict(Counter(dev_grades.tolist())), 'group_links':links,
              'selected_primary_experiment':best['experiment'],'selection_bias':'选模和报告同一组CV，分数可能乐观；非最终独立测试结论',
              'hyperparameter_search':'本轮固定参数，无搜索；下一轮仅在训练折内调参',
              'historical_test_note':'历史测试曾在早期流程使用过；本轮不读取其目标进行拟合、选模或评估',
              'all_empty_feature_handling':'保留列并由插补器填0，仅为数值占位，缺失指示器保留，非材料真实为0',
              'grouping_limit':'共享论文/研究组或完整原表配比签名合组；未证明能识别所有跨文献近似重复',
              'formal_reliability_promoted':False}
    write_json(output/'manifest.json',manifest)
    lines=['# 双轨探索基线报告','','本报告全部为探索性结果；原始标签按用户许可暂定系数换算，不代表正式可信数据库。',
           f"候选{len(rows)}条，开发{len(dev)}条，历史留出{manifest['reserved_n']}条；开发等级{manifest['development_grades']}。",
           '同一套来源分组五折；AB与ABC使用共同的分组折，shared_AB_metrics可用于相同AB验证样本上的比较。',
           '中位数和标准化KNN插补均限训练折。保留所有有有效标签的开发记录，不按缺失率或高强度删样本。',
           '材料零值默认未知；零值照录另做敏感性实验。SP/AWA混合单位只做探索对照，不能作可迁移物理定量解释。','',
           '|实验|n|MAE MPa|RMSE MPa|R²|论文组平均MAE|','|---|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"|{r['experiment']}|{r['n']}|{r['MAE_MPa']:.3f}|{r['RMSE_MPa']:.3f}|{r['R2']:.3f}|{r['macro_group_MAE_MPa']:.3f}|")
    lines += ['',f"主要候选（不含混合单位外加剂）最低CV MAE：{best['experiment']}。",
              '已保存折外误差、核查优先记录、验证折置换重要性及开发集拟合模型。选模后的CV分数有选择偏差；尚无新独立测试结论。',
              '下一轮：针对最高折外误差论文核查单位/范围，补充无量纲配合比；内层分组调参，再做版本间同折比较。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return manifest
