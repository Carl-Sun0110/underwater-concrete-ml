"""特征白名单、未知值视图和样本保留敏感性；不拟合预处理器或模型。"""
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
from statistics import median
from .reliability import number

# 物理含义明确的候选名称集合；准许分析不等于值已证实可靠。
CANDIDATE_INPUTS={'Water','Cem','SF','FA','GGBS','MK','NP','AH','RS','SS','CA','SCA',
                  'SP','AWA','Retarder','Emulsion','StF','SyF','w/c','w/b'}


def validate_features(fields):
    """只允许明确列出的配合比字段；目标、强度派生及身份信息无法进入X。"""
    if len(fields)!=len(set(fields)) or not fields:raise ValueError('特征为空或重复')
    prohibited=set(fields)-CANDIDATE_INPUTS
    if prohibited:raise ValueError(f'非白名单或泄漏字段：{sorted(prohibited)}')


def diagnostic_value(raw):
    """输出独立视图；0仅标未知，不宣称是真缺失；不做任何填补。"""
    n=number(raw)
    if n is None:return None,'非数值/缺失'
    if n<0:return None,'负值待核查'
    if n==0:return None,'零值语义未知'
    return n,None


def scenario(rows,fields,threshold,require_core=True):
    """仅按预定字段规则统计保留数量；不根据目标值或得分选择样本。"""
    validate_features(fields)
    if not 0<=threshold<=1:raise ValueError('缺失比例阈值不在[0,1]')
    kept=[]
    for r in rows:
        if require_core and not r['core_positive']:continue
        missing=sum(r['diagnostic_values'].get(f) is None for f in fields)
        if missing/len(fields)<=threshold+1e-12:kept.append(r)
    ids=[r['sample_id'] for r in kept]
    if len(set(ids))!=len(ids):raise ValueError('样本ID重复')
    groups=Counter(r['research_group'] for r in kept)
    return {'threshold':threshold,'denominator_features':len(fields),'kept':len(kept),
            'excluded_from_458':len(rows)-len(kept),'groups':len(groups),'source_distribution':dict(groups),
            'sample_ids':ids}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(labels_path,policy_path,output):
    labels_path,policy_path,output=map(Path,(labels_path,policy_path,output))
    before={str(p.resolve()):digest(p) for p in [labels_path,policy_path]}
    policy=json.loads(policy_path.read_text());source=json.loads(labels_path.read_text())
    if policy['imputation'] is not None or policy['scaling'] is not None or policy['model_training']:
        raise ValueError('本阶段禁止填补、缩放及训练')
    if policy['selected_missing_threshold'] is not None:raise ValueError('本轮不选定缺失阈值')
    core=policy['core_features'];expanded=policy['expanded_features']
    ratios=policy['optional_ratio_features']
    for fields in [core,expanded,ratios,*policy['diagnostic_positive_sets'].values()]:validate_features(fields)
    all_fields=sorted(set(expanded+ratios+[f for fs in policy['diagnostic_positive_sets'].values() for f in fs]))
    rows=[]
    for src in source:
        raw=src['original_values'];values={};reasons={}
        for f in all_fields:
            values[f],why=diagnostic_value(raw.get(f))
            if why:reasons[f]=why
        # 目标及所有派生标签只存在源审计表；X诊断视图不复制目标。
        r={'sample_id':src['sample_id'],'source_row':src['source_row'],
           'research_group':raw['Research Group'],'pdf_filename':raw['PDF filename'],
           'raw_feature_values':{f:raw.get(f) for f in all_fields},'diagnostic_values':values,
           'unknown_reasons':reasons,'core_positive':all(values[f] is not None for f in core),
           'original_reliability_status':src['prior_reliability_status'],
           'interpretation':'探索性诊断，正数未自动核实，零值未知未确认是缺失'}
        rows.append(r)
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('源样本ID重复')
    scenarios={}
    for name,fields in policy['diagnostic_positive_sets'].items():
        scenarios[name]=scenario(rows,fields,0,False)
    thresholds=[scenario(rows,expanded,t) for t in policy['missing_fraction_thresholds']]
    profiles=[]
    for field in expanded+ratios:
        c=Counter(r['unknown_reasons'].get(field,'正值，来源待核查') for r in rows)
        profiles.append({'field':field,'counts':dict(c),'positive_is_not_verified':True})
    summary={'source_records':len(rows),'core_positive_records':scenarios['core_positive']['kept'],
             'core_unknown_or_invalid_records':len(rows)-scenarios['core_positive']['kept'],
             'positive_scenario_counts':{k:v['kept'] for k,v in scenarios.items()},
             'expanded_features':expanded,'threshold_counts':[{k:r[k] for k in ('threshold','kept','groups')} for r in thresholds],
             'missing_threshold_selected':False,'imputation_fitted':False,'model_trained':False,
             'new_trusted_samples':0,'inputs_unchanged':all(digest(Path(k))==v for k,v in before.items())}
    if not summary['inputs_unchanged']:raise RuntimeError('输入文件改变')
    output.mkdir(parents=True,exist_ok=False)
    for name,obj in [('diagnostic_views.json',rows),('scenarios.json',scenarios),('threshold_sensitivity.json',thresholds),
                     ('field_status_profiles.json',profiles),('summary.json',summary),('feature_policy_snapshot.json',policy),
                     ('manifest.json',{'inputs_sha256':before,'script_sha256':digest(Path(__file__)),
                                       'created_at_utc':datetime.now(timezone.utc).isoformat()})]:
        (output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    return summary
