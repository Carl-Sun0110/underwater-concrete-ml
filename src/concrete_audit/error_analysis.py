"""固定训练折外预测的来源误差、折内分布外输入和输入碰撞诊断。"""
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from .baselines import regression_metrics,digest,write_json
from .reliability import number


def input_vector(row):
    """与现有模型相同的两个输入和计算比值，不使用目标。"""
    raw=row['original_values'];water,cement=number(raw['Water']),number(raw['Cem'])
    if water is None or cement is None or water<=0 or cement<=0:raise ValueError('无效核心输入')
    return np.array([water,cement,water/cement])


def domain_flags(vector,reference):
    """只用该外折训练输入定义范围；标记不是删除依据，也不代表外推必错。"""
    vector=np.asarray(vector,dtype=float);reference=np.asarray(reference,dtype=float)
    if reference.ndim!=2 or reference.shape[1]!=len(vector) or len(reference)==0:raise ValueError('参考输入维度不匹配')
    if not np.isfinite(reference).all() or not np.isfinite(vector).all():raise ValueError('输入非有限')
    low,high=reference.min(axis=0),reference.max(axis=0)
    return {'reference_min':low.tolist(),'reference_max':high.tolist(),
            'outside_axes':np.flatnonzero((vector<low-1e-10)|(vector>high+1e-10)).tolist()}


def source_error_table(items):
    """来源SSE占比加总为1，不因大误差少数来源而修改样本。"""
    if not items:raise ValueError('空误差记录')
    total=sum((r['prediction']-r['actual'])**2 for r in items)
    by=defaultdict(list)
    for r in items:by[r['research_group']].append(r)
    result=[]
    for source,rows in by.items():
        y=[r['actual'] for r in rows];pred=[r['prediction'] for r in rows]
        sse=sum((a-b)**2 for a,b in zip(y,pred))
        result.append({'research_group':source,**regression_metrics(y,pred),
                       'SSE':float(sse),'SSE_share':sse/total if total else 0,
                       'input_outside_fold_range':sum(bool(r['outside_input_fields']) for r in rows),
                       'source_rows':[r['source_row'] for r in rows]})
    return sorted(result,key=lambda r:r['SSE'],reverse=True)


def run(labels_path,split_path,old_dir,new_dir,policy_path,raw_path,output):
    labels_path,split_path,old_dir,new_dir,policy_path,raw_path,output=map(Path,(labels_path,split_path,old_dir,new_dir,policy_path,raw_path,output))
    policy=json.loads(policy_path.read_text());split=json.loads(split_path.read_text())
    paths=[labels_path,split_path,policy_path,raw_path,old_dir/'training_oof_predictions.json',new_dir/'training_oof_predictions.json',
           old_dir/'nested_cv_results.json',new_dir/'nested_cv_results.json']
    protected={str(p.resolve()):digest(p) for p in paths}
    for p,h in json.loads((labels_path.parent/'manifest.json').read_text())['inputs_sha256'].items():
        if digest(p)!=h:raise ValueError('上游数据变化')
        protected[p]=h
    allrows={r['sample_id']:r for r in json.loads(labels_path.read_text())}
    train={r['sample_id'] for r in split['rows'] if r['split']=='train'}
    test={r['sample_id'] for r in split['rows'] if r['split']=='test'}
    if train&test:raise ValueError('固定划分重叠')
    predictions=json.loads((old_dir/'training_oof_predictions.json').read_text())+json.loads((new_dir/'training_oof_predictions.json').read_text())
    results=json.loads((old_dir/'nested_cv_results.json').read_text())+json.loads((new_dir/'nested_cv_results.json').read_text())
    rows_out=[];source_tables={};case_tables={};model_metrics={}
    for model in policy['models']:
        matched=[r for r in results if r['model']==model and r['feature_set']==policy['feature_set']]
        if len(matched)!=1:raise ValueError('模型结果有歧义')
        info=matched[0];fold_by_id={sid:f for f in info['folds'] for sid in f['validation_sample_ids']}
        if set(fold_by_id)!=train:raise ValueError('外层折范围不符')
        model_pred=[r for r in predictions if r['model']==model and r['feature_set']==policy['feature_set']]
        if len(model_pred)!=len(train) or {r['sample_id'] for r in model_pred}!=train:raise ValueError('训练预测覆盖不符')
        references={}
        for f in info['folds']:
            validation=set(f['validation_sample_ids']);fit_ids=train-validation
            references[f['fold']]=np.array([input_vector(allrows[sid]) for sid in sorted(fit_ids)])
        diagnostics=[]
        for p in model_pred:
            sid=p['sample_id']
            if sid in test:raise ValueError('错误分析包含测试记录')
            r=allrows[sid];fold=fold_by_id[sid];vector=input_vector(r)
            actual=r['Provisional_Converted_CS_28d'];pred=p['oof_prediction_MPa']
            if abs(actual-p['actual_provisional_MPa'])>1e-9:raise ValueError('目标映射不一致')
            flags=domain_flags(vector,references[fold['fold']])
            fit_ids=train-set(fold['validation_sample_ids'])
            fit_y=[allrows[x]['Provisional_Converted_CS_28d'] for x in fit_ids]
            raw=r['original_values']
            diagnostics.append({'sample_id':sid,'source_row':r['source_row'],'research_group':raw['Research Group'],
               'pdf_filename':raw['PDF filename'],'specimen_id':raw['Specimen ID'],'model':model,'outer_fold':fold['fold'],
               'actual':actual,'prediction':pred,'signed_error_mpa':pred-actual,'absolute_error_mpa':abs(pred-actual),
               'large_error_diagnostic_only':abs(pred-actual)>policy['absolute_error_flag_mpa'],
               'outside_input_fields':[policy['domain_features'][j] for j in flags['outside_axes']],
               'fold_training_input_range':{f:[flags['reference_min'][j],flags['reference_max'][j]] for j,f in enumerate(policy['domain_features'])},
               'input_values':dict(zip(policy['domain_features'],vector.tolist())),
               'fold_training_target_range':[min(fit_y),max(fit_y)],
               'actual_outside_fold_training_target_range_posthoc':actual<min(fit_y) or actual>max(fit_y),
               'material_values_unverified':{f:raw.get(f) for f in policy['inspect_material_fields']},
               'reliability_status_unchanged':r['prior_reliability_status'],
               'interpretation':'误差和分布诊断，不判定原始数据错误，不改变可信状态'})
        rows_out+=diagnostics
        source_tables[model]=source_error_table(diagnostics)
        case_tables[model]=sorted(diagnostics,key=lambda r:r['absolute_error_mpa'],reverse=True)[:policy['top_cases_per_model']]
        model_metrics[model]={'n':len(diagnostics),'large_errors_over_20':sum(r['large_error_diagnostic_only'] for r in diagnostics),
            'outside_fold_input_range':sum(bool(r['outside_input_fields']) for r in diagnostics),
            'max_error_mpa':max(r['absolute_error_mpa'] for r in diagnostics)}
    # 相同输入但不同目标：只在同一外折验证范围内比较，避免跨模型拟合的歧义。
    ref=next(r for r in results if r['model']=='xgb' and r['feature_set']==policy['feature_set'])
    by=defaultdict(list)
    for f in ref['folds']:
        for sid in f['validation_sample_ids']:
            v=input_vector(allrows[sid]);by[(f['fold'],v[0],v[1])].append(allrows[sid])
    collisions=[]
    for (fold,water,cement),items in by.items():
        targets=[r['Provisional_Converted_CS_28d'] for r in items]
        if len(items)>1 and max(targets)-min(targets)>1e-9:
            varying={}
            for field in policy['inspect_material_fields']:
                values={str(r['original_values'].get(field)) for r in items}
                if len(values)>1:varying[field]=sorted(values)
            collisions.append({'outer_fold':fold,'Water':water,'Cem':cement,'records':len(items),
                'target_min':min(targets),'target_max':max(targets),'target_span':max(targets)-min(targets),
                'source_rows':[r['source_row'] for r in items],
                'research_groups':sorted({r['original_values']['Research Group'] for r in items}),
                'varying_recorded_materials_unverified':varying,'diagnostic_only':True})
    # 题录只标为数据库元信息，不能冒充已读取这些论文全文。
    import openpyxl
    wb=openpyxl.load_workbook(raw_path,read_only=True,data_only=False)
    source_meta={}
    for rr in wb['Paper List'].iter_rows(min_row=2,values_only=True):
        if rr[0]:source_meta[rr[0]]={'title_recorded':rr[3],'doi_recorded':rr[6],'existing_notes':rr[8],
                                  'provenance':'原版Paper List，未在本轮独立核实全文'}
    wb.close()
    priority_sources=[]
    xgb_by_source={r['research_group']:r for r in source_tables['xgb']}
    mlp_by_source={r['research_group']:r for r in source_tables['mlp']}
    prioritized=sorted(xgb_by_source,key=lambda source:max(xgb_by_source[source]['SSE_share'],mlp_by_source[source]['SSE_share']),reverse=True)
    for source in prioritized[:5]:
        item=xgb_by_source[source]
        related=next(r for r in rows_out if r['research_group']==item['research_group'])
        filename=related['pdf_filename'];provided=any(p.name==filename for p in Path('/workspace/attachments').rglob('*.pdf'))
        priority_sources.append({**item,'pdf_filename':filename,**source_meta.get(filename,{}),'matching_pdf_present':provided,
            'mlp_error_metrics':mlp_by_source[item['research_group']]})
    summary={'training_samples':len(train),'protected_test_samples':len(test),'test_evaluated':False,
             'large_error_threshold_mpa':policy['absolute_error_flag_mpa'],'per_model':model_metrics,
             'same_inputs_different_target_groups':len(collisions),
             'records_in_same_input_groups':sum(r['records'] for r in collisions),
             'removed_records':0,'retrained_models':False,'new_trusted_samples':0,
             'inputs_unchanged':all(digest(Path(k))==h for k,h in protected.items())}
    if not summary['inputs_unchanged']:raise RuntimeError('输入变化')
    output.mkdir(parents=True,exist_ok=False)
    for filename,obj in [('diagnostics.json',rows_out),('source_error_contributions.json',source_tables),
                         ('largest_errors.json',case_tables),('same_input_collisions.json',collisions),
                         ('priority_sources.json',priority_sources),('summary.json',summary),
                         ('manifest.json',{'inputs_sha256':protected,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                                           'source_code_sha256':digest(Path(__file__))})]:write_json(output/filename,obj)
    return summary
