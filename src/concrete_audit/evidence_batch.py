"""小批原表证据线索与水泥零值假设：不把算术、标题或模型分数当作原文核实。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request
import urllib.parse
import openpyxl
from openpyxl.utils import get_column_letter
import numpy as np
from .baseline_v1 import read_source
from .baseline_refinement import load_training, matrix, pipeline
from .baselines import grouped_folds, regression_metrics
from .parallel_exploration import numeric, digest, write_json


def flags(raw,notes):
    """只建立待核查线索；不从Cem0或题名自动确认无水泥体系。"""
    out=[]
    if numeric(raw.get('Cem'))==0 and (numeric(raw.get('AlkaliAct')) or 0)>0 and any((numeric(raw.get(f)) or 0)>0 for f in ['GGBS','FA','MK']):
        out.append('水泥零值可能为真实未掺；矿物胶材及碱激发剂正值仅为原表线索，需PDF确认')
    if 'mortar' in str(raw.get('CastMethod','')).lower() or '砂浆' in notes:
        out.append('原表提示砂浆/预置骨料灌浆体系；强度标签代表砂浆还是成型混凝土待确认')
    if '推定' in notes or 'inferred' in notes.lower():
        out.append('原表文献备注明确提示成型环境为推定，不能视为原文已确认水下成型')
    return out


def request_metadata(url,timeout):
    """有限次只读请求；代理拒绝与出版商内容状态分开记，不绕过访问限制。"""
    try:
        with urllib.request.urlopen(url,timeout=timeout) as r:
            content=r.read(1024*1024)
            return {'url':url,'status':'retrieved','http_status':r.status,'resolved_url':r.url,
                    'content_type':r.headers.get('Content-Type'),'bytes_read':len(content),'body_preview':content[:120].decode('utf-8','replace')}
    except Exception as exc:
        return {'url':url,'status':'request_failed','exception_type':type(exc).__name__,'error':str(exc),
                'interpretation':'请求/代理失败，不能据此推断出版商是否开放或DOI是否有效'}


def run(config_path,output):
    config_path,output=Path(config_path),Path(output);config=json.loads(config_path.read_text());output.mkdir(exist_ok=False)
    base=Path(config['baseline']);refine=Path(config['refinement'])
    previous=json.loads((refine/'manifest.json').read_text())
    for p,expected in previous['input_hashes_after'].items():
        if digest(p)!=expected:raise ValueError(f'先前输入变化：{p}')
    paths=[config_path,Path(__file__),Path(config['raw']),refine/'manifest.json',refine/'results.json',
        refine/'training_oof_predictions.json',refine/'cohort_and_training_folds.json',base/'paper_error_analysis.json']
    paths += [Path(p) for p in previous['input_hashes_after']]
    before={str(p.resolve()):digest(p) for p in paths}
    source,dictionary=read_source(config['raw']);byid={r['sample_id']:r for r in source}
    w=openpyxl.load_workbook(config['raw'],read_only=True,data_only=False)
    metadata={}
    for ri,row in enumerate(w['Paper List'].iter_rows(values_only=True),1):
        if row and row[0] in config['papers']:
            metadata[row[0]]={'paper':row[0],'authors':row[2],'title':row[3],'journal':row[5],'doi_as_recorded':row[6],
                'notes_as_recorded':row[8],'paper_list_row':ri,'provenance':'existing_excel_statement_not_independent_PDF_evidence'}
    columns={r[3]:r[0]+1 for r in w['Parameter Explanation'].iter_rows(min_row=2,values_only=True) if isinstance(r[0],int) and r[3]}
    w.close()
    errors=json.loads((base/'paper_error_analysis.json').read_text())
    selected=json.loads((base/'modeling_records.json').read_text());modelids={r['sample_id'] for r in selected}
    local_pdfs=[str(p) for p in Path('/workspace').rglob('*.pdf')]
    packets=[]
    for r in source:
        raw=r['original_values'];paper=raw.get('PDF filename')
        if paper not in config['papers'] or r['sample_id'] not in modelids:continue
        m=metadata[paper];notes=str(m.get('notes_as_recorded') or '')
        fields=['Specimen ID','CastingEnv','CastMethod','CS_28d','CS_Spec','Conversion Factor','Converted CS_28d',
                'Water','Cem','GGBS','FA','MK','AlkaliAct','RS','CA','AWA','w/c','w/b']
        packets.append({'sample_id':r['sample_id'],'source_row':r['source_row'],'paper':paper,
            'recorded_fields':{f:raw.get(f) for f in fields},'pending_flags':flags(raw,notes),
            'excel_cell_refs':{f:f'Data!{get_column_letter(columns[f])}{r["source_row"]}' for f in fields},
            'metadata':m,'evidence_level':'原始Excel线索，未获对应PDF独立核实','formal_status_changed':False,
            'local_corresponding_pdf_found':False})
    # DOI照录用于尝试访问，不擅自纠错或声称已验证。网络失败也保留完整盘点结果。
    urls=['https://api.crossref.org/works?'+urllib.parse.urlencode({'query.title':metadata[config['papers'][0]]['title'],'rows':1}),
          'https://doi.org/'+str(metadata[config['papers'][0]]['doi_as_recorded'])]
    network=[request_metadata(u,config['network_timeout_seconds']) for u in urls]
    write_json(output/'network_attempts.json',network);write_json(output/'record_review_packets.json',packets)
    paper_summary=[]
    for paper,m in metadata.items():
        pp=[p for p in packets if p['paper']==paper];ee=[e for e in errors if e['paper']==paper and e['cohort']=='exploratory_underwater']
        paper_summary.append({**m,'baseline_records':len(pp),'baseline_mean_model_groupCV_MAE_MPa':float(np.mean([e['MAE_MPa'] for e in ee])),
            'source_rows':[p['source_row'] for p in pp],'pending_flags':sorted({f for p in pp for f in p['pending_flags']}),
            'corresponding_pdf_available':False,'reason':'当前11份本地PDF无此对应题名/年份来源，网络访问失败不能补文献证据'})
    write_json(output/'paper_review_summary.json',paper_summary)
    # 单一假设：只针对指定文献且原表同时满足水泥0、激发剂/矿物正值的训练记录。
    rows,groups,split=load_training(base,'exploratory_underwater')
    refined_config=previous['config'];baseline_config=json.loads((base/'manifest.json').read_text())['config']
    candidate_ids=[r['sample_id'] for r in rows if r['paper']==config['zero_hypothesis_paper'] and
        flags(byid[r['sample_id']]['original_values'],'') and numeric(byid[r['sample_id']]['original_values'].get('Cem'))==0]
    y=np.array([r['target'] for r in rows]);folds=grouped_folds(np.zeros((len(y),1)),y,groups,min(refined_config['cv_folds'],len(set(groups))))
    results=[];predictions=[]
    for fs in config['feature_sets']:
        fields=refined_config['features'][fs];baseX=matrix(rows,fields)
        for variant in ['baseline_zero_unknown','hypothesis_cement_zero_recorded']:
            X=baseX.copy()
            if variant.startswith('hypothesis'):
                for i,r in enumerate(rows):
                    if r['sample_id'] in candidate_ids:X[i,fields.index('Cem')]=0.
            for model in config['models']:
                pred=np.full(len(y),np.nan)
                for t,v in folds:
                    pipe=pipeline(model,'median',refined_config,baseline_config);pipe.fit(X[t],y[t]);pred[v]=pipe.predict(X[v])
                gs=[{'group':str(g),**regression_metrics(y[groups==g],pred[groups==g])} for g in sorted(set(groups))]
                name=f'{fs}__{model}__{variant}';result={'experiment':name,'feature_set':fs,'model':model,'variant':variant,
                    **regression_metrics(y,pred),'macro_source_MAE_MPa':float(np.mean([g['MAE_MPa'] for g in gs])),'groups':gs}
                results.append(result)
                for i,r in enumerate(rows):predictions.append({'experiment':name,'sample_id':r['sample_id'],'source_row':r['source_row'],
                    'paper':r['paper'],'group':str(groups[i]),'actual':float(y[i]),'predicted':float(pred[i]),
                    'zero_hypothesis_applied':r['sample_id'] in candidate_ids and variant.startswith('hypothesis')})
                print(f'{name}: MAE={result["MAE_MPa"]:.4f} R2={result["R2"]:.4f} sourceMAE={result["macro_source_MAE_MPa"]:.4f}',flush=True)
    write_json(output/'sensitivity_results.json',results);write_json(output/'training_oof_predictions.json',predictions)
    write_json(output/'training_cohort.json',{'n':len(rows),'source_groups':len(set(groups)),'training_ids':split['train_ids'],
        'heldout_ids':split['test_ids'],'candidate_ids':candidate_ids,'candidate_rows':[byid[i]['source_row'] for i in candidate_ids]})
    after={p:digest(p) for p in before}
    if after!=before:raise RuntimeError('原始输入变更')
    write_json(output/'manifest.json',{'experiment_id':config['experiment_id'],'created_utc':datetime.now(timezone.utc).isoformat(),
        'config':config,'input_hashes_before':before,'input_hashes_after':after,'local_pdf_count':len(local_pdfs),
        'reviewed_baseline_records':len(packets),'papers':len(metadata),'new_pdfs_acquired':0,
        'formal_reliability_changes':0,'labels_changed':False,'heldout_prediction_count':0,
        'zero_hypothesis_count':len(candidate_ids),'limitations':'零值处理仅假设敏感性，未获PDF证实；分组CV仍有跨轮适应性偏差，不按分数认定假设为真。'})
