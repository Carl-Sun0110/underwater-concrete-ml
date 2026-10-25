"""固定外层队列：来源组等权选参与文献输入版本的2×2对照。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from .baselines import grouped_folds, regression_metrics
from .nested_exploration import estimator, load_development
from .parallel_exploration import digest, feature_matrix, write_json

SOURCE_FIELDS=['source_water_cement_ratio','source_fa_cement_ratio','source_sf_cement_ratio',
               'source_ggbs_cement_ratio','source_fineagg_cement_ratio','source_water_defined_components_ratio']


def source_ratios(rows, annotations, sand_annotations):
    """仅同单位、正值且来源定义明确的数值作比；未知成分不补0求胶材和。"""
    byrow={a['source_row']:a for a in annotations};sand={a['source_row']:a for a in sand_annotations}
    values=np.full((len(rows),len(SOURCE_FIELDS)),np.nan)
    for i,row in enumerate(rows):
        a=byrow.get(row['source_row'])
        if a:
            cem=a['components']['Cem'];water=a.get('water')
            for j,component in enumerate(['water','FA','SF','GGBS']):
                entry=water if component=='water' else a['components'].get(component)
                if entry and entry['value'] is not None and cem['value'] and entry['value']>0 and entry['unit']==cem['unit']:
                    values[i,j]=entry['value']/cem['value']
            defined=a.get('defined_binder_components')
            if defined and water:
                parts=[a['components'][f] for f in defined]
                if all(p['value'] is not None and p['value']>0 and p['unit']==water['unit'] for p in parts):
                    values[i,5]=water['value']/sum(p['value'] for p in parts)
        s=sand.get(row['source_row'])
        if s and s['numerator_unit']==s['denominator_unit'] and s['numerator']>0 and s['denominator']>0:
            values[i,4]=s['numerator']/s['denominator']
    return values


def versioned_inputs(rows,fields,evidence,sand):
    """只生成新矩阵；单位敏感性设置未知，不改变原表、标签或样本数量。"""
    recorded=feature_matrix(rows,fields);masked=recorded.copy();changes=[]
    annotations={a['source_row']:a for a in evidence['annotations']}
    for i,row in enumerate(rows):
        a=annotations.get(row['source_row'])
        if not a:continue
        relative=a['components']['Cem']['unit']=='relative_cement_mass'
        water=a.get('water')
        for j,field in enumerate(fields):
            reason=None
            if relative and np.isfinite(recorded[i,j]):
                reason='原文按相对水泥质量报告；现有绝对kg/m³基准未核实'
            elif field=='Water' and water and water['unit']=='L/m3' and np.isfinite(recorded[i,j]):
                reason='原文明示L/m³；本敏感性版本不假定密度将其作为kg/m³'
            if reason:
                masked[i,j]=np.nan
                changes.append({'sample_id':row['sample_id'],'source_row':row['source_row'],
                    'field':field,'recorded_value':float(recorded[i,j]),'new_value':None,'reason':reason,
                    'evidence_paper':a['paper'],'evidence_pages':sorted({e['page'] for e in a['evidence']})})
    ratios=source_ratios(rows,evidence['annotations'],sand['annotations'])
    versions={'recorded':(recorded,fields),'unit_mask':(masked,fields),
              'recorded_plus_source_ratios':(np.column_stack([recorded,ratios]),fields+SOURCE_FIELDS),
              'unit_mask_plus_source_ratios':(np.column_stack([masked,ratios]),fields+SOURCE_FIELDS)}
    return versions,changes


def selection_scores(y,pred,groups,fold_indices):
    """旧目标为等权折MAE；新目标为每个来源连通组MAE等权，不给大组更多票。"""
    errors=np.abs(np.asarray(y)-np.asarray(pred))
    if not np.isfinite(errors).all():raise ValueError('内层预测不完整')
    return {'fold_mean':float(np.mean([errors[v].mean() for _,v in fold_indices])),
            'source_macro':float(np.mean([errors[groups==g].mean() for g in sorted(set(groups))])),
            'pooled_sample':float(errors.mean())}


def paired_search(X,y,groups,model,config):
    """相同内层预测同时计算两个选参目标，避免额外搜索和随机差异。"""
    folds=grouped_folds(X,y,groups,config['inner_folds'])
    base=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                   ('model',estimator(model,config['seed']))])
    candidates=[]
    for params in ParameterGrid(config['grids'][model]):
        pred=np.full(len(y),np.nan)
        for t,v in folds:
            pipe=clone(base).set_params(**params);pipe.fit(X[t],y[t]);pred[v]=pipe.predict(X[v])
        candidates.append({'params':params,'inner_scores':selection_scores(y,pred,groups,folds)})
    chosen={objective:min(candidates,key=lambda c:c['inner_scores'][objective]) for objective in config['objectives']}
    return base,folds,candidates,chosen


def run(config_path,output):
    config_path,output=Path(config_path),Path(output);config=json.loads(config_path.read_text())
    previous,meta,rows,groups,y,folds=load_development(config['baseline_run'])
    output.mkdir(exist_ok=False)
    evidence=json.loads(Path(config['evidence_config']).read_text());sand=json.loads(Path(config['sand_evidence']).read_text())
    paths=[config_path,Path(__file__),Path('src/concrete_audit/nested_exploration.py'),Path('src/concrete_audit/baselines.py'),
        Path(config['evidence_config']),Path(config['sand_evidence']),Path(config['comparison_run'])/'results.json',
        Path(config['comparison_run'])/'oof_predictions.json',Path(config['baseline_run'])/'manifest.json',
        Path(config['baseline_run'])/'quality_snapshot.json',Path(config['baseline_run'])/'development_folds.json']
    paths += [Path(p) for p in previous['input_hashes_after']]
    for paper in evidence['papers'].values():
        p=Path(paper['path'])
        if digest(p)!=paper['sha256']:raise ValueError('证据PDF版本变化')
        paths.append(p)
    before={str(p.resolve()):digest(p) for p in paths}
    fields=previous['config']['feature_sets']['materials'];versions,changes=versioned_inputs(rows,fields,evidence,sand)
    write_json(output/'input_changes.json',changes)
    write_json(output/'feature_versions.json',{v:{'fields':names,'observed_n_by_field':dict(zip(names,np.isfinite(X).sum(axis=0).tolist())),
        'feature_matrix_sha256':__import__('hashlib').sha256(np.nan_to_num(X,nan=-1.).astype('<f8').tobytes()).hexdigest()}
        for v,(X,names) in versions.items()})
    write_json(output/'cohort_and_folds.json',[{**m,'outer_fold':next(k for k,_,v in folds if i in v)} for i,m in enumerate(meta)])
    results=[];predictions=[];traces=[]
    for version in config['data_versions']:
        X,names=versions[version]
        for model in config['models']:
            oof={objective:np.full(len(y),np.nan) for objective in config['objectives']}
            scores={objective:[] for objective in config['objectives']}
            for k,t,v in folds:
                base,inner,candidates,chosen=paired_search(X[t],y[t],groups[t],model,config)
                for objective in config['objectives']:
                    pipe=clone(base).set_params(**chosen[objective]['params']);pipe.fit(X[t],y[t])
                    p=pipe.predict(X[v]);oof[objective][v]=p;scores[objective].append({'outer_fold':k,**regression_metrics(y[v],p)})
                    traces.append({'version':version,'model':model,'objective':objective,'outer_fold':k,
                        'best_params':chosen[objective]['params'],'candidates':candidates,
                        'outer_train_groups':sorted(set(groups[t])),'outer_validation_groups':sorted(set(groups[v])),
                        'inner_folds':[{'train_groups':sorted(set(groups[t][it])),
                                        'validation_groups':sorted(set(groups[t][iv]))} for it,iv in inner]})
            for objective,pred in oof.items():
                name=f'{model}__{version}__{objective}'
                gs=[{'group':str(g),**regression_metrics(y[groups==g],pred[groups==g])} for g in sorted(set(groups))]
                result={'experiment':name,'model':model,'version':version,'objective':objective,
                    **regression_metrics(y,pred),'macro_group_MAE_MPa':float(np.mean([r['MAE_MPa'] for r in gs])),
                    'worst_group_MAE_MPa':max(r['MAE_MPa'] for r in gs),'groups':gs,'folds':scores[objective]}
                results.append(result)
                for i,row in enumerate(rows):predictions.append({'experiment':name,'sample_id':row['sample_id'],
                    'source_row':row['source_row'],'paper':meta[i]['paper'],'group':str(groups[i]),
                    'outer_fold':next(k for k,_,v in folds if i in v),'observed':float(y[i]),
                    'predicted':float(pred[i]),'absolute_error':float(abs(y[i]-pred[i]))})
                print(f'{name}: MAE={result["MAE_MPa"]:.4f} RMSE={result["RMSE_MPa"]:.4f} R2={result["R2"]:.4f} macro={result["macro_group_MAE_MPa"]:.4f}',flush=True)
            write_json(output/'partial_results.json',results)
    write_json(output/'results.json',results);write_json(output/'oof_predictions.json',predictions);write_json(output/'inner_search_traces.json',traces)
    contrasts=[]
    def pair(kind,a,b):
        aa={g['group']:g['MAE_MPa'] for g in a['groups']};bb={g['group']:g['MAE_MPa'] for g in b['groups']}
        contrasts.append({'contrast':kind,'before':a['experiment'],'after':b['experiment'],
            'delta_sample_MAE_MPa':b['MAE_MPa']-a['MAE_MPa'],
            'delta_source_macro_MAE_MPa':b['macro_group_MAE_MPa']-a['macro_group_MAE_MPa'],
            'source_groups_improved':sum(bb[g]<aa[g] for g in aa),'source_groups_worsened':sum(bb[g]>aa[g] for g in aa),
            'per_group_MAE_delta':{g:bb[g]-aa[g] for g in aa}})
    index={(r['model'],r['version'],r['objective']):r for r in results}
    for model in config['models']:
        for version in config['data_versions']:pair('来源等权选参',index[model,version,'fold_mean'],index[model,version,'source_macro'])
        for objective in config['objectives']:
            pair('单位证据处理_原字段',index[model,'recorded',objective],index[model,'unit_mask',objective])
            pair('单位证据处理_含来源比值',index[model,'recorded_plus_source_ratios',objective],index[model,'unit_mask_plus_source_ratios',objective])
            pair('增加来源比值_原字段',index[model,'recorded',objective],index[model,'recorded_plus_source_ratios',objective])
            pair('增加来源比值_单位处理后',index[model,'unit_mask',objective],index[model,'unit_mask_plus_source_ratios',objective])
    write_json(output/'paired_contrasts.json',contrasts)
    after={p:digest(p) for p in before}
    if before!=after:raise RuntimeError('原始或版本输入发生变化')
    write_json(output/'manifest.json',{'experiment_id':config['experiment_id'],'created_utc':datetime.now(timezone.utc).isoformat(),
        'config':config,'data_version':previous['data_version'],'n':len(y),'groups':len(set(groups)),
        'input_hashes_before':before,'input_hashes_after':after,'reserved_test_predictions':0,
        'changed_cells_in_sensitivity_version':len(changes),'changed_rows':len({r['sample_id'] for r in changes}),
        'formal_reliability_changed':False,'labels_changed':False,
        'selection_target':'来源宏平均MAE改变选参目标，不给训练样本加权；来源是原有防泄漏连通组，Nasr两论文合组。',
        'evidence_limit':'单位处理是敏感性版本，不宣称所有其余原始单位已核实；有矛盾的胶材分项比例按定义保留标记，不纠正原表。',
        'exploration_limit':'多轮复用开发集且文献优先级受既往误差启发，仍有适应性偏差；不是最终独立测试。'})
