"""Baseline V1.1：只用固定论文留出划分中的训练样本，比较缺失处理并保存预测模型。"""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted

from .baseline_v1 import model_pipeline
from .baselines import grouped_folds, regression_metrics
from .parallel_exploration import digest, write_json


class ScaledKNNImputer(TransformerMixin,BaseEstimator):
    """距离以训练集观察值标准化；填补后还原原单位，缺失指示器独立保留。"""
    def __init__(self,n_neighbors=5):self.n_neighbors=n_neighbors

    def fit(self,X,y=None):
        X=check_array(X,ensure_all_finite='allow-nan');self.n_features_in_=X.shape[1]
        observed=np.isfinite(X);n=observed.sum(axis=0)
        self.mean_=np.divide(np.where(observed,X,0.).sum(axis=0),n,out=np.zeros(X.shape[1]),where=n>0)
        variance=np.divide(np.where(observed,(X-self.mean_)**2,0.).sum(axis=0),n,out=np.zeros(X.shape[1]),where=n>0)
        self.scale_=np.where(variance>0,np.sqrt(variance),1.)
        self.imputer_=KNNImputer(n_neighbors=self.n_neighbors,weights='distance',add_indicator=True,keep_empty_features=True)
        self.imputer_.fit((X-self.mean_)/self.scale_);return self

    def transform(self,X):
        check_is_fitted(self,'imputer_');X=check_array(X,ensure_all_finite='allow-nan')
        if X.shape[1]!=self.n_features_in_:raise ValueError('输入列数不匹配')
        result=self.imputer_.transform((X-self.mean_)/self.scale_)
        result[:,:self.n_features_in_]=result[:,:self.n_features_in_]*self.scale_+self.mean_
        return result


def load_training(baseline,cohort):
    """只返回Baseline V1论文留出的训练记录；预测文件仅取分组身份元数据。"""
    baseline=Path(baseline)
    split=next(r for r in json.loads((baseline/'split_manifest.json').read_text()) if r['cohort']==cohort and r['protocol']=='paper_group_holdout')
    record_byid={r['sample_id']:r for r in json.loads((baseline/'modeling_records.json').read_text())}
    metadata=json.loads((baseline/'predictions.json').read_text())
    groupmap={r['sample_id']:r['paper_group'] for r in metadata if r['cohort']==cohort and r['protocol']=='paper_group_cv'}
    rows=[record_byid[i] for i in split['train_ids']]
    if set(split['train_ids'])&set(split['test_ids']):raise ValueError('训练测试身份重叠')
    return rows,np.array([groupmap[r['sample_id']] for r in rows]),split


def matrix(rows,fields):
    return np.array([[np.nan if r['features'][f] is None else r['features'][f] for f in fields] for r in rows],float)


def pipeline(model,imputation,config,baseline_config):
    estimator=clone(model_pipeline(model,baseline_config)['model'])
    imp=(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True) if imputation=='median'
         else ScaledKNNImputer(n_neighbors=config['knn_neighbors']))
    return Pipeline([('impute',imp),('model',estimator)])


def run(config_path,output):
    config_path,output=Path(config_path),Path(output);config=json.loads(config_path.read_text())
    baseline=Path(config['baseline']);previous=json.loads((baseline/'manifest.json').read_text())
    for p,expected in previous['input_hashes_after'].items():
        if digest(p)!=expected:raise ValueError(f'Baseline V1输入发生变化：{p}')
    output.mkdir(exist_ok=False)
    paths=[config_path,Path(__file__),baseline/'manifest.json',baseline/'split_manifest.json',
        baseline/'modeling_records.json',baseline/'predictions.json',Path('src/concrete_audit/baseline_v1.py'),
        Path('src/concrete_audit/baselines.py')]+[Path(p) for p in previous['input_hashes_after']]
    before={str(p.resolve()):digest(p) for p in paths}
    log=(output/'training.log').open('x')
    def message(s):print(s,flush=True);log.write(s+'\n');log.flush()
    results=[];predictions=[];cohorts=[];selections=[]
    for cohort in ['primary_direct','exploratory_underwater']:
        rows,groups,split=load_training(baseline,cohort);y=np.array([r['target'] for r in rows])
        folds=grouped_folds(np.zeros((len(y),1)),y,groups,min(config['cv_folds'],len(set(groups))))
        cohorts.append({'cohort':cohort,'train_n':len(y),'training_source_groups':len(set(groups)),
            'cv_folds':len(folds),'heldout_n':len(split['test_ids']),'train_ids':split['train_ids'],
            'heldout_ids':split['test_ids'],'folds':[{'fold':k,'train_ids':[rows[i]['sample_id'] for i in t],
            'validation_ids':[rows[i]['sample_id'] for i in v],'train_groups':sorted(set(groups[t])),
            'validation_groups':sorted(set(groups[v]))} for k,(t,v) in enumerate(folds)]})
        cases=[('core','median','mean')]+[(fs,imp,model) for fs in config['features'] for imp in config['imputations'] for model in config['models']]
        for fs,imp,model in cases:
            fields=config['features'][fs];X=matrix(rows,fields);prediction=np.full(len(y),np.nan);fold_scores=[]
            for k,(t,v) in enumerate(folds):
                pipe=pipeline(model,imp,config,previous['config']);pipe.fit(X[t],y[t]);prediction[v]=pipe.predict(X[v])
                fold_scores.append({'fold':k,**regression_metrics(y[v],prediction[v])})
            gs=[{'group':str(g),**regression_metrics(y[groups==g],prediction[groups==g])} for g in sorted(set(groups))]
            experiment=f'{cohort}__{fs}__{imp}__{model}'
            result={'experiment':experiment,'cohort':cohort,'feature_set':fs,'imputation':imp,'model':model,
                **regression_metrics(y,prediction),'macro_source_MAE_MPa':float(np.mean([s['MAE_MPa'] for s in gs])),
                'groups':gs,'folds':fold_scores,'cv_folds':len(folds)}
            results.append(result)
            for i,r in enumerate(rows):predictions.append({'experiment':experiment,'sample_id':r['sample_id'],
                'source_row':r['source_row'],'paper':r['paper'],'group':str(groups[i]),
                'fold':next(k for k,(_,v) in enumerate(folds) if i in v),'actual':float(y[i]),
                'predicted':float(prediction[i]),'absolute_error':float(abs(y[i]-prediction[i]))})
            message(f'{experiment} n={len(y)} MAE={result["MAE_MPa"]:.4f} R2={result["R2"]:.4f} sourceMAE={result["macro_source_MAE_MPa"]:.4f}')
        eligible=[r for r in results if r['cohort']==cohort and r['model']!='mean']
        chosen=min(eligible,key=lambda r:r[config['selection_metric']]);fields=config['features'][chosen['feature_set']]
        X=matrix(rows,fields);pipe=pipeline(chosen['model'],chosen['imputation'],config,previous['config']);pipe.fit(X,y)
        ranges={}
        for j,f in enumerate(fields):
            known=X[np.isfinite(X[:,j]),j];ranges[f]={'min':float(known.min()) if len(known) else None,
                'max':float(known.max()) if len(known) else None,'observed_training_n':len(known)}
        bundle={'pipeline':pipe,'features':fields,'training_ranges':ranges,'selected_experiment':chosen['experiment'],
            'training_n':len(y),'cohort':cohort,'target':'Converted CS_28d','target_unit':'MPa',
            'zero_policy':'非正值和占位视未知','input_unit_policy':'与原库一致的名义配合比单位；单位问题仍待核，不自动换算L与kg或密度',
            'exploratory_only':True,'selection_metric':config['selection_metric'],'heldout_evaluated_this_round':False}
        model_path=output/f'{cohort}__selected.joblib';joblib.dump(bundle,model_path)
        selections.append({**{k:chosen[k] for k in ['cohort','experiment','feature_set','imputation','model','MAE_MPa','RMSE_MPa','R2','macro_source_MAE_MPa']},
            'model_file':str(model_path),'training_n':len(y),'heldout_n':len(split['test_ids'])})
        # 只从训练记录提取演示输入，不把重预测训练样本当独立测试。
        with (output/f'{cohort}__demo_inputs.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=['case_id']+fields);writer.writeheader()
            for r in rows[:3]:writer.writerow({'case_id':f'existing_training_row_{r["source_row"]}',**{field:r['features'][field] for field in fields}})
    write_json(output/'results.json',results);write_json(output/'training_oof_predictions.json',predictions)
    write_json(output/'cohort_and_training_folds.json',cohorts);write_json(output/'selected_models.json',selections)
    after={p:digest(p) for p in before}
    if before!=after:raise RuntimeError('原始或Baseline V1输入发生变化')
    write_json(output/'manifest.json',{'experiment_id':config['experiment_id'],'created_utc':datetime.now(timezone.utc).isoformat(),
        'config':config,'input_hashes_before':before,'input_hashes_after':after,'heldout_evaluated_this_round':False,
        'new_holdout_prediction_count':0,'limits':config['evaluation_note'],'demo_inputs_are_existing_training_rows':True})
    log.close()
