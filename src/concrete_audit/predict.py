"""读取项目训练的可信本地joblib，将配合比CSV转换为带适用范围标记的探索预测。"""
import csv
from pathlib import Path
import joblib
import numpy as np
from .parallel_exploration import numeric


def predict_csv(model_path,input_path,output_path):
    """只读取模型白名单配合比，忽略其他列；原输入只读，不覆盖输出。"""
    model_path,input_path,output_path=map(Path,(model_path,input_path,output_path))
    if output_path.resolve() in {model_path.resolve(),input_path.resolve()}:raise ValueError('不能覆盖模型或输入')
    if output_path.exists():raise FileExistsError('输出已存在，请使用新文件名')
    bundle=joblib.load(model_path);fields=bundle['features']
    with input_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader);columns=reader.fieldnames or []
    if not rows:raise ValueError('输入CSV没有记录')
    X=np.full((len(rows),len(fields)),np.nan);diagnostics=[]
    for i,r in enumerate(rows):
        missing=[];outside=[];unknown_basis=[]
        for j,field in enumerate(fields):
            value=numeric(r.get(field))
            if value is None or value<=0:missing.append(field);continue
            X[i,j]=value;limits=bundle['training_ranges'][field]
            if limits['min'] is None:unknown_basis.append(field)
            elif value<limits['min'] or value>limits['max']:outside.append(field)
        diagnostics.append((missing,outside,unknown_basis))
    predictions=bundle['pipeline'].predict(X)
    names=['case_id','predicted_Converted_CS_28d_MPa','missing_inputs','outside_training_range',
           'no_observed_training_support','input_status','unused_columns','research_status']
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for i,(r,p,(missing,outside,unknown)) in enumerate(zip(rows,predictions,diagnostics)):
            status=('all_inputs_missing' if len(missing)==len(fields) else
                    'extrapolation_or_unsupported' if outside or unknown else
                    'contains_imputed_inputs' if missing else 'within_univariate_training_ranges')
            writer.writerow({'case_id':r.get('case_id') or str(i+1),'predicted_Converted_CS_28d_MPa':float(p),
                'missing_inputs':'|'.join(missing),'outside_training_range':'|'.join(outside),
                'no_observed_training_support':'|'.join(unknown),'input_status':status,
                'unused_columns':'|'.join(c for c in columns if c not in fields and c!='case_id'),
                'research_status':'exploratory_only;range_check_is_not_reliability_proof;units_not_auto_converted'})
    return len(rows)
