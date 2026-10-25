"""对探索标签子集作描述性画像，不填补、缩放、筛选或输出训练矩阵。"""
from pathlib import Path
from decimal import Decimal,InvalidOperation
import argparse,json

FEATURES=['Water','Cem','SF','FA','GGBS','MK','NP','AH','GP','QP','LS','StP','RS','MfS','SS',
          'CA','SCA','SP','AWA','Retarder','Emulsion','StF','SyF','w/c','w/b']


def numeric(value):
    if isinstance(value,bool):return None
    try:
        x=Decimal(str(value));return x if x.is_finite() else None
    except (InvalidOperation,ValueError,TypeError):return None


def run(source,output):
    rows=json.loads(source.read_text());profiles=[]
    for field in FEATURES:
        values=[numeric(r['original_values'].get(field)) for r in rows]
        profiles.append({'field':field,'records':len(rows),'numeric':sum(x is not None for x in values),
          'zero_semantics_unresolved':sum(x==0 for x in values),
          'negative':sum(x is not None and x<0 for x in values),
          'non_numeric_or_missing':sum(x is None for x in values),
          'unique_numeric_values':len({x for x in values if x is not None}),
          'treatment':'只作画像，零值未改为真零或缺失，未填补或缩放'})
    with output.open('x',encoding='utf-8') as f:json.dump(profiles,f,ensure_ascii=False,indent=2)
    return profiles

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print('字段画像数：',len(run(args.source,args.output)))
