"""运行特征与零值策略分析，不训练。"""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from concrete_audit.feature_plan import run
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--labels',type=Path,default=ROOT/'outputs/provisional-labels-v1/exploratory_labels.json')
 p.add_argument('--policy',type=Path,default=ROOT/'configs/feature_policy.json')
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();print(json.dumps(run(a.labels,a.policy,a.output),ensure_ascii=False,indent=2))
