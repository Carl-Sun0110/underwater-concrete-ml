"""生成独立探索性强度标签，不训练模型。"""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from concrete_audit.provisional import run
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--inputs',type=Path,default=ROOT/'configs/provisional_inputs.json')
 p.add_argument('--policy',type=Path,default=ROOT/'configs/provisional_strength_policy.json')
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();print(json.dumps(run(a.inputs,a.policy,a.output),ensure_ascii=False,indent=2))
