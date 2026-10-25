"""探索性基线运行入口，不执行四算法搜索。"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from concrete_audit.baselines import run
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--labels',type=Path,default=ROOT/'outputs/provisional-labels-v1/exploratory_labels.json')
 p.add_argument('--config',type=Path,default=ROOT/'configs/baseline_experiment.json')
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();print(json.dumps(run(a.labels,a.config,a.output),ensure_ascii=False,indent=2))
