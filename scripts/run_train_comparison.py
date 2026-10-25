"""仅用已有训练部分进行模型比较，测试集保持封存。"""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from concrete_audit.train_compare import run
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--labels',type=Path,default=ROOT/'outputs/provisional-labels-v1/exploratory_labels.json')
 p.add_argument('--split',type=Path,default=ROOT/'outputs/exploratory-baselines-v2/split_manifest.json')
 p.add_argument('--config',type=Path,default=ROOT/'configs/train_only_comparison.json')
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();print(json.dumps(run(a.labels,a.split,a.config,a.output),ensure_ascii=False,indent=2))
