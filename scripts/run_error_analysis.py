"""仅训练部分的折外误差诊断入口。"""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from concrete_audit.error_analysis import run
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cfg=json.loads((ROOT/'configs/provisional_inputs.json').read_text())
 print(json.dumps(run(ROOT/'outputs/provisional-labels-v1/exploratory_labels.json',
   ROOT/'outputs/exploratory-baselines-v2/split_manifest.json',ROOT/'outputs/train-only-comparison-v1',
   ROOT/'outputs/advanced-train-only-v1',ROOT/'configs/error_analysis.json',cfg['raw'],a.output),ensure_ascii=False,indent=2))
