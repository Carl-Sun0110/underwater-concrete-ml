"""复现材料语义候选层；不训练，不覆盖任何输入。"""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.material_features import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT / 'outputs/provisional-labels-v1/exploratory_labels.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/material_feature_evidence.json')
    parser.add_argument('--split', type=Path, default=ROOT / 'outputs/exploratory-baselines-v2/split_manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.labels, args.config, args.split, args.output), ensure_ascii=False, indent=2))
