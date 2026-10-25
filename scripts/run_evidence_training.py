"""运行固定严格范围的证据输入扩展比较，不评估测试集。"""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.evidence_train import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/fine_ratio_train_only_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(ROOT / 'outputs/provisional-labels-v1/exploratory_labels.json',
                         ROOT / 'outputs/exploratory-baselines-v2/split_manifest.json', args.config, args.output), ensure_ascii=False, indent=2))
