"""运行只读范围审查，输出独立视图及隔离清单。"""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.scope_gate import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(ROOT / 'outputs/provisional-labels-v1/exploratory_labels.json',
                         ROOT / 'outputs/exploratory-baselines-v2/split_manifest.json',
                         ROOT / 'configs/casting_scope_evidence.json',
                         ROOT / 'outputs/train-only-comparison-v1/nested_cv_results.json', args.output), ensure_ascii=False, indent=2))
