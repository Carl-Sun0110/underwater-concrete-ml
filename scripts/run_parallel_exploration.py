"""在仓库根目录运行：PYTHONPATH=src python scripts/run_parallel_exploration.py。"""
import argparse
from concrete_audit.parallel_exploration import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='核查与建模并行：独立探索基线')
    parser.add_argument('--config',default='configs/parallel_exploration_v1.json')
    parser.add_argument('--output',default='outputs/parallel-exploration-v1')
    args=parser.parse_args()
    run(args.config,args.output)
