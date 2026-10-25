import argparse
from concrete_audit.neural_comparison import run

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/parallel_svr_mlp_v1.json')
    parser.add_argument('--output',default='outputs/parallel-svr-mlp-v1');args=parser.parse_args()
    run(args.config,args.output)
