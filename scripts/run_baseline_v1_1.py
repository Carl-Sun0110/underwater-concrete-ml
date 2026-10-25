import argparse
from concrete_audit.baseline_refinement import run

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/baseline_v1_1.json')
    parser.add_argument('--output',default='outputs/baseline_v1_1')
    args=parser.parse_args();run(args.config,args.output)
