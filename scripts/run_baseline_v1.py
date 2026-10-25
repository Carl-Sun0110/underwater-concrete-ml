import argparse
from concrete_audit.baseline_v1 import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baseline_v1.json');p.add_argument('--output',default='outputs/baseline_v1')
    a=p.parse_args();run(a.config,a.output)
