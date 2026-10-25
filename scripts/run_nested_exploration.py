import argparse
from concrete_audit.nested_exploration import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='第二轮：嵌套论文分组CV与配比特征')
    parser.add_argument('--config',default='configs/parallel_nested_v2.json')
    parser.add_argument('--output',default='outputs/parallel-nested-v2')
    args=parser.parse_args();run(args.config,args.output)
