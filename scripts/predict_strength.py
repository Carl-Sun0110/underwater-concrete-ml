"""PYTHONPATH=src python scripts/predict_strength.py --model ... --input ... --output ..."""
import argparse
from concrete_audit.predict import predict_csv

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(f'已输出{predict_csv(a.model,a.input,a.output)}条探索预测：{a.output}')
