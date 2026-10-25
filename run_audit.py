"""从显式配置运行审计；不修改配置中指向的输入文件。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from concrete_audit.audit import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'configs/inputs.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    print(json.dumps(run(config['raw'],config['screening'],args.output,
                         config.get('evidence',[]),config.get('decisions')),ensure_ascii=False,indent=2))
