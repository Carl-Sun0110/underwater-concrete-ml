"""运行本地证据核查试点；不需要外部API。"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from concrete_audit.field_review import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/pilot_evidence.json')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(run(a.config,a.output),ensure_ascii=False,indent=2))
