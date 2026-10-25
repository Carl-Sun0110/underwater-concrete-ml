"""按原始文件哈希定位材料，生成新的本地配置，不改历史配置/清单哈希。"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(repo,materials_root=None,output_dir=None):
    repo=Path(repo).resolve();materials_root=Path(materials_root or repo/'data/raw/attachments').resolve()
    output_dir=Path(output_dir or repo/'configs/local').resolve()
    if output_dir.exists():raise FileExistsError('本地配置目录已存在；请用--output-dir指定新目录，避免覆盖')
    manifest=json.loads((repo/'migration/materials_manifest.json').read_text())
    available={}
    if materials_root.exists():
        for p in materials_root.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.pdf','.xlsx','.xls','.txt'}:
                available.setdefault(sha(p),[]).append(p)
    matches=[];mapping={}
    for item in manifest['files']:
        paths=sorted(available.get(item['sha256'],[]));path=paths[0] if paths else None
        matches.append({'file':item['display_name'],'sha256':item['sha256'],
                        'status':'hash_matched' if path else 'missing','local_path':str(path) if path else None})
        if path:mapping[item['cloud_path']]=str(path)
    cloud=json.loads((repo/'configs/baseline_v1.json').read_text())
    scope=repo/'migration/scope_registry_snapshot.json'
    baseline_ready=cloud['raw'] in mapping and scope.is_file()
    output_dir.mkdir(parents=True,exist_ok=False)
    if baseline_ready:
        local=dict(cloud);local['raw']=mapping[cloud['raw']];local['scope_registry']=str(scope)
        local['experiment_id']='baseline_v1_desktop_local'
        local['historical_note']=cloud['historical_note']+'；本地配置由原始字节哈希定位生成，历史结果未重签。'
        (output_dir/'baseline_v1.json').write_text(json.dumps(local,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        refinement=json.loads((repo/'configs/baseline_v1_1.json').read_text())
        refinement['baseline']=str(repo/'outputs/desktop_baseline_v1')
        refinement['experiment_id']='baseline_v1_1_desktop_local'
        (output_dir/'baseline_v1_1.json').write_text(json.dumps(refinement,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report={'repository':str(repo),'materials_root':str(materials_root),'materials':matches,
        'prediction_ready':(repo/'models/exploratory_v1_1.joblib').is_file(),'baseline_ready':baseline_ready,
        'local_configs_created':baseline_ready,'historical_configs_modified':False,'historical_hashes_rewritten':False,
        'missing_requested_original_workbooks':manifest['missing_requested_original_workbooks'],
        'note':'legacy历史脚本仍含云端路径，应从本地新配置重新执行V1/V1.1，不能批量重写旧清单来伪称复现成功。'}
    (output_dir/'desktop_setup_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--materials-root');parser.add_argument('--output-dir');args=parser.parse_args()
    report=prepare(args.repo,args.materials_root,args.output_dir)
    print(json.dumps({'prediction_ready':report['prediction_ready'],'baseline_ready':report['baseline_ready'],
        'materials_found':sum(r['status']=='hash_matched' for r in report['materials']),
        'materials_total':len(report['materials'])},ensure_ascii=False))
