"""生成私下转移的完整状态及分片原始材料ZIP，不把原始PDF/Excel推到Git。"""
import hashlib
import json
from pathlib import Path
import zipfile


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_archive(target,entries):
    files={}
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for source,name in entries:
            archive.write(source,name);files[name]={'sha256':digest(source),'bytes':source.stat().st_size}
        archive.writestr('MIGRATION_CONTENTS.json',json.dumps({'files':files,'note':'按条目SHA256核对；原始文件字节未改。'},ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name,value in files.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==value['sha256']
    return {'file':target.name,'sha256':digest(target),'bytes':target.stat().st_size,'entries':len(files),'verified':True}


def build():
    repo=Path(__file__).resolve().parents[1];out=repo/'migration_archives';out.mkdir(exist_ok=False)
    materials=json.loads((repo/'migration/materials_manifest.json').read_text())
    # 状态包保留代码、原有配置及所有历史输出；不包含.git、环境、原始附件或其他本地秘密文件。
    entries=[]
    for p in sorted(repo.rglob('*')):
        relative=p.relative_to(repo)
        if not p.is_file() or any(part in {'.git','__pycache__','.pytest_cache','migration_archives'} or part.startswith('.venv') for part in relative.parts):continue
        if p.name.startswith('.env'):continue
        if relative.parts[:2]==('data','raw'):continue
        if relative.parts[0]=='configs' and len(relative.parts)>1 and relative.parts[1].startswith('local'):continue
        if str(relative)=='migration/archive_index.json':continue
        entries.append((p,relative.as_posix()))
    archives=[{'kind':'project_state',**make_archive(out/'project_state.zip',entries)}]
    limit=28*1024*1024;groups=[]
    for item in sorted(materials['files'],key=lambda r:-r['bytes']):
        if item['bytes']>limit:raise ValueError('单个原始材料超过28MiB分片预算')
        dest=next((g for g in groups if g['bytes']+item['bytes']<=limit),None)
        if dest is None:dest={'bytes':0,'files':[]};groups.append(dest)
        dest['files'].append(item);dest['bytes']+=item['bytes']
    for i,group in enumerate(groups,1):
        entries=[(Path(r['cloud_path']),r['desktop_relative_path']) for r in group['files']]
        for path,name in entries:
            item=next(r for r in group['files'] if r['cloud_path']==str(path))
            if digest(path)!=item['sha256']:raise ValueError('原始字节变化')
        archives.append({'kind':'original_materials','part':i,**make_archive(out/f'original_materials_{i:02}.zip',entries)})
    if any(a['bytes']>32*1024*1024 for a in archives):raise ValueError('归档超过32MiB下载分片预算')
    index={'archives':archives,'original_materials_count':len(materials['files']),
        'instructions':'首次克隆后把所有ZIP解压到仓库根目录，材料在data/raw/attachments，历史在outputs；不批量重写历史哈希。',
        'raw_materials_committed_to_git':False}
    (repo/'migration/archive_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'archive_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(archives,ensure_ascii=False))


if __name__=='__main__':build()
