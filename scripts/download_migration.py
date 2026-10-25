"""通过Git取迁移包，绕开聊天文件下载；恢复历史/材料，不覆盖本地代码或改动文件。"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def restore_archive(archive_path,repo,kind):
    """只恢复outputs或原始材料目录，校验内容清单，不改Git跟踪代码与旧哈希。"""
    repo=Path(repo).resolve();counts={'created':0,'same_content_skipped':0,'conflicts':[]}
    prefix='outputs/' if kind=='project_state' else 'data/raw/attachments/'
    with zipfile.ZipFile(archive_path) as archive:
        contents=json.loads(archive.read('MIGRATION_CONTENTS.json'))['files']
        for name,info in contents.items():
            if not name.startswith(prefix):continue
            relative=PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('归档含非法路径')
            dest=(repo/Path(*relative.parts)).resolve()
            if not dest.is_relative_to(repo):raise ValueError('目标超出仓库')
            if dest.exists():
                if dest.is_file() and digest(dest)==info['sha256']:counts['same_content_skipped']+=1
                else:counts['conflicts'].append(name)
                continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dest.parent,prefix='.migration-',delete=False) as f:
                temporary=Path(f.name)
                with archive.open(name) as source:shutil.copyfileobj(source,f)
            try:
                if digest(temporary)!=info['sha256']:raise ValueError('解压内容哈希不匹配：'+name)
                temporary.replace(dest);counts['created']+=1
            finally:
                if temporary.exists():temporary.unlink()
    return counts


def download(repo,extract=False,branch='desktop-transfer-20261025'):
    repo=Path(repo).resolve();index=json.loads((repo/'migration/archive_index.json').read_text())
    out=repo/'migration_archives';out.mkdir(exist_ok=True)
    needed=[item for item in index['archives'] if not (out/item['file']).is_file() or digest(out/item['file'])!=item['sha256']]
    revision=None
    if needed:
        subprocess.run(['git','fetch','--no-tags','origin',branch],cwd=repo,check=True,timeout=300)
        revision=subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=repo,text=True).strip()
    report={'transfer_branch':branch,'fetched_revision':revision,'archives':[],'extracted':extract,'historical_hashes_rewritten':False}
    for item in index['archives']:
        name=item['file'];dest=out/name
        if '/' in name or '\\' in name or name in {'.','..'}:raise ValueError('非法归档文件名')
        if dest.exists() and digest(dest)!=item['sha256']:
            raise ValueError('已有同名迁移包哈希不同，保留该文件：'+str(dest))
        if not dest.exists():
            with tempfile.NamedTemporaryFile(dir=out,prefix='.download-',delete=False) as f:
                temporary=Path(f.name)
            try:
                with temporary.open('wb') as f:
                    subprocess.run(['git','show',revision+':migration_downloads/'+name],cwd=repo,stdout=f,check=True,timeout=120)
                if digest(temporary)!=item['sha256']:raise ValueError('Git下载内容哈希不匹配：'+name)
                temporary.replace(dest)
            finally:
                temporary.unlink(missing_ok=True)
        row={'file':name,'verified':True,'bytes':dest.stat().st_size}
        if extract:row['restore']=restore_archive(dest,repo,item['kind'])
        report['archives'].append(row)
        print('已校验并'+('恢复：' if extract else '下载：')+name,flush=True)
    (out/'download_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--extract',action='store_true')
    a=p.parse_args();r=download(a.repo,a.extract)
    conflicts=sum(len(x.get('restore',{}).get('conflicts',[])) for x in r['archives'])
    print(f'完成{len(r["archives"])}个包；保留本地不同内容文件{conflicts}个。接着运行scripts/prepare_desktop.py。')
