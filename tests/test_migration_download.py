import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec=importlib.util.spec_from_file_location('download_migration',Path(__file__).resolve().parents[1]/'scripts/download_migration.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class RestoreArchiveTests(unittest.TestCase):
    def test_restores_history_but_does_not_overwrite_code_or_different_content(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);repo=root/'repo';repo.mkdir();(repo/'README.md').write_text('current')
            (repo/'outputs').mkdir();(repo/'outputs/existing.json').write_text('local-edited')
            files={'README.md':b'old-code','outputs/new.json':b'new','outputs/existing.json':b'old'}
            archive=root/'test.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for name,data in files.items():z.writestr(name,data)
                z.writestr('MIGRATION_CONTENTS.json',json.dumps({'files':{name:{'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()}}))
            result=module.restore_archive(archive,repo,'project_state')
            self.assertEqual((repo/'README.md').read_text(),'current');self.assertEqual((repo/'outputs/existing.json').read_text(),'local-edited')
            self.assertEqual((repo/'outputs/new.json').read_text(),'new');self.assertEqual(result['conflicts'],['outputs/existing.json'])

    def test_rejects_traversal_even_under_allowed_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive=root/'test.zip';name='outputs/../../escape.txt'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr(name,b'x');z.writestr('MIGRATION_CONTENTS.json',json.dumps({'files':{name:{'sha256':hashlib.sha256(b'x').hexdigest()}}}))
            with self.assertRaises(ValueError):module.restore_archive(archive,root,'project_state')


if __name__=='__main__':unittest.main()
