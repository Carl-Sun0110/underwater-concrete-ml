import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('prepare_desktop',Path(__file__).resolve().parents[1]/'scripts/prepare_desktop.py')
setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)


class DesktopSetupTests(unittest.TestCase):
    def test_hash_lookup_generates_new_configs_without_touching_history(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);(repo/'migration').mkdir();(repo/'configs').mkdir();(repo/'models').mkdir()
            material=repo/'data/raw/attachments/renamed.xlsx';material.parent.mkdir(parents=True);material.write_bytes(b'original')
            original='/workspace/attachments/x/source.xlsx';digest=hashlib.sha256(b'original').hexdigest()
            (repo/'migration/materials_manifest.json').write_text(json.dumps({'files':[{'display_name':'source.xlsx','sha256':digest,'cloud_path':original}],
                'missing_requested_original_workbooks':['missing.xlsx']}))
            (repo/'migration/scope_registry_snapshot.json').write_text('{"records":[]}')
            (repo/'configs/baseline_v1.json').write_text(json.dumps({'raw':original,'scope_registry':'old','historical_note':'old'}))
            (repo/'configs/baseline_v1_1.json').write_text('{"baseline":"old"}')
            (repo/'models/exploratory_v1_1.joblib').write_bytes(b'placeholder')
            before=(repo/'configs/baseline_v1.json').read_bytes();r=setup.prepare(repo)
            self.assertTrue(r['baseline_ready']);self.assertTrue(r['prediction_ready'])
            local=json.loads((repo/'configs/local/baseline_v1.json').read_text())
            self.assertEqual(local['raw'],str(material));self.assertEqual(before,(repo/'configs/baseline_v1.json').read_bytes())
            with self.assertRaises(FileExistsError):setup.prepare(repo)

    def test_wrong_bytes_are_not_accepted_by_filename(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);(repo/'migration').mkdir();(repo/'configs').mkdir();raw=repo/'data/raw/attachments';raw.mkdir(parents=True)
            (raw/'source.xlsx').write_bytes(b'wrong')
            (repo/'migration/materials_manifest.json').write_text(json.dumps({'files':[{'display_name':'source.xlsx','sha256':'0'*64,'cloud_path':'old'}],
                'missing_requested_original_workbooks':[]}))
            (repo/'configs/baseline_v1.json').write_text('{"raw":"old"}')
            r=setup.prepare(repo);self.assertFalse(r['baseline_ready']);self.assertFalse(r['historical_hashes_rewritten'])


if __name__=='__main__':unittest.main()
