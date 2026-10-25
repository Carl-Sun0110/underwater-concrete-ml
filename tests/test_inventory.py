"""盘点过程不能把未知当真零，也要支持空表和无缓存公式。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import openpyxl

spec=importlib.util.spec_from_file_location('inventory',Path(__file__).resolve().parents[1]/'scripts/inventory_stage12.py')
inventory=importlib.util.module_from_spec(spec);spec.loader.exec_module(inventory)

class InventoryTests(unittest.TestCase):
    def test_missing_zero_formula_separate(self):
        p=inventory.profile([None,'','-','/',0,False,'=1+1',2])
        self.assertEqual(p,{'cells':8,'blank':2,'placeholder_text':2,'numeric_zero':1,'formula':1})

    def test_empty_sheet_and_formula_without_cache(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'inputs';root.mkdir()
            w=openpyxl.Workbook();s=w.active;s.title='Data';s['A7']='=1+1'
            w.create_sheet('Empty');w.save(root/'source.xlsx')
            before=inventory.digest(root/'source.xlsx')
            inventory.run(root,Path(t)/'outputs')
            books=json.loads((Path(t)/'outputs/workbooks.json').read_text())
            main=books[0]['sheets'][0]
            self.assertEqual(main['formula_missing_cache'],1)
            self.assertEqual(main['sample_count'],1)
            self.assertEqual(inventory.digest(root/'source.xlsx'),before)
            with self.assertRaises(FileExistsError):inventory.run(root,Path(t)/'outputs')
