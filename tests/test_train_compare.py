"""训练内比较的隔离与派生输入测试。"""
import copy
import unittest
import numpy as np
from concrete_audit.train_compare import CoreRepresentation,select_train_records

class TrainCompareTests(unittest.TestCase):
    def test_ratio_is_computed_from_X_only(self):
        X=np.array([[200.,400.],[180.,600.]])
        old=X.copy();p=CoreRepresentation(add_ratio=True).fit(X,np.array([999.,-999.]))
        result=p.transform(X)
        np.testing.assert_allclose(result,[[200,400,.5],[180,600,.3]])
        np.testing.assert_array_equal(X,old)
        self.assertEqual(result.shape[1],3)

    def test_no_unknown_or_extra_fields_accepted(self):
        for X in [[[0,400]],[[200,-1]],[[np.nan,400]],[[200,400,100]]]:
            with self.subTest(X=X),self.assertRaises(ValueError):CoreRepresentation().fit(X)

    def test_held_out_targets_cannot_change_training_selection(self):
        labels=[{'sample_id':'a','source_row':7,'original_values':{'Research Group':'r1','PDF filename':'p1'},'target':10},
                {'sample_id':'b','source_row':8,'original_values':{'Research Group':'r2','PDF filename':'p2'},'target':20}]
        split=[{'sample_id':'a','source_row':7,'split':'train','leakage_group':'g1','research_group':'r1','pdf_filename':'p1'},
               {'sample_id':'b','source_row':8,'split':'test','leakage_group':'g2','research_group':'r2','pdf_filename':'p2'}]
        original=copy.deepcopy(select_train_records(labels,split)[0])
        labels[1]['target']=10000000
        self.assertEqual(original,select_train_records(labels,split)[0])
        self.assertEqual(select_train_records(labels,split)[2],{'b'})

    def test_bad_identity_or_group_overlap_fails(self):
        labels=[{'sample_id':'a','source_row':7,'original_values':{'Research Group':'r1','PDF filename':'p1'}}]
        split=[{'sample_id':'a','source_row':7,'split':'train','leakage_group':'g1','research_group':'r1','pdf_filename':'p1'},
               {'sample_id':'b','source_row':8,'split':'test','leakage_group':'g2','research_group':'r2','pdf_filename':'p2'}]
        bad=copy.deepcopy(split);bad[1]['leakage_group']='g1'
        with self.assertRaises(ValueError):select_train_records(labels,bad)
        bad=copy.deepcopy(split);bad[0]['source_row']=100
        with self.assertRaises(ValueError):select_train_records(labels,bad)
