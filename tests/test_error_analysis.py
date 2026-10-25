"""误差贡献统计与折内输入范围边界验证。"""
import copy
import unittest
import numpy as np
from concrete_audit.error_analysis import domain_flags,source_error_table

class ErrorAnalysisTests(unittest.TestCase):
    def test_input_domain_uses_supplied_training_range(self):
        reference=np.array([[100,300,.3],[200,500,.5]])
        flag=domain_flags([605,1202,.503],reference)
        self.assertEqual(flag['outside_axes'],[0,1,2])
        self.assertEqual(flag['reference_max'],[200,500,.5])
        self.assertEqual(domain_flags([200,500,.5],reference)['outside_axes'],[])

    def test_group_sse_contributions_preserve_all_cases(self):
        rows=[{'research_group':'a','actual':10,'prediction':12,'outside_input_fields':[],'source_row':7},
              {'research_group':'b','actual':20,'prediction':14,'outside_input_fields':['Water'],'source_row':8}]
        before=copy.deepcopy(rows);report=source_error_table(rows)
        self.assertEqual(sum(r['n'] for r in report),2)
        self.assertAlmostEqual(sum(r['SSE_share'] for r in report),1)
        self.assertEqual(report[0]['research_group'],'b')
        self.assertAlmostEqual(report[0]['SSE_share'],.9)
        self.assertEqual(rows,before)

    def test_perfect_predictions_do_not_divide_by_zero(self):
        rows=[{'research_group':'a','actual':10,'prediction':10,'outside_input_fields':[],'source_row':7}]
        result=source_error_table(rows)
        self.assertEqual(result[0]['SSE_share'],0)
        self.assertIsNone(result[0]['R2'])

    def test_invalid_reference_cannot_be_silently_used(self):
        with self.assertRaises(ValueError):domain_flags([1,2],[[1]])
        with self.assertRaises(ValueError):domain_flags([1,2],[[float('nan'),2]])
        with self.assertRaises(ValueError):source_error_table([])
