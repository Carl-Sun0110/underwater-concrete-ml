"""核查门槛测试：单位、计算、证据、图读数和可信状态必须保持分离。"""
import unittest
from concrete_audit.field_review import LocalEvidenceBackend,validate_evidence,STATUSES,canonical_unit

class FieldReviewTests(unittest.TestCase):
    def setUp(self):self.backend=LocalEvidenceBackend()

    def test_same_number_different_units_is_not_verified(self):
        claim={'mode':'direct','unit':'L/m3','comparison_value':12}
        result=self.backend.review(12,'kg/m3',claim)
        self.assertEqual(result['status'],'已核实但需转换')
        self.assertIsNone(result['computed_value'])
        self.assertNotEqual(canonical_unit('L/m3'),canonical_unit('kg/m³'))

    def test_direct_strength_conflict(self):
        claim={'mode':'direct','unit':'MPa','comparison_value':59.2}
        self.assertEqual(self.backend.review(59.2,'MPa',claim)['status'],'已核实')
        self.assertEqual(self.backend.review(63,'MPa',claim)['status'],'发现不一致')

    def test_percent_cement_not_binder(self):
        r=self.backend.review(2.66,'kg/m3',{'mode':'calculated','formula_id':'percent_mass',
                            'parameters':{'percent':.5,'mass_kg_m3':532.8}})
        self.assertAlmostEqual(r['computed_value'],2.664)
        self.assertEqual(r['status'],'已核实但需转换')
        self.assertIn('未审批',r['value_category'])

    def test_unknown_density_never_invents_mass(self):
        r=self.backend.review(1.57,'kg/m3',{'mode':'calculated','formula':'V*rho',
                                          'parameters':{'density':None}})
        self.assertIsNone(r['computed_value'])
        self.assertEqual(r['status'],'已核实但需转换')

    def test_missing_evidence_not_same_as_not_reported(self):
        r=self.backend.review(0,'kg/m3',None)
        self.assertEqual(r['status'],'证据不足')
        self.assertIn('不等于论文未报告',r['difference'])
        with self.assertRaises(ValueError):
            self.backend.review(0,'kg/m3',{'mode':'not_found'})
        r=self.backend.review(0,'kg/m3',{'mode':'not_found','search_scope':'仅PDF p3文本'})
        self.assertEqual(r['status'],'文献中未找到')

    def test_fabricated_or_wrong_page_quote_rejected(self):
        good={'evidence':[{'page':1,'quote':'Table 3\nUCR1 59.2'}]}
        validate_evidence(good,['Table 3  UCR1 59.2'])
        for bad in [{'evidence':[{'page':2,'quote':'Table 3'}]},
                    {'evidence':[{'page':1,'quote':'UCR1 100'}]}, {'evidence':[]}]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                validate_evidence(bad,['Table 3 UCR1 59.2'])

    def test_all_six_statuses_supported_without_forcing_occurrence(self):
        self.assertEqual(len(STATUSES),6)
        self.assertEqual(self.backend.review(0,'-',{'mode':'ambiguous'})['status'],'待人工确认')
        with self.assertRaises(ValueError):self.backend.review(0,'-',{'mode':'invented'})

    def test_component_sum_kept_as_proposed_computation(self):
        r=self.backend.review(0,'kg/m3',{'mode':'calculated','formula_id':'sum_components',
                                       'parameters':{'components_kg_m3':[356,830]}})
        self.assertEqual(r['computed_value'],1186)
        self.assertNotEqual(r['status'],'已核实')
        with self.assertRaises(ValueError):
            self.backend.review(0,'kg/m3',{'mode':'calculated','formula_id':'sum_components',
                                         'parameters':{'components_kg_m3':[356,None]}})
