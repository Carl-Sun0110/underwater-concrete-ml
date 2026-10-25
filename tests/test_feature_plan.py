"""验证零值诊断、泄漏拦截和阈值比较不修改输入或依赖目标。"""
import copy
import unittest
from concrete_audit.feature_plan import diagnostic_value,validate_features,scenario

class FeaturePlanTests(unittest.TestCase):
    def test_zero_is_unknown_and_source_preserved(self):
        original={'Water':0,'Cem':400}
        before=copy.deepcopy(original)
        self.assertEqual(diagnostic_value(original['Water']),(None,'零值语义未知'))
        self.assertEqual(diagnostic_value(original['Cem']),(400.0,None))
        self.assertEqual(original,before)
        self.assertEqual(diagnostic_value(-1),(None,'负值待核查'))
        self.assertEqual(diagnostic_value('NaN'),(None,'非数值/缺失'))

    def test_target_and_identifiers_cannot_enter_features(self):
        for field in ['CS_28d','CS_7d','Conversion Factor','Converted CS_28d',
                      'Provisional_Converted_CS_28d','Research Group','Specimen ID','ConvertFlag','Density']:
            with self.subTest(field=field),self.assertRaises(ValueError):validate_features(['Water',field])
        validate_features(['Water','Cem','SF'])
        with self.assertRaises(ValueError):validate_features(['Water','Water'])

    def test_threshold_boundary_and_core_gating(self):
        rows=[{'sample_id':'a','research_group':'g1','core_positive':True,
               'diagnostic_values':{'Water':200,'Cem':400,'SF':None,'FA':None}},
              {'sample_id':'b','research_group':'g2','core_positive':False,
               'diagnostic_values':{'Water':None,'Cem':400,'SF':10,'FA':10}}]
        fields=['Water','Cem','SF','FA']
        self.assertEqual(scenario(rows,fields,.49)['kept'],0)
        self.assertEqual(scenario(rows,fields,.5)['sample_ids'],['a'])
        self.assertEqual(scenario(rows,fields,1)['sample_ids'],['a'])
        self.assertEqual(scenario(rows,fields,.5,False)['kept'],2)

    def test_target_values_do_not_affect_retention(self):
        rows=[{'sample_id':'a','research_group':'g','core_positive':True,
               'diagnostic_values':{'Water':200,'Cem':400},'target':10}]
        a=scenario(rows,['Water','Cem'],0)
        rows[0]['target']=100000
        self.assertEqual(a,scenario(rows,['Water','Cem'],0))

    def test_duplicate_id_and_invalid_threshold_rejected(self):
        r={'sample_id':'a','research_group':'g','core_positive':True,'diagnostic_values':{'Water':200}}
        with self.assertRaises(ValueError):scenario([r,r],['Water'],0)
        with self.assertRaises(ValueError):scenario([r],['Water'],-.1)
