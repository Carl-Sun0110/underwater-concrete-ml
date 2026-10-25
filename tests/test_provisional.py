"""暂定标签边界：精确系数、不补目标、不恢复候选外记录。"""
from decimal import Decimal
import unittest
from concrete_audit.provisional import provisional_strength,eligible_reasons

ALLOWED=[.95,.8,.76]

class ProvisionalTests(unittest.TestCase):
    def test_preserves_unrounded_decimal(self):
        result,reasons=provisional_strength(59.2,.76,ALLOWED)
        self.assertEqual(result,Decimal('44.992'))
        self.assertEqual(reasons,[])
        self.assertEqual(provisional_strength(17,.95,ALLOWED)[0],Decimal('16.15'))

    def test_nearby_or_identity_factor_not_substituted(self):
        for factor in [.766,.74,.84,.94,.73,.696,1]:
            with self.subTest(factor=factor):
                result,reasons=provisional_strength(40,factor,ALLOWED)
                self.assertIsNone(result)
                self.assertIn('原有系数不在本次授权白名单',reasons)

    def test_invalid_strength_or_factor_no_imputation(self):
        for strength in [None,'-','/',0,-1,True,'NaN','Infinity','=A1']:
            with self.subTest(strength=strength):self.assertIsNone(provisional_strength(strength,.8,ALLOWED)[0])
        for factor in [None,'-',True,'NaN']:
            with self.subTest(factor=factor):self.assertIsNone(provisional_strength(40,factor,ALLOWED)[0])

    def test_existing_numeric_text_is_accepted_without_guessing(self):
        self.assertEqual(provisional_strength('40','0.80',ALLOWED)[0],Decimal('32.00'))

    def test_candidate_and_environment_constraints(self):
        row={'CastingEnv':'Underwater','ConvertFlag':0,'CS_Spec':'Cu100'}
        self.assertEqual(eligible_reasons(row,True,[]),[])
        self.assertIn('未与原527候选身份一致关联',eligible_reasons(row,False,[]))
        self.assertIn('表内未标为Underwater',eligible_reasons({**row,'CastingEnv':'Air'},True,[]))
        self.assertIn('ConvertFlag未明确为0',eligible_reasons({**row,'ConvertFlag':1},True,[]))
        self.assertIn('表内试件规格缺失或占位',eligible_reasons({**row,'CS_Spec':'-'},True,[]))
        self.assertIn('invalid',eligible_reasons(row,True,['invalid']))

    def test_invalid_policy_rejected(self):
        for allowed in [[None],[0],[-.8]]:
            with self.subTest(allowed=allowed),self.assertRaises(ValueError):
                provisional_strength(40,.8,allowed)
