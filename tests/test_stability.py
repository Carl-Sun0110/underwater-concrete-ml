import unittest
from concrete_audit.stability import paired_source_stability


class StabilityTests(unittest.TestCase):
    def rows(self):
        model=[{'sample_id':str(i),'actual':10.,'predicted':9. if i<8 else 1.,'group':'a' if i<8 else 'b','paper':'p'+('a' if i<8 else 'b')} for i in range(10)]
        baseline=[{**r,'predicted':7.} for r in model];return model,baseline

    def test_preserves_source_and_sample_weighting_and_pairing(self):
        m,b=self.rows();r=paired_source_stability(m,b,resamples=100,seed=42)
        self.assertAlmostEqual(r['model_sample_MAE_MPa'],2.6)
        self.assertAlmostEqual(r['model_macro_MAE_MPa'],5.)
        self.assertEqual(r['sources_improved'],1);self.assertEqual(r['sources_worsened'],1)

    def test_deterministic_fixed_seed(self):
        m,b=self.rows();self.assertEqual(paired_source_stability(m,b,100,42),paired_source_stability(m,b,100,42))

    def test_mismatched_labels_or_duplicate_ids_rejected(self):
        m,b=self.rows();b[0]['actual']=999
        with self.assertRaises(ValueError):paired_source_stability(m,b)
        m,b=self.rows()
        with self.assertRaises(ValueError):paired_source_stability(m+[m[0]],b)


if __name__=='__main__':unittest.main()
