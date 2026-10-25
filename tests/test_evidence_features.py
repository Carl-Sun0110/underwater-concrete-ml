"""质量比单位、骨料列身份、未知值和折内预处理隔离测试。"""
import copy
import unittest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from concrete_audit.evidence_features import (mass_ratio, positive_table_columns, build_feature_records,
                                             EvidenceFeatureTransformer, fold_support, RATIO_FEATURE)


class EvidenceFeatureTests(unittest.TestCase):
    def test_ratio_from_mass_or_normalized_ratio_never_assumes_density(self):
        self.assertEqual(mass_ratio(1.05, 1., 'relative_cement_mass', 'relative_cement_mass'), 1.05)
        self.assertAlmostEqual(mass_ratio(564, 1202, 'kg/m3', 'kg/m3'), 564/1202)
        for n, d, u, v in [(5, 429, 'L/m3', 'kg/m3'), (0, 429, 'kg/m3', 'kg/m3'),
                            (1, 0, 'kg/m3', 'kg/m3'), (None, 1, 'kg/m3', 'kg/m3')]:
            with self.assertRaises(ValueError):
                mass_ratio(n, d, u, v)

    def test_yousri_fine_aggregate_not_fly_ash_or_water(self):
        text = 'M–0.5–10FA 532.8 – 59.2 213.5 607 603 2.5 .50 –'
        result = positive_table_columns(text, {'M-0.5-10FA'}, [1, 6])
        self.assertEqual(result['M-0.5-10FA']['values'], [532.8, 603])
        with self.assertRaises(ValueError):
            positive_table_columns(text, {'M-0.5-10FA'}, [1, 2])
        with self.assertRaises(ValueError):
            positive_table_columns(text + '\n' + text, {'M-0.5-10FA'}, [1, 6])

    def test_transformer_and_scaler_fit_training_only(self):
        train = np.array([[200., 400., 1.5], [180., 600., 1.]])
        val = np.array([[10000., 9000., 30.]])
        pipe = Pipeline([('inputs', EvidenceFeatureTransformer(add_water_cement_ratio=True)), ('scale', StandardScaler())])
        pipe.fit(train, [99999, -99999])
        np.testing.assert_allclose(pipe['scale'].mean_, [190., 500., 1.25, .4])
        before = pipe['scale'].mean_.copy()
        self.assertEqual(pipe.transform(val).shape, (1, 4))
        np.testing.assert_array_equal(before, pipe['scale'].mean_)
        for bad in [[[0, 400, 1]], [[200, 400, np.nan]], [[200, 400, 1, 7]]]:
            with self.assertRaises(ValueError):
                EvidenceFeatureTransformer().fit(bad)

    def test_feature_records_do_not_depend_on_target_or_include_test(self):
        labels = [{'sample_id': 'a', 'source_row': 1, 'original_values': {'Water': 200, 'Cem': 400,
                    'Specimen ID': 'M1', 'PDF filename': 'p', 'CS_28d': 99}, 'prior_reliability_status': '待核查'}]
        annotation = {'source_row': 1, 'specimen_id': 'M1', 'registry_pdf_filename': 'p', 'paper': 'p',
                      'numerator': 600, 'denominator': 400, 'numerator_unit': 'kg/m3', 'denominator_unit': 'kg/m3',
                      'evidence': [{'page': 1, 'quote': 'Sand 600 Cement 400'}]}
        original = copy.deepcopy(labels)
        args = ([annotation], ['a'], {'test'}, {'p': ['Sand 600 Cement 400']}, {'p': {'sha256': 'hash'}})
        first = build_feature_records(labels, *args)
        self.assertEqual(labels, original)
        labels[0]['original_values']['CS_28d'] = -100000
        self.assertEqual(first, build_feature_records(labels, *args))
        self.assertEqual(set(first[0]['candidate_features']), {'Water', 'Cem', RATIO_FEATURE})
        self.assertFalse(first[0]['trusted_qualification_granted'])
        with self.assertRaises(ValueError):
            build_feature_records(labels, [annotation], ['a'], {'a'}, args[3], args[4])

    def test_source_only_feature_cannot_use_held_out_evidence(self):
        rows = [{'sample_id': 'a', 'candidate_values': {'env': 'FW'}},
                {'sample_id': 'b', 'candidate_values': {'env': None}}]
        folds = [{'fold': 1, 'validation_sample_ids': ['a']}, {'fold': 2, 'validation_sample_ids': ['b']}]
        profile = fold_support(rows, ['a', 'b'], {'a': 'A', 'b': 'B'}, folds, ['env'])[0]
        self.assertTrue(profile['fold_checks'][0]['held_out_only_evidence'])
        self.assertFalse(profile['automatic_admission'])
