"""新增证据输入必须保持同范围同网格，拒绝身份错配和目标列混入。"""
import copy
import unittest
import numpy as np
from concrete_audit.evidence_train import (INPUTS, FEATURE_SETS, FIXED_KEYS, validate_fixed_config,
                                          align_feature_matrix, extended_estimator)


class EvidenceTrainTests(unittest.TestCase):
    def configs(self):
        baseline = {k: ['fixed'] for k in FIXED_KEYS}
        baseline.update(scope_gate={'mode': 'strict_evidence_scope', 'registry': 'p', 'registry_sha256': 'h'},
                        base_features=['Water', 'Cem'], target='Provisional_Converted_CS_28d')
        config = copy.deepcopy(baseline)
        config.update(training_execution_authorized=True, test_evaluation_allowed=False,
                      material_imputation_allowed=False, evidence_input_features=INPUTS, feature_sets=list(FEATURE_SETS))
        return config, baseline

    def test_fixed_grid_seed_and_scope_cannot_change_with_new_feature(self):
        config, baseline = self.configs()
        validate_fixed_config(config, baseline)
        for field, value in [('seed', 123), ('svr_C', [999]), ('models', ['other']),
                             ('test_evaluation_allowed', True), ('material_imputation_allowed', True),
                             ('evidence_input_features', ['Water', 'Cem', 'CS_28d'])]:
            bad = copy.deepcopy(config)
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_fixed_config(bad, baseline)

    def fixture(self):
        rows = [{'sample_id': 'a', 'source_row': 7, 'original_values': {'Water': 200, 'Cem': 400}}]
        features = [{'sample_id': 'a', 'source_row': 7,
                     'candidate_features': {'Water': 200, 'Cem': 400, INPUTS[2]: 1.5}}]
        return rows, features

    def test_evidence_identity_columns_and_core_values_are_bound(self):
        rows, features = self.fixture()
        np.testing.assert_allclose(align_feature_matrix(rows, features, {'test'}), [[200, 400, 1.5]])
        variants = [features * 2, [], [features[0] | {'source_row': 99}],
                    [features[0] | {'candidate_features': features[0]['candidate_features'] | {'CS_28d': 60}}],
                    [features[0] | {'candidate_features': features[0]['candidate_features'] | {'Water': 999}}]]
        for bad in variants:
            with self.assertRaises(ValueError):
                align_feature_matrix(rows, bad, {'test'})
        with self.assertRaises(ValueError):
            align_feature_matrix(rows, features, {'a'})

    def test_target_changes_do_not_change_feature_alignment(self):
        rows, features = self.fixture()
        before = align_feature_matrix(rows, features, {'test'})
        rows[0]['target'] = -99999
        rows[0]['original_values']['CS_28d'] = 999999
        np.testing.assert_array_equal(before, align_feature_matrix(rows, features, {'test'}))

    def test_new_transformer_is_inside_fold_pipeline_with_scaler(self):
        config = {'svr_C': [1], 'svr_gamma': ['scale'], 'svr_epsilon': [.1]}
        pipe, grid = extended_estimator('svr', FEATURE_SETS[1], config)
        X = np.array([[200., 400., 1.5], [180., 600., 1.], [220., 500., 1.2]])
        pipe.fit(X, [20., 30., 40.])
        np.testing.assert_allclose(pipe['scale'].mean_, [200., 500., 3.7/3, (.5 + .3 + .44)/3])
        mean = pipe['scale'].mean_.copy()
        self.assertTrue(np.isfinite(pipe.predict([[10000., 9000., 30.]] )).all())
        np.testing.assert_array_equal(mean, pipe['scale'].mean_)
        self.assertEqual(set(grid), {'model__C', 'model__gamma', 'model__epsilon'})
