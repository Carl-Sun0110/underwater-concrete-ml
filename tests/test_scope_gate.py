"""范围未知不得自动通过严格门禁；不得改变标签、来源分组或历史结果。"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concrete_audit.scope_gate import (DIRECT, NON_DIRECT, UNKNOWN, build_records, scope_allows,
                                      select_training_scope, check_training_request, load_training_gate)


class ScopeGateTests(unittest.TestCase):
    def labels(self):
        return [{'sample_id': f'id{n}', 'source_row': n, 'prior_reliability_status': '待核查',
                 'original_values': {'Specimen ID': f'M{n}', 'PDF filename': 'paper', 'CastingEnv': 'Underwater'},
                 'target': 1000 * n} for n in (1, 2, 3)]

    def claims(self):
        return [{'source_row': n, 'specimen_id': f'M{n}', 'registry_pdf_filename': 'paper',
                 'paper': 'p', 'scope_state': state, 'reason': 'test scope only',
                 'evidence': [{'page': 1, 'quote': quote}]} for n, state, quote in
                [(1, DIRECT, 'cast underwater'), (2, NON_DIRECT, 'remolded after exposure')]]

    def records(self):
        return build_records(self.labels(), self.claims(), {'p': ['cast underwater; remolded after exposure']}, {'p': {'sha256': 'hash'}})

    def test_unknown_underwater_tag_not_promoted(self):
        records = self.records()
        self.assertEqual([r['scope_state'] for r in records], [DIRECT, NON_DIRECT, UNKNOWN])
        self.assertTrue(all(not r['trusted_qualification_granted'] for r in records))
        self.assertTrue(all(r['prior_reliability_status'] == '待核查' for r in records))
        self.assertFalse(scope_allows(UNKNOWN, 'strict_evidence_scope'))
        self.assertTrue(scope_allows(UNKNOWN, 'isolate_only_exploratory'))

    def test_wrong_pdf_identity_duplicate_and_false_quote_rejected(self):
        for field, bad in [('specimen_id', 'other'), ('registry_pdf_filename', 'other')]:
            claims = self.claims()
            claims[0][field] = bad
            with self.assertRaises(ValueError):
                build_records(self.labels(), claims, {'p': ['cast underwater']}, {'p': {'sha256': 'hash'}})
        with self.assertRaises(ValueError):
            build_records(self.labels(), self.claims() * 2, {'p': ['cast underwater; remolded after exposure']}, {'p': {'sha256': 'hash'}})
        with self.assertRaises(ValueError):
            build_records(self.labels(), self.claims(), {'p': ['not the cited quote']}, {'p': {'sha256': 'hash'}})

    def test_gate_does_not_read_targets_or_reassign_groups(self):
        labels = self.labels()
        original = copy.deepcopy(labels)
        records = self.records()
        kept, groups, excluded = select_training_scope(labels, ['g1', 'g2', 'g3'], records, 'strict_evidence_scope')
        self.assertEqual([r['sample_id'] for r in kept], ['id1'])
        self.assertEqual(groups, ['g1'])
        self.assertEqual(len(excluded), 2)
        labels[0]['target'] = -999999
        again = select_training_scope(labels, ['g1', 'g2', 'g3'], records, 'strict_evidence_scope')
        self.assertEqual([r['sample_id'] for r in again[0]], ['id1'])
        self.assertEqual(original[1:], labels[1:])
        kept, _, excluded = select_training_scope(labels, ['g1', 'g2', 'g3'], records, 'isolate_only_exploratory')
        self.assertEqual([r['sample_id'] for r in kept], ['id1', 'id3'])
        self.assertEqual(excluded[0]['sample_id'], 'id2')

    def test_missing_or_mismatched_registry_blocks_training_selection(self):
        with self.assertRaises(ValueError):
            select_training_scope(self.labels(), ['a', 'b', 'c'], self.records()[:1], 'strict_evidence_scope')
        bad = self.records()
        bad[0]['specimen_id'] = 'other'
        with self.assertRaises(ValueError):
            select_training_scope(self.labels(), ['a', 'b', 'c'], bad, 'isolate_only_exploratory')

    def test_gate_version_and_dataset_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels, split, registry = [root / n for n in ('labels.json', 'split.json', 'registry.json')]
            labels.write_text('labels')
            split.write_text('split')
            h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            registry.write_text(json.dumps({'label_file_sha256': h(labels), 'split_file_sha256': h(split),
                                           'protected_inputs_sha256': {str(labels): h(labels)}, 'records': self.records()}))
            request = {'registry': str(registry), 'registry_sha256': h(registry), 'mode': 'strict_evidence_scope'}
            self.assertEqual(len(load_training_gate(request, labels, split)[0]), 3)
            labels.write_text('changed')
            with self.assertRaises(ValueError):
                load_training_gate(request, labels, split)
            with self.assertRaises(ValueError):
                load_training_gate(request | {'registry_sha256': 'wrong'}, labels, split)

    def test_training_entry_fails_before_model_without_gate(self):
        from concrete_audit.train_compare import run
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            config.write_text('{}')
            with patch('concrete_audit.train_compare.make_estimator') as model:
                with self.assertRaisesRegex(ValueError, 'scope_gate'):
                    run('not-opened-labels', 'not-opened-split', config, Path(tmp) / 'output')
                model.assert_not_called()
                self.assertFalse((Path(tmp) / 'output').exists())
        self.assertIsNone(check_training_request({'historical_scope_reproduction_only': True}))
        with self.assertRaises(ValueError):
            check_training_request({'scope_gate': {}, 'historical_scope_reproduction_only': True})

    def test_disabled_training_plan_and_legacy_entry_cannot_fit(self):
        from concrete_audit.train_compare import run
        from concrete_audit.baselines import run as legacy_run
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            config.write_text(json.dumps({'scope_gate': {'mode': 'strict_evidence_scope',
                              'registry': 'not-opened', 'registry_sha256': 'not-opened'},
                              'training_execution_authorized': False}))
            with patch('concrete_audit.train_compare.make_estimator') as model:
                with self.assertRaisesRegex(ValueError, '训练预案尚未启用'):
                    run('not-opened', 'not-opened', config, Path(tmp) / 'output')
                model.assert_not_called()
            config.write_text('{}')
            with self.assertRaisesRegex(ValueError, '历史范围复现'):
                legacy_run('not-opened-labels', config, Path(tmp) / 'output')
