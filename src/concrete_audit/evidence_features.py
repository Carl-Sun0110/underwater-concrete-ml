"""基于原文的细骨料/水泥比及训练折内预处理检查，不拟合预测模型。"""
from collections import Counter
import math
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .field_review import validate_evidence

RATIO_FEATURE = 'fine_aggregate_to_cement_mass_ratio'


def positive_table_columns(text, expected_ids, columns):
    """只读指定原表列；统一排版短横线，不将其他列的横线转换为0。"""
    result = {}
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        key = tokens[0].replace('–', '-').replace('−', '-')
        if key not in expected_ids:
            continue
        if key in result or len(tokens) <= max(columns):
            raise ValueError('原表配方重复或列不完整')
        try:
            values = [float(tokens[i]) for i in columns]
        except ValueError as exc:
            raise ValueError('所需质量列不是明确数值，禁止把横线当0') from exc
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('所需质量列无效')
        result[key] = {'values': values, 'source_line': line}
    if set(result) != set(expected_ids):
        raise ValueError('原表配方集合未完整匹配')
    return result


def mass_ratio(numerator, denominator, numerator_unit, denominator_unit):
    """同单位质量或同基准归一化配比相除；禁止体积/质量混算及假定密度。"""
    if numerator_unit != denominator_unit or numerator_unit not in ('kg/m3', 'relative_cement_mass'):
        raise ValueError('质量比的单位或归一化基准不一致')
    for value in (numerator, denominator):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('当前正质量比证据无效，未知或0不能直接代入')
    return numerator / denominator


def build_feature_records(labels, annotations, expected_ids, test_ids, pages, papers):
    """只构造已固定严格训练ID的X候选，不复制目标或自动提升可靠状态。"""
    by_id = {r['sample_id']: r for r in labels}
    if len(by_id) != len(labels) or set(expected_ids) & set(test_ids):
        raise ValueError('身份重复或训练/测试重叠')
    by_row = {r['source_row']: r for r in labels if r['sample_id'] in expected_ids}
    selected = {}
    for a in annotations:
        row = a['source_row']
        if row not in by_row or row in selected:
            raise ValueError('证据包含非训练行或重复行')
        raw = by_row[row]['original_values']
        if (a['specimen_id'], a['registry_pdf_filename']) != (raw['Specimen ID'], raw['PDF filename']):
            raise ValueError('细骨料比证据身份错配')
        validate_evidence(a, pages[a['paper']], papers[a['paper']]['sha256'])
        ratio = mass_ratio(a['numerator'], a['denominator'], a['numerator_unit'], a['denominator_unit'])
        selected[row] = ratio
    if set(selected) != set(by_row) or len(by_row) != len(expected_ids):
        raise ValueError('原文质量比证据没有完整覆盖预定训练记录')
    records = []
    for sample_id in expected_ids:
        src = by_id[sample_id]
        values = [src['original_values']['Water'], src['original_values']['Cem'], selected[src['source_row']]]
        EvidenceFeatureTransformer().fit(np.array([values]))
        records.append({'sample_id': sample_id, 'source_row': src['source_row'],
                        'candidate_features': {'Water': values[0], 'Cem': values[1], RATIO_FEATURE: values[2]},
                        'prior_reliability_status': src['prior_reliability_status'],
                        'human_confirmation': '待核查', 'trusted_qualification_granted': False,
                        'ratio_does_not_verify_absolute_masses': True})
    return records


class EvidenceFeatureTransformer(BaseEstimator, TransformerMixin):
    """固定三列输入，可附加Water/Cem；不学习填补值，不接收强度派生字段。"""
    def __init__(self, add_water_cement_ratio=False):
        self.add_water_cement_ratio = add_water_cement_ratio

    def fit(self, X, y=None):
        self._check(X)
        self.n_features_in_ = 3
        return self

    def _check(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all() or np.any(X <= 0):
            raise ValueError('仅接受Water、Cem、原文细骨料比三列正数，不填补未知')
        return X

    def transform(self, X):
        X = self._check(X)
        return np.column_stack([X, X[:, 0]/X[:, 1]]) if self.add_water_cement_ratio else X.copy()


def fold_support(candidate_rows, expected_ids, group_by_id, folds, fields):
    """只用固定训练部分检查证据覆盖；找出仅验证折有值的列，禁止据此填补训练。"""
    by_id = {r['sample_id']: r['candidate_values'] for r in candidate_rows if r['sample_id'] in expected_ids}
    if set(by_id) != set(expected_ids):
        raise ValueError('候选字段视图未覆盖固定训练身份')
    result = []
    for field in fields:
        known = [sid for sid, values in by_id.items() if values.get(field) is not None]
        checks = []
        for f in folds:
            valid = set(f['validation_sample_ids'])
            train = set(expected_ids) - valid
            train_known = [sid for sid in train if by_id[sid].get(field) is not None]
            valid_known = [sid for sid in valid if by_id[sid].get(field) is not None]
            train_values = {by_id[sid][field] for sid in train_known}
            checks.append({'fold': f['fold'], 'train_known': len(train_known), 'validation_known': len(valid_known),
                           'train_known_groups': len({group_by_id[sid] for sid in train_known}),
                           'held_out_only_evidence': not train_known and bool(valid_known),
                           'unseen_validation_values': sum(by_id[sid][field] not in train_values for sid in valid_known)})
        result.append({'field': field, 'known_training_records': len(known), 'training_denominator': len(expected_ids),
                       'known_groups': len({group_by_id[sid] for sid in known}),
                       'complete_on_training': len(known) == len(expected_ids), 'fold_checks': checks,
                       'automatic_admission': False,
                       'status': '验证折独有证据，当前方案不可直接入模' if any(c['held_out_only_evidence'] for c in checks) else
                                 '需另定缺失/类型/来源混杂处理，尚未入模'})
    return result
