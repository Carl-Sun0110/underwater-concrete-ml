"""独立接入原文证据特征；固定严格训练范围/外折/网格，不访问测试目标。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import numpy as np
from sklearn.model_selection import GridSearchCV

from .baselines import digest, grouped_folds, regression_metrics, write_json
from .evidence_features import RATIO_FEATURE, EvidenceFeatureTransformer, build_feature_records
from .scope_gate import check_training_request, load_training_gate, select_training_scope
from .scope_results import validate_fold_layout
from .train_compare import select_train_records, make_estimator

FEATURE_SETS = ('core_plus_fine_ratio', 'core_plus_fine_and_water_cement_ratio')
INPUTS = ['Water', 'Cem', RATIO_FEATURE]
FIXED_KEYS = ('models', 'base_features', 'target', 'outer_folds', 'inner_folds', 'seed', 'n_jobs', 'scoring',
              'ridge_alpha', 'svr_C', 'svr_gamma', 'svr_epsilon', 'rf_n_estimators', 'rf_max_depth',
              'rf_min_samples_leaf', 'xgb_learning_rate', 'xgb_n_estimators', 'xgb_max_depth',
              'mlp_hidden_sizes', 'mlp_epochs', 'mlp_batch_size', 'mlp_dropout')


def validate_fixed_config(config, baseline):
    """只允许预先约定的输入扩展；不顺便换范围、网格、种子或评估测试。"""
    request = check_training_request(config)
    if request is None or request['mode'] != 'strict_evidence_scope':
        raise ValueError('新证据特征仅绑定既定严格训练范围')
    if config.get('training_execution_authorized') is not True:
        raise ValueError('证据特征训练尚未启用')
    if config.get('test_evaluation_allowed') is not False or config.get('material_imputation_allowed') is not False:
        raise ValueError('禁止测试评价和未知材料填补')
    if config.get('base_features') != ['Water', 'Cem'] or config.get('target') != 'Provisional_Converted_CS_28d':
        raise ValueError('基础输入或暂定目标不符合本轮方案')
    if config.get('evidence_input_features') != INPUTS or config.get('feature_sets') != list(FEATURE_SETS):
        raise ValueError('证据输入或表示不在固定白名单')
    if any(config.get(k) != baseline.get(k) for k in FIXED_KEYS) or config['scope_gate'] != baseline['scope_gate']:
        raise ValueError('输入扩展外的原比较方案已改变')


def align_feature_matrix(rows, feature_records, test_ids):
    """按训练ID/原行号对齐固定三列；拒绝目标字段、漏行、额外行和测试行。"""
    by_id = {r['sample_id']: r for r in feature_records}
    expected = {r['sample_id'] for r in rows}
    if len(by_id) != len(feature_records) or set(by_id) != expected or expected & set(test_ids):
        raise ValueError('证据输入未精确覆盖固定训练ID或包含测试')
    values = []
    for row in rows:
        record = by_id[row['sample_id']]
        features = record['candidate_features']
        if record['source_row'] != row['source_row'] or set(features) != set(INPUTS):
            raise ValueError('证据原行号或特征列不一致')
        if any(features[f] != row['original_values'][f] for f in ('Water', 'Cem')):
            raise ValueError('原核心输入被改变')
        values.append([features[f] for f in INPUTS])
    X = np.asarray(values, dtype=float)
    EvidenceFeatureTransformer().fit(X)
    return X


def extended_estimator(name, feature_set, config):
    """复用原模型及网格，只替换为固定证据输入变换；缩放仍在折内Pipeline中。"""
    if feature_set not in FEATURE_SETS:
        raise ValueError('未知证据表示')
    estimator, grid = make_estimator(name, 'core2', config)
    estimator.set_params(representation=EvidenceFeatureTransformer(
        add_water_cement_ratio=feature_set == 'core_plus_fine_and_water_cement_ratio'))
    return estimator, grid


def run(labels_path, split_path, config_path, output):
    labels_path, split_path, config_path, output = map(Path, (labels_path, split_path, config_path, output))
    config = json.loads(config_path.read_text())
    reference = Path(config['reference_training_folder'])
    ref_config = json.loads((reference / 'config_snapshot.json').read_text())
    validate_fixed_config(config, ref_config)
    protected = {str(p.resolve()): digest(p) for p in (labels_path, split_path, config_path)}
    for p, h in json.loads((reference / 'manifest.json').read_text())['inputs_sha256'].items():
        if digest(p) != h:
            raise ValueError('原对比训练输入已变更')
        protected[p] = h
    labels = json.loads(labels_path.read_text())
    split = json.loads(split_path.read_text())['rows']
    rows, groups, test_ids = select_train_records(labels, split)
    records, gate_protected = load_training_gate(config['scope_gate'], labels_path, split_path)
    rows, groups, excluded = select_training_scope(rows, groups, records, 'strict_evidence_scope')
    groups = np.asarray(groups)
    protected.update(gate_protected)
    ref_scope = json.loads((reference / 'training_scope.json').read_text())
    if ref_scope['train_sample_ids'] != [r['sample_id'] for r in rows] or set(ref_scope['protected_test_ids']) != test_ids:
        raise ValueError('原训练/测试身份不同，不能开展新增输入比较')
    feature_folder = Path(config['evidence_feature_folder'])
    feature_manifest = feature_folder / 'manifest.json'
    if digest(feature_manifest) != config['evidence_manifest_sha256']:
        raise ValueError('证据特征来源清单版本改变')
    protected[str(feature_manifest.resolve())] = digest(feature_manifest)
    for p, h in json.loads(feature_manifest.read_text())['inputs_sha256'].items():
        if digest(p) != h:
            raise ValueError('证据特征上游文件改变')
        protected[p] = h
    feature_path = feature_folder / 'candidate_training_features.json'
    if digest(feature_path) != config['evidence_features_sha256']:
        raise ValueError('证据特征版本改变')
    evidence_path = feature_folder / 'evidence_snapshot.json'
    if digest(evidence_path) != config['evidence_snapshot_sha256']:
        raise ValueError('原文证据快照版本改变')
    evidence = json.loads(evidence_path.read_text())
    pages = {}
    for key, paper in evidence['papers'].items():
        if digest(paper['path']) != paper['sha256']:
            raise ValueError('原文版本改变')
        pages[key] = subprocess.run(['pdftotext', '-layout', paper['path'], '-'], check=True,
                                     capture_output=True).stdout.decode().split('\f')
    rebuilt = build_feature_records(labels, evidence['annotations'], [r['sample_id'] for r in rows], test_ids, pages, evidence['papers'])
    supplied = json.loads(feature_path.read_text())
    if rebuilt != supplied:
        raise ValueError('证据输入不能由原文声明完整复现')
    X = align_feature_matrix(rows, supplied, test_ids)
    y = np.asarray([r[config['target']] for r in rows], dtype=float)
    if not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError('无效暂定目标')
    folds = grouped_folds(X, y, groups, config['outer_folds'])
    ref_reports = json.loads((reference / 'nested_cv_results.json').read_text())
    validate_fold_layout(ref_reports, {r['sample_id'] for r in rows})
    if [f['validation_sample_ids'] for f in ref_reports[0]['folds']] != [[rows[i]['sample_id'] for i in va] for _, va in folds]:
        raise ValueError('新增输入的外折与原对照不同')
    for path in [feature_path, evidence_path, reference / 'manifest.json', reference / 'nested_cv_results.json',
                 reference / 'training_scope.json', reference / 'config_snapshot.json']:
        protected[str(path.resolve())] = digest(path)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'training_scope.json', {'train_sample_ids': [r['sample_id'] for r in rows],
        'protected_test_ids': sorted(test_ids), 'input_features': INPUTS, 'scope_mode': 'strict_evidence_scope',
        'excluded_by_scope_gate': excluded, 'test_predictions_generated': False,
        'evidence_scope_is_not_trusted_qualification': True})
    reports, predictions = [], []
    for feature_set in config['feature_sets']:
        for model in config['models']:
            oof = np.full(len(rows), np.nan)
            details = []
            for fold, (tr, va) in enumerate(folds, 1):
                estimator, grid = extended_estimator(model, feature_set, config)
                if grid:
                    inner = grouped_folds(X[tr], y[tr], groups[tr], config['inner_folds'])
                    fitted = GridSearchCV(estimator, grid, scoring=config['scoring'], cv=inner,
                        n_jobs=config['n_jobs'], error_score='raise', refit=True).fit(X[tr], y[tr])
                    params, inner_mae = fitted.best_params_, float(-fitted.best_score_)
                else:
                    fitted = estimator.fit(X[tr], y[tr])
                    params, inner_mae = {}, None
                oof[va] = fitted.predict(X[va])
                details.append({'fold': fold, 'train_n': len(tr), 'validation_n': len(va),
                    'train_groups': sorted(set(groups[tr])), 'validation_groups': sorted(set(groups[va])),
                    'validation_sample_ids': [rows[i]['sample_id'] for i in va], 'best_params': params,
                    'inner_selection_MAE': inner_mae, 'metrics': regression_metrics(y[va], oof[va])})
            if not np.isfinite(oof).all():
                raise RuntimeError('折外预测未完整覆盖')
            score = regression_metrics(y, oof)
            reports.append({'feature_set': feature_set, 'model': model, 'pooled_nested_oof': score, 'folds': details})
            predictions += [{'sample_id': r['sample_id'], 'source_row': r['source_row'],
                'research_group': r['original_values']['Research Group'], 'feature_set': feature_set, 'model': model,
                'actual_provisional_MPa': float(y[i]), 'oof_prediction_MPa': float(oof[i])} for i, r in enumerate(rows)]
            print(feature_set, model, score, flush=True)
    unchanged = all(digest(p) == h for p, h in protected.items())
    if not unchanged:
        raise RuntimeError('训练期间保护输入改变')
    summary = {'training_samples': len(rows), 'training_groups': len(set(groups)), 'sealed_test_samples': len(test_ids),
        'experiments': len(reports), 'outer_folds': len(folds), 'test_set_evaluated': False,
        'fixed_reference_folds_verified': True, 'input_or_reliability_changed': False, 'new_trusted_samples': 0,
        'formal_model_claimed': False, 'scope_mode': 'strict_evidence_scope'}
    for name, data in [('nested_cv_results.json', reports), ('training_oof_predictions.json', predictions),
                       ('summary.json', summary), ('config_snapshot.json', config),
                       ('manifest.json', {'inputs_sha256': protected, 'code_sha256': digest(__file__),
                        'created_at_utc': datetime.now(timezone.utc).isoformat(),
                        'representation_code_sha256': digest(Path(__file__).with_name('evidence_features.py')),
                        'model_factory_sha256': digest(Path(__file__).with_name('train_compare.py')),
                        'advanced_model_code_sha256': digest(Path(__file__).with_name('advanced_models.py'))})]:
        write_json(output / name, data)
    return summary
