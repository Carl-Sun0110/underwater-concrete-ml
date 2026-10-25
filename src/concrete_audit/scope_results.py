"""范围审查后训练内结果审计：每个范围分别汇总，不把不同样本成绩直接排名。"""
from collections import Counter, defaultdict
import math
from statistics import mean

from .baselines import regression_metrics


def validate_fold_layout(reports, expected_ids, planned_cv=None):
    """核对外折覆盖、来源隔离和所有算法相同折；可绑定事前保存的CV预案。"""
    if not reports:
        raise ValueError('训练报告为空')
    reference = None
    for report in reports:
        layout = [f['validation_sample_ids'] for f in report['folds']]
        flat = [sample for ids in layout for sample in ids]
        if len(flat) != len(set(flat)) or set(flat) != set(expected_ids):
            raise ValueError('外层验证未唯一覆盖预定训练记录')
        if reference is None:
            reference = layout
        elif layout != reference:
            raise ValueError('同一范围算法使用了不同验证折')
        for fold in report['folds']:
            if set(fold['train_groups']) & set(fold['validation_groups']):
                raise ValueError('外层来源组泄漏')
        if planned_cv is not None and layout != [f['validation_sample_ids'] for f in planned_cv]:
            raise ValueError('实际外折与事前预案不一致')


def aggregate_predictions(predictions, group_by_id, expected_ids, protected_test_ids, pairs):
    """每项配置精确覆盖训练ID；按样本及按来源等权分别统计误差，保留全部预测。"""
    if set(expected_ids) & set(protected_test_ids):
        raise ValueError('预定训练与测试身份重叠')
    buckets = defaultdict(list)
    for row in predictions:
        if row['sample_id'] in protected_test_ids or row['sample_id'] not in expected_ids:
            raise ValueError('折外结果含测试或非预定样本')
        pair = row['feature_set'], row['model']
        if pair not in pairs:
            raise ValueError('未计划的模型/表示配置')
        if not all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) for k in
                   ('actual_provisional_MPa', 'oof_prediction_MPa')):
            raise ValueError('预测或暂定目标非有限')
        buckets[pair].append(row)
    if set(buckets) != set(pairs):
        raise ValueError('缺少计划配置的折外预测')
    metrics, groups = [], []
    for feature_set, model in sorted(pairs):
        selected = buckets[feature_set, model]
        ids = [r['sample_id'] for r in selected]
        if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
            raise ValueError('配置预测重复或未覆盖全部训练记录')
        grouped = defaultdict(list)
        for row in selected:
            if row['sample_id'] not in group_by_id:
                raise ValueError('缺少来源组身份')
            grouped[group_by_id[row['sample_id']]].append(row)
        source_mae = []
        for group, members in sorted(grouped.items()):
            score = regression_metrics([r['actual_provisional_MPa'] for r in members],
                                       [r['oof_prediction_MPa'] for r in members])
            source_mae.append(score['MAE_MPa'])
            groups.append({'feature_set': feature_set, 'model': model, 'leakage_group': group,
                           'research_groups': sorted({r['research_group'] for r in members}), **score})
        pooled = regression_metrics([r['actual_provisional_MPa'] for r in selected],
                                    [r['oof_prediction_MPa'] for r in selected])
        metrics.append({'feature_set': feature_set, 'model': model, **pooled,
                        'source_macro_MAE_MPa': mean(source_mae), 'worst_source_MAE_MPa': max(source_mae),
                        'source_groups': len(grouped)})
    return metrics, groups
