"""同一样本的特征增量比较；来源组等权与样本加权分开，不作显著性推断。"""
from collections import defaultdict
from statistics import mean
import math


def paired_error_changes(before, after, group_by_id):
    """按ID配对，强度必须一致；改变量为新增输入后减去原模型，负数表示MAE下降。"""
    left = {r['sample_id']: r for r in before}
    right = {r['sample_id']: r for r in after}
    if not left or len(left) != len(before) or len(right) != len(after) or set(left) != set(right):
        raise ValueError('比较样本未唯一配对')
    grouped = defaultdict(list)
    errors = []
    for sid, a in left.items():
        b = right[sid]
        if sid not in group_by_id:
            raise ValueError('缺少配对来源身份')
        if not math.isclose(a['actual_provisional_MPa'], b['actual_provisional_MPa'], rel_tol=0, abs_tol=1e-10):
            raise ValueError('新增输入比较中的暂定目标改变')
        values = [a['actual_provisional_MPa'], a['oof_prediction_MPa'], b['oof_prediction_MPa']]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('配对预测不是有限值')
        ea, eb = abs(values[0] - values[1]), abs(values[0] - values[2])
        grouped[group_by_id[sid]].append((ea, eb))
        errors.append((ea, eb))
    source_changes = []
    for group, pairs in sorted(grouped.items()):
        ea, eb = mean(v[0] for v in pairs), mean(v[1] for v in pairs)
        source_changes.append({'leakage_group': group, 'n': len(pairs), 'MAE_before_MPa': ea,
                               'MAE_after_MPa': eb, 'delta_MAE_after_minus_before_MPa': eb - ea})
    result = {'n': len(errors), 'source_groups': len(source_changes),
              'delta_sample_MAE_MPa': mean(v[1] for v in errors) - mean(v[0] for v in errors),
              'delta_source_macro_MAE_MPa': mean(r['delta_MAE_after_minus_before_MPa'] for r in source_changes),
              'source_groups_improved': sum(r['delta_MAE_after_minus_before_MPa'] < -1e-10 for r in source_changes),
              'source_groups_worsened': sum(r['delta_MAE_after_minus_before_MPa'] > 1e-10 for r in source_changes),
              'source_changes': source_changes, 'statistical_significance_claimed': False}
    return result
