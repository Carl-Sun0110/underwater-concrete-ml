"""胶材/纤维组成的未知保留诊断，材料家族标签不自动成为模型输入。"""
from collections import defaultdict
import math
from .evidence_features import mass_ratio

MINERALS = ('Cem', 'FA', 'SF', 'GGBS')


def complete_component_sum(components, names):
    """仅对明确给定的组分定义求和；缺项/未知即未知，不把横线变0。"""
    if not names or 'Cem' not in names or len(set(names)) != len(names):
        return None, '胶材定义未知或不完整'
    selected = [components.get(name) for name in names]
    if any(c is None or c.get('value') is None for c in selected):
        return None, '定义中的组分值仍未知，禁止以0代替求和'
    if any(c.get('status') != 'reported_positive' for c in selected):
        return None, '当前求和只允许明确正值，零值需要另行独立证据'
    if len({c['unit'] for c in selected}) != 1:
        raise ValueError('胶材组分单位或归一化基准不一致')
    if any(isinstance(c['value'], bool) or not isinstance(c['value'], (int, float)) or
           not math.isfinite(c['value']) or c['value'] <= 0 for c in selected):
        raise ValueError('胶材数值无效')
    return sum(c['value'] for c in selected), None


def ratio_if_known(numerator, cement):
    """只计算已有明确值的类型独立质量比，其他情况保持None。"""
    if not numerator or numerator.get('value') is None or not cement or cement.get('value') is None:
        return None
    return mass_ratio(numerator['value'], cement['value'], numerator['unit'], cement['unit'])


def derive_composition(recipe):
    """定义内组分和、原文总量、候选水胶比分开，不修正原文矛盾。"""
    c = recipe['components']
    cement = c['Cem']
    ratios = {name.lower() + '_to_cement_mass_ratio': ratio_if_known(c.get(name), cement) for name in ('FA', 'SF', 'GGBS')}
    for name in ('steel', 'pva', 'pp'):
        ratios[name + '_to_cement_mass_ratio'] = ratio_if_known(recipe['fibers'].get(name), cement)
    total, reason = complete_component_sum(c, recipe.get('defined_binder_components'))
    printed = recipe.get('printed_binder_total')
    conflict = total is not None and printed is not None and not math.isclose(total, printed, rel_tol=0, abs_tol=1e-9)
    fraction, wb = None, None
    water = recipe.get('water')
    if total is not None:
        fraction = cement['value'] / total
        if water and water.get('value') is not None and water['unit'] == cement['unit']:
            wb = water['value'] / total
    ratios['cement_fraction_from_defined_reported_components'] = fraction
    ratios['water_over_defined_reported_binder_components'] = wb
    return {'candidate_ratios': ratios, 'binder_component_sum': total, 'binder_sum_unit': cement['unit'] if total else None,
            'binder_unknown_reason': reason, 'source_printed_binder_total': printed, 'binder_total_conflict': conflict,
            'water_source_unit': water['unit'] if water else None,
            'water_binder_model_admission': False,
            'water_binder_block_reason': '原文分项和与总量矛盾' if conflict else
                '水量与胶材单位不同，不假定密度换算' if total is not None and wb is None else
                '候选需人工复核，尚未入模' if wb is not None else reason,
            'automatic_model_admission': False, 'trusted_qualification_granted': False}


def family_confounding(records, group_by_id, folds):
    """统计原文声明家族与来源的混杂，报告验证中未见家族，不将来源名编码进X。"""
    families = defaultdict(list)
    by_id = {r['sample_id']: r['declared_mix_family'] for r in records}
    for r in records:
        families[r['declared_mix_family']].append(r['sample_id'])
    rows = [{'declared_mix_family': family, 'records': len(ids),
             'independent_groups': sorted({group_by_id[sid] for sid in ids}),
             'group_count': len({group_by_id[sid] for sid in ids})} for family, ids in sorted(families.items())]
    checks = []
    for fold in folds:
        valid = set(fold['validation_sample_ids'])
        if not valid <= set(by_id):
            raise ValueError('家族检查与固定训练折不对应')
        train = set(by_id) - valid
        seen = {by_id[sid] for sid in train}
        checks.append({'fold': fold['fold'], 'validation_records': len(valid),
                       'unseen_family_validation_records': sum(by_id[sid] not in seen for sid in valid)})
    return {'families': rows, 'folds': checks, 'all_families_single_group': all(r['group_count'] == 1 for r in rows),
            'family_labels_are_not_mutually_exclusive_physical_taxonomy': True,
            'automatic_categorical_feature_admission': False}
