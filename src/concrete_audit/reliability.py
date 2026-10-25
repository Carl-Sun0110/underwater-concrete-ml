"""可靠状态与数值校验分离，任何自动检查均不能授予可信资格。"""
import math

UNKNOWN = '未知'
PENDING = '待核查'
VERIFIED = '已核实可用'
EXCLUDED = '已排除'
ASPECTS = ('source', 'strength_basis', 'density_assumption', 'feature_units', 'zero_semantics')


def number(value):
    """只接受有限数值；布尔值、Excel公式和占位文本都不是数值。"""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def conversion_check(row, tolerance=0.1):
    """检查乘法和必要字段；返回机械结果，绝不判断来源真实性。"""
    a, b, y = (number(row.get(k)) for k in ('CS_28d', 'Conversion Factor', 'Converted CS_28d'))
    reasons = []
    if a is None or a <= 0:
        reasons.append('原始28d强度缺失、非数值或非正')
    if b is None or b <= 0:
        reasons.append('换算系数缺失、非数值或非正')
    if y is None or y <= 0:
        reasons.append('换算28d强度缺失、非数值或非正')
    delta = abs(a*b-y) if None not in (a, b, y) else None
    if delta is not None and delta > tolerance + 1e-9:
        reasons.append('换算差值超出0.1 MPa')
    if number(row.get('ConvertFlag')) != 0:
        reasons.append('ConvertFlag未明确为0')
    if row.get('CS_Spec') in (None, '', '-', '/', 0):
        reasons.append('试件规格缺失或占位')
    return {'passed': not reasons, 'absolute_delta_mpa': delta, 'reasons': reasons}


def apply_review(record, decision, available_evidence):
    """校验人工决策：绑定文件哈希/记录，要求核查人、日期及真实可用证据。"""
    if decision.get('record_id') != record['record_id'] or decision.get('source_sha256') != record['source_sha256']:
        raise ValueError('人工决策的记录ID或源文件哈希不匹配')
    if not all(decision.get(k) for k in ('reviewer', 'reviewed_at', 'reason', 'evidence')):
        raise ValueError('缺少核查人、日期、理由或证据')
    for item in decision['evidence']:
        if item.get('sha256') not in available_evidence or not item.get('location'):
            raise ValueError('证据文件未提供或缺少页码/表格定位')
    state = decision.get('status')
    if state not in (PENDING, VERIFIED, EXCLUDED):
        raise ValueError('无效可靠状态')
    if state == VERIFIED:
        if record['casting_env'] != 'Underwater' or not record['conversion']['passed']:
            raise ValueError('机械条件或水下浇筑条件未通过')
        if any(decision.get('checks', {}).get(k) != '已核实' for k in ASPECTS):
            raise ValueError('五项证据核查未全部完成')
        if decision.get('density_assumption') != '已核实未使用假设密度反算':
            raise ValueError('密度假设核查未通过')
        if decision.get('unresolved_reasons') != []:
            raise ValueError('未解决问题必须显式为空列表')
    updated = {**record, 'status': state, 'manual_review': decision,
               'density_assumption_status': decision.get('density_assumption', UNKNOWN)}
    if state == VERIFIED:
        updated.update(source_evidence_status='已核实',strength_basis_status='已核实',
                       feature_units_status='已核实',zero_semantics_status='已核实',
                       prior_pending_reasons=record.get('pending_reasons',[]),pending_reasons=[])
    return updated
