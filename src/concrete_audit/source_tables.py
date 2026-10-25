"""新原文的确定性表格读取：配方、龄期、环境和暴露处理分别绑定。"""
import math
import re


def case_rows(text, columns, expected, aliases=None):
    """只解析给定表格片段和明确配方集合；缺列、重复、非有限数均失败。"""
    aliases = aliases or {}
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = aliases.get(parts[0], parts[0])
        if key not in expected:
            continue
        if key in rows or len(parts) < columns + 1:
            raise ValueError('表格配方重复或缺列')
        try:
            values = [float(x.replace('−', '-')) for x in parts[1:columns + 1]]
        except ValueError as exc:
            raise ValueError('表格数字无效') from exc
        if not all(math.isfinite(x) for x in values):
            raise ValueError('表格包含非有限值')
        rows[key] = {'values': values, 'source_line': line, 'printed_id': parts[0]}
    if set(rows) != set(expected):
        raise ValueError(f'表格配方集合不完整：{sorted(set(expected) - set(rows))}')
    return rows


def nasr_identity(specimen_id, moving=False):
    """完整匹配配方和FW/SW后缀；不由强度值反推环境。"""
    pattern = r'(M[1-4]C[12]V[1-4])-(FW|SW)' if moving else r'(M[1-7]C[0-3])-(FW|SW)'
    match = re.fullmatch(pattern, specimen_id)
    if not match:
        raise ValueError('Nasr配方/水环境身份不完整')
    return match.groups()


def nasr_strength(values, environment, age=28):
    """表2按显式列位选择实测强度，不能使用空气强度或相对强度充当目标。"""
    if len(values) != 10 or age not in (7, 28) or environment not in ('FW', 'SW'):
        raise ValueError('强度列位/环境/龄期无效')
    offset = 0 if age == 7 else 5
    return values[offset + {'FW': 0, 'SW': 1}[environment]]


def assaad_protocol(cast_method):
    """按原记录的实验方法绑定已阅读的原文分支，不按目标大小择列。"""
    wash = 'CRD C61 washout (plunge)' in cast_method
    infiltration = 'Water infiltration simulated' in cast_method
    if wash == infiltration:
        raise ValueError('冲洗/渗水分支不唯一，不能核实强度列')
    return 'washout_then_remolded' if wash else 'permeability_then_remolded'


def assaad_strength(values, protocol):
    """Table5九个数值字段中，第3/6列为两种实验后的强度。"""
    if len(values) != 9 or protocol not in ('washout_then_remolded', 'permeability_then_remolded'):
        raise ValueError('Assaad表5或处理分支无效')
    return values[2 if protocol == 'washout_then_remolded' else 5]
