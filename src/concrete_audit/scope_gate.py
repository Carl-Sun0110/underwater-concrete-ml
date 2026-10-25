"""水下浇筑范围门禁：只判断实验范围，不授予整条数据可信资格。"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import openpyxl

from .field_review import digest, validate_evidence
from .material_features import append_excel_row

DIRECT = '有直接水下浇筑证据_待人工确认'
NON_DIRECT = '非直接水下浇筑实验_隔离'
UNKNOWN = '浇筑范围未知_待核查'
MODES = ('strict_evidence_scope', 'isolate_only_exploratory')


def build_records(labels, claims, pages, papers):
    """逐配方校验来源和引文，未编录记录留未知，不由Underwater标签推定。"""
    by_row = {r['source_row']: r for r in labels}
    if len(by_row) != len(labels) or len({r['sample_id'] for r in labels}) != len(labels):
        raise ValueError('范围输入身份重复')
    evidence = {}
    for claim in claims:
        row = claim['source_row']
        if row not in by_row or row in evidence:
            raise ValueError('范围声明行不存在或重复/冲突')
        raw = by_row[row]['original_values']
        if (claim['specimen_id'], claim['registry_pdf_filename']) != (raw['Specimen ID'], raw['PDF filename']):
            raise ValueError('范围声明配方或论文身份错配')
        if claim['scope_state'] not in (DIRECT, NON_DIRECT):
            raise ValueError('已编录范围声明状态无效')
        paper = claim['paper']
        validate_evidence(claim, pages[paper], papers[paper]['sha256'])
        evidence[row] = claim
    records = []
    for label in labels:
        row, raw = label['source_row'], label['original_values']
        claim = evidence.get(row)
        records.append({'sample_id': label['sample_id'], 'source_row': row,
                        'specimen_id': raw['Specimen ID'], 'registry_pdf_filename': raw['PDF filename'],
                        'scope_state': claim['scope_state'] if claim else UNKNOWN,
                        'scope_reason': claim['reason'] if claim else '尚未编录适用于该配方的直接浇筑程序证据；不等于原文没有报告。',
                        'paper': claim['paper'] if claim else None, 'evidence': claim['evidence'] if claim else [],
                        'prior_reliability_status': label['prior_reliability_status'],
                        'human_confirmation': '待核查', 'trusted_qualification_granted': False})
    return records


def scope_allows(state, mode):
    """严格视图拒绝未知；仅隔离视图保留未知但只能标为探索性。"""
    if mode not in MODES or state not in (DIRECT, NON_DIRECT, UNKNOWN):
        raise ValueError('范围模式/状态无效')
    return state == DIRECT if mode == 'strict_evidence_scope' else state != NON_DIRECT


def select_training_scope(rows, groups, records, mode):
    """过滤固定训练部分，身份必须完全对应；不读取目标或改变分组。"""
    by_id = {r['sample_id']: r for r in records}
    if len(by_id) != len(records) or len(rows) != len(groups):
        raise ValueError('范围档案重复或分组长度错配')
    selected, selected_groups, excluded = [], [], []
    for row, group in zip(rows, groups):
        gate = by_id.get(row['sample_id'])
        raw = row['original_values']
        if gate is None or (gate['source_row'], gate['specimen_id'], gate['registry_pdf_filename']) != (
                row['source_row'], raw['Specimen ID'], raw['PDF filename']):
            raise ValueError('训练记录不在范围档案或身份错配')
        if scope_allows(gate['scope_state'], mode):
            selected.append(row)
            selected_groups.append(group)
        else:
            excluded.append({'sample_id': row['sample_id'], 'source_row': row['source_row'], 'scope_state': gate['scope_state']})
    if not selected:
        raise ValueError('范围门禁后无训练记录')
    return selected, selected_groups, excluded


def check_training_request(config):
    """未来训练须显式指定范围；历史复现例外必须单独声明，不能默认绕过。"""
    request = config.get('scope_gate')
    if config.get('historical_scope_reproduction_only') is True:
        if request is not None:
            raise ValueError('历史复现与新范围配置不能混用')
        return None
    if not isinstance(request, dict) or request.get('mode') not in MODES:
        raise ValueError('新训练必须指定scope_gate；旧范围仅能显式声明historical_scope_reproduction_only用于历史复现')
    if not request.get('registry') or not request.get('registry_sha256'):
        raise ValueError('范围门禁缺少档案路径或版本指纹')
    return request


def load_training_gate(request, labels_path, split_path):
    """把范围审查绑定标签/划分和原文版本；文件变更必须重新审查。"""
    path = Path(request['registry'])
    if digest(path) != request['registry_sha256']:
        raise ValueError('范围档案版本指纹不匹配')
    registry = json.loads(path.read_text())
    if registry['label_file_sha256'] != digest(labels_path) or registry['split_file_sha256'] != digest(split_path):
        raise ValueError('范围审查不适用于当前标签或划分')
    protected = registry['protected_inputs_sha256']
    if any(digest(p) != expected for p, expected in protected.items()):
        raise ValueError('范围审查上游输入已变更')
    return registry['records'], {**protected, str(path.resolve()): digest(path)}


def run(labels_path, split_path, config_path, historical_cv_path, output):
    """只读构建范围档案、两种清单和训练预案；不创建模型或预测。"""
    labels_path, split_path, config_path, historical_cv_path, output = map(
        Path, (labels_path, split_path, config_path, historical_cv_path, output))
    config = json.loads(config_path.read_text())
    protected = [labels_path, split_path, config_path, historical_cv_path,
                 *map(Path, config['protected_inputs']), *[Path(p['path']) for p in config['papers'].values()]]
    before = {str(p.resolve()): digest(p) for p in protected}
    pages = {}
    for key, paper in config['papers'].items():
        if digest(paper['path']) != paper['sha256']:
            raise ValueError('范围证据PDF版本不符')
        result = subprocess.run(['pdftotext', '-layout', paper['path'], '-'], check=True, capture_output=True)
        pages[key] = result.stdout.decode().split('\f')
    labels = json.loads(labels_path.read_text())
    split = json.loads(split_path.read_text())['rows']
    by_id = {r['sample_id']: r for r in labels}
    split_by_id = {r['sample_id']: r for r in split}
    if len(split_by_id) != len(split) or set(split_by_id) - set(by_id):
        raise ValueError('划分存在重复/未知样本')
    if any(r['split'] not in ('train', 'test') for r in split):
        raise ValueError('划分状态错误')
    train, test = [[r for r in split if r['split'] == name] for name in ('train', 'test')]
    if any({r[field] for r in train} & {r[field] for r in test}
           for field in ('leakage_group', 'research_group', 'pdf_filename')):
        raise ValueError('固定划分来源泄漏')
    records = build_records(labels, config['claims'], pages, config['papers'])
    for record in records:
        record['historical_split'] = split_by_id.get(record['sample_id'], {}).get('split', 'core_unknown_not_in_split')
    historical = json.loads(historical_cv_path.read_text())[0]['folds']
    folds = {f['fold']: set(f['validation_sample_ids']) for f in historical}
    train_ids = {r['sample_id'] for r in train}
    if set().union(*folds.values()) != train_ids or sum(map(len, folds.values())) != len(train_ids):
        raise ValueError('历史折未完整且唯一覆盖固定训练部分')
    views = {}
    for mode in MODES:
        kept = [r for r in records if scope_allows(r['scope_state'], mode)]
        kept_ids = {r['sample_id'] for r in kept}
        train_kept = [r for r in train if r['sample_id'] in kept_ids]
        test_kept = [r for r in test if r['sample_id'] in kept_ids]
        historical_counts = {str(k): len(v & kept_ids) for k, v in folds.items()}
        views[mode] = {'scope_sample_ids': [r['sample_id'] for r in kept], 'scope_records': len(kept),
                       'train_sample_ids': [r['sample_id'] for r in train_kept], 'train_records': len(train_kept),
                       'train_groups': dict(Counter(r['leakage_group'] for r in train_kept)),
                       'test_scope_ids_for_audit_only': [r['sample_id'] for r in test_kept],
                       'test_scope_count_for_audit_only': len(test_kept),
                       'protected_original_test_ids': sorted(r['sample_id'] for r in test),
                       'historical_outer_validation_counts': historical_counts,
                       'historical_folds_reusable': all(historical_counts.values()),
                       'formal_trusted_dataset': False, 'training_executed': False}
    summary = {'source_records': len(labels), 'scope_counts': dict(Counter(r['scope_state'] for r in records)),
               'historical_train_records': len(train), 'protected_test_records': len(test),
               'new_trusted_samples': 0, 'models_trained': False, 'test_evaluated': False,
               'inputs_unchanged': all(digest(p) == h for p, h in before.items())}
    if not summary['inputs_unchanged']:
        raise RuntimeError('范围审查输入变化')
    output.mkdir(parents=True, exist_ok=False)
    registry = {'schema_version': 1, 'label_file_sha256': digest(labels_path), 'split_file_sha256': digest(split_path),
                'protected_inputs_sha256': before, 'records': records,
                'scope_is_not_reliability_approval': True, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    for name, data in [('scope_registry.json', registry), ('scope_views.json', views), ('summary.json', summary),
                       ('config_snapshot.json', config), ('manifest.json', {'inputs_sha256': before, 'module_sha256': digest(__file__)})]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = '458条范围审查'
    append_excel_row(sheet, ['原始行', '配方', '论文', '历史划分', '范围状态', '原因', '原可靠状态', '新增可信资格'])
    for r in records:
        append_excel_row(sheet, [r['source_row'], r['specimen_id'], r['registry_pdf_filename'], r['historical_split'],
                                 r['scope_state'], r['scope_reason'], r['prior_reliability_status'], '否'])
    for title, state in [('隔离清单', NON_DIRECT), ('范围未知', UNKNOWN)]:
        sheet = book.create_sheet(title)
        append_excel_row(sheet, ['原始行', '配方', '论文', '原因', 'PDF页码'])
        for r in records:
            if r['scope_state'] == state:
                append_excel_row(sheet, [r['source_row'], r['specimen_id'], r['registry_pdf_filename'], r['scope_reason'],
                                         ','.join(str(e['page']) for e in r['evidence'])])
    for sheet in book:
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
    book.save(output / '浇筑范围审查与隔离清单.xlsx')
    return summary
