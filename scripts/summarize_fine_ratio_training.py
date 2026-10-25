"""同128条、同折的新增质量比配对比较，保留弱结果与来源差异。"""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile
import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.baselines import digest
from concrete_audit.scope_results import validate_fold_layout, aggregate_predictions
from concrete_audit.paired_features import paired_error_changes
from concrete_audit.material_features import append_excel_row


def run(output):
    output = Path(output)
    reference = ROOT / 'outputs/scope-strict-train-only-v1'
    extended = ROOT / 'outputs/fine-ratio-train-only-v1'
    names = ['training_scope.json', 'nested_cv_results.json', 'training_oof_predictions.json', 'summary.json', 'manifest.json', 'config_snapshot.json']
    protected = {}
    data = []
    for folder in [reference, extended]:
        values = []
        for name in names:
            path = folder / name
            protected[str(path)] = digest(path)
            values.append(json.loads(path.read_text()))
        scope, reports, predictions, summary, manifest, config = values
        assert not summary['test_set_evaluated'] and not summary['input_or_reliability_changed']
        for p, h in manifest['inputs_sha256'].items():
            assert digest(p) == h, p
            protected[p] = h
        data.append(values)
    before_scope, before_reports, before_predictions, _, _, before_config = data[0]
    after_scope, after_reports, after_predictions, _, _, after_config = data[1]
    assert before_scope['train_sample_ids'] == after_scope['train_sample_ids']
    assert before_scope['protected_test_ids'] == after_scope['protected_test_ids']
    expected, test_ids = set(before_scope['train_sample_ids']), set(before_scope['protected_test_ids'])
    fold_plan_path = ROOT / 'outputs/casting-scope-gate-v1/training_plan.json'
    fold_plan = json.loads(fold_plan_path.read_text())['plans']['strict_evidence_scope']['cv']
    protected[str(fold_plan_path)] = digest(fold_plan_path)
    validate_fold_layout(before_reports + after_reports, expected, fold_plan)
    split_path = ROOT / 'outputs/exploratory-baselines-v2/split_manifest.json'
    groups = {r['sample_id']: r['leakage_group'] for r in json.loads(split_path.read_text())['rows']}
    protected[str(split_path)] = digest(split_path)
    before_scores, before_groups = aggregate_predictions(before_predictions, groups, expected, test_ids,
        {(f, m) for f in before_config['feature_sets'] for m in before_config['models']})
    after_scores, after_groups = aggregate_predictions(after_predictions, groups, expected, test_ids,
        {(f, m) for f in after_config['feature_sets'] for m in after_config['models']})
    pairs = [('core2', 'core_plus_fine_ratio'), ('core_plus_computed_ratio', 'core_plus_fine_and_water_cement_ratio')]
    comparisons, source_changes, fold_changes, paired_rows = [], [], [], []
    for before_set, after_set in pairs:
        for model in before_config['models']:
            old = [r for r in before_predictions if (r['feature_set'], r['model']) == (before_set, model)]
            new = [r for r in after_predictions if (r['feature_set'], r['model']) == (after_set, model)]
            change = paired_error_changes(old, new, groups)
            a = next(r for r in before_scores if (r['feature_set'], r['model']) == (before_set, model))
            b = next(r for r in after_scores if (r['feature_set'], r['model']) == (after_set, model))
            comparisons.append({'baseline_representation': before_set, 'extended_representation': after_set, 'model': model,
                'n': len(expected), 'MAE_before_MPa': a['MAE_MPa'], 'MAE_after_MPa': b['MAE_MPa'],
                'delta_MAE_after_minus_before_MPa': change['delta_sample_MAE_MPa'],
                'R2_before': a['R2'], 'R2_after': b['R2'], 'MSE_after_MPa2': b['MSE_MPa2'],
                'RMSE_after_MPa': b['RMSE_MPa'], 'MAPE_after_percent': b['MAPE_percent'],
                'source_macro_MAE_before_MPa': a['source_macro_MAE_MPa'],
                'source_macro_MAE_after_MPa': b['source_macro_MAE_MPa'],
                'delta_source_macro_MAE_MPa': change['delta_source_macro_MAE_MPa'],
                'source_groups_improved': change['source_groups_improved'],
                'source_groups_worsened': change['source_groups_worsened'],
                'negative_predictions_after': sum(r['oof_prediction_MPa'] < 0 for r in new)})
            source_changes += [{'baseline_representation': before_set, 'model': model, **r} for r in change['source_changes']]
            left, right = [{r['sample_id']: r for r in values} for values in [old, new]]
            for sid in before_scope['train_sample_ids']:
                aa, bb = left[sid], right[sid]
                paired_rows.append({'sample_id': sid, 'source_row': aa['source_row'], 'research_group': aa['research_group'],
                    'leakage_group': groups[sid], 'baseline_representation': before_set, 'model': model,
                    'actual_provisional_MPa': aa['actual_provisional_MPa'],
                    'prediction_before_MPa': aa['oof_prediction_MPa'], 'prediction_after_MPa': bb['oof_prediction_MPa']})
                if model == 'mean' and not np.isclose(aa['oof_prediction_MPa'], bb['oof_prediction_MPa'], rtol=0, atol=1e-12):
                    raise ValueError('均值对照预测发生变化，可能并非同样本同折')
            ar = next(r for r in before_reports if (r['feature_set'], r['model']) == (before_set, model))
            br = next(r for r in after_reports if (r['feature_set'], r['model']) == (after_set, model))
            for f, g in zip(ar['folds'], br['folds']):
                fold_changes.append({'baseline_representation': before_set, 'model': model, 'fold': f['fold'],
                    'validation_n': f['validation_n'], 'validation_groups': f['validation_groups'],
                    'MAE_before_MPa': f['metrics']['MAE_MPa'], 'MAE_after_MPa': g['metrics']['MAE_MPa'],
                    'delta_MAE_MPa': g['metrics']['MAE_MPa'] - f['metrics']['MAE_MPa'],
                    'R2_before': f['metrics']['R2'], 'R2_after': g['metrics']['R2'],
                    'best_params_before': f['best_params'], 'best_params_after': g['best_params']})
    output.mkdir(parents=True, exist_ok=False)
    summary = {'training_records': len(expected), 'source_groups': len({groups[s] for s in expected}),
               'new_experiments': len(after_reports), 'new_outer_validation_folds': 60,
               'new_oof_predictions': len(after_predictions), 'paired_comparisons': len(comparisons),
               'protected_test_records': len(test_ids), 'test_evaluated': False, 'new_trusted_samples': 0,
               'mean_control_predictions_unchanged': True, 'same_fold_membership_verified': True,
               'best_new_by_MAE_exploratory_only': min(after_scores, key=lambda r: r['MAE_MPa']),
               'statistical_significance_claimed': False}
    for name, value in [('paired_model_metrics.json', comparisons), ('paired_source_changes.json', source_changes),
                        ('paired_fold_changes.json', fold_changes), ('paired_oof_predictions.json', paired_rows),
                        ('new_model_metrics.json', after_scores), ('new_source_metrics.json', after_groups),
                        ('summary.json', summary), ('manifest.json', {'inputs_sha256': protected, 'script_sha256': digest(__file__)})]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    book = openpyxl.Workbook()
    book.remove(book.active)
    for title, rows in [('加入前后模型指标', comparisons), ('来源组变化', source_changes),
                        ('相同外折与超参', fold_changes), ('逐记录配对预测', paired_rows), ('新模型完整指标', after_scores)]:
        sheet = book.create_sheet(title)
        keys = list(rows[0])
        append_excel_row(sheet, keys)
        for r in rows:
            append_excel_row(sheet, [json.dumps(r[k], ensure_ascii=False) if isinstance(r[k], (dict, list)) else r[k] for k in keys])
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
    book.save(output / '细骨料质量比加入前后对比.xlsx')
    lines = ['# 细骨料/水泥质量比加入前后的配对验证', '',
             '实际完成12项新增实验、60个外层验证折、1536条新折外预测；原有12项核心输入实验原样复用。两组结果均为相同128条严格范围训练记录、相同5个来源组、相同5外折×4内折及既定网格。原145条测试未预测/评分，原始与可靠状态未改。', '',
             '新增物理特征有原文依据，但仍待人工确认；没有自动授予整条数据可信资格。下列均为训练内折外成绩，不是最终泛化成绩。']
    for before_set, after_set in pairs:
        lines += ['', '## ' + ('不加计算水灰比' if before_set == 'core2' else '保留计算水灰比'), '',
                  '|模型|原MAE|新MAE|MAE变化（新−原）|原R²|新R²|新来源等权MAE|改善来源组/5|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for model in before_config['models']:
            r = next(r for r in comparisons if r['baseline_representation'] == before_set and r['model'] == model)
            lines.append(f'|{model}|{r["MAE_before_MPa"]:.3f}|{r["MAE_after_MPa"]:.3f}|{r["delta_MAE_after_minus_before_MPa"]:+.3f}|{r["R2_before"]:.4f}|{r["R2_after"]:.4f}|{r["source_macro_MAE_after_MPa"]:.3f}|{r["source_groups_improved"]}|')
    best = summary['best_new_by_MAE_exploratory_only']
    lines += ['', '## 本轮结论和限制', '',
              f'新增方案按MAE最小的是{best["model"]}/{best["feature_set"]}，MAE={best["MAE_MPa"]:.3f}MPa，R²={best["R2"]:.4f}；来源等权MAE={best["source_macro_MAE_MPa"]:.3f}MPa，最差来源组MAE={best["worst_source_MAE_MPa"]:.3f}MPa。',
              '本轮所有新增配置的R²仍为负。部分配置MAE下降不等于已具备可靠解释或外推能力，也不能宣布所有算法都受益。不同指标和不同来源的变化必须同时查看。',
              '均值模型的逐条预测完全相同，为同样本同折对照检查。算法、输入/MLP目标缩放仍在每个内训练折拟合，未用前一轮外折缩放器替代内折拟合，也未改网格或种子追分。',
              '原文细骨料比覆盖完整，但仍只有5组，Nasr占104/128。个别来源改善不足以证明可泛化特征增益；不以128条记录假装128个独立来源，不做显著性声明。',
              '大误差及负预测均保留，没有截断后重新报分或按误差删样本。暂定强度换算、原核心用量和材料体系差异仍未全部解决。', '',
              '## 代码与复现', '',
              'evidence_train.run逐一核对范围、原训练ID、证据清单和PDF版本，重建原文质量比并比较缓存值；align_feature_matrix只接受固定三列且拒绝测试/目标混入；extended_estimator复用旧模型网格，仅替换输入变换器。',
              'paired_error_changes按同一ID和同一暂定目标配对，分别统计样本加权和来源组等权误差。全体对照与新模型的外折均核对事前预案。', '',
              '```bash',
              '/workspace/.venv-underwater-ml/bin/python scripts/run_evidence_training.py --output outputs/new-fine-ratio-training',
              'PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v',
              '```', '',
              '实际训练结果在outputs/fine-ratio-train-only-v1；本脚本只读汇总该目录与outputs/scope-strict-train-only-v1。Excel含模型指标、来源变化、相同外折超参和1536条逐记录配对预测。', '',
              '下一步宜先按来源解释误差及数据适用范围，检查可追溯的胶凝材料组成/纤维等关键条件；暂不继续扩大调参或查看封存测试。']
    cache = Path(tempfile.gettempdir()) / 'underwater-concrete-plot-cache'
    cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, (before_set, _) in zip(axes, pairs):
        rows = [next(r for r in comparisons if r['baseline_representation'] == before_set and r['model'] == m) for m in before_config['models']]
        x = np.arange(len(rows))
        ax.bar(x-.18, [r['MAE_before_MPa'] for r in rows], .36, label='Before')
        ax.bar(x+.18, [r['MAE_after_MPa'] for r in rows], .36, label='Add evidence fine/cement ratio')
        ax.set_xticks(x, before_config['models'], rotation=25)
        ax.set_ylabel('Nested grouped OOF MAE (MPa)')
        ax.set_title('Same 128 samples: ' + before_set)
        ax.legend(fontsize=8)
    fig.savefig(output / 'paired_fine_ratio_mae.png', dpi=180)
    plt.close(fig)
    lines += ['', '![同样本同折的MAE对比](paired_fine_ratio_mae.png)']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    if any(digest(p) != h for p, h in protected.items()):
        raise RuntimeError('汇总过程中源文件改变')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
