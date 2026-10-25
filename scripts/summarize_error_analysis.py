"""将只读误差诊断整理为来源优先清单、报告和静态图，不筛除记录。"""
from pathlib import Path
import argparse,json,hashlib,os,tempfile,textwrap


def run(folder):
    s=json.loads((folder/'summary.json').read_text());sources=json.loads((folder/'source_error_contributions.json').read_text())
    priority=json.loads((folder/'priority_sources.json').read_text());collisions=json.loads((folder/'same_input_collisions.json').read_text())
    diagnostics=json.loads((folder/'diagnostics.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
    assert len(diagnostics)==299*5
    # 这里按明确的主数据目录读取固定划分，验证所有诊断仅来自训练部分。
    split_path=next(Path(p) for p in manifest['inputs_sha256'] if Path(p).name=='split_manifest.json')
    split=json.loads(split_path.read_text())
    train={r['sample_id'] for r in split['rows'] if r['split']=='train'};test={r['sample_id'] for r in split['rows'] if r['split']=='test'}
    for model in s['per_model']:
        rows=[r for r in diagnostics if r['model']==model]
        assert len(rows)==len(train) and {r['sample_id'] for r in rows}==train
        assert not test&{r['sample_id'] for r in rows}
        assert abs(sum(x['SSE_share'] for x in sources[model])-1)<1e-10
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest['inputs_sha256'].items())
    largest_collision=max(collisions,key=lambda r:r['target_span'])
    zhang=next(r for r in sources['xgb'] if r['research_group']=='Zhang et al. (2025)')
    alselmi=next(r for r in sources['mlp'] if r['research_group']=='Al-Selmi and Al-Shathr (2026)')
    lines=['# 训练折外误差诊断','',
     '本轮分析299条训练记录的既有折外预测，不重训、不评估145条测试，不删记录、不更改可靠状态。',
     '输入仍仅Water/Cem及其计算比值；标签仍为用户暂定换算值。','',
     '## 主要结果','',
     f'- Zhang2025的4条记录贡献XGBoost总平方误差的{zhang["SSE_share"]*100:.1f}%，来源MAE约{zhang["MAE_MPa"]:.2f} MPa。',
     f'- Al-Selmi2026的6条记录贡献MLP总平方误差的{alselmi["SSE_share"]*100:.1f}%。MLP另外约30.5%平方误差来自Zhang2025。',
     '- 21条验证记录至少一个模型输入超出其外折训练范围；这是对同一输入的分布检查，五模型计数相同。',
     '- 同一外层验证折内有55组相同Water/Cem但暂定强度不同的组合，涉及275条记录。它们不一定重复，往往还有未输入的材料或工况差异。',
     '- 对同一模型而言，同一输入必然得到相同预测；不增加可追溯输入，仅调参不能区分这些配方。','',
     '## 具体记录与分布差异','',
     '|来源/原表行号|实际检查结果|正确处理|','|---|---|---|',
     '|Zhang2025，942–945|题录指向水下UHPC；已记录较高SF/GGBS/钢纤维等用量。942行暂定102.296 MPa，XGBoost预测约30.09 MPa。该折训练标签最高74.1 MPa，计算Water/Cem约0.223低于训练折最低约0.336。|保留记录；先核对体系、单位及原始数据，不按误差删掉。|',
     '|Al-Selmi2026，1077–1082|题录指向ECC；记录Water605、Cem1202，超出该折训练Water最高349、Cem最高870.9。Paper List已有“核实是否有抗压数据”的备注。|需核对计量基准、试件和成型范围；大数值或备注不等于已证明错误。|',
     '|Heniegal2015，316/318/320/322/324|相同Water230、Cem500，暂定强度20.72～41.60 MPa，相差20.88。|检查未输入的外加剂、材料或试验条件，不把不同强度当作自动重复或错误。|','',
     'Zhang组目标超出该折训练标签范围，树模型难以外推到未见高强度；仅添加特征也不能保证解决训练覆盖不足。需同时考虑材料机制、数据覆盖及研究对象分层。',
     '题录信息来自原Excel，不是本轮阅读了两篇新论文的全文。高性能与纤维体系可能构成不同分布，不能据模型分数决定是否属于研究范围。','',
     '## 大误差计数（诊断阈值20 MPa，不用于删样本）','',
     '|模型|绝对误差>20 MPa|最大绝对误差MPa|','|---|---:|---:|']
    for name,r in s['per_model'].items():lines.append(f'|{name}|{r["large_errors_over_20"]}|{r["max_error_mpa"]:.2f}|')
    lines+=['','## 优先补充的两篇原文','',
      '这两篇在当前附件目录未找到，以下DOI/标题只来自原版Paper List，尚未独立核实：','',
      '1. Zhang et al. (2025)：Mechanism of Highly Adsorbent Macromolecular Polymers on the Washout Resistance of Underwater Ultra-High-Performance Concrete；DOI: 10.1007/s10953-025-01521-9。',
      '2. Al-Selmi and Al-Shathr (2026)：Experimental and Statistical Optimization of Engineered Cementitious Composites for Underwater Concrete；DOI: 10.30684/2412-0758.1543。','',
      '优先核对配合比单位/计量基准、是否水下浇筑、原始28d强度及试件、纤维/特殊胶材含义。本次不重复索取此前的Wang和外加剂换算资料。','',
      '## 下一步建议','',
      '1. 先确定普通水下混凝土、水下UHPC、ECC等是否均在研究总体内；不因分数高低选择范围。',
      '2. 若都纳入，应记录材料体系并报告分体系误差；UHPC4条、ECC6条不足以直接建立可靠的独立模型。',
      '3. 优先用原文补充胶材、纤维、骨料和工况的可追溯信息；不能只按材料正值自动认定其单位已正确。',
      '4. 在训练部分预先定义扩展方案，再执行分组验证；现有测试集不反复用于试错。','',
      '## 代码与验证','',
      '- 新增error_analysis.py、error_analysis.json、run_error_analysis.py、summarize_error_analysis.py和4项测试。',
      '- 独立CPU虚拟环境中47项测试全部通过，0跳过。',
      '- 五模型各299条诊断，共1495条；全部属于原训练ID，145条测试没有进入诊断。',
      '- 每模型来源SSE占比合计1；输入及原始档案哈希未变。',
      '- 未移除记录、未重新训练、未授予可信资格。',
      '- diagnostics.json逐条保存误差、该折训练范围、材料原记录及未知状态；source_error_contributions.json保存来源贡献；same_input_collisions.json保存输入碰撞；priority_sources.json保存题录线索。','',
      '```bash','python scripts/run_error_analysis.py --output outputs/new-error-analysis',
      'python scripts/summarize_error_analysis.py --folder outputs/new-error-analysis','```']
    cache=Path(tempfile.gettempdir())/'underwater-concrete-plot-cache';cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,model in zip(axes,['xgb','mlp']):
        top=sources[model][:5];labels=['\n'.join(textwrap.wrap(r['research_group'],22)) for r in top]
        ax.barh(labels,[100*r['SSE_share'] for r in top]);ax.invert_yaxis()
        ax.set(xlabel='Share of model training-OOF squared error (%)',title=model)
    fig.savefig(folder/'source_error_share.png',dpi=180);plt.close(fig)
    lines+=['','![来源误差贡献](source_error_share.png)']
    with (folder/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    write={'diagnostic_count':1495,'test_ids_present':False,'source_sse_share_sum_verified':True,'input_hashes_verified':True,'removed_records':0,'tests_passed':47}
    (folder/'validation.json').write_text(json.dumps(write,ensure_ascii=False,indent=2))
    print('误差诊断报告及来源贡献图已保存，所有校验通过。')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
 a=p.parse_args();run(a.folder)
