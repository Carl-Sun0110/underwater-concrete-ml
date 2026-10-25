# 小批文献核查线索与水泥零值假设

本轮根据Baseline V1既有折外误差优先整理四篇缺少原文的来源；没有再次使用留出成绩选择模型。全部引用是原表元数据，未冒充PDF核实。

实际整理4篇、26条基线记录。网络请求返回代理Tunnel 403，未下载到新论文；不能判断出版商是否开放或DOI是否正确。

已发现的原表线索：
- Zhang 2023：8条水下基线记录中，747/751/755/759行为Cem0、GGBS550、AlkaliAct189。与水泥为550的A系列并列，提示C系列可能为无水泥碱激发体系；不能仅凭这些列或题名确认真实未掺。
- 751行Water64，其余上述C系列记录Water128；应核对原文表和有效水/激发剂溶液定义，不能按邻行相似性直接改成128。
- Wang 2024：790行CastMethod明确含Mortar、预置骨料骨架灌浆，Paper List备注也称砂浆强度；需确认标签代表砂浆还是整体混凝土。尚不据此删样本或排除论文。
- Horszczaruk 2017：Paper List备注称CastingEnv=Underwater为推定。当前5条记录仍是扩展探索记录，不视为直接水下成型已核实。
- Horszczaruk 2023：12条高影响记录需核对水下修补、压力养护与成型方式及原始强度表；不能由论文题名概括整组条件。

## 单一假设的训练内对照

复用Baseline V1.1扩展训练320条、23来源组和同一五折，锁定147条论文留出。仅对指定4条记录测试Cem0照录；其他未知零值规则不动。无参数搜索，无标签改动，不部署假设模型。
|特征|模型|基线MAE MPa|假设MAE MPa|差值|假设R²|
|---|---|---:|---:|---:|---:|
|core|rf|10.033|9.970|-0.063|-0.039|
|core|xgb|9.118|9.018|-0.100|0.092|
|core|catboost|9.466|9.330|-0.135|0.087|
|materials|rf|10.051|9.819|-0.232|-0.070|
|materials|xgb|9.666|9.530|-0.135|-0.004|
|materials|catboost|9.756|9.627|-0.129|0.024|

两特征XGBoost MAE9.118→9.018 MPa，变化较小。分数不能证明真零；方案受既往误差启发且在同一训练CV评估，存在适应性偏差，非最终泛化结果。
125项测试通过，六组旧流水线逐条预测精确复现，原始输入/标签/正式可靠状态和已发布预测模型均未变。

## 最优先需要的材料

Zhongzhe Zhang等，2023，Effect of antiwashout admixture on compressive strength of alkali-activated underwater non-dispersible concrete，Magazine of Concrete Research。原表DOI：10.1680/jmacr.22.00239（本轮无法联网核验）。
优先核查配合比表A/C系列、水泥和矿渣列、激发剂溶液组成与水量定义、原始28天强度表及150mm试件。取得这一篇原文即可先解决当前4条水泥零值的关键含义，无须等待全部论文。
其余三篇题名、DOI照录、需要核查的记录和单元格位置见Excel与paper_review_summary.json。

新增文件：configs/priority_evidence_zero_v1.json、src/concrete_audit/evidence_batch.py、scripts/run_priority_evidence_zero.py、scripts/summarize_priority_evidence_zero.py、tests/test_evidence_batch.py；README增加本轮说明。
函数说明：flags只标注原表待核查线索；request_metadata真实记录请求或失败；run绑定原表单元格并执行单一训练内假设；summarize复算指标并检查旧流水线、留出隔离及原始哈希。
