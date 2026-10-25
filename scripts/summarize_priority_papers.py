"""汇总本轮两篇原文证据，原文矛盾与数据库差异分开，绝不改标签。"""
from pathlib import Path
from decimal import Decimal
import json,hashlib
import openpyxl

ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'outputs/priority-paper-review-v1';checksdir=ROOT/'outputs/priority-paper-field-check-v2'
meta=json.loads((folder/'papers.json').read_text());config=json.loads((ROOT/'configs/priority_paper_evidence.json').read_text())
checks=json.loads((checksdir/'field_checks.json').read_text());summary=json.loads((checksdir/'summary.json').read_text());manifest=json.loads((checksdir/'manifest.json').read_text())
for p,h in manifest['inputs_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
for paper in meta:assert hashlib.sha256(Path(paper['path']).read_bytes()).hexdigest()==paper['sha256']
assert len(checks)==168 and len({c['check_id'] for c in checks})==168
assert all(c['human_confirmation']=='待确认' for c in checks)
for row,y in [(942,134.6),(943,128.8),(944,122),(945,97.5),(1080,78)]:
 c=next(c for c in checks if c['source_row']==row and c['field']=='CS_28d')
 assert c['paper_value']==y and c['excel_value']==y and c['status']=='已核实'
assert all(next(c for c in checks if c['source_row']==n and c['field']=='CS_28d')['status']=='待人工确认' for n in [1077,1078,1079,1081,1082])
issues=[{'id':'AS-T4-01','type':'原文内部整数表述不一致','paper':'AlSelmi2026','pdf_page':6,'location':'Table4及表注',
 'observed':'表中OPC1202+FA301=1503 kg/m³，表注写总胶材1502 kg/m³。',
 'treatment':'记录原表和表注两种表述，不擅自改成1502或1503；可能涉及取整/笔误，但原因未知。'},
 {'id':'AS-EQ-01','type':'回归方程与数据表达待澄清','paper':'AlSelmi2026','pdf_page':16,'location':'回归方程及实际百分比直接代入说明',
 'observed':'正文方程72.17−3.50A+2.17B并称直接代实际百分比；A=0.7、B=1.75得73.5175 MPa，而PDF p10明确同配方28d为78 MPa。',
 'treatment':'只记录不一致；不可擅自认定A/B一定应为编码变量，不用方程预测替代实测标签。'},
 {'id':'Z-T3-01','type':'绝对用量来源仍未知','paper':'Zhang2025','pdf_page':5,'location':'Table3表注',
 'observed':'All mix proportions are normalized to cement mass (1.0)。',
 'treatment':'相对配比有据不代表kg/m³有据；保留旧绝对值且标未知，不自行按混凝土密度反算。'},
 {'id':'Z-YEAR-01','type':'题录年份分开记录，非判错','paper':'Zhang2025','pdf_page':1,'location':'首页题录和online日期',
 'observed':'文件/原档案按2025标识；在线日期2025-10-22，印刷卷期为2026年第55卷481–499。',
 'treatment':'保留原文件名和ID，分别记录online_year=2025、print_year=2026，不因年份差异认定为另一来源。'}]
calc={'type':'源文方程复算，不作为标签','formula':'72.17 - 3.50*A_percent + 2.17*B_percent',
      'A_percent':.7,'B_percent':1.75,'computed_MPa':float(Decimal('72.17')-Decimal('3.50')*Decimal('.7')+Decimal('2.17')*Decimal('1.75')),
      'paper_explicit_28d_MPa':78,'used_to_replace_label':False}
assert calc['computed_MPa']==73.5175
# 逐行来源特征标签只在建议层存储，不修改主档/训练数据。
proposals=[]
for n in range(942,946):proposals.append({'source_row':n,'proposed_material_system':'UN-UHPC','casting_evidence':'PDF p5 §2.3',
 'absolute_dosage_basis':'仍未知','action':'保留原始/探索记录，不提升kg/m³可信资格','human_confirmation':'待确认'})
for n in range(1077,1083):proposals.append({'source_row':n,'proposed_material_system':'ECCUWC','casting_evidence':'PDF p6 §2.2',
 'absolute_dosage_basis':'PDF p6 Table4直接标kg/m³；表内矛盾另记','action':'保留原始记录，不因605/1202数值大而删样本','human_confirmation':'待确认'})
for name,obj in [('source_issues.json',issues),('equation_check_not_labels.json',calc),('proposed_system_tags.json',proposals)]:
 (folder/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
text='''# 两篇重点来源原文核查

本轮已实际读取新上传的Zhang与Al-Selmi原文，核查10条来源记录、168个字段槽位。其中138项有定位证据，30项尚未编录；“有定位”不等于支持当前值。
字段辅助状态为81项一致、66项证据不足、21项待人工确认；所有结果仍待必要人工复核，新增正式可信样本0。
未改原始Excel/PDF、可信状态、暂定标签、固定训练/测试划分或任何模型；没有训练或重评测试集。

## 结论一：大误差不能作为删数据依据

### Zhang，Data第942–945行

- 原文§2.3（PDF p5）明确“cast underwater”；§2.4.2（p6）明确100×100×100mm立方体，7/28天。
- Fig8(a)（p15）斜线柱明确标注28天强度：SX1=134.6、SX2=128.8、SX3=122、SX4=97.5 MPa；与Excel原始CS_28d逐条一致。
- 上述为图上明确数字标签，经目视核对，不是从柱高估读；空白柱125.4/119.6/112.6/88.6是7天，不能混用。
- 因此，102.296 MPa等暂定换算标签对应的高原始强度有原文出处，不应仅因模型预测30MPa左右就删样本。
- 但Table3（p5）是相对水泥质量归一化配比，不是kg/m³。原Excel的Cem870.9、Water194.2等绝对值的基准，尚未由本轮原文核实。
- Table1（p4）给出部分材料密度，不能自动代替完整每立方米基准。不得自行假设混凝土总密度或缺失材料密度重建绝对用量。
- 本轮支持的比例包括Water/Cem0.223、SF/Cem0.18、FA/Cem0.06、GGBS/Cem0.19、sand/Cem1.05、steel/Cem0.16、SP/Cem0.035。
- 正文§2.2支持聚合物0/1.5/2.5/3%水泥质量；表行0/0.015/0.025/0.030要结合正文理解，不能当成0.015%。本库AWA_Dos单位是“% or kg/m³”，仍须逐行绑定%及by cement。
- 文件2025标识与2026卷期并不自动矛盾：p1明确在线发表于2025-10-22，印刷为2026年第55卷。旧文件名与ID未改。

### Al-Selmi，Data第1077–1082行

- PDF p6 Table4明确单位kg/m³，Water605、OPC1202、FA301、sand564均直接报告。
- 同表SP18/19.5/21/19.5/21/22.5，AWA10.5/13.5/16.5/10.5/13.5/16.5，PVA19.2或22.4 kg/m³与Excel对应值一致。
- 大用量是论文原表内容，不能仅因超出模型训练范围就判为录入错误；直接有据也不代表原文物理/计量无任何疑问。
- p6正文明确水下浇筑；p7明确100×200mm抗压圆柱和28、90、180、360天，支持Excel的Cy100表示。
- p10正文明确G0.7–1.75在28天为78 MPa、360天为99.6 MPa，因此第1080行原始78有明确依据，不能以摘要长期99.6替代28天。
- 另外5条28天强度来自p12 Fig6无数值标签的蓝柱，本轮目视检查大致高度和龄期，但未独立数字化，不能据此确认73.5、70等小数精度。
- p6命名说明：AWA为胶材质量百分比0.7/0.9/1.1，PVA为体积分数1.5/1.75%；不能把PVA体积分数直接当质量百分比。
- 原文PVA密度1.28g/cm³（p5 Table3）与1.5/1.75vol%对应19.2/22.4kg/m³；这里只记录该具体产品的依据，不跨来源移用密度。

## 结论二：原文内部仍有待澄清之处

1. Al-Selmi p6：表中1202+301=1503，但表注写总胶材1502。保留两者，不猜测是哪一个笔误。
2. 表注水胶比约0.40；605/(1202+301)约0.4025，不能把约值当作无误差的精确等式。
3. p16回归方程写72.17−3.50A+2.17B，并称A/B直接代实际百分比。代0.7与1.75得到73.5175 MPa，与p10该配方实测78不同。回归本身允许残差，此例与“很低残差”的表述及变量尺度需进一步澄清，不能直接断定方程错误或强行改为编码变量。
4. 这些疑点不授权删除原样本或用论文回归预测制造新标签。source_issues.json保存独立问题，不冒充数据库值与原文不一致。

## 对既有模型结果的解释边界

- 两篇确实属于水下特殊材料体系，不只是题名猜测；原文说明分别为UN-UHPC和ECCUWC。
- Zhang主要改变聚合物，Al-Selmi主要改变AWA/PVA，而当前模型只输入Water/Cem，所以遗漏了原文明示的变化因素。
- 这支持后续优先完善材料体系、抗分散剂/聚合物掺量与基准、纤维类型/体积分数等字段；不证明加入后一定提高泛化成绩。
- 两来源仅4条和6条，不能直接各自建立可靠独立模型；也不能仅依据误差高低决定研究范围。
- 本次不更改先前实验，不把“新核查有证据”误写成“旧模型是在全部可信输入上训练”。

## 输出和验证

- ../priority-paper-field-check-v2/小批字段核查.xlsx与field_checks.json：10条、168字段的原值、批注值、原文值、单位、位置、证据、状态与人工确认标记。
- 本目录原文提取文本和关键页图、papers.json、source_issues.json、equation_check_not_labels.json、proposed_system_tags.json。
- 图中明确数值用图像证据、PDF页码、原PDF与图像SHA-256绑定；图注与摘录也匹配真实提取文本。
- 新增3项图像来源/指纹/文本匹配测试；完整测试结果见本目录validation.json。所有原始文件与可靠状态档案哈希不变。
- 此前field-check-v1将复合单位差异机械归为“需转换”；v2改为“待人工确认单位绑定”，不凭数值一致就静默选择%或kg/m³。

## 下一步建议（本轮不执行）

1. 保留全部10条，不按预测误差删数据。特殊体系标签作为提议层，等待研究范围确认。
2. 先将有原文支持的相对掺量、材料体系、纤维含量/基准整理成统一字段映射；未确认的kg/m³继续未知。
3. 扩展到其他来源时逐篇检查定义和单位，不能仅给这10条填写新特征而把其他来源缺失默认为0。
4. 暂不重训或重看测试集；先完成覆盖范围和零/未知语义说明，再制定训练内比较方案。
'''
with (folder/'report.md').open('x',encoding='utf-8') as f:f.write(text)
print('两篇原文核查报告、内部问题和体系建议已保存；输入及原可靠状态保持不变。')
