"""本轮两篇原文阅读证据编录；分清直接值、图上标签、比例和绝对用量。"""
from pathlib import Path
import json,hashlib
import openpyxl
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'outputs/priority-paper-review-v1'
papers={p['key']:{'path':p['path'],'sha256':p['sha256']} for p in json.loads((folder/'papers.json').read_text())}
pages={key:(folder/(key+'.txt')).read_text().split('\f') for key in papers}
inputs=json.loads((ROOT/'configs/provisional_inputs.json').read_text())
registry={r['source_row']:r for r in json.loads(Path(inputs['registry']).read_text())}

def ref(key,page,location,anchor,before=0,after=4,image=False):
 lines=pages[key][page-1].splitlines();i=next(i for i,line in enumerate(lines) if anchor in line)
 d={'page':page,'location':location,'quote':'\n'.join(lines[max(0,i-before):i+after])}
 if image:
  p=folder/f'{key}-p{page}.png';d.update(image_path=str(p),image_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_pdf_sha256=papers[key]['sha256'])
 return d

def direct(value,unit,refs,note='',method='正文/表格明确值'):
 return {'mode':'direct','paper_value':value,'comparison_value':value,'unit':unit,'evidence':refs,'note':note,
         'method':method,'action':'保存字段级辅助一致结论，待人工复核，不提升整条可靠状态。'}

def unresolved(value,unit,refs,note,mode='insufficient',method='定位阅读编录'):
 return {'mode':mode,'paper_value':value,'unit':unit,'evidence':refs,'note':note,'method':method,
         'action':'保留原始值，分开保存证据和疑点，不自动重算/覆盖或删除。'}

samples=[]
# Zhang：原文中不含每立方米绝对基准，不能把表3比例当绝对用量。
zt=ref('Zhang2025',5,'Table3，SX1–SX4及水泥归一化表注','Table 3',after=11,image=True)
zm=ref('Zhang2025',5,'§2.3，水下浇筑','the mixture is cast underwater',before=2,after=5)
zs=ref('Zhang2025',6,'§2.4.2，100mm立方体及龄期','compressive strength tests',before=1,after=4)
zgraph=ref('Zhang2025',15,'Fig8(a)，图上明示28天数值标签','Fig. 8',after=2,image=True)
zpoly=ref('Zhang2025',5,'§2.2，聚合物掺量与基准','set at 0, 0.015',before=5,after=7)
for n,strength,dose in [(942,134.6,0),(943,128.8,1.5),(944,122,2.5),(945,97.5,3)]:
 claims={'CS_28d':direct(strength,'MPa',[zgraph],f'Fig8(a)的SX{n-941}斜线柱为28days，空白柱为7days；读取明确数值标签，非曲线估读。',method='图中明确标注值，助手目视读取，非估读'),
         'CS_Spec':direct('Cu100','-',[zs],'原文100×100×100mm立方体；未验证固定换算系数的方法学依据。'),
         'CastingEnv':direct('Underwater','-',[zm]),
         'w/c':direct(.223,'-',[zt],'原文Water/Cement=0.223。'),
         'AWA_Dos':unresolved(dose,'%', [zpoly],'正文给0%、1.5%、2.5%、3%，与当前数字一致。但本库该列单位为% or kg/m³，需逐条确认绑定为%；不是需要任意缩放数字。',mode='ambiguous'),
         'AWA_Dos_Base':direct('by cement','-',[zpoly],'正文明确掺量相对水泥质量。')}
 for f,ratio in [('Cem',1),('Water',.223),('SF',.18),('FA',.06),('GGBS',.19),('SS',1.05),('StF',.16),('SP',.035)]:
  claims[f]=unresolved(ratio,'相对水泥质量的比例',[zt],
       '原文Table3仅给比例；表注normalized to cement mass(1.0)。当前Excel为kg/m³，未找到足以核实其绝对基准的链条；不以材料密度表代替混凝土每立方米基准。')
 claims['AWA']=unresolved(dose,'%水泥质量',[zpoly,zt],
       '原文聚合物掺量可定位，质量kg/m³仍依赖未核实绝对水泥基准。SX1明确不掺聚合物，但不能以此自动证明其他记录零值。')
 s={'source_row':n,'paper':'Zhang2025','specimen_id':registry[n]['specimen_id'],
    'registry_pdf_filename':registry[n]['pdf_filename'],'fields':['CastingEnv','CS_Spec','CS_28d','w/c','Cem','Water','SF','FA','GGBS','SS','StF','SP','AWA','AWA_Dos','AWA_Dos_Base','Density','Conversion Factor','Converted CS_28d'],
    'claims':claims}
 samples.append(s)
# Al-Selmi：直接保留表4的kg/m³，即使数值大；5个强度值仍仅图像近似。
at=ref('AlSelmi2026',6,'Table4，6配方及kg/m³表头','Table 4.',after=15,image=True)
am=ref('AlSelmi2026',6,'§2.2–2.3，水下浇筑和命名','cast underwater',before=1,after=8)
aSpec=ref('AlSelmi2026',7,'§2.3，抗压试件与龄期','100 × 200',before=2,after=4,image=True)
aGraph=ref('AlSelmi2026',12,'Fig6，蓝柱为28d、其他为90/180/360d','Fig. 6.',after=3,image=True)
a78=ref('AlSelmi2026',10,'§3.2，G0.7–1.75：28d78、360d99.6','78 MPa at 28',before=1,after=4)
for n,sp,awa,pva,dose,vf in [(1077,18,10.5,19.2,.7,1.5),(1078,19.5,13.5,19.2,.9,1.5),
                            (1079,21,16.5,19.2,1.1,1.5),(1080,19.5,10.5,22.4,.7,1.75),
                            (1081,21,13.5,22.4,.9,1.75),(1082,22.5,16.5,22.4,1.1,1.75)]:
 claims={k:direct(v,'kg/m3',[at],'按原文Table4直接报告单位保留；不因数值偏大判错。') for k,v in {'Cem':1202,'FA':301,'RS':564,'Water':605,'SP':sp,'AWA':awa,'SyF':pva}.items()}
 claims['RS']['note']+='S列与细砂对应；仍可保留具体砂种名称核查。'
 claims['CastingEnv']=direct('Underwater','-',[am])
 claims['CS_Spec']=direct('Cy100','-',[aSpec],'原文100×200mm圆柱；不把28d与360d混用。')
 if n==1080:
  claims['CS_28d']=direct(78,'MPa',[a78,aGraph],'正文明确78 MPa为28天、99.6 MPa为360天，不是依据回归方程生成。')
 else:
  claims['CS_28d']=unresolved('Fig6蓝柱，无精确数值标签','MPa',[aGraph],
      '本轮目视确认龄期及大致高度，但不以近似柱高证明Excel中的小数精度；未新生成估读数值。',mode='ambiguous',method='图像读数待复核，非图中明确数字标签')
 claims['w/b']=unresolved('approximately0.40','-',[at],
      '表注给约0.40，按605/(1202+301)约0.4025；记录0.4为近似口径，保留原文而不反向强行精确。',mode='ambiguous')
 claims['AWA_Dos']=unresolved(dose,'%', [am],'第一部分是胶材质量百分比，与当前数字一致；本库单位为% or kg/m³，需逐条绑定为%，不可混用水泥基准。',mode='ambiguous')
 claims['AWA_Dos_Base']=direct('by binder','-',[am])
 s={'source_row':n,'paper':'AlSelmi2026','specimen_id':registry[n]['specimen_id'],
    'registry_pdf_filename':registry[n]['pdf_filename'],
    'fields':['CastingEnv','CS_Spec','CS_28d','Cem','FA','RS','Water','SP','AWA','SyF','w/b','AWA_Dos','AWA_Dos_Base','Density','Conversion Factor','Converted CS_28d'],
    'claims':claims}
 samples.append(s)
config={**{k:inputs[k] for k in ['raw','annotated','registry']},'raw_sha256':hashlib.sha256(Path(inputs['raw']).read_bytes()).hexdigest(),
        'papers':papers,'samples':samples,'scope':'10条来源记录的字段级辅助核查；无人工最终签字；不改原始数据、状态、训练集或模型'}
(ROOT/'configs/priority_paper_evidence.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('已编录10条记录，',sum(len(s['fields']) for s in samples),'字段槽位；',sum(len(s['claims']) for s in samples),'项有定位证据。')
