"""本会话阅读证据的结构化编录；不是自动从整篇论文提取全部数据。"""
from pathlib import Path
import hashlib,json
import openpyxl

ROOT=Path(__file__).resolve().parents[1]
prior=ROOT/'outputs/paper-evidence-01'
papers={x['key']:{'path':x['path'],'sha256':x['sha256']} for x in json.loads((prior/'papers.json').read_text())}
pages={key:(prior/(key+'.txt')).read_text().split('\f') for key in papers}
raw=next(Path('/workspace/attachments/22c7b11d-170d-42eb-97c6-2593982771b2').glob('*.xlsx'))
annotated=next(Path('/workspace/attachments/c841de1c-56a6-40b4-91f2-b1c87ed10763').glob('*.xlsx'))
registry=ROOT/'outputs/final-audit/registry.json'
records={r['source_row']:r for r in json.loads(registry.read_text())}
w=openpyxl.load_workbook(raw,data_only=False)


def ref(key,page,location,anchor,before=0,after=3):
    """摘录直接取自既有PDF文本；找不到锚点就报错，不生成引用内容。"""
    lines=pages[key][page-1].splitlines()
    n=next(i for i,line in enumerate(lines) if anchor in line)
    return {'page':page,'location':location,'quote':'\n'.join(lines[max(0,n-before):n+after])}


def direct(value,unit,evidence,note=''):
    return {'mode':'direct','paper_value':value,'comparison_value':value,'unit':unit,
            'evidence':evidence,'note':note,'action':'字段级一致，仍需人工复核；不升级整条可信资格。'}


def uncertain(value,unit,evidence,note,mode='insufficient',action='保留原值，待人工核查。',method=None):
    d={'mode':mode,'paper_value':value,'unit':unit,'evidence':evidence,'note':note,'action':action}
    if method:d['method']=method
    return d


def calculated(value,unit,evidence,formula,parameters,note,formula_id=None):
    return {'mode':'calculated','paper_value':value,'unit':unit,'evidence':evidence,
            'formula':formula,'formula_id':formula_id,'parameters':parameters,'note':note,
            'action':'独立保存候选计算和参数来源，不回写Excel；人工确认后方可应用。'}


def sample(row,key):
    return {'source_row':row,'paper':key,'specimen_id':records[row]['specimen_id'],
            'registry_pdf_filename':records[row]['pdf_filename'],
            'fields':['CastingEnv','CS_Spec','CS_28d','Cem','Water','RS','CA','SP','AWA','Density',
                      'Conversion Factor','Converted CS_28d'],'claims':{}}

samples=[]
# Sonebi：100%cement、w/cm0.41下的两组明确表格记录。
for row,sp,cs in [(8,1.3,17),(16,3.5,27)]:
 s=sample(row,'Sonebi');c=s['claims'];s['fields']+=['Retarder','w/b']
 table=ref('Sonebi',4,'Table 1，100%cement、w/cm=0.41列','Cement, kg/m3',after=14)
 strength=ref('Sonebi',5,'Table 3，100%cement、对应HRWR剂量行；f_c,uw列',f'{sp:.1f} ',after=2)
 # 1.3/3.5的唯一行也记录该表的列头，避免空气/水下列混淆。
 header=ref('Sonebi',5,'Table 3表头','Table 3',after=7)
 c.update({k:direct(v,'kg/m3',[table]) for k,v in {'Cem':592,'Water':243,'RS':693,'CA':816}.items()})
 c['CS_28d']=direct(cs,'MPa',[header,strength],'括号值为变异系数，未当成强度或误差修正。')
 c['CastingEnv']=direct('Underwater','-',[header,strength,ref('Sonebi',3,'Mixing and test methods','The underwater strength',after=4)])
 c['CS_Spec']=direct('Cy100','-',[ref('Sonebi',3,'Mixing and test methods','100 x 200 mm cylinders',before=1,after=3)],'原文完整尺寸100×200 mm圆柱；Cy100仅为本库表示法。')
 c['w/b']=direct(.41,'-',[header,table],'本配方为100%水泥，胶材定义对应w/cm；未推广至其他胶材。')
 c['SP']=calculated(sp,'L/m3',[header,strength,ref('Sonebi',3,'Materials，HRWR类型与相对密度','a naphthalene-based HRWR',before=2,after=3)],
     'mass = volume × relative_density × reference_water_density',
     {'volume_l_m3':sp,'relative_density':1.21,'reference_water_density_kg_l':None},
     '原文给体积及相对密度；尚需明确相对密度参照/温度和原表质量换算取整规则，当前不生成质量结果。')
 ret=ref('Sonebi',4,'Table 1，缓凝剂行','Set retarder',after=4)
 c['Retarder']=uncertain(85,'mL/100 kg胶材',[ret],
     '原文明确使用缓凝剂；Excel 0不能解释为确认未掺。未取得可用产品密度，85不可直接作kg/m3。',
     mode='conflict',action='标记0的语义冲突；质量用量保持未知，不自动改值。')
 if row==16:
  c['AWA']=calculated(.07,'%胶材质量',[header,strength,table],
      'mass_kg_m3 = percent / 100 × binder_mass_kg_m3',{'percent':.07,'mass_kg_m3':592},
      '原文为Welan gum 0.07%胶材质量；候选结果0.4144与Excel0.41需按精度规则比较。','percent_mass')
 else:
  c['AWA']=uncertain('None（Table 3）','-',[header,strength],
      '原表None适用于该control；需人工确认该0代表未添加AWA而非缺失占位。',mode='ambiguous')
 samples.append(s)
# Yousri：2个普通配方加1个VA特殊配方。
ytable=ref('Yousri',3,'Table 2，表头和完整配方行','Table 2.',after=24)
ymethod=ref('Yousri',3,'Experimental programme续','150 mm cubes',before=1,after=12)
ytext=ref('Yousri',3,'Materials，HRWR质量百分比','0.35 to 0.42%',before=2,after=3)
ygraph=ref('Yousri',8,'Fig.4–5，强度曲线所在页','Concrete compressive strength',after=7)
for row,cement,fa,water,awa,sp in [(197,592,0,211.5,.5,2.1),(203,532.8,59.2,213.5,.5,2.5),(211,503.2,59.2,233,None,None)]:
 s=sample(row,'Yousri');c=s['claims'];s['fields']+=['FA','w/b']
 c.update({k:direct(v,'kg/m3',[ytable]) for k,v in {'Cem':cement,'RS':603,'CA':607}.items()})
 if fa:c['FA']=direct(fa,'kg/m3',[ytable])
 else:c['FA']=uncertain('—','kg/m3',[ytable],'表格横线的真零含义需人工确认，未自动当作数值0。',mode='ambiguous')
 c['Water']=direct(water,'L/m3',[ytable],'原文体积单位；不默认采用1 kg/L。')
 c['CS_Spec']=direct('Cu150','-',[ymethod],'原文150 mm立方体。')
 c['CastingEnv']=direct('Underwater','-',[ymethod],'核查的是同配方水下分支；曲线中空气对照不能混用。')
 c['CS_28d']=uncertain('曲线，未独立读数','N/mm²',[ygraph],
      '本轮未独立数字化28d曲线，不能确认Excel小数精度；N/mm²与MPa等价不解决估读问题。',
      mode='ambiguous',method='图像估读待复核，未新生成数值')
 c['SP']=uncertain(sp if sp is not None else '—','L/m3',[ytable,ytext],
      '表格体积与正文质量百分比关系未厘清；Excel0不能当确认无HRWR。' if sp else
      'M–10FA–VA的HRWR为横线，另列0.6% viscosity agent；不能沿用2.5。',
      mode='insufficient' if sp else 'ambiguous')
 if awa:
  c['AWA']=calculated(awa,'%水泥质量',[ytable],
      'mass_kg_m3 = percent / 100 × cement_mass_kg_m3',{'percent':awa,'mass_kg_m3':cement},
      'Table2标明by wt of cement；不将其误作总胶材基准。','percent_mass')
 else:
  c['AWA']=uncertain('AWA —；viscosity agent 0.6%','%水泥质量',[ytable],
      'VA是否应归入现有AWA/其他外加剂字段未确认；不能把横线统一转真零。',mode='ambiguous')
 c['w/b']=uncertain('Table4 w/c ratio；正文water-cementitious ratio','-',
      [ref('Yousri',5,'Table4及w/c列','Table 4.',after=16),
       ref('Yousri',2,'Experimental programme，水胶比术语','water-cementitious',before=2,after=10)],
      '论文w/c称谓与本库w/c、w/b定义需要显式映射；本轮不凭列名同义认定。',mode='ambiguous')
 samples.append(s)
# Brzozowski：水下两组、空气对照一组。
btable=ref('Brzozowski',3,'Table1，表头及UCR1–3行','Table 1',after=11)
bstrength=ref('Brzozowski',5,'Table3，f_c,uw与f_c,air；28d','Table 3',after=12)
bmethod=ref('Brzozowski',2,'§2.2.1','specimens of 100 mm',before=3,after=7)
bconflict=ref('Brzozowski',2,'§2.1，正文specific weight与掺量','specific weight of 1.16',before=2,after=4)
for row,cs,env,sp,awa in [(514,59.2,'Underwater',12,4.3),(515,63,'Air',12,4.3),(518,41.4,'Underwater',6,6.5)]:
 s=sample(row,'Brzozowski');c=s['claims'];s['fields']+=['Emulsion']
 c.update({k:direct(v,'kg/m3',[btable]) for k,v in {'Cem':429,'RS':754,'CA':1115,'AWA':awa}.items()})
 c['Water']=direct(171,'L/m3',[btable],'原文表格单位L/m³。')
 c['CS_28d']=direct(cs,'MPa',[bstrength],'明确区分f_c,uw与f_c,air，未拿正文64 MPa代替表3的63 MPa空气值。')
 c['CS_Spec']=direct('Cu100','-',[bmethod],'水下试块切割得到100 mm立方体；参考样本为100 mm模具。')
 c['CastingEnv']=direct(env,'-',[bstrength,bmethod], '按表3水下或空气对应列识别，不按标题推断。')
 c['SP']=uncertain(sp,'L/m3',[btable,bconflict],
      'Excel按kg/m3记录12/6，原文表格为L/m3；正文specific weight单位异常且掺量冲突，无法可靠换算。',
      mode='conflict',action='保留原值并标记单位/基准冲突；查产品资料或勘误。')
 c['Emulsion']=uncertain(5 if row==518 else '—','L/m3',[btable],
      'UCR3乳液5 L/m3不是5 kg/m3，产品密度尚缺。' if row==518 else '横线未自动解释为真零。',
      mode='conflict' if row==518 else 'ambiguous')
 samples.append(s)
# Wang：PC/NS1的绝对基准与材料映射。
wtable=ref('Wang',3,'Table3，PC、NS1行与Nano SiO2列','Table 3.',after=14)
wratio=ref('Wang',3,'§2.2','water:binder ratio',before=1,after=8)
wmethod=ref('Wang',4,'Fig4及§2.3–2.4.2','The perfusion method',after=12)
wgraph=ref('Wang',5,'§3.2及Fig7','shown in Fig. 7',before=1,after=5)
for row in [520,521]:
 s=sample(row,'Wang');c=s['claims'];s['fields']+=['SF','w/b']
 c['w/b']=direct(.48,'-',[wratio])
 c['CS_Spec']=direct('Cu100','-',[wmethod],'原文100×100×100 mm立方体。')
 c['CastingEnv']=direct('Underwater','-',[wmethod],'水下灌注需结合已目视核对的Fig4；该段有海水养护描述，不能只凭养护判断浇筑。')
 c['CS_28d']=uncertain('Fig7的28d柱形图，未独立估读','MPa',[wgraph],
      '尚未独立读图，不确认当前23.1/33.4的小数精度。',mode='ambiguous',method='图像估读待复核，未新生成数值')
 for field in ['Cem','Water','RS','CA','AWA']:
  c[field]=uncertain('质量比例；无本轮确认的kg/m3基准','比例',[wtable,wratio],
      '所查§2.2与Table3不能唯一验证当前绝对用量；是否使用过假设密度仍未知。',mode='insufficient')
 if row==521:
  c['SF']=uncertain('NS1: Nano SiO2=0.01（质量比）','比例',[wtable],
      '字典SF表示silica fume，原文材料为nano-SiO2；不能把SF列4.3直接视为已核实硅灰，需核实分类及绝对换算。',mode='ambiguous')
 samples.append(s)
# Chaikasatsin：最终修补配方与明确不同研究分支的顶管配方。
chtable=ref('Chaikasatsin',60,'Table4.3，最终repair mix行及表注','*Concrete mix for repair',before=1,after=6)
chmethod=ref('Chaikasatsin',54,'§4.2.3.5','Compressive strength of concrete according',after=5)
chseg=ref('Chaikasatsin',53,'§4.2.3.3，静态离析检查','150 x 300 mm',before=1,after=4)
for row in [510,511]:
 s=sample(row,'Chaikasatsin');c=s['claims'];s['fields']+=['FA','StF']
 if row==510:
  c['CS_Spec']=uncertain('ASTM C39；对应抗压尺寸未明确','-',[chmethod,chseg],
       'Total明确写尺寸未报告，批注版为默认Cy150；150×300 mm来自静态离析检查，非抗压尺寸直接证据。',mode='ambiguous')
  for field in ['Cem','Water','RS','CA','SP','AWA','FA']:
   c[field]=uncertain('最终配方比例及相对掺量','比例/%',[chtable],
       'Table4.3没有逐项完整绝对kg/m3基准；原表0不能确认未使用。',mode='insufficient')
  c['CastingEnv']=uncertain('抗压章节同时讨论水下和空气分支','-',[chmethod,chtable],
       '需将该repair mix的具体强度曲线与水下分支绑定，方法章节总体描述不等于完成逐条绑定。',mode='ambiguous')
 else:
  t=ref('Chaikasatsin',72,'Table5.1，OPC1+3D+FA10行与表头','Table 5.1',after=17)
  c.update({k:direct(v,'kg/m3',[t]) for k,v in {'Cem':332,'FA':37,'RS':911,'StF':50,'Water':104}.items()})
  c['CA']=calculated('Gravel 3/4=356；Gravel3/8=830','kg/m3',[t],
       'CA = Gravel_3_4 + Gravel_3_8',{'components_kg_m3':[356,830]},
       '原表CA=0但论文报告两档粗骨料；候选合计1186，分类及适用范围需确认。','sum_components')
  c['SP']=direct(3140,'mL/m3',[t],'当前SP0与原文3140 mL/m3不能按同单位比较，仍缺质量换算依据。')
  c['CS_Spec']=uncertain('ASTM C39；该段未报告尺寸','-',
       [ref('Chaikasatsin',71,'§5.2.2.2','Compressive strength according',after=5)],
       '不能用抗折棱柱试件或其他章节圆柱尺寸替代本分支抗压尺寸。',mode='insufficient')
  c['CastingEnv']=uncertain('dry-cast steel fiber concrete for jacking pipe','-',
       [ref('Chaikasatsin',72,'§5.3.1.1','dry-cast steel fiber concrete',before=2,after=6)],
       '该分支不是自动的水下浇筑；原表为Air，本轮不改变；成型环境的直接原文说明仍待补。',mode='ambiguous')
  c['CS_28d']=uncertain('PDF第73页Fig5.4，未独立读图','MPa',
       [ref('Chaikasatsin',73,'§5.3.1.3及Fig5.4','550',before=2,after=9)],
       '章节交叉引用Fig5.3而图题为Fig5.4；61.7尚未独立读图核对。',mode='ambiguous',method='图像估读待复核，未新生成数值')
 samples.append(s)
config={'raw':str(raw),'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
        'annotated':str(annotated),'registry':str(registry),'papers':papers,'samples':samples,
        'scope':'12条定向选择，非随机样本；不估计全库通过率；不改变任何可靠状态。'}
(ROOT/'configs/pilot_evidence.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
print('已编录',len(samples),'条样本；',sum(len(s['claims']) for s in samples),'项有定位的字段证据。')
