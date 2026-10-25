"""重读两个原始PDF中的配比证据；保存引用匹配和原有冲突，不回写数据。"""
import json
from pathlib import Path
import re
import subprocess

from concrete_audit.parallel_exploration import digest, write_json


def recheck(output='outputs/two-paper-recheck-v1'):
    output=Path(output);output.mkdir(exist_ok=False)
    config_path=Path('configs/composition_review_evidence.json');config=json.loads(config_path.read_text())
    selected=[a for a in config['annotations'] if a['paper'] in {'Zhang2025','AlSelmi2026'}]
    evidence=[];cache={};seen=set();before={str(config_path):digest(config_path)}
    for row in selected:
        paper=config['papers'][row['paper']];path=Path(paper['path'])
        before[str(path)]=digest(path)
        if before[str(path)]!=paper['sha256']:raise ValueError('原文PDF版本变化')
        for ref in row['evidence']:
            key=(row['paper'],ref['page'],ref['quote'])
            if key in seen:continue
            seen.add(key);page_key=key[:2]
            if page_key not in cache:
                cache[page_key]=subprocess.check_output(['pdftotext','-layout','-f',str(ref['page']),'-l',str(ref['page']),str(path),'-']).decode('utf-8')
                (output/f'{row["paper"]}-p{ref["page"]}.txt').write_text(cache[page_key])
            text=cache[page_key]
            def normalized(t):return re.sub(r'\s+',' ',t).strip()
            exact=normalized(ref['quote']) in normalized(text)
            evidence.append({'paper':row['paper'],'page':ref['page'],'location':ref['location'],
                             'quote':ref['quote'],'whitespace_normalized_quote_match':exact,
                             'source_pdf_sha256':before[str(path)]})
    if not all(r['whitespace_normalized_quote_match'] for r in evidence):
        raise ValueError('旧引用无法在当前原始PDF页重现，不能声称完成复查')
    records=[]
    for row in selected:
        components=row['defined_binder_components'];values=[row['components'][f]['value'] for f in components]
        total=sum(values) if all(v is not None for v in values) else None
        printed=row['printed_binder_total'];water=row['water']['value']
        records.append({'source_row':row['source_row'],'paper':row['paper'],'specimen_id':row['specimen_id'],
            'binder_components':components,'component_sum':total,'printed_total':printed,
            'conflict':printed is not None and total is not None and abs(total-printed)>1e-8,
            'basis':row['components']['Cem']['unit'],'water_value':water,'water_unit':row['water']['unit'],
            'water_over_component_sum_candidate':water/total if total and water is not None else None,
            'notes':row['notes'],'formal_status_change':False,'used_to_change_labels':False})
    write_json(output/'quote_recheck.json',evidence);write_json(output/'composition_recheck.json',records)
    after={p:digest(p) for p in before}
    if before!=after:raise ValueError('原文或配置发生变化')
    write_json(output/'manifest.json',{'source_files_sha256':before,'source_files_after_sha256':after,
        'script_sha256':digest(__file__),'record_count':len(records),'distinct_quote_count':len(evidence),
        'all_quotes_reproduced':True,'formal_status_changed':False,'labels_changed':False})
    (output/'report.md').write_text('# 两篇配比证据复查\n\n'
        '重新从原始PDF提取Zhang第5页Table3及Al-Selmi第6页Table4；旧引用逐条在当前PDF页重现，PDF哈希未变。\n\n'
        '- Zhang四条按水泥质量归一化，胶材分项和1+.06+.18+.19=1.43。水/该分项和=.223/1.43≈.15594；这是同一相对质量基准的候选比值，不能据此核实库中绝对kg/m³。\n'
        '- Al-Selmi六条表头为kg/m³，OPC1202+FA301=1503，表注明写1502。605/1503≈.40253；保留两个总量，不能仅凭算术替作者修正表注。\n'
        '- 复查只支持引用及上述字段解释，不表示整篇实验数据或28天换算依据全部已核实。未将稀疏候选水胶比强制加入本批模型，两个研究体系的缺失与冲突不阻断SVR/MLP实验。\n')
    print(json.dumps({'records':len(records),'quotes':len(evidence),'matches':True,'conflicts':sum(r['conflict'] for r in records)},ensure_ascii=False))


if __name__=='__main__':recheck()
