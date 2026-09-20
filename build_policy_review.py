"""Create a visual inspection index from the policy regression artifacts."""
import html
import json
from pathlib import Path


def build(root=Path('testdata/policy'), report_name='evaluation.json', result_dir='results', index_name='review.html'):
    report = json.loads((root / report_name).read_text())
    links = []
    for case in report['cases']:
        if case['suite'] != 'policy':
            continue
        folder = Path(result_dir) / case['case'] / case['variant']
        result = json.loads((root / folder / 'styles.json').read_text())
        lines = []
        for line in result['lines']:
            spans = []
            for span in line['spans']:
                style = 'font-weight:700;' if span['render_bold'] else ''
                if span['underline']:
                    style += 'text-decoration:underline;'
                if span['color']:
                    style += f"color:{span['color']};"
                spans.append(f'<span style="{style}">{html.escape(span["text"])}</span>')
            lines.append('<div>' + ''.join(spans) + '</div>')
        boxes = []
        for error in case['bold_errors']:
            x1, y1, x2, y2 = error['box']
            color = '#e22' if error['predicted_bold'] else '#1683ff'
            boxes.append(f'<i title="{html.escape(error["text"])}" style="left:{x1}px;top:{y1}px;width:{x2-x1}px;height:{y2-y1}px;border-color:{color}"></i>')
        title = f"{case['case']} / {case['variant']}"
        if case.get('bold_observable') is False:
            title += '（正文的源字重在像素中不可区分）'
        body = f'''<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif;margin:24px}}main{{display:flex;gap:30px;align-items:flex-start}}
.source{{position:relative;flex-shrink:0}}img{{display:block}}i{{position:absolute;border:1px solid;box-sizing:border-box;pointer-events:none}}
input:not(:checked)~main i{{display:none}}article{{min-width:400px;line-height:1.65}}article div{{margin-bottom:10px}}</style>
<h1>{title}</h1><p>左：输入截图；右：本地样式结果（DOM 字符框，不含 OCR/表格结构还原）。红框为粗体误报，蓝框为漏报。</p>
<a href="styled.md">Markdown 文件</a> · <a href="input.png">输入图片</a>
<p>粗体指标：{html.escape(str(case['metrics']['after']['bold']))}</p>
<input type="checkbox" id="errors"><label for="errors">显示粗体错误框</label>
<main><div class="source"><img src="input.png">{''.join(boxes)}</div><article>{''.join(lines)}</article></main>'''
        (root / folder / 'review.html').write_text(body, encoding='utf-8')
        links.append(f'<li><a href="{folder.as_posix()}/review.html">{title}</a></li>')
    count=len({case['case'] for case in report['cases'] if case['suite']=='policy'})
    (root / index_name).write_text(f'<!doctype html><meta charset="utf-8"><h1>密集协议样式回归</h1><p>{count} 类虚构文档，共 {len(links)} 页。每页可对照原图并显示粗体错误。不可见字重对照的错误框仅代表源格式差异，不代表视觉识别错误。</p><ul>' + ''.join(links) + '</ul>', encoding='utf-8')


if __name__ == '__main__':
    build()
