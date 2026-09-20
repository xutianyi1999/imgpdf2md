"""Re-render stored style decisions without OCR calls or model inference."""
import argparse
import json
from pathlib import Path
from markdown_it import MarkdownIt
from style_demo import Unit,enrich_vl_markdown


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('page',type=Path)
    args=parser.parse_args()
    result=json.loads((args.page/'styles.json').read_text())
    units=[Unit(**u) for line in result['lines'] for u in line['units']]
    markdown,coverage=enrich_vl_markdown((args.page/'vl.md').read_text(),units)
    assert abs(coverage-result['vl_alignment_coverage'])<1e-8
    (args.page/'styled.md').write_text(markdown,encoding='utf-8')
    body=MarkdownIt('commonmark',{'html':True}).enable('table').render(markdown)
    (args.page/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>Markdown 渲染预览</title><style>body{font-family:sans-serif;max-width:1000px;margin:30px auto;line-height:1.6}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}</style>'+body,encoding='utf-8')
    print(args.page/'preview.html')


if __name__=='__main__':main()
