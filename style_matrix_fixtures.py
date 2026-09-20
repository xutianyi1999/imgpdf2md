#!/usr/bin/env python3
"""Generate a parameterized style matrix independent of the dense fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from dense_fixtures import CHROME, COLLECT


CASES = [
    ("sans_12_weight600", "Noto Sans CJK SC", 12, 600, "light"),
    ("sans_14_weight700", "Noto Sans CJK SC", 14, 700, "light"),
    ("sans_16_weight900", "Noto Sans CJK SC", 16, 900, "light"),
    ("sans_20_weight700", "Noto Sans CJK SC", 20, 700, "warm"),
    ("serif_12_weight600", "Noto Serif CJK SC", 12, 600, "light"),
    ("serif_14_weight700", "Noto Serif CJK SC", 14, 700, "light"),
    ("serif_16_weight900", "Noto Serif CJK SC", 16, 900, "warm"),
    ("serif_20_weight700", "Noto Serif CJK SC", 20, 700, "dark"),
    ("sans_13_weight600_dark", "Noto Sans CJK SC", 13, 600, "dark"),
    ("sans_15_weight700_gray", "Noto Sans CJK SC", 15, 700, "gray"),
    ("serif_13_weight600_gray", "Noto Serif CJK SC", 13, 600, "gray"),
    ("mixed_latin_digits", "Noto Sans CJK SC", 15, 700, "light"),
]


PHRASES = [
    ("身份核验与账号安全", "设备标识与操作日志"),
    ("个人信息处理规则", "撤回授权与删除记录"),
    ("第三方共享接收方", "保存期限与安全措施"),
    ("自动续费及关闭方式", "未成年人信息保护"),
]


def case_html(name: str, family: str, size: int, bold_weight: int, theme: str, index: int) -> str:
    paragraphs = []
    for row in range(7):
        primary, secondary = PHRASES[(index + row) % len(PHRASES)]
        paragraphs.append(
            f"""<p>第{row + 1}条普通说明开始，<span class='target'>{primary}</span>属于重点内容；
            <span class='medium'>这里使用五百字重但不应判为粗体</span>。您可以查看
            <a href='#'>{secondary}</a>，并选择<span class='single target'>{'是' if row % 2 else '否'}</span>。
            组合提示为<a href='#' class='target'><u>彩色粗体下划线</u></a>，结尾恢复普通正文。</p>"""
        )
    table = "".join(
        f"<tr><td>{row:02d}</td><td class='target'>{PHRASES[row % 4][0]}</td>"
        f"<td>普通说明</td><td><a href='#'>{'查看清单' if row % 2 else '单独同意'}</a></td></tr>"
        for row in range(1, 7)
    )
    return f"""<section class='matrix-page {theme}' data-case='{name}'
      style='font-family:"{family}";font-size:{size}px;--bold-weight:{bold_weight}'>
      <div class='meta'>字体：{family}　字号：{size}px　目标字重：{bold_weight}</div>
      <h1>隐私授权与重点条款</h1>
      {''.join(paragraphs)}
      <table><thead><tr><th>序号</th><th>处理场景</th><th>说明</th><th>操作</th></tr></thead>
      <tbody>{table}</tbody></table>
      <p>英文和数字边界：普通 API v2.1，<span class='target'>SDK 2026 与 OCR-3.0</span>，
      普通 privacy@example.cn。</p>
    </section>"""


def html() -> str:
    pages = "".join(case_html(*case, index) for index, case in enumerate(CASES))
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><style>
    *{{box-sizing:border-box}} html,body{{margin:0;background:#cfd4dc;color:#252930}}
    .matrix-page{{width:760px;min-height:1050px;margin:18px;padding:30px 36px;background:#fff;overflow:hidden}}
    .matrix-page p{{line-height:1.62;margin:6px 0;text-align:justify}}
    .matrix-page h1{{font-size:1.65em;line-height:1.3;margin:14px 0}}
    .meta{{font-size:.86em;color:#667085;border-bottom:1px solid #d0d5dd;padding-bottom:8px}}
    .target{{font-weight:var(--bold-weight)}} .medium{{font-weight:500}}
    a{{color:#0866d9;text-decoration:underline;text-underline-offset:2px}}
    table{{width:100%;border-collapse:collapse;margin-top:14px;font-size:.88em}}
    th,td{{border:1px solid #cfd4dc;padding:5px 6px;text-align:left}} th{{font-weight:700;background:#eef1f5}}
    .warm{{background:#fffaf0;color:#342d26}} .gray{{background:#edf0f3;color:#30343b}}
    .dark{{background:#171a20;color:#edf0f5}} .dark h1{{color:#fff}}
    .dark .meta{{color:#aab2c0;border-color:#475467}} .dark th{{background:#292e38}}
    .dark th,.dark td{{border-color:#475467}} .dark a{{color:#53b1fd}}
    </style></head><body>{pages}</body></html>"""


def generate_style_matrix(output_dir: Path = Path("testdata/style_matrix")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "source.html"
    source.write_text(html(), encoding="utf-8")
    cases = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=CHROME, headless=True)
        page = browser.new_page(viewport={"width": 900, "height": 1200})
        page.goto(source.resolve().as_uri(), wait_until="networkidle")
        elements = page.locator(".matrix-page")
        for index in range(elements.count()):
            element = elements.nth(index)
            name = element.get_attribute("data-case")
            _, family, size, bold_weight, theme = CASES[index]
            units = element.evaluate(COLLECT)
            image = f"{name}.png"
            element.screenshot(path=str((output_dir / image).resolve()))
            box = element.bounding_box()
            cases.append({
                "name": name,
                "image": image,
                "width": round(box["width"]),
                "height": round(box["height"]),
                "font_family": family,
                "font_size": size,
                "bold_weight": bold_weight,
                "theme": theme,
                "units": units,
            })
        browser.close()
    path = output_dir / "ground_truth.json"
    path.write_text(json.dumps({
        "description": "Font, size, weight, theme, and boundary style matrix",
        "cases": cases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(generate_style_matrix())
