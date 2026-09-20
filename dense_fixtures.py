#!/usr/bin/env python3
"""Generate source-labeled dense-text screenshots in several layouts."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = "/usr/bin/google-chrome"


POLICY_PARAGRAPHS = [
    "为向您提供账号注册、身份核验和文档解析服务，我们会处理设备信息、操作日志及您主动上传的文件。",
    "上传内容仅用于完成本次解析任务；除非取得您的另行授权，我们不会将文件用于广告推荐或与服务无关的模型训练。",
    "您可以在控制台查询处理记录并删除历史任务。删除请求提交后，在线副本通常会在二十四小时内完成清理。",
    "当服务由受托处理方协助完成时，我们会通过协议约束其处理目的、期限、方式和安全措施，并开展必要审计。",
    "如发生个人信息安全事件，我们将依法告知事件原因、可能影响、已采取的措施以及您可自主防范的风险。",
]


def marked_paragraph(index: int, text: str) -> str:
    keywords = ["身份核验", "仅用于完成本次解析任务", "删除历史任务", "必要审计", "安全事件"]
    keyword = keywords[index % len(keywords)]
    marked = text.replace(keyword, f"<strong>{keyword}</strong>")
    if index % 3 == 1:
        marked += " 详情请阅读<a href='#'>数据处理规则</a>。"
    if index % 5 == 2:
        marked += " <u>本条款自页面所示日期起生效</u>。"
    return marked


def dense_mobile() -> str:
    sections = []
    for index in range(1, 13):
        first = marked_paragraph(index, POLICY_PARAGRAPHS[index % len(POLICY_PARAGRAPHS)])
        second = marked_paragraph(index + 1, POLICY_PARAGRAPHS[(index + 2) % len(POLICY_PARAGRAPHS)])
        sections.append(f"<h2>{index}. {['收集范围','使用目的','共享规则','保存期限'][index % 4]}</h2><p>{first}</p><p>{second}</p>")
    return f"""<section class='page mobile' data-case='dense_mobile_policy'>
      <header><b>示例应用隐私政策</b><span>更新日期：2026-09-20</span></header>
      <h1>隐私政策与个人信息处理清单</h1>
      <p class='lead'>请您在使用服务前完整阅读。<strong class='red'>粗体及彩色内容</strong>与您的权益密切相关，蓝色下划线文字可查看补充说明。</p>
      {''.join(sections)}
    </section>"""


def dense_user_agreement() -> str:
    blocks = []
    titles = ["服务内容与开通", "账号与身份认证", "用户行为规范", "费用及续费", "数据与隐私保护", "违约责任"]
    for index in range(1, 15):
        text = POLICY_PARAGRAPHS[(index * 2) % len(POLICY_PARAGRAPHS)]
        text = marked_paragraph(index, text)
        extra = "<strong>自动续费服务可在扣款前关闭</strong>。" if index == 4 else ""
        blocks.append(f"<h2>{index}. {titles[(index-1)%len(titles)]}</h2><p>{text}{extra}</p>")
    return f"""<section class='page agreement' data-case='dense_user_service_agreement'>
      <div class='kicker'>用户签署文件　协议版本：2026.09</div>
      <h1>应用服务协议与个人信息授权书</h1>
      <p class='party'><b>甲方（用户）：张弛</b>　　<b>乙方：示例网络科技有限公司</b></p>
      <p class='lead'>点击“同意并继续”即表示您已阅读并接受全部条款。请特别关注
      <strong class='red'><u>责任限制、自动续费和个人信息授权</u></strong>等与您权益密切相关的内容。</p>
      {''.join(blocks)}
      <p class='fine-print'>小字说明：本协议中的技术术语应按通常含义解释；法律法规另有强制规定的，从其规定。</p>
      <div class='sign'><span>用户签名：<b>张　弛</b></span><span>签署日期：<b>2026-09-20</b></span></div>
    </section>"""


def dense_table() -> str:
    rows = []
    for index in range(1, 31):
        necessary = "是" if index % 4 else "否"
        level_class = "red" if necessary == "是" else "green"
        status = "查看规则" if index % 3 else "单独同意"
        rows.append(
            f"<tr><td>{index:02d}</td><td><b>{['账号注册','文档上传','安全保障','客户服务'][index%4]}</b></td>"
            f"<td>{['手机号码','设备标识','上传文件','操作日志'][index%4]}</td><td>完成对应功能并保障账号与服务安全。</td>"
            f"<td class='{level_class}'><b>{necessary}</b></td><td><a href='#'>{status}</a></td><td>{7 + index % 24} 天</td></tr>"
        )
    return f"""<section class='page table-page' data-case='dense_personal_information_table'>
      <div class='kicker'>隐私政策附件一</div><h1>个人信息收集与使用清单</h1>
      <p>以下清单列明处理场景、信息类型、使用目的及保存期限。标记为<span class='red'><b>必要</b></span>的项目是实现基础功能所必需，详情链接带下划线。</p>
      <table><thead><tr><th>序号</th><th>场景</th><th>信息类型</th><th>处理目的</th><th>必要</th><th>说明</th><th>期限</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>"""


def html() -> str:
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><style>
    *{{box-sizing:border-box}} html,body{{margin:0;background:#d9dee5;font-family:'Noto Sans CJK SC',sans-serif;color:#24272d}}
    .page{{background:#fff;position:relative;overflow:hidden;margin:20px}}
    .mobile{{width:760px;height:3000px;padding:34px 40px}}
    .agreement{{width:900px;height:2500px;padding:48px 54px}}
    .table-page{{width:1400px;height:1900px;padding:44px 48px}}
    header{{display:flex;justify-content:space-between;border-bottom:1px solid #d0d5dd;padding-bottom:12px;font-size:14px;color:#667085}}
    h1{{font-size:27px;line-height:1.35;margin:22px 0 14px;color:#101828}} h2{{font-size:16px;line-height:1.5;margin:14px 0 3px}}
    p{{font-size:14px;line-height:1.62;margin:3px 0 7px;text-align:justify}} .mobile p{{font-size:16px;line-height:1.58}}
    .mobile h2{{font-size:17px;margin-top:16px}} .lead{{background:#f8fafc;padding:12px;border-left:3px solid #1570ef}}
    a{{color:#0866d9;text-decoration:underline;text-underline-offset:2px}} .red{{color:#c11574}} .green{{color:#067647}} .amber{{color:#b54708}}
    .kicker{{font-size:13px;color:#475467;border-bottom:1px solid #d0d5dd;padding-bottom:10px}}
    .agreement p{{font-size:15px;line-height:1.7}} .party{{background:#f2f4f7;padding:10px 12px}}
    .fine-print{{font-size:11px!important;color:#667085;margin-top:18px}}
    .sign{{display:flex;justify-content:space-between;border-top:1px solid #98a2b3;margin-top:25px;padding-top:18px;font-size:14px}}
    table{{width:100%;border-collapse:collapse;font-size:12px;line-height:1.4}} th,td{{border:1px solid #cfd4dc;padding:6px 7px}}
    th{{font-weight:700;background:#eef1f5;text-align:left}} td:nth-child(1),td:nth-child(3),td:nth-child(5),td:nth-child(6){{white-space:nowrap}}
    </style></head><body>{dense_mobile()}{dense_user_agreement()}{dense_table()}</body></html>"""


COLLECT = r"""(page) => {
  const pageRect = page.getBoundingClientRect(), units = [];
  const walker = document.createTreeWalker(page, NodeFilter.SHOW_TEXT); let node;
  while ((node = walker.nextNode())) {
    const style = getComputedStyle(node.parentElement); let offset = 0;
    for (const char of Array.from(node.textContent)) {
      const length = char.length;
      if (!/^\s$/u.test(char)) {
        const range = document.createRange(); range.setStart(node,offset); range.setEnd(node,offset+length);
        const r = range.getBoundingClientRect();
        if (r.width>0 && r.height>0 && r.bottom>pageRect.top && r.top<pageRect.bottom) {
          const rgb=style.color.match(/\d+/g).slice(0,3).map(Number); let under=false, el=node.parentElement;
          while(el){if(getComputedStyle(el).textDecorationLine.includes('underline')){under=true;break} if(el===page)break;el=el.parentElement}
          units.push({text:char,box:[Math.round(r.left-pageRect.left),Math.round(r.top-pageRect.top),Math.round(r.right-pageRect.left),Math.round(r.bottom-pageRect.top)],bold:Number(style.fontWeight)>=600,underline:under,color:'#'+rgb.map(x=>x.toString(16).padStart(2,'0')).join(''),colored:Math.max(...rgb)-Math.min(...rgb)>=35&&Math.max(...rgb)>=75,font_size:parseFloat(style.fontSize),tag:node.parentElement.tagName.toLowerCase()});
        }
      } offset+=length;
    }
  } return units;
}"""


def generate_dense(output_dir: Path = Path("testdata/dense")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "source.html"
    source.write_text(html(), encoding="utf-8")
    cases = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=CHROME, headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 2000})
        page.goto(source.resolve().as_uri(), wait_until="networkidle")
        elements = page.locator(".page")
        for index in range(elements.count()):
            element = elements.nth(index)
            name = element.get_attribute("data-case")
            units = element.evaluate(COLLECT)
            image = f"{name}.png"
            element.screenshot(path=str((output_dir / image).resolve()))
            box = element.bounding_box()
            cases.append({"name": name, "image": image, "width": round(box["width"]), "height": round(box["height"]), "units": units})
        browser.close()
    path = output_dir / "ground_truth.json"
    path.write_text(json.dumps({"description":"Dense browser screenshots with DOM truth","cases":cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(generate_dense())
