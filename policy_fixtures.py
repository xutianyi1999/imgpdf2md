"""Dense synthetic agreements; DOM labels, not model-generated labels."""
from pathlib import Path
import json

from playwright.sync_api import sync_playwright
from dense_fixtures import CHROME, COLLECT


CLAUSES = [
    ("收集与使用", "我们会在您主动开启相关功能时处理必要信息，包括设备型号、系统版本和操作记录。", "拒绝提供非必要信息不影响基础服务"),
    ("权限与授权", "相机用于扫描您选择的文件，相册用于读取指定图片。权限关闭后，相关功能将停止访问。", "您可以随时撤回已授予的权限"),
    ("共享与委托", "为完成支付及消息推送，我们可能委托服务商处理订单编号和推送标识，详细范围见附表。", "未经单独同意不向其他接收方提供敏感信息"),
    ("保存与删除", "账号注销后，我们将按照适用的保存要求处理记录，无法立即删除的备份限制其他用途。", "您可以申请查询、更正和删除个人信息"),
    ("付费与续订", "订阅周期、价格和权益以确认页面展示为准，您可以通过账户设置查看订单及关闭续订。", "请在确认扣款前核对金额与自动续费选项"),
    ("争议与通知", "服务调整时会在应用内通知，您可以通过客服入口反馈意见或查询协议的历史版本。", "请特别阅读责任限制及争议解决条款"),
]

# Content and layout cases, separate from the parameter-only font matrix.
CASES = [
    ("privacy_narrow_inline", 360, 15, "inline", "light"),
    ("agreement_full_bold", 414, 16, "paragraph", "light"),
    ("privacy_plain_negative", 390, 14, "plain", "light"),
    ("agreement_single_boundary", 375, 15, "single", "light"),
    ("sdk_table_wrapped", 720, 14, "table", "light"),
    ("consent_small_gray", 360, 12, "inline", "gray"),
    ("privacy_dark_combined", 414, 16, "combined", "dark"),
    ("agreement_serif_dense", 720, 15, "serif", "light"),
]


def html() -> str:
    pages = []
    for name, width, size, mode, theme in CASES:
        blocks = []
        for index, (title, body, emphasis) in enumerate(CLAUSES):
            marked = emphasis if mode == "plain" else f"<strong>{emphasis}</strong>"
            if mode == "single":
                marked = "您可选择<strong>同</strong>意或<strong>不</strong>同意，普通说明仍按正常字重显示。"
            if mode == "combined":
                marked = f"<strong><a><u>{emphasis}</u></a></strong>"
            paragraph = f"{body}{marked}。{body}"
            if mode == "paragraph":
                paragraph = f"<strong>{paragraph}</strong>" if index % 2 == 0 else paragraph
            link = "查看补充说明" if mode == "plain" else "<a><u>查看补充说明</u></a>"
            blocks.append(f"<h2>{index+1}、{title}</h2><p>{paragraph}</p><p>{link}。处理期限：30天；SDK v2.1；privacy@example.cn。</p>")
        if mode == "table":
            rows = "".join(f"<tr><td>示例组件 {i+1}</td><td>{body}<strong>{emphasis}</strong></td><td><a><u>处理规则</u></a></td></tr>" for i, (_, body, emphasis) in enumerate(CLAUSES))
            blocks.insert(2, f"<table><tr><th>接收方</th><th>信息处理目的与范围</th><th>链接</th></tr>{rows}</table>")
        pages.append(f"<section data-case='{name}' class='{mode} {theme}' style='width:{width}px;font-size:{size}px'><header>示例应用 · 签署文件</header><h1>隐私政策与用户服务协议</h1><p>此页为样式识别测试用虚构文档。更新日期：2026年9月20日。</p>{''.join(blocks)}<footer>阅读确认：我已阅读上述条款。版本号 3.2.0</footer></section>")
    return """<!doctype html><meta charset='utf-8'><style>
    *{box-sizing:border-box}body{margin:0;background:#ddd;color:#252525;font-family:'Noto Sans CJK SC'}
    section{padding:20px;margin:16px;background:white}h1{font-size:1.4em}h2{font-size:1em;margin:12px 0 3px}
    p{line-height:1.55;margin:4px 0;text-align:justify;overflow-wrap:anywhere}strong{font-weight:700}
    header,footer{font-size:.8em;color:#666}a{color:#0866d9}u{text-underline-offset:2px}
    table{border-collapse:collapse;width:100%;font-size:.9em}td,th{border:1px solid #aaa;padding:5px}
    th:first-child{width:20%}th:last-child{width:15%}.serif{font-family:'Noto Serif CJK SC'}
    .plain *{font-weight:400}.gray{background:#eee;color:#555}.dark{background:#191919;color:#eee}
    .dark header,.dark footer{color:#aaa}.dark a{color:#63b3ff}
    </style>""" + "".join(pages)


def generate_policy(output_dir: Path = Path("testdata/policy")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "source.html"
    source.write_text(html(), encoding="utf-8")
    cases = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True)
        page = browser.new_page(viewport={"width": 850, "height": 1000})
        page.goto(source.resolve().as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        for element in page.locator("section").all():
            name = element.get_attribute("data-case")
            units = element.evaluate(COLLECT)
            box = element.bounding_box()
            element.screenshot(path=str(output_dir / f"{name}.png"))
            assert all(0 <= u['box'][0] < u['box'][2] <= round(box['width']) for u in units)
            cases.append({"name": name, "image": f"{name}.png", "width": round(box['width']), "height": round(box['height']), "units": units})
        browser.close()
    path = output_dir / "ground_truth.json"
    path.write_text(json.dumps({"description": "Synthetic policy and agreement screenshots, independent DOM truth", "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(generate_policy())
