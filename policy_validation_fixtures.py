"""Preassigned development/holdout agreement layouts with DOM ground truth."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from dense_fixtures import CHROME, COLLECT

TEXTS = {
 'development': [
  ('订阅服务说明','您可以在会员页面查看订阅周期和下一次扣费时间，订单成功后通过账户设置管理服务。','取消续订不会影响已支付周期内的使用权益'),
  ('位置权限说明','仅在您使用附近门店功能时读取所选位置，退出功能后不持续获取当前位置。','拒绝定位权限仍可手动选择城市'),
  ('注销申请流程','提交申请前请确认账户余额、未完成订单和售后事项，处理进度通过消息中心通知。','注销完成后部分历史内容无法恢复'),
  ('消息推送规则','提醒内容包括订单状态及您订阅的活动信息，可以分别调整通知开关。','关闭营销通知不影响接收必要服务通知'),
 ],
 'holdout': [
  ('健康数据授权','运动记录由您主动录入或选择同步，可以在设备管理页面断开连接并查询授权状态。','健康记录仅在所选服务范围内使用'),
  ('儿童账号管理','监护人可以设置使用时段和可访问内容，变更后会同步到已绑定的终端。','涉及儿童信息的新增用途需要再次确认'),
  ('云端备份条款','备份任务在您设置的网络条件下执行，可在文件列表查看完成时间并选择恢复版本。','删除本地文件不会自动删除云端备份'),
  ('实名认证说明','身份验证用于确认账号归属，提交前请核对证件信息并阅读处理范围。','未通过验证时可申请人工复核'),
 ]}
MODES=('wrapped','mostly_bold','sparse','plain','table','small','serif','dark')

CALIBRATE = r'''element => {
 const checks=new Map();
 for(const el of element.querySelectorAll('*')){
  const s=getComputedStyle(el);
  if(Number(s.fontWeight)<600)continue;
  const key=[s.fontFamily,s.fontSize,s.fontWeight].join('|');
  if(checks.has(key))continue;
  const render=weight=>{const c=document.createElement('canvas');c.width=800;c.height=100;
   const x=c.getContext('2d');x.font=weight+' '+s.fontSize+' '+s.fontFamily;
   x.fillText('健康记录仅在所选服务范围内使用隐私授权',10,60);
   return x.getImageData(0,0,800,100).data;};
  const a=render('400'),b=render(s.fontWeight);let difference=0;
  for(let i=3;i<a.length;i+=4)difference+=Math.abs(a[i]-b[i]);
  checks.set(key,{family:s.fontFamily,size:s.fontSize,weight:s.fontWeight,alpha_difference:difference});
 }
 return Array.from(checks.values());
}'''


def document(split,mode,index):
    width=(352 if split=='development' else 402) if mode!='table' else 680
    size=12 if mode=='small' else (15 if split=='development' else 16)
    blocks=[]
    for n in range(8):
        title,body,key=TEXTS[split][n%4]
        emphasis=key if mode=='plain' or (mode=='sparse' and n!=3) else f'<strong>{key}</strong>'
        text=f'{body}{emphasis}。{body}'
        if mode=='wrapped' and n%2==0:
            text=f'<strong>{key}。{body}{key}。</strong>{body}'
        if mode=='mostly_bold':
            text=f'<strong>{body}{key}。{body}</strong>这里恢复普通说明。'
        if mode=='table':
            blocks.append(f'<tr><td>{n+1}</td><td>{title}</td><td>{text}</td><td><a><u>查看详情</u></a></td></tr>')
        else:
            link='' if mode=='plain' else '<a><u>补充说明与联系方式</u></a>。'
            blocks.append(f'<h2>{n+1}、{title}</h2><p>{text}{link}</p>')
    body=''.join(blocks)
    if mode=='table':body=f'<table><tr><th>序号</th><th>功能</th><th>处理说明</th><th>规则</th></tr>{body}</table>'
    name=f'{split}_{mode}'
    return f'''<section data-case='{name}' data-split='{split}' class='{mode}' style='width:{width}px;font-size:{size}px'>
    <header>演示应用 / 文件阅读</header><h1>{'服务订阅与授权协议' if split=='development' else '个人信息保护与账号使用规则'}</h1>
    <p>本页为虚构测试文档。版本：2.{index}；更新日期：2026年9月。</p>{body}
    <footer>联系邮箱 support@example.cn　文档编号 2026-{index}<br><span style='font-weight:500'>请确认您已阅读并了解以上说明。</span></footer></section>'''


def generate(root=Path('testdata/policy_validation')):
    root.mkdir(parents=True,exist_ok=True)
    pages=''.join(document(split,mode,i) for split in TEXTS for i,mode in enumerate(MODES))
    source=root/'source.html'
    source.write_text('''<!doctype html><meta charset='utf-8'><style>
    *{box-sizing:border-box}body{margin:0;background:#ccc;font-family:'Noto Sans CJK SC';color:#242424}
    section{padding:18px;margin:18px;background:white}p{margin:3px 0 12px;line-height:1.5;text-align:justify;overflow-wrap:anywhere}
    h1{font-size:1.4em}h2{font-size:1em;margin:14px 0 3px}strong{font-weight:700}
    a{color:#0872cc}u{text-underline-offset:2px}header,footer{font-size:.8em;color:#555}
    .plain *{font-weight:400!important}.serif{font-family:'AR PL UMing CN'}
    .small{background:#f2f2f2;color:#555}.dark{background:#202020;color:#eee}.dark a{color:#69afff}
    .dark header,.dark footer{color:#aaa}table{width:100%;border-collapse:collapse;font-size:.9em}
    td,th{border:1px solid #aaa;padding:5px}td:first-child{width:6%}td:last-child{width:14%}
    </style>'''+pages,encoding='utf-8')
    cases=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=CHROME,headless=True)
        page=browser.new_page(viewport={'width':850,'height':1100})
        page.goto(source.resolve().as_uri(),wait_until='networkidle')
        page.evaluate('document.fonts.ready')
        for el in page.locator('section').all():
            name=el.get_attribute('data-case')
            assert el.evaluate('(e)=>e.scrollWidth<=e.clientWidth')
            units=el.evaluate(COLLECT)
            checks=el.evaluate(CALIBRATE)
            el.screenshot(path=str(root/f'{name}.png'))
            cases.append(dict(name=name,split=el.get_attribute('data-split'),image=f'{name}.png',
                              bold_observable=all(c['alpha_difference']>0 for c in checks),
                              font_checks=checks,units=units))
        browser.close()
    path=root/'ground_truth.json'
    path.write_text(json.dumps({'protocol':'Preassigned by document family; all variants stay in the same split. No threshold tuning on holdout.', 'cases':cases},ensure_ascii=False,indent=2),encoding='utf-8')
    return path


if __name__=='__main__':print(generate())
