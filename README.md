# APP 密集文档富文本恢复 Demo

这是一个可行性 Demo，目标输入是已经分页/切图的 APP 隐私政策、用户协议、
授权书和个人信息收集清单截图。它只补足 OCR 通常丢失的三类行内样式：

- 粗体；
- 文字颜色；
- 下划线。

标题、段落、列表、表格及阅读顺序由 AI Studio `PaddleOCR-VL-1.6` 提供；
`PP-OCRv6 + returnWordBox=true` 提供细粒度坐标。本地使用 OpenCV 提取颜色、
检测连续下划线。粗体由 TexTAR 上下文模型、FontDNA-V2 和同一行相对笔画
特征共同判断，并使用受行、标点和字符间距约束的连续片段补齐，再把样式对齐回
VL Markdown。所有文档图片只发送到国内的
AI Studio；TexTAR 和视觉算法均在本地运行。

## 运行

`.env` 需要包含 AI Studio Token：

```dotenv
TOKEN=AI Studio access token
```

```bash
uv sync
uv run python style_demo.py input.png
```

启用当前准确率更高的 TexTAR 后端：

```bash
uv sync --group textar
uv run --group textar python style_demo.py input.png --textar
```

首次运行会下载约 9 MB 的 FontDNA-V2 INT8 ONNX 模型到
`.cache/fontdna/`。启用 TexTAR 时还会一次性下载约 66 MB 的公开权重到
`.cache/textar/`；之后推理完全本地执行。也可提前下载并通过
`--textar-model` 指定本地路径。每页输出位于
`demo_output/<输入名>/page-0001/`：

- `styled.md`：保留 VL 结构并回填样式的最终 Markdown；
- `styles.json`：逐字符/词坐标、三态粗体、颜色和下划线证据；
- `styled_ocr.md`：仅按 PP-OCR 行生成的调试结果；
- `vl.md`：PaddleOCR-VL 原始 Markdown；
- `overlay.jpg`：样式框可视化；
- `ppocr.jsonl`、`vl.jsonl`：API 原始结果缓存。

粗体使用 `true / false / null`。只有高置信证据才写成 `true` 并渲染为
`**...**`；中间区域保留为 `null`，避免把识别失败解释成普通字重。

## 目标场景测试

```bash
uv run python dense_fixtures.py
uv run --group textar python evaluate_dense.py
uv run --group textar python evaluate_style_matrix.py
uv run --group textar python evaluate_policy.py
uv run python build_policy_review.py
uv run --group textar python evaluate_noise.py
uv run python evaluate_api_dense.py
uv run pytest -q
```

`testdata/dense/` 包含六类目标文档：

1. 760×3000 APP 隐私政策长截图；
2. 用户服务协议与个人信息授权书；
3. 密集个人信息收集与使用清单表格；
4. 640 px 窄屏、小字号的授权确认页；
5. 深色模式隐私政策；
6. 行内粗体边界压力页，覆盖单字粗体、中等字重、相邻粗体和组合样式。

用户协议正文使用衬线字体，其他用例使用无衬线字体；内容同时覆盖中文、数字、
英文邮箱、局部粗体、彩色字和下划线链接。

`testdata/style_matrix/` 另外包含 12 组参数化页面，交叉覆盖：

- Noto Sans/Noto Serif CJK；
- 12、13、14、15、16、20 px；
- 600、700、900 目标粗体，以及 500 字重负例；
- 白色、暖色、灰色和深色背景；
- 单字、短语、长片段、表格、中文/英文/数字和组合样式边界。

噪声版本覆盖 0.85/0.67 倍缩放、中度/强度 JPEG、模糊噪点和低对比度色偏。

`testdata/policy/` 新增 8 类协议布局：360 px 窄屏政策、整段加粗协议、纯普通
正文负例、单字强调、换行 SDK 表格、12 px 灰底说明、深色组合样式和衬线协议。
每类包含原图、JPEG Q55、0.85 倍缩放，共 24 个测试输入。
`evaluate_policy.py` 还回归已有 6 个密集文档和 12 个字体矩阵页面，总计 42 次
样式测试；前后版本共享模型预测，单独比较连续片段置信度修复的影响。

运行上述评测及对照页生成命令后，打开 `testdata/policy/review.html` 查看目录。
每页提供原图、样式预览、可切换的粗体错误框，以及 `styled.md` 链接；实际文件在
`testdata/policy/results/<用例>/<clean|jpeg_q55|resize_085>/`。
这些结果使用 DOM 真值字符框，不调用 API，不代表真实 OCR 的端到端准确率，
也不评估表格结构还原。生成的详细输出已忽略，可通过命令重建。

## 范围边界

本 Demo 不重复实现长截图分页或拼接。字体名、精确 CSS 字号、斜体、删除线、
上标下标和背景高亮不在当前核心范围；`visual_font_height_px` 与
`relative_size` 只作为坐标诊断值。压缩和缩小会显著降低粗体识别精度，详见
`VALIDATION.md`。TexTAR 的公开代码采用 MIT License，但模型仓库没有填写
许可证元数据，生产使用前仍需向发布方确认权重授权。
