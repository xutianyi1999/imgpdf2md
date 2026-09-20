# APP 密集文档富文本恢复 Demo

这是一个可行性 Demo，目标输入是已经分页/切图的 APP 隐私政策、用户协议、
授权书和个人信息收集清单截图。它只补足 OCR 通常丢失的三类行内样式：

- 粗体；
- 文字颜色；
- 下划线。

标题、段落、列表、表格及阅读顺序由 AI Studio `PaddleOCR-VL-1.6` 提供；
`PP-OCRv6 + returnWordBox=true` 提供细粒度坐标。本地使用 OpenCV 提取颜色、
检测连续下划线，使用 FontDNA-V2 和同一行相对笔画特征判断粗体，再把样式
对齐回 VL Markdown。

## 运行

`.env` 需要包含 AI Studio Token：

```dotenv
TOKEN=AI Studio access token
```

```bash
uv sync
uv run python style_demo.py input.png
```

首次运行会下载约 9 MB 的 FontDNA-V2 INT8 ONNX 模型到
`.cache/fontdna/`。每页输出位于 `demo_output/<输入名>/page-0001/`：

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
uv run python evaluate_dense.py
uv run python evaluate_noise.py
uv run python evaluate_api_dense.py
uv run pytest -q
```

`testdata/dense/` 只包含三类目标文档：

1. 760×3000 APP 隐私政策长截图；
2. 用户服务协议与个人信息授权书；
3. 密集个人信息收集与使用清单表格。

噪声版本只覆盖缩放、聊天软件 JPEG、模糊/噪点和低对比度色偏。

## 范围边界

本 Demo 不重复实现长截图分页或拼接。字体名、精确 CSS 字号、斜体、删除线、
上标下标和背景高亮不在当前核心范围；`visual_font_height_px` 与
`relative_size` 只作为坐标诊断值。压缩和缩小会显著降低粗体识别精度，详见
`VALIDATION.md`，不能把当前模型当成生产级中文字重识别器。
