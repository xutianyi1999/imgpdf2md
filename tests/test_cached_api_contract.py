from pathlib import Path

from style_demo import extract_ppocr, extract_vl_markdown, read_first_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_ppocr_word_box_contract_from_real_response():
    payload = read_first_jsonl(
        ROOT / "api_test_results/PP-OCRv6-word-box/包含粗体和蓝色字体.jsonl"
    )
    result = extract_ppocr(payload)
    assert result["return_word_box"] is True
    assert len(result["text_word"]) == len(result["text_word_boxes"])
    assert "示" in result["text_word"][10]


def test_vl_real_response_contains_detected_underline():
    payload = read_first_jsonl(
        ROOT / "api_test_results/PaddleOCR-VL-1.6/下划线测试.jsonl"
    )
    markdown = extract_vl_markdown(payload)
    assert "\\underline" in markdown
    assert "行中三次或多次" in markdown
