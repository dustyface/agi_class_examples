from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# extracting test from pdf, reference:
# - extract text sample: https://pdfminersix.readthedocs.io/en/latest/tutorial/extract_pages.html
# - pdfminer.six 解析pdf layout的算法: https://pdfminersix.readthedocs.io/en/latest/topic/converting_pdf_to_text.html
# - pdfminer.six 区分段落的逻辑发生在整体算法的第2阶段, 它是判断line间距小于line的高度时,认为是同一个段, 否则分段
def parse_paragraph_from_pdf(in_file, page_numbers:list[int]=None):
    full_text = ""
    for i, page_layout in enumerate(extract_pages(in_file)):
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            # element 可以是LTTextBox, LTFigure, LTImage, LTRect, LTLine
            # LTTextContainer本身就是一个段落, 每句话后面都有一个\n
            # 因此每段最后添加一个\n, 然后split("\n\n")可以区分每段;
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + "\n"
    return full_text.split("\n\n")

# _split_paragraph_by_lastline_length区分段落的逻辑如下:
# 1. 形成一段: 当line的内容大于last_line_length时，认为这是一段中的内容，不断的加到buffer中;
# 2. 一段的结束: 当line的内容，小于last_line_length时，认为这段是上一段最后一line，将line加到buffer，添加一个paragraph
# bug: 依据last_line_length来分段，是可能存在问题的，
# e.g. 当一段的最后一行，其文字长度超过last_line_length时，会导致2段的合并;
def _split_paragraph_by_lastline_length(lines: list[str], last_line_length=1):
    paragraphs = []
    buffer = ""
    for text in lines:
        if len(text) >= last_line_length:
            buffer += (" " + text) if not text.endswith("-") else text.strip("-")
        elif buffer:
            paragraphs.append(buffer)
            buffer = ""
        else:  # bugfix: 当text < min_line_length, 且buffer为空，说明这是一段的开始，直接将text加到buffer中
            paragraphs.append(text)
    if buffer:
        paragraphs.append(buffer)
    return paragraphs

def parse_paragraph_from_pdf_v2(in_file, page_numbers:list[int]=None):
    paras = parse_paragraph_from_pdf(in_file, page_numbers)
    lines = "\n".join(paras).split("\n")
    return _split_paragraph_by_lastline_length(lines, last_line_length=10)