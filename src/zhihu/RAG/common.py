from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from nltk.tokenize import sent_tokenize

from zhihu.common.util import write_log_file

# extracting test from pdf, reference:
# - extract text sample: https://pdfminersix.readthedocs.io/en/latest/tutorial/extract_pages.html
# - pdfminer.six 解析pdf layout的算法: https://pdfminersix.readthedocs.io/en/latest/topic/converting_pdf_to_text.html
# - pdfminer.six 区分段落的逻辑发生在整体算法的第2阶段, 它是判断line间距小于line的高度时,认为是同一个段, 否则分段
def parse_paragraph_from_pdf(in_file, page_numbers:list[int]=None, *, sep:str="\n\n"):
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
    return full_text.split(sep)

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
            # paragraphs.append(text)
            # 对于开头少于last_line_length的line, 直接忽略
            pass
    if buffer:
        paragraphs.append(buffer)
    return paragraphs

def parse_paragraph_from_pdf_v2(in_file, page_numbers:list[int]=None):
    # sep="\n",是为了使用课件中的分段算法，该方法要求输入lines数据每段最后的line后面跟随一个空行
    paras = parse_paragraph_from_pdf(in_file, page_numbers, sep="\n")
    write_log_file("lines_1", paras)
    return _split_paragraph_by_lastline_length(paras, last_line_length=5)


def cascade_split_text(paragraph, chunk_size=300, overlap_size=100):
    """
    将paragraph按照chunk_size进行切分，每个chunk之间有overlap_size的重叠
    不再按paragraph去组织文本，而是打散成按句子组织的粒度，而且每个句子之前之后，加上一段重叠的文本
    """
    sentence = [s.strip() for p in paragraph for s in sent_tokenize(p)]
    chunks = []
    i = 0
    while i < len(sentence):
        chunk = sentence[i]
        overlap = ""
        prev = i - 1
        while prev >= 0 and len(sentence[prev]) + len(overlap) <= overlap_size:
            overlap = sentence[prev] + " " + overlap
            prev -= 1
        chunk = overlap + chunk
        next = i + 1
        while next < len(sentence) and len(chunk) + len(sentence[next]) < chunk_size:
            chunk += " " + sentence[next]
            next += 1
        
        chunks.append(chunk)
        i = next    # i设定为已添加完next的句子的下一个位置
    return chunks

