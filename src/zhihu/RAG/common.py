""" This module is the common utility for RAG pipeline"""
import re
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import jieba
from zhihu.common.util import write_log_file

nltk.download("punkt")              # 英文切词、词根、切句等方法
nltk.download("stopwords")          # 英文停用词库

# extracting test from pdf, reference:
# - extract text sample: https://pdfminersix.readthedocs.io/en/latest/tutorial/extract_pages.html
# - pdfminer.six 解析pdf layout的算法:
#   https://pdfminersix.readthedocs.io/en/latest/topic/converting_pdf_to_text.html
# - pdfminer.six 区分段落的逻辑发生在整体算法的第2阶段, 它是判断line间距小于line的高度时,认为是同一个段, 否则分段


def parse_paragraph_from_pdf(in_file, page_numbers: list[int] = None, *, sep: str = "\n\n"):
    """ Parse the pdf file, ouptut the paragraphs """
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
    """ Split the lines into paragraphs by the last line length """
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


def parse_paragraph_from_pdf_v2(in_file, page_numbers: list[int] = None):
    """
    Parse the pdf file, ouptut the paragraphs,
    using the splitting method provided by zhihu
    """
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
        next_pos = i + 1
        while next_pos < len(sentence) and len(chunk) + len(sentence[next_pos]) < chunk_size:
            chunk += " " + sentence[next_pos]
            next_pos += 1

        chunks.append(chunk)
        i = next_pos    # i设定为已添加完next_pos的句子的下一个位置
    return chunks


def to_keywrod_en(input_string):
    """ Convert the input string to keywords """
    # 使用正则表达式替换所有非字母数字的字符为空格
    no_symblos = re.sub(r"[^a-zA-Z0-9\s]", " ", input_string)
    word_tokens = word_tokenize(no_symblos)
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    # 去停用词，取词根
    filtered_sentence = [
        ps.stem(w) for w in word_tokens if not w.lower() in stop_words]
    return " ".join(filtered_sentence)


def to_keyword_cn(input_string):
    """ 将句子转成检索关键词序列 """
    # 按搜索引擎模式分词
    word_tokens = jieba.cut_for_search(input_string)
    # 加载停用词表
    stop_words = set(stopwords.words("chinese"))
    # 去除停用词
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)


def sent_tokenize_cn(input_string):
    """按标点断句"""
    # 按标点切分
    sentences = re.split(r'(?<=[。！？；?!])', input_string)
    # 去掉空字符串
    return [sentence for sentence in sentences if sentence.strip()]


def rrf(ranks, k=1):
    """ 混合检索, 基于倒数排序的融合算法 """
    ret = dict()
    for rank in ranks:
        for _id, val in rank.items():
            if id not in ret:
                ret[_id] = {"score": 0, "text": val["text"]}
            # 计算rrf公式的单项的值
            ret[_id]["score"] = 1.0/(k + val["rank"])
    return dict(sorted(ret.items(), key=lambda x: x[1]["score"], reverse=True))
