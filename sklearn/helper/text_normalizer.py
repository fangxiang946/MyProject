import re
import string
import jieba
import unicodedata
from bs4 import BeautifulSoup


# 分词
def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


# 加载停用词
def get_stopword(path_stopword):
    with open(path_stopword, encoding="utf8") as f:
        stopword_list = f.readlines()
    return stopword_list


# 去掉特殊符号和空白格
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    text = re.sub(' +', ' ', text)
    return text


# 取出停用词
def remove_stopwords(text, stopword_list):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def normalize_corpus(corpus, tokenize=False):
    path_stopword = "helper/stop_words.utf8"
    stopword_list = get_stopword(path_stopword)

    # 声明一个变量用来存储标准化后的数据
    normalized_corpus = []
    for text in corpus:
        text = text.lower()
        text = strip_html_tags(text)
        text = remove_accented_chars(text)
        # 移除多余的换行符
        text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
        # 在特殊字符之间插入空白符，使得后面可以简单地将它们移除
        special_char_pattern = re.compile(r'([{.(-)!}])')
        text = special_char_pattern.sub(" \\1 ", text)
        # 去掉特殊符号
        text = remove_special_characters(text)
        # 取出停用词
        text = remove_stopwords(text, stopword_list)

        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)

    return normalized_corpus


def normalize_text(text, tokenize=True):
    path_stopword = "helper/stop_words.utf8"
    stopword_list = get_stopword(path_stopword)
    # 声明一个变量用来存储标准化后的数据
    normalized_text = []
    text = text.lower()
    text = strip_html_tags(text)
    text = remove_accented_chars(text)
    # 移除多余的换行符
    text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
    # 在特殊字符之间插入空白符，使得后面可以简单地将它们移除
    special_char_pattern = re.compile(r'([{.(-)!}])')
    text = special_char_pattern.sub(" \\1 ", text)
    # 去掉特殊符号
    text = remove_special_characters(text)
    # 取出停用词
    text = remove_stopwords(text, stopword_list)

    if tokenize:
        text = tokenize_text(text)
    return text