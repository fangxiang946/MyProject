'''
    二分类问题：对垃圾邮件进行二分类
    数据集：垃圾邮件数据集
        ham_data.txt  正常邮件
        spam_data.txt 垃圾邮件
        stop_words.utf8  停词器
    步骤：
         1.获取数据源
        2.获取停词器
        3.拆分数据集
        4.处理文本
            分词
            过滤停词
        5.tfidf特征抽取
        6.模型训练
        7.模型评估

    打印结果：
        基于tfidf的【贝叶斯】模型
            准确率: 0.79
            精度: 0.85
            召回率: 0.79
            F1得分: 0.78
        基于tfidf的【决策树】模型
            准确率: 0.97
            精度: 0.97
            召回率: 0.97
            F1得分: 0.97
        基于tfidf的【随机森林】模型
            准确率: 0.96
            精度: 0.97
            召回率: 0.96
            F1得分: 0.96
'''

import numpy as np
import re
import string
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB


# 获取数据
def get_data(path_ham,
             path_spam):
    '''
    获取数据
    :return: 文本数据，对应的labels
    '''
    with open(path_ham, encoding="utf8") as ham_f, open(path_spam, encoding="utf8") as spam_f:
        # 正常的邮件数据
        ham_data = ham_f.readlines()
        # 垃圾邮件数据
        spam_data = spam_f.readlines()

        # 正常数据标记为1
        ham_label = np.ones(len(ham_data)).tolist()
        # 垃圾邮件数据标记为0
        spam_label = np.zeros(len(spam_data)).tolist()

        # 数据集集合
        corpus = ham_data + spam_data

        # 标记数据集合
        labels = ham_label + spam_label

    return corpus, labels

# 数据分割
def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    '''
    :param corpus: 文本数据
    :param labels: label数据
    :param test_data_proportion:测试数据占比
    :return: 训练数据,测试数据，训练label,测试label
    '''
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels,
                                                        test_size=test_data_proportion, random_state=42)
    return train_X, test_X, train_Y, test_Y

# 移除空的数据
def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels

# 加载停用词
def get_stopword(path_stopword):
    with open(path_stopword, encoding="utf8") as f:
        stopword_list = f.readlines()
    return stopword_list

# 分词
def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens

# 去掉特殊符号
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# 取出停用词
def remove_stopwords(text,stopword_list):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text

# 使用tfidf特征提取
def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


# 标准化数据集
def normalize_corpus(corpus,stopword_list, tokenize=False):
    # 声明一个变量用来存储标准化后的数据
    normalized_corpus = []
    for text in corpus:
            # 去掉特殊符号
        text = remove_special_characters(text)
            # 取出停用词
        text = remove_stopwords(text,stopword_list)
        normalized_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)

    return normalized_corpus

# 模型性能预测函数
def get_metrics(true_labels, predicted_labels):
    print('准确率:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels),
        2))
    print('精度:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'),
        2))
    print('召回率:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'),
        2))
    print('F1得分:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'),
        2))


# 模型调用函数，这样做的好处是，可以自己人选模型
def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # 模型构建
    classifier.fit(train_features, train_labels)
    # 使用哪个模型做预测
    predictions = classifier.predict(test_features)
    # 评估模型预测性能
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions


def main():
    '''
        1.获取数据源
        2.获取停词器
        3.拆分数据集
        4.处理文本
            分词
            过滤停词
        5.tfidf特征抽取
        6.模型训练
        7.模型评估
    '''
    path_ham = '../../MyDataSets/分类数据集/垃圾邮件分类/ham_data.txt'
    path_spam = '../../MyDataSets/分类数据集/垃圾邮件分类/spam_data.txt'

    corpus, labels = get_data(path_ham,path_spam)  # 1.获取数据集
    corpus, labels = remove_empty_docs(corpus, labels) #数据集取出空值

    path_stopword = "helper/stop_words.utf8"
    stopword_list = get_stopword(path_stopword)  #2.获取停词器


    # 3.对数据进行划分
    train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus,
                                                                            labels,
                                                                            test_data_proportion=0.3)


    #4.处理文本
    norm_train_corpus = normalize_corpus(train_corpus,stopword_list)
    norm_test_corpus = normalize_corpus(test_corpus,stopword_list)

    #5.tfidf特征抽取
    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

    #6.模型训练
    mnb = MultinomialNB()
    # 基于tfidf的多项式朴素贝叶斯模型
    print("基于tfidf的贝叶斯模型")
    mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)



    return None



if __name__ == "__main__":
    main()