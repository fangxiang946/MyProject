'''
    多分类问题：根据文本信息进行类别预测分类
    数据集：动漫csv
    步骤：
        1.获取数据
        2.数据预处理
        3.特征工程
            分词 + tfidf
        4.模型训练
        5.模型预测 + 模型评估
'''

import numpy as np
import pandas as pd
import jieba
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def GetData():
    '''
        返回：特征值、目标值
    '''
    path = '../../MyDataSets/分类数据集/动漫分类/动漫.csv'
    df = pd.read_csv(path, names=["x1", "x2", "x3", "x4", "x5"])

    df = df.dropna()
    df = df[:10000]  # 取前10000条数据来训练和测试
    x_data = df.x5
    y_data = df.x1
    return x_data, y_data

#中文需要自己分词，然后再进行文本提取
def cuttext(text):
    return " ".join(list(jieba.cut(text)))

def dealText(data):
    temp = []
    for s in data:
        temp.append(cuttext(str(s)))
    return temp


def DataPreprocessing(x_data, y_data):
    '''
           一、对【特征值】进行处理
               1.分词
           二、对【目标值】进行处理
               1.将类型文本值转换为数值
           返回：x_data, y_data
       '''
    x_data = dealText(x_data)

    encoder = LabelEncoder()
    y_data = encoder.fit_transform(y_data)
    return x_data, y_data



def FeatureEngineering(x_data, y_data):
    '''
        1、拆分数据集
        2、对特征值进行tfidf分类
        返回：x_train, x_test, y_train, y_test
    '''
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,random_state=22)
    tranfer = TfidfVectorizer()
    x_train = tranfer.fit_transform(x_train)
    x_test = tranfer.transform(x_test)
    return x_train, x_test, y_train, y_test

def Model_fit_predict(x_train, x_test, y_train, y_test):
    '''
        1、训练模型+预测
        2、模型评估
    '''
    y_predict = fit_predict_byRandomForest(x_train, x_test, y_train)

    report = model_Report(y_test,y_predict)

    print(report)



def fit_predict_byRandomForest(x_train, x_test, y_train):
    estimator = RandomForestClassifier()
    # 训练
    estimator.fit(x_train, y_train)
    # 预测
    y_predict = estimator.predict(x_test)
    return y_predict

def model_Report(y_test, y_predict):
    report = classification_report(y_test, y_predict)
    return report


if __name__ == "__main__":
    x_data,y_data = GetData()

    x_data,y_data=DataPreprocessing(x_data,y_data)

    x_train, x_test, y_train, y_test = FeatureEngineering(x_data,y_data)

    Model_fit_predict(x_train, x_test, y_train, y_test)