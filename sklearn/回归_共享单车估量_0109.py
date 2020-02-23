'''
    回归问题：预测共享单车需求量
    数据集：共享单车需求量数据集
        https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
    步骤：
        1.选择所需要的特征列
        2.拆分数据集
        3.数据预处理
        4.模型训练
        5.模型评估

    打印结果：
        训练集模型评分: 0.9805292537195031
        训练集10折的评分: [0.86968582 0.85763376 0.82866578 0.89411031 0.79693482 0.84995064
         0.78911578 0.81704367 0.89001661 0.89258841]
        训练集10折的均方误差: 728.7311562988785
        测试集的均方误差: 650.7268352426523
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


def GetData():
    path = '../../MyDataSets/回归数据集/Bike-Sharing-Dataset/day.csv'
    df = pd.read_csv(path)
    return df

def Fix_tranform_oht(df, col_names):
    '''
        将某些列转化成one-hot编码---训练集用！！！
                返回
                    字典：包含onehot列、各列对应的转换器
    '''
    dic = {}
    for col_name in col_names:
        scaler = LabelBinarizer()
        returnObj = ft_single(df, scaler, col_name)
        dic[str(returnObj[2])] = returnObj[0]
        dic["scaler_" + str(returnObj[2])] = returnObj[1]
    return dic

def ft_single(df, scaler, col_name):
    '''
        将某列转化成one-hot编码
        返回
            df_new：新表
            scaler：处理器
            col_name：列名
    '''
    feature_arr = scaler.fit_transform(df[col_name])
    if feature_arr.shape[1] > 1:
        fearure_labels = ['is_' + col_name + '_' + str(cls_label) for cls_label in scaler.classes_]
    else:
        fearure_labels = ['is_' + col_name]
    df_new = pd.DataFrame(data=feature_arr, columns=fearure_labels)
    return df_new, scaler, col_name

def Tranform(df,scaler_dic):
    '''
         将某些列转化成one-hot编码---测试集用！！！
            返回
                字典：包含onehot列、各列对应的转换器
    '''
    dic = {}
    for i in scaler_dic:
        if 'scaler_' in i:
            scaler = scaler_dic[i]
            col_name = i.split('_')[1]
            feature_arr = scaler.transform(df[[col_name]])
            if feature_arr.shape[1] > 1:
                fearure_labels = ['is_'+col_name+'_'+str(cls_label) for cls_label in scaler.classes_]
            else:
                fearure_labels = ['is_'+col_name]
            df_new = pd.DataFrame(data=feature_arr,columns=fearure_labels)
            dic[str(col_name)]=df_new
    return dic





def main():
    #1.获取数据集
    df = GetData()
    #换一些顺眼的列名
    data = df.rename(columns={
        'yr': 'year',
        'mnth': 'month',
        'holiday': 'isholiday',
        'workingday': 'isworkingday',
        'weathersit': 'weather',
        'cnt': 'total'
    })
    myfeaturename = ['season', 'year', 'month', 'isholiday', 'weekday', 'isworkingday', 'weather', 'temp', 'atemp',
                     'hum', 'windspeed']
    #2.拆分数据集
    x_train, x_test, y_train, y_test = train_test_split(data[myfeaturename], data.total, random_state=42)

    #3.数据预处理
    cat_featurename = ['season', 'year', 'month', 'weekday', 'weather']
    tranferObj = Fix_tranform_oht(x_train, cat_featurename)

    x_train_deal = x_train.reset_index()  # 索引要重排一下
    x_train_deal.drop("index", axis=1, inplace=True)
    for i in tranferObj:
        if i in cat_featurename:
            x_train_deal = pd.concat([x_train_deal, tranferObj[i]], axis=1)

    x_train_ready = x_train_deal[[i for i in x_train_deal.columns if i not in cat_featurename]]

    # 4.训练模型
    estimator = RandomForestRegressor()
    estimator.fit(x_train_ready, y_train)

    # 5.模型评估
    score_train = estimator.score(x_train_ready, y_train)
    print("训练集模型评分:", score_train)

    score_By10_train = cross_val_score(estimator, x_train_ready, y=y_train, cv=10)
    print("训练集10折的评分:", score_By10_train)

    mse_By10_train = cross_val_score(estimator, x_train_ready, y=y_train, cv=10,scoring="neg_mean_squared_error")
    print("训练集10折的均方误差:", np.mean(np.sqrt(-mse_By10_train)))

    # 6.开始对测试集下手
    tranferObj_test = Tranform(x_test, tranferObj)

    x_test_deal = x_test.reset_index()  # 索引要重排一下
    x_test_deal.drop("index", axis=1, inplace=True)
    for i in tranferObj_test:
        if i in cat_featurename:
            x_test_deal = pd.concat([x_test_deal, tranferObj_test[i]], axis=1)

    x_test_ready = x_test_deal[[i for i in x_test_deal.columns if i not in cat_featurename]]

    # 7.开始预测！
    y_predict = estimator.predict(x_test_ready)

    mse = mean_squared_error(y_predict,y_test)

    print("测试集的均方误差:", np.mean(np.sqrt(mse)))

    return None


if __name__ == '__main__':
    main()
