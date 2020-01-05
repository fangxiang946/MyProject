'''
    回归问题：预测房价
    数据集：美国加州房屋价格
    步骤：
        1.数据集拆分
             1)找到相关性最高的那个特征
             2)按照那个特征进行分层抽样
        2.数据预处理
             1)缺失值处理-无
             2)类别转one-hot编码-无
             3)特征放缩:标准化
        3.
'''

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def Split_train_test(data_set, test_size=0.2, target_name='fangjia'):
    '''
        1.找到与房价最相关的特征名称
        2.将该特征值离散化
        3.按照相关性最高的特征的分布情况进行对整个数据集进行分层拆分
        返回:训练集,测试集
    '''

    featurename_impotances = np.absolute(data_set.corr()[target_name]).sort_values(ascending=False).index[1]

    data_set["temp"] = np.ceil(data_set[featurename_impotances])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    for train_index, test_index in sss.split(data_set, data_set["temp"]):
        train_set, test_set = data_set.loc[train_index], data_set.loc[test_index]

    train_set.drop("temp", axis=1, inplace=True)
    test_set.drop("temp", axis=1, inplace=True)
    return train_set, test_set


def main():
    california_housing = fetch_california_housing()
    target_name = 'fangjia'
    df = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
    df[target_name] = california_housing.target


    train_set, test_set = Split_train_test(df, target_name=target_name)

    X_train_set = train_set.iloc[:, :-1]
    Y_train_set = train_set[target_name]
    X_test_set = test_set.iloc[:, :-1]
    Y_test_set = test_set[target_name]

    scaler = StandardScaler()
    X_train_set = scaler.fit_transform(X_train_set)
    X_test_set = scaler.transform(X_test_set)


    estimator = RandomForestRegressor()
    estimator.fit(X_train_set,Y_train_set)

    Y_predict = estimator.predict(X_train_set)

    mse = mean_squared_error(Y_train_set,Y_predict)
    print("训练集上：均方误差为：",mse)

    cvs = cross_val_score(estimator, Y_predict.reshape(-1, 1), Y_train_set, scoring="neg_mean_squared_error", cv=10)
    print("训练集上10折：均方误差为：", np.absolute(cvs))

    Y_predict_test = estimator.predict(X_test_set)
    test_mse = mean_squared_error(Y_test_set,Y_predict_test)
    print("测试集上：均方误差为：", test_mse)



if __name__ == '__main__':
    main()