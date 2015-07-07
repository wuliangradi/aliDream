#coding=utf-8
# 训练-预测-验证
"""
---- author = "liang wu" ----
---- time = "20150405" ----
---- Email = "wl062345@gmail.com" ----
"""
import csv
import pandas as pd
from sklearn.linear_model import  LogisticRegression
from datetime import datetime as dt

def normalize(mat):
    mean = mat.mean(axis=0)   # 每一列数据均值
    dif = mat - mean
    std = mat.std(axis=0)     # 每一列数据方差
    norMat = dif/std          # 计算归一化值
    return norMat

def logisticRegression():
    train = pd.read_csv("train.csv", sep=',', header=None)
    array = train.values
    y = array[:,17]
    X = array[:,[0,1,2,3,6,7,9,12,13,15]]
    X = normalize(X)
    print X[:10,:]

    model = LogisticRegression()
    model = model.fit(X, y)
    print model.score(X, y)
    #from sklearn import svm
    #clf = svm.SVC()
    #clf.fit(X, y)
    print "*******训练完成*******"

    test = pd.read_csv("right_week_feature.csv", sep=',', header=0)
    test = test.drop([test.columns[0]], axis=1)
    test_array = test.values
    test_array = test_array[:, [2,3,4,5,8,9,11,14,15,17]]
    test_array = normalize(test_array)

    print "*******读取待分类数据完成*******"
    predict = model.predict(test_array)
    #predict = clf.predict(test_array)

    print predict[:1000]
    print "*******预测完成*******"
    csv_file = open('predict.csv', 'w')
    m = csv.writer(csv_file, delimiter=',')
    for i in range(predict.shape[0]):
        if predict[i] == 1:
            item = [test.iloc[i]['user_id'], test.iloc[i]['item_id']]
            m.writerow(item)

def validation():
    df = pd.read_csv('tianchi_mobile_recommend_train_user.csv', sep=',', parse_dates=True,
                        header=0, index_col=5)
    print "数据导入成功"
    df = df.ix['2014 12 18']
    df = df[df['behavior_type']==4]

    num_correct = 0
    num_real = df.shape[0]
    num_predict = 0

    with open("predict.csv") as fp:
        for line in fp.readlines():
            line = line.strip()
            u_id, i_id = line.split(',')
            u_id = int(float(u_id))
            i_id = int(float(i_id))
            u = df[df.user_id == u_id]
            i = u[u.item_id == i_id]
            if i.shape[0]>0:
                num_correct = num_correct + 1
            num_predict = num_predict + 1
    fp.close()
    print num_real
    print num_predict
    print "准确率"
    p = float(num_correct)/num_predict
    print p
    print "召回率"
    c = float(num_correct)/num_real
    print c
    print "得分"
    print 2*p*c/(p+c)

logisticRegression()
validation()