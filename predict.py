#coding=utf-8
# 阿里大数据竞赛--初始版本
"""
---- author = "liang wu" ----
---- time = "20150705" ----
---- Email = "wl062345@gmail.com" ----
"""
import csv
import pandas as pd
from sklearn.linear_model import  LogisticRegression
from datetime import datetime as dt

def loadDatabydate():
    '''
    按照时间分割数据
    :return:
    '''
    all_df = pd.read_csv('tianchi_mobile_recommend_train_user.csv', sep=',', parse_dates=True,
                        header=0, index_col=5)
    df = all_df.ix['2014 12 11']
    df_buy = df[df['behavior_type']==4]
    df_other = df[df['behavior_type']!=4]
    df_buy.to_csv('11_buy.csv', sep=',', encoding='utf-8')                 # 用于训练的标签数据（正样本）
    num = df_buy.shape[0]*2
    df_other[:num].to_csv('11_other.csv', sep=',', encoding='utf-8')       # 用于训练的标签数据（负样本）
    df_latest_week = all_df['2014, 12, 12':'2014, 12, 18']
    df_latest_week.to_csv('latest_week.csv', sep=',', encoding='utf-8')    # 用于预测19号的19号之前的一周数据
    df_1 = all_df['2014, 12, 4':'2014, 12, 10']
    df_1.to_csv('week_one.csv', sep=',' ,encoding='utf-8')                 # 用于训练的训练数据


def featureExtraction(filename):
    """
    提取特征
    :param filename:
    :return:
    """
    df = pd.read_csv(filename, sep=',', header=0)
    df = df.drop_duplicates()
    csv_file = open(filename[0:-4]+'_num.csv', 'w')
    a = csv.writer(csv_file, delimiter=',')
    for (u, i), group in df.groupby(['user_id', 'item_id']):
        num_skim = 0
        num_collect = 0
        num_cart = 0
        num_buy = 0
        dict =  pd.value_counts(group.behavior_type, sort=False)
        if (1 in dict):
            num_skim = int(dict[1])
        if (2 in dict):
            num_collect = int(dict[2])
        if 3 in dict:
            num_cart = int(dict[3])
        if 4 in dict:
            num_buy = int(dict[4])
        item = [u, i, num_skim, num_collect, num_cart, num_buy]
        a.writerow(item)

def getDataMat():
    '''
    获得训练数据和测试数据的矩阵X,y
    :return:
    '''
    unames = ['user_id', 'item_id', 'skim', 'collect', 'cart', 'buy']
    csv_file = open('train.csv', 'w')
    a = csv.writer(csv_file, delimiter=',')
    test = pd.read_csv("week_one_num.csv", sep=',', header=None, names=unames)
    print "读取数据完成"
    with open("11_buy.csv") as fp:
        for line in fp.readlines()[1:]:
            line = line.strip()
            list = line.split(',')
            u_id = int(float(list[1]))
            i_id = int(float(list[2]))
            y_value = int(float(list[3]))
            u = test[test.user_id == u_id]
            i = u[u.item_id == i_id]
            if i.shape[0]>0:
                item = [i.iloc[0]['skim'], i.iloc[0]['collect'], i.iloc[0]['cart'], i.iloc[0]['buy'], 1]
                a.writerow(item)
    print '正数据处理完成'
    with open("11_other.csv") as fp:
        for line in fp.readlines()[1:]:
            line = line.strip()
            list = line.split(',')
            u_id = int(float(list[1]))
            i_id = int(float(list[2]))
            u = test[test.user_id == u_id]
            i = u[u.item_id == i_id]
            if i.shape[0]>0:
                item = [i.iloc[0]['skim'], i.iloc[0]['collect'], i.iloc[0]['cart'], i.iloc[0]['buy'], 0]
                a.writerow(item)
    print "负数据处理完成"
    fp.close()

def logisticRegression():
    '''
    利用逻辑回归模型预测
    :return:
    '''
    train = pd.read_csv("train.csv", sep=',', header=None)
    array = train.values
    y = array[:,4]
    X = array[:,0:4]
    model = LogisticRegression()
    model = model.fit(X, y)
    print model.score(X, y)
    print "*******训练完成*******"
    unames = ['user_id', 'item_id', 'skim', 'collect', 'cart', 'buy']
    test = pd.read_csv("latest_week_num.csv", sep=',', header=None, names=unames)
    test_array = test.values
    test_array = test_array[:, 2:6]
    print "*******读取待分类数据完成*******"
    predict = model.predict(test_array)
    print predict[:1000]
    print "*******预测完成*******"
    csv_file = open('predict.csv', 'w')
    m = csv.writer(csv_file, delimiter=',')
    for i in range(predict.shape[0]):
        if predict[i] == 1:
            item = [test.iloc[i]['user_id'], test.iloc[i]['item_id']]
            m.writerow(item)

def validation():
    '''
    验证预测精度
    :return:
    '''
    df = pd.read_csv('tianchi_mobile_recommend_train_user.csv', sep=',', parse_dates=True,
                        header=0, index_col=5)
    print "数据导入成功"
    df = df.ix['2014 12 18']
    df = df[df['behavior_type']==4]

    num_correct = 0
    num_real = df.shape[0]
    num_predict = 0

    with open("new_predict.csv") as fp:
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
    print float(num_correct)/num_predict
    print "召回率"
    print float(num_correct)/num_real

def main():
    start = dt.now()
    #loadDatabydate()           # 根据时间范围获取数据：包括train data, test data; 仅为原始样本的csv格式，需要进行清洗和特征提取
    print "数据分割完成..."
    #featureExtraction("week_one.csv")   # 提取特征
    print "数据块1  特征提取完成..."
    #featureExtraction("latest_week.csv")
    print "数据块2  特征提取完成..."
    #getDataMat()           # 得到矩阵数据
    print "矩阵生成"
    #logisticRegression()  # 训练逻辑回归模型并进行预测
    validation()           # 验证预测精度
    end = dt.now()
    print end-start
    
if __name__=='__main__':main()
