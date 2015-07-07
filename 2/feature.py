#coding=utf-8
# 提取特征
"""
---- author = "liang wu" ----
---- time = "20150405" ----
---- Email = "wl062345@gmail.com" ----
"""
import csv
import pandas as pd
from sklearn.linear_model import  LogisticRegression
from datetime import datetime as dt

def splitData():
    all_df = pd.read_csv("tianchi_mobile_recommend_train_user.csv", sep=',', parse_dates=True,
                        header=0, index_col=5)

    df_latest_week = all_df['2014, 11, 27':'2014, 12, 3']
    df_latest_week.to_csv('week_left.csv', sep=',', encoding='utf-8')

    df_latest_3days = all_df['2014, 12, 1':'2014, 12, 3']
    df_latest_3days.to_csv('3days_left.csv', sep=',', encoding='utf-8')

    df_latest_week = all_df['2014, 12, 11':'2014, 12, 17']
    df_latest_week.to_csv('week_right.csv', sep=',', encoding='utf-8')

    df_latest_3days = all_df['2014, 12, 15':'2014, 12, 17']
    df_latest_3days.to_csv('3days_right.csv', sep=',', encoding='utf-8')

    df_left_month = all_df['2014, 11, 18':'2014, 12, 3']
    df_left_month.to_csv('month_left.csv', sep=',', encoding='utf-8')

    df_right_month = all_df['2014, 12, 4':'2014, 12, 17']
    df_right_month.to_csv('month_right.csv', sep=',', encoding='utf-8')


def U_I_behaviorSum(filename):
    """
    用户-商品 行为累计数
    """
    df = pd.read_csv(filename, sep=',', header=0)
    csv_file = open(filename[0:-4]+'_sum.csv', 'w')
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

        conversion_rate = 0
        conversion_rate = (num_buy*1.0)/dict.sum()
        conversion_rate = float('%.4f'% conversion_rate)
        item = [u, i, num_skim, num_collect, num_cart, num_buy, conversion_rate]
        a.writerow(item)

def feature_itemBased(filename):

    df = pd.read_csv(filename, sep=',', header=0)
    csv_file = open(filename[0:-4]+'_item.csv', 'w')
    a = csv.writer(csv_file, delimiter=',')
    a.writerow(['item_id', 'num_skim', 'num_collect', 'num_cart', 'num_buy', 'conversion_rate', 'num_people_buy'])
    for item_id, group in df.groupby('item_id'):
        dict =  pd.value_counts(group.behavior_type, sort=False)
        num_skim = 0
        num_collect = 0
        num_cart = 0
        num_buy = 0
        if (1 in dict):
            num_skim = int(dict[1])               # 浏览数
        if (2 in dict):
            num_collect = int(dict[2])            # 收藏数
        if 3 in dict:
            num_cart = int(dict[3])               # 加入购物车数
        if 4 in dict:
            num_buy = int(dict[4])                # 购买数

        conversion_rate = 0
        conversion_rate = (num_buy*1.0)/dict.sum()
        conversion_rate = float('%.4f'% conversion_rate)

        buy_group = group[group['behavior_type']==4]
        num_people_buy = (pd.value_counts(buy_group.user_id)).shape[0]   # 购买人数

        item = [item_id, num_skim, num_collect, num_cart, num_buy, conversion_rate, num_people_buy]
        a.writerow(item)

def feature_userBased(filename):

    df = pd.read_csv(filename, sep=',', header=0)
    csv_file = open(filename[0:-4]+'_user.csv', 'w')
    a = csv.writer(csv_file, delimiter=',')
    for user_id, group in df.groupby('user_id'):
        dict =  pd.value_counts(group.behavior_type, sort=False)
        num_skim = 0
        num_collect = 0
        num_cart = 0
        num_buy = 0
        if (1 in dict):
            num_skim = int(dict[1])               # 浏览数
        if (2 in dict):
            num_collect = int(dict[2])            # 收藏数
        if 3 in dict:
            num_cart = int(dict[3])               # 加入购物车数
        if 4 in dict:
            num_buy = int(dict[4])                # 购买数

        conversion_rate = 0
        conversion_rate = (num_buy*1.0)/dict.sum()
        conversion_rate = float('%.4f'% conversion_rate)


        buy_group = group[group['behavior_type']==4]
        num_item_buy = (pd.value_counts(buy_group.item_id)).shape[0]   # 该客户购买的商品种数
        item = [user_id, num_skim, num_collect, num_cart, num_buy, conversion_rate, num_item_buy]
        a.writerow(item)

start = dt.now()
# splitData()
#U_I_behaviorSum("week_left.csv")
#U_I_behaviorSum("week_right.csv")
#U_I_behaviorSum("3days_left.csv")
#U_I_behaviorSum("3days_rightm.csv")
#U_I_behaviorSum("tianchi_mobile_recommend_train_user.csv")
#U_I_behaviorSum("week_4-10.csv")
#U_I_behaviorSum("1-7.csv")
#U_I_behaviorSum("month_right.csv")
U_I_behaviorSum("15-17.csv")
print "第一项完成"

#feature_itemBased("week_left.csv")
#feature_itemBased("3days_left.csv")
#feature_itemBased("3days_right.csv")
#feature_itemBased("week_right.csv")
#feature_itemBased("week_4-10.csv")
feature_itemBased("15-17.csv")
print "第二项完成"

#feature_userBased("3days_left.csv")
#feature_userBased("3days_right.csv")
#feature_userBased("week_left.csv")
#feature_userBased("week_right.csv")
#feature_userBased("week_4-10.csv")
feature_userBased("15-17.csv")
print "第三项完成"

#feature_itemBased("month_right.csv")
#feature_itemBased("tianchi_mobile_recommend_train_user.csv")




