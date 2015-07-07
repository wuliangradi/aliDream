#coding=utf-8
# 构造特征矩阵X，y
"""
---- author = "liang wu" ----
---- time = "20150405" ----
---- Email = "wl062345@gmail.com" ----
"""
import csv
import pandas as pd
from sklearn.linear_model import  LogisticRegression
from datetime import datetime as dt


def four_ten():
    df_item = pd.read_csv("week_4-10_item.csv", sep=',', header=0)
    df_user = pd.read_csv("week_4-10_user.csv", sep=',', header=0)
    df_user_item = pd.read_csv("week_4-10_sum.csv", sep=',', header=0)
    df_right_month = pd.read_csv("month_right_item.csv", sep=',', header=0)
    merge_four_ten = pd.merge(pd.merge(df_user_item, df_user),df_right_month)
    merge_four_ten.to_csv('four_ten_feature_month.csv', sep=',', encoding='utf-8')

def week_left():
    df_item = pd.read_csv("week_left_item.csv", sep=',', header=0)
    df_user = pd.read_csv("week_left_user.csv", sep=',', header=0)
    df_user_item = pd.read_csv("week_left_sum.csv", sep=',', header=0)
    df_right_month = pd.read_csv("month_right_item.csv", sep=',', header=0)
    merge_left_week = pd.merge(pd.merge(df_user_item, df_user),df_right_month)
    merge_left_week.to_csv('left_week_feature_month.csv', sep=',', encoding='utf-8')

def week_right():
    df_item = pd.read_csv("week_right_item.csv", sep=',', header=0)
    df_user = pd.read_csv("week_right_user.csv", sep=',', header=0)
    df_user_item = pd.read_csv("week_right_sum.csv", sep=',', header=0)
    df_right_month = pd.read_csv("month_right_item.csv", sep=',', header=0)
    merge_left_week = pd.merge(pd.merge(df_user_item, df_user),df_right_month)
    merge_left_week.to_csv('111.csv', sep=',', encoding='utf-8')

def ten_17():
    df_user = pd.read_csv("8-17_user.csv", sep=',', header=0)
    df_user_item = pd.read_csv("8-17_sum.csv", sep=',', header=0)
    df_right_month = pd.read_csv("month_right_item.csv", sep=',', header=0)
    print "----"
    df_1 = pd.merge(df_user_item, df_user)
    print "++++++++"
    merge_left_week = pd.merge(df_1,df_right_month)
    print "-------"
    merge_left_week.to_csv('10-17_feature.csv', sep=',', encoding='utf-8')


def nine_18():
    df_user = pd.read_csv("9-18_user.csv", sep=',', header=0)
    df_user_item = pd.read_csv("9-18_sum.csv", sep=',', header=0)
    df_right_month = pd.read_csv("month_right_item.csv", sep=',', header=0)
    print "----"
    df_1 = pd.merge(df_user_item, df_user)
    print "++++++++"
    merge_left_week = pd.merge(df_1,df_right_month)
    print "-------"
    merge_left_week.to_csv('9-18_feature.csv', sep=',', encoding='utf-8')

def one_seven():
    df_user = pd.read_csv("1-7_user.csv", sep=',', header=0)
    df_user_item = pd.read_csv("1-7_sum.csv", sep=',', header=0)
    df_right_month = pd.read_csv("month_right_item.csv", sep=',', header=0)
    merge_left_week = pd.merge(pd.merge(df_user_item, df_user),df_right_month)
    merge_left_week.to_csv('one_seven_feature_month.csv', sep=',', encoding='utf-8')

def loadLabelData():
    all_df = pd.read_csv('tianchi_mobile_recommend_train_user.csv', sep=',', parse_dates=True,
                        header=0, index_col=5)

    df_11 = all_df.ix['2014 12 11']
    df_11_buy = df_11[df_11['behavior_type']==4]
    df_11_other = df_11[df_11['behavior_type']!=4]

    num = int(df_11_buy.shape[0]*0.3)
    df_11_buy = df_11_buy[:num]
    df_11_buy.to_csv('11_buy.csv', sep=',', encoding='utf-8')                 # 用于训练的标签数据（正样本）
    num = df_11_buy.shape[0]*9
    df_11_other[:num].to_csv('11_other.csv', sep=',', encoding='utf-8')       # 用于训练的标签数据（负样本）

    df = all_df.ix['2014 12 4']
    df_buy = df[df['behavior_type']==4]
    df_other = df[df['behavior_type']!=4]
    num = int(df_buy.shape[0]*0.3)
    df_buy = df_buy[:num]
    df_buy.to_csv('4_buy.csv', sep=',', encoding='utf-8')                 # 用于训练的标签数据（正样本）
    num = df_buy.shape[0]*9
    df_other[:num].to_csv('4_other.csv', sep=',', encoding='utf-8')       # 用于训练的标签数据（负样本）

def getTrain():
    csv_file = open('train_month.csv', 'w')
    a = csv.writer(csv_file, delimiter=',')

    left_week = pd.read_csv("left_week_feature_month.csv", sep=',', header=0)
    left_week = left_week.drop([left_week.columns[0]], axis=1)
    with open("4_buy.csv") as fp:
        for line in fp.readlines()[1:]:
            line = line.strip()
            list = line.split(',')
            u_id = int(float(list[1]))
            i_id = int(float(list[2]))
            y_value = int(float(list[3]))
            u = left_week[left_week.user_id == u_id]
            i = u[u.item_id == i_id]
            if i.shape[0]>0:
                item = [i.iloc[0]['u_i_skim'], i.iloc[0]['u_i_collect'], i.iloc[0]['u_i_cart'], i.iloc[0]['u_i_buy'],
                        i.iloc[0]['u_i_cr'], i.iloc[0]['u_skim'], i.iloc[0]['u_collect'], i.iloc[0]['u_cart'],
                        i.iloc[0]['u_buy'], i.iloc[0]['u_cr'], i.iloc[0]['u_item'], i.iloc[0]['i_skim'],
                        i.iloc[0]['i_collect'], i.iloc[0]['i_cart'], i.iloc[0]['i_buy'], i.iloc[0]['i_cr'],i.iloc[0]['i_people'],1]
                a.writerow(item)
    print 'left_week 正样本处理完成'
    with open("4_other.csv") as fp:
        for line in fp.readlines()[1:]:
            line = line.strip()
            list = line.split(',')
            u_id = int(float(list[1]))
            i_id = int(float(list[2]))
            u = left_week[left_week.user_id == u_id]
            i = u[u.item_id == i_id]
            if i.shape[0]>0:
                item = [i.iloc[0]['u_i_skim'], i.iloc[0]['u_i_collect'], i.iloc[0]['u_i_cart'], i.iloc[0]['u_i_buy'],
                        i.iloc[0]['u_i_cr'], i.iloc[0]['u_skim'], i.iloc[0]['u_collect'], i.iloc[0]['u_cart'],
                        i.iloc[0]['u_buy'], i.iloc[0]['u_cr'], i.iloc[0]['u_item'], i.iloc[0]['i_skim'],
                        i.iloc[0]['i_collect'], i.iloc[0]['i_cart'], i.iloc[0]['i_buy'], i.iloc[0]['i_cr'],i.iloc[0]['i_people'],0]
                a.writerow(item)
    print "left_week 负样本处理完成"

    fourTen = pd.read_csv('four_ten_feature_month.csv', sep=',', header=0)
    fourTen = fourTen.drop([fourTen.columns[0]], axis=1)
    with open("11_buy.csv") as fp:
        for line in fp.readlines()[1:]:
            line = line.strip()
            list = line.split(',')
            u_id = int(float(list[1]))
            i_id = int(float(list[2]))
            u = fourTen[fourTen.user_id == u_id]
            i = u[u.item_id == i_id]
            if i.shape[0]>0:
                item = [i.iloc[0]['u_i_skim'], i.iloc[0]['u_i_collect'], i.iloc[0]['u_i_cart'], i.iloc[0]['u_i_buy'],
                        i.iloc[0]['u_i_cr'], i.iloc[0]['u_skim'], i.iloc[0]['u_collect'], i.iloc[0]['u_cart'],
                        i.iloc[0]['u_buy'], i.iloc[0]['u_cr'], i.iloc[0]['u_item'], i.iloc[0]['i_skim'],
                        i.iloc[0]['i_collect'], i.iloc[0]['i_cart'], i.iloc[0]['i_buy'], i.iloc[0]['i_cr'],i.iloc[0]['i_people'],1]
                a.writerow(item)
    print 'four_ten 正样本处理完成'
    with open("11_other.csv") as fp:
        for line in fp.readlines()[1:]:
            line = line.strip()
            list = line.split(',')
            u_id = int(float(list[1]))
            i_id = int(float(list[2]))
            u = fourTen[fourTen.user_id == u_id]
            i = u[u.item_id == i_id]
            if i.shape[0]>0:
                item = [i.iloc[0]['u_i_skim'], i.iloc[0]['u_i_collect'], i.iloc[0]['u_i_cart'], i.iloc[0]['u_i_buy'],
                        i.iloc[0]['u_i_cr'], i.iloc[0]['u_skim'], i.iloc[0]['u_collect'], i.iloc[0]['u_cart'],
                        i.iloc[0]['u_buy'], i.iloc[0]['u_cr'], i.iloc[0]['u_item'], i.iloc[0]['i_skim'],
                        i.iloc[0]['i_collect'], i.iloc[0]['i_cart'], i.iloc[0]['i_buy'], i.iloc[0]['i_cr'],i.iloc[0]['i_people'],0]
                a.writerow(item)
    print "four_ten 负样本处理完成"
#loadLabelData()
#getTrain()
#week_left()
#week_right()
#four_ten()
#week_right()
#nine_18()