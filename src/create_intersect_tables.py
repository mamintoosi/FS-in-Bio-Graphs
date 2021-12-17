# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021-2029 by
#    Mahmood Amintoosi <m.amintoosi@gmail.com>
#    All rights reserved.
#    MIT license.
"""Main for feature selection"""

# برنامه انتخاب ویژگی‌هایی از گراف که در انتخاب نودهای ضد سرطان مؤثرتر هستند

from pandas import ExcelWriter
import pandas as pd
import numpy as np
# from itertools import tee
from tqdm.notebook import tqdm
import networkx as nx
from pandas import DataFrame 
pd.options.display.float_format = "{:.2f}".format
import random
from itertools import combinations, chain
# import ml_metrics
import argparse
# from bio_graph_utils import * #make_graph_from_df
# from feature_selection import powerset, obj_fun
# import seaborn as sns

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type = str, default = 'stomach')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # برای اجرای محلی 
    # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # در اجرای محلی پارامترهای رشته‌ای ارسالی به مین نباید داخل تک کوتیشن باشند

    # global args 
    args = get_args()


    data_dir = 'data/'
    output_dir = 'results/'
    working_dir = 'data/'

    # فایل زیر حاوی متابولیت‌هایی است که خاصیت ضدسرطانی آنها اثبات شده است، 
    # به همراه تعدادی از گیاهان دارای این متابولیت‌ها
    working_file_name = 'AC.xlsx'

    # فایلهای زیر حاوی گیاهانی هست که خاصیت ضد سرطانی آنها اثبات است
    # GT_file_name = 'LR_Met_Plant.xlsx'
    if args.dataset_name.lower() == 'stomach':
        GT_file_name = 'Stomach.xlsx'
    elif 'wound' in args.dataset_name.lower():
        GT_file_name = 'Wound_healing.xlsx'
    elif 'breast' in args.dataset_name.lower():
        GT_file_name = 'Breast.xlsx'
    else:
        print('Unknown dataset')
        exit(1)
    node_objects = 'Plant'
    edge_objects = 'Met'

    working_file = working_dir+working_file_name
    wdf = pd.read_excel(working_file, engine="openpyxl") 

    # در فرم نود=گیاه، گیاهانی که حداقل مین‌کانت بار در دیتابیس اومدن، نگه داشته میشن
    # به عبارت دیگه حداقل مین‌کانت متابولیت داشتن
    # نگهداری ستونهای با بیش از یا مساوی با ۵ عنصر
    # min_count = 2
    min_count_list = [1] # np.arange(1,3).tolist() # 1,8

    GT_file = working_dir+GT_file_name
    true_df = pd.read_excel(GT_file, engine="openpyxl") 

    # در فرم نود=گیاه، لیست گیاهانی که خاصیت ضد سرطانی آنها اثبات شده است
    true_list = list(true_df[node_objects].unique())
    # true_list = list(true_df.keys().values)

    # رسم نمودار تکی 
    xls_file_name = 'results/FS_{}_{}_{}_{}_mc{:d}-{:d}.xlsx'.format(working_file_name[:2], GT_file_name[:2], \
        node_objects, edge_objects, min_count_list[0], min_count_list[-1])
    # writer = ExcelWriter(xls_file_name)
    xl = pd.ExcelFile(xls_file_name) 
    # jadval.to_excel(writer, sheet_name='jadval')  # , index=False)
    # gf_df_sorted.to_excel(writer, sheet_name='gf_df_sorted')  # , index=False)
    # gf_df.to_excel(writer, sheet_name='gf_df')  # , index=False)
    
    # intersected_nodes = set(wdf['Plant'].unique()).intersection(set(true_df['Plant'].unique()))

    true_list = list(true_df['Plant'].unique())
    # true_list
    
    # SDF_list = xl.parse('gf_df_max_sorted')  # , index=False)
    # recom_list = list(SDF_list.iloc[:,0].values[:20])
    # # recom_list
    # print("Best Recom List:")
    # for p in recom_list:
    #     flag = 1 if p in true_list else 0
    #     print(p, flag)
        
    SDF_list = xl.parse('gf_df_min_sorted')  # , index=False)
    recom_list = list(SDF_list.iloc[:,0].values[:20])        
    print("Worst Recom List:")
    for p in recom_list:
        flag = 1 if p in true_list else 0
        print(p, flag)        