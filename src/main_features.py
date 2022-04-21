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
from bio_graph_utils import * #make_graph_from_df
# from feature_selection import powerset, obj_fun
import seaborn as sns

# # توابع زیر رو از دو فایل دیگه اینجا آوردم که ایمپورت نخواسته باشه و اون بسته دیگه ای که در ویندوز ندارم

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def obj_fun(true_list, pred_list,index):
    [apk,_] = compute_metrics(true_list, pred_list, apk_ranges=index)
    return np.mean(apk) #, np.mean(ark)

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

    minCount = min(min_count_list)

    GT_file = working_dir+GT_file_name
    true_df = pd.read_excel(GT_file, engine="openpyxl") 

    # در فرم نود=گیاه، لیست گیاهانی که خاصیت ضد سرطانی آنها اثبات شده است
    true_list = list(true_df[node_objects].unique())
    # true_list = list(true_df.keys().values)
    index = np.arange(1,10)

    jadval = pd.DataFrame(columns=['Min Count', 'Set No', 'k', 'AP@k'])

    print('Table info: ', wdf.shape)
    print('Number of unique nodes (Plants) in main file: ', len(wdf[node_objects].unique()))
    jadval_row_no = 0

    with tqdm(total=len(min_count_list)) as progress_bar:
        for row_no,min_count in enumerate(min_count_list):
            df = wdf.copy()
            if min_count>1:
                s = df[node_objects].value_counts()
                df = df[df[node_objects].isin(s[s >= min_count].index)]
                # print('Number of nodes after prunning those nodes which have < min count={} elements is: {}'.format(min_count,len(df[node_objects].unique())))

            G,dfct,bow = make_graph_from_df(df,node_objects,edge_objects)

            # اضافه کردن وزن لبه‌ی متابولیت‌های مشترک
            # در حال حاضر فقط یک واحد اضافه میشه که میشه به نسبت تکرارش در مجموعه دوم اضافه بشه
            # print('Number of Metabolites in dfct: ', len(dfct.columns))
            second_graph_edges = list(true_df[edge_objects].unique())
            # print('Number of Metabolites in Wound healing: ',len(second_graph_edges))
            intersected_edges = set(dfct.columns).intersection(set(second_graph_edges))
            print('Number of intersected Meabolites:', len(intersected_edges))
            for row in range(dfct.shape[0]):
                for e in intersected_edges:
                    if dfct.iloc[row][e] != 0:
                        dfct.iloc[row][e] += 1

            # پیدا کردن بزرگترین زیرگراف همبند
            # print('Computing the largest connected graph...\n')
            subG = largest_con_com(dfct, G)
            # print('Number of sub graph nodes:', len(subG.nodes()))
            # # nx.draw_shell(subG,with_labels=True)
            subG_ix = list(subG.nodes())
            dfct_subG = dfct.loc[subG_ix]
            # print('Sub graph info before dropping: ', dfct_subG.shape)
            # drop columns with zero sum
            dfct_subG = dfct_subG.loc[:, (dfct_subG != 0).any(axis=0)]
            # dfct = pd.crosstab(df[node_objects], df[edge_objects])
            # print('Sub graph info: ', dfct_subG.shape)

            # print('Computing the graph features...\n')
            gf_df, gf_df_sorted = rank_using_graph_features(subG, weight='weight') #, min_count, node_objects, \
                # edge_objects, data_dir, output_dir, working_file_name)

            n_features = gf_df.shape[1]-1
            indices = list(powerset(range(n_features)))
            # remove the empty subset
            indices.pop(0)

            # print(indices)
            APK = []
            scores = []
            DF_list= list()
            SDF_list= list()
            for set_no, idx in enumerate(indices):
                cols_s = list(idx) # columns with feature_sum
                cols_s.append(-1)
                # SettingWithCopyWarning دیپ کپی برای اجتناب از خطای 
                selected_df = gf_df.iloc[:, cols_s].copy(deep=True)
                # print(cols,cols_s,selected_df.shape)#,cols_s)
                n_cols = selected_df.shape[1]-1
                features_sum = selected_df.iloc[:,:n_cols].sum(axis=1)
                selected_df['features_sum'] = features_sum
                selected_df_sorted = selected_df.sort_values(by='features_sum', ascending=False)
                
                DF_list.append(selected_df)
                SDF_list.append(selected_df_sorted)
                
                pred_list = list(selected_df_sorted.index.values)
                # print(obj_fun(true_list,pred_list))

                for i_apk_index in index:
                    apk_index = np.arange(1,i_apk_index+1) # +۱ باید باشه وگرنه اولیش تهی هست
                    score = obj_fun(true_list, pred_list, apk_index)
                    scores.append(score)
                    jadval_row_no += 1
                    jadval.loc[jadval_row_no] = [min_count, set_no, i_apk_index, score]
            progress_bar.update(1) # update progress


    # رسم نمودار تکی 
    min_count = 1
    # m, n = 2, 3
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))        

    #  show error bands
    df = jadval[(jadval['Min Count']==min_count)]
    # ax=axes[1, 0]
    sns.lineplot(ax=ax,
                x="Set No", y="AP@k",
                # hue="k",# 
                #  style="Min Count",
                data=df)
    ax.set_title('k='+str(index))
    # ax.set(ylim=(0, 1))

    f = {'AP@k':'mean'} #'Min Count':'first',
    df_mean_score = df.groupby(['Set No'], as_index=False).agg(f)
    # نمودار وسطی قبلی با همین میانگین یکی هست
    # sns.lineplot(ax=axes[1, 1],
    #             x="Set No", y="AP@k",
    #             data=df_mean_score)
    x, y = df_mean_score['AP@k'].idxmax(),df_mean_score['AP@k'].max()
    # ax=axes[1, 0]
    ax.scatter(x, y, marker='o', color='g', s=100)
    f_names = list(gf_df.keys().format())
    best_idx = int(x)
    best_features = [f_names[i] for i in indices[best_idx]]
    features_list = '{:d}th subset,\nwhich contains\n{:d} elements:'.format(best_idx,len(best_features))
    for i,s in enumerate(best_features):
        # if i==0:
        #     features_list = s
        # else:
        features_list += "\n"+s
    ax.text(x, 0, features_list, fontsize=12, color='g') #add text

    x, y = df_mean_score['AP@k'].idxmin(),df_mean_score['AP@k'].min()
    # ax=axes[1, 0]
    ax.scatter(x, y, marker='o', color='r', s=100)
    f_names = list(gf_df.keys().format())
    best_idx = int(x)
    best_features = [f_names[i] for i in indices[best_idx]]
    features_list = '{:d}th subset,\nwhich contains\n{:d} elements:'.format(best_idx,len(best_features))
    for i,s in enumerate(best_features):
        features_list += "\n"+s
    ax.text(x, 0, features_list, fontsize=12, color='r') #add text

    png_file_name = 'results/FS_{}_{}_{}_{}_mc{:d}_k{:d}-{:d}_apk.png'.format(working_file_name[:2], GT_file_name[:2], \
        node_objects, edge_objects, min_count, index[0],index[-1])
    plt.savefig(png_file_name)


    # # رسم نمودار مین‌کانت‌های مختلف رو در اینجا در گوگل کولب دارم که فعلا حذف می کنم
    # df_max_score = jadval.groupby(['Set No', 'Min Count'])['AP@k'].mean().reset_index(name='mAP@k')

    # توجه: در اینجا فقط ویژگی‌های آخرین زیرگراف ذخیره می‌شوند 
    xls_file_name = 'results/FS_{}_{}_{}_{}_mc{:d}-{:d}.xlsx'.format(working_file_name[:2], GT_file_name[:2], \
        node_objects, edge_objects, min_count_list[0], min_count_list[-1])
    writer = ExcelWriter(xls_file_name)
    jadval.to_excel(writer, sheet_name='jadval')  # , index=False)
    # gf_df_sorted.to_excel(writer, sheet_name='gf_df_sorted')  # , index=False)
    # gf_df.to_excel(writer, sheet_name='gf_df')  # , index=False)

    x_max, x_min = df_mean_score['AP@k'].idxmax(), df_mean_score['AP@k'].idxmin()
    SDF_list[x_max].to_excel(writer, sheet_name='gf_df_max_sorted')  # , index=False)
    SDF_list[x_min].to_excel(writer, sheet_name='gf_df_min_sorted')  # , index=False)
    DF_list[x_max].to_excel(writer, sheet_name='gf_df_max')  # , index=False)
    writer.save()

    # چاپ لیست ویژگی‌ها
    # all_features_sets = []
    # for idx in range(len(indices)):
    #     all_features_sets.append([f_names[i] for i in indices[idx]])
    # for i,item in enumerate(all_features_sets):
    #     print(i,item)

    # s1 = gf_fim_df_sorted['degree_cent']
    # s2 = gf_fim_df_sorted['degree_fim']
    # print('Correlation of degree and degree_fim:',s1.corr(s2))
