# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021-2029 by
#    Mahmood Amintoosi <m.amintoosi@gmail.com>
#    All rights reserved.
#    BSD license.
"""Main for feature selection"""

# برنامه انتخاب ویژگی‌هایی از گراف که در انتخاب نودهای ضد سرطان مؤثرتر هستند

## جدول رپ بر اساس مین کانت درست کنم
# Using QR Decomposition

from pandas import ExcelWriter
import pandas as pd
import numpy as np
from itertools import tee
from tqdm.notebook import tqdm
import networkx as nx
from pandas import DataFrame 
pd.options.display.float_format = "{:.2f}".format
import random
from itertools import combinations, chain
import ml_metrics
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
    parser.add_argument('--ds_name', type = str, default = 'stomach')
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
    working_file_name = 'AC_Met_Plant.xlsx'
    # GT_file_name = 'LR_Met_Plant.xlsx'
    if args.ds_name.lower() == 'stomach':
        GT_file_name = 'Stomach AntiCancer Metabolites.xlsx'
    node_objects = 'Plant'
    edge_objects = 'Met'

    working_file = working_dir+working_file_name
    wdf = pd.read_excel(working_file, engine="openpyxl") 

    # for col in df.columns:
    #    df[col] = df[col].apply(lambda x: float(x) if not pd.isna(x) else x)

    # min_max_scaler = preprocessing.MinMaxScaler()

    # در فرم نود=گیاه، گیاهانی که حداقل مین‌کانت بار در دیتابیس اومدن، نگه داشته میشن
    # به عبارت دیگه حداقل مین‌کانت متابولیت داشتن
    # نگهداری ستونهای با بیش از یا مساوی با ۵ عنصر
    # min_count = 2
    min_count_list = np.arange(1,8).tolist()

    minCount = min(min_count_list)

    GT_file = working_dir+GT_file_name
    true_df = pd.read_excel(GT_file, engine="openpyxl") 

    # در فرم نود=گیاه، لیست گیاهانی که خاصیت ضد سرطانی آنها اثبات شده است
    true_list = list(true_df[node_objects].unique())
    # true_list = list(true_df.keys().values)
    index = np.arange(1,10)

    # jadval = pd.DataFrame(
    #     {'k': index,
    #      'Min Count': '',
    #      'AP@k': 0 #apk_gf
    #     #  'mAR@k': ark_gf
    #     })
    # jadval = pd.DataFrame(columns=['k', 'Min Count', 'Best Features Numbers', 'AP@k'])
    # jadval = pd.DataFrame(columns=['k', 'Set No', 'Min Count', 'AP@k'])
    jadval = pd.DataFrame(columns=['Min Count', 'Set No', 'k', 'AP@k'])

    # jadval = pd.DataFrame(columns=['Min Count', 'Best Features Numbers', 'mAP@k'])

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
            gf_df, gf_df_sorted = rank_using_graph_features(subG) #, min_count, node_objects, \
                # edge_objects, data_dir, output_dir, working_file_name)

            n_features = gf_df.shape[1]-1
            indices = list(powerset(range(n_features)))
            # remove the empty subset
            indices.pop(0)

            # print(indices)
            APK = []
            scores = []
            for set_no, idx in enumerate(indices):
                # print()
                # cols = list(idx)
                cols_s = list(idx) # columns with feature_sum
                cols_s.append(-1)
                selected_df = gf_df.iloc[:, cols_s]
                # print(cols,cols_s,selected_df.shape)#,cols_s)
                n_cols = selected_df.shape[1]-1
                features_sum = selected_df.iloc[:,:n_cols].sum(axis=1)
                selected_df['features_sum'] = features_sum
                selected_df_sorted = selected_df.sort_values(by='features_sum', ascending=False)
                pred_list = list(selected_df_sorted.index.values)
                # print(obj_fun(true_list,pred_list))

                for i_apk_index in index:
                    apk_index = np.arange(1,i_apk_index+1) # +۱ باید باشه وگرنه اولیش تهی هست
                    score = obj_fun(true_list, pred_list, apk_index)
                    scores.append(score)
                    jadval_row_no += 1
                    jadval.loc[jadval_row_no] = [min_count, set_no, i_apk_index, score]
                    # [apk, _] = compute_metrics(true_list, pred_list, apk_ranges=index)
                    # APK.append(apk)
                    # tmp_df = pd.DataFrame(
                    #     {'k': i_apk_index,
                    #     'Set No': set_no,
                    #     'Min Count': min_count,
                    #     # 'Best Features Numbers': num_best_features,
                    #     'AP@k': score,#best_apk #apk_gf
                    #     #  'mAR@k': ark_gf
                    #     })
                    # jadval = jadval.append(tmp_df, ignore_index=True)


                # best_score = np.max(np.array(scores))
                # best_idx = np.argmax(np.array(scores))
                # # best_idx,scores[best_idx],indices[best_idx]
                # f_names = list(gf_df.keys().format())
                # best_features = [f_names[i] for i in indices[best_idx]]
                # print('Best Features are: ', best_features)
                # best_apk = APK[best_idx]
                # num_best_features = len(indices[best_idx])

                # jadval.loc[row_no] = [min_count, num_best_features, best_score]

                # tmp_df = pd.DataFrame(
                #     {'k': index,
                #     'Min Count': min_count,
                #     'Best Features Numbers': num_best_features,
                #     'AP@k': best_apk #apk_gf
                #     #  'mAR@k': ark_gf
                #     })
                # jadval = jadval.append(tmp_df, ignore_index=True)
            
            progress_bar.update(1) # update progress


    # رسم نمودار تکی 
    min_count = 1
    m, n = 2, 3
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('AP@k for all subsets of features, Min Count = '+str(min_count))
    for i in np.arange(m):
        for j in np.arange(n):
            ax = axes[i,j]
            ax.set(ylim=(-.1, 1.1))

    index_subset = [1,5,9]
    for x, k in enumerate(index_subset):
        df = jadval[(jadval['k']==k) & (jadval['Min Count']==min_count)]
        # print(df)
        # plt.figure()
        i, j = (x)//3, (x)%3
        ax=axes[i, j]
        sns.lineplot(ax=ax,
                    x="Set No", y="AP@k",
                    #  hue="k",# 
                    #  style="Min Count",
                    data=df)
        ax.set_title('k='+str(k))
        

    #  show error bands
    df = jadval[(jadval['Min Count']==min_count)]
    ax=axes[1, 0]
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
    ax.scatter(x, y, marker='o', color='r', s=100)
    f_names = list(gf_df.keys().format())
    best_idx = int(x)
    best_features = [f_names[i] for i in indices[best_idx]]
    features_list = '{:d}th subset,\nwhich contains\n{:d} elements:'.format(best_idx,len(best_features))
    for i,s in enumerate(best_features):
        # if i==0:
        #     features_list = s
        # else:
        features_list += "\n"+s
    ax.text(40, 0,features_list  , fontsize=12) #add text

    #  show error bars and plot the 68% confidence interval (standard error):
    df = jadval[(jadval['Min Count']==min_count)]
    sns.lineplot(ax=axes[1, 1],
                x="Set No", y="AP@k",
                data=df, err_style="bars", ci=68)
    ax=axes[1, 1]
    # ax.set(ylim=(0, 1))
    ax.scatter(x, y, marker='o', color='r', s=100)
    ax.set_title('k='+str(index))
    ax.text(40, 0,features_list  , fontsize=12) #add text
    sns.lineplot(ax=axes[1, 1],
                x="Set No", y="AP@k",
                data=df_mean_score)

    png_file_name = 'results/FS_{}_{}_{}_{}_mc{:d}_k{:d}-{:d}_apk.jpg'.format(working_file_name[:2], GT_file_name[:2], \
        node_objects, edge_objects, min_count, index[0],index[-1])
    plt.savefig(png_file_name)


    # رسم نمودار مین‌کانت‌های مختلف
    m, n = 2, 3
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('AP@k for all subsets of features, Min Count = '+str(min_count))
    for i in np.arange(m):
        for j in np.arange(n):
            ax = axes[i,j]
            ax.set(ylim=(-.1, 1.1))

    min_count_list = [1,2,3,4,5,6]
    for x, min_count in enumerate(min_count_list):
        i, j = (x)//3, (x)%3
        ax=axes[i, j]
        df = jadval[(jadval['Min Count']==min_count)]
        ax=axes[i, j]
        sns.lineplot(ax=ax,
                    x="Set No", y="AP@k",
                    data=df)
        ax.set_title('k='+str(index))
        

        f = {'AP@k':'mean'} #'Min Count':'first',
        df_mean_score = df.groupby(['Set No'], as_index=False).agg(f)
        x, y = df_mean_score['AP@k'].idxmax(),df_mean_score['AP@k'].max()
        ax.scatter(x, y, marker='o', color='r', s=100)
        f_names = list(gf_df.keys().format())
        best_idx = int(x)
        best_features = [f_names[i] for i in indices[best_idx]]
        features_list = '{:d}th subset,\nwhich contains\n{:d} elements:'.format(best_idx,len(best_features))
        for i,s in enumerate(best_features):
            features_list += "\n"+s
        ax.text(40, 0,features_list  , fontsize=12) #add text

    png_file_name = 'results/FS_{}_{}_{}_{}_mc{:d}-{:d}_k{:d}-{:d}_apk.jpg'.format(working_file_name[:2], GT_file_name[:2], \
        node_objects, edge_objects, min_count_list[0], min_count_list[-1], index[0],index[-1])
    plt.savefig(png_file_name)

    # print(jadval)
    # plt.figure()
    # sns.lineplot(x="k", y="AP@k",
    #             #  hue="Method Name",# style="min_count",
    #              data=jadval)
    # png_file_name = 'results/FS_{}_{}_{}_{}_mc{:d}-{:d}_apk.png'.format(working_file_name[:2], GT_file_name[:2], \
    #     node_objects, edge_objects, min_count_list[0], min_count_list[-1])
    # plt.savefig(png_file_name)

    df_max_score = jadval.groupby(['Set No', 'Min Count'])['AP@k'].mean().reset_index(name='mAP@k')
    df_max_score
    plt.figure()
    sns.lineplot(x="Set No", y="mAP@k",
                hue="Min Count",# style="min_count",
                data=df_max_score) 


    # for min_count in min_count_list:
    #     tmp_df = jadval[jadval['Min Count']==min_count]
    #     if min_count == min_count_list[0]:
    #         df_max_score = tmp_df.groupby(['Set No'])['AP@k'].mean().reset_index(name='mAP@k')
    #     else:
    #         df_max_score.append(tmp_df.groupby(['Set No'])['AP@k'].mean().reset_index(name='mAP@k'),\
    #             , ignore_index=True)

    # plt.figure()
    # sns.lineplot(x="Set No", y="mAP@k",
    #             #  hue="Method Name",# style="min_count",
    #              data=df_max_score)





    # plt.figure()
    # sns.lineplot(x="k", y="AR@k",
    #              hue="Method Name",# style="min_count",
    #              data=jadval)
    # png_file_name = 'results/FS_{}_{}_{}_{}_mc{:d}-{:d}_ark.png'.format(working_file_name[:2], GT_file_name[:2], \
    #     node_objects, edge_objects, min_count_list[0], min_count_list[-1])
    # plt.savefig(png_file_name)

    # توجه: در اینجا فقط ویژگی‌های آخرین زیرگراف ذخیره می‌شوند 
    xls_file_name = 'results/FS_{}_{}_{}_{}_mc{:d}-{:d}.xlsx'.format(working_file_name[:2], GT_file_name[:2], \
        node_objects, edge_objects, min_count_list[0], min_count_list[-1])
    writer = ExcelWriter(xls_file_name)
    # gf_df_sorted.to_excel(writer, sheet_name='gf_df_sorted')  # , index=False)
    # gf_fim_df_sorted.to_excel(writer, sheet_name='gf_fim_df_sorted')  # , index=False)
    jadval.to_excel(writer, sheet_name='jadval')  # , index=False)
    gf_df.to_excel(writer, sheet_name='gf_df')  # , index=False)
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
