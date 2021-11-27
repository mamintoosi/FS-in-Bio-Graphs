# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021-2029 by
#    Mahmood Amintoosi <m.amintoosi@gmail.com>
#    All rights reserved.
#    BSD license.
"""Dataset Statistics"""

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
from matplotlib_venn import venn2, venn2_circles


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
    working_file_name = 'AC_Met_Plant.xlsx'
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

    GT_file = working_dir+GT_file_name
    true_df = pd.read_excel(GT_file, engine="openpyxl") 

    # در فرم نود=گیاه، لیست گیاهانی که خاصیت ضد سرطانی آنها اثبات شده است
    working_node_set = set(wdf[node_objects].unique())
    gt_node_set = set(true_df[node_objects].unique())
    venn2([working_node_set ,gt_node_set], set_labels=(working_file_name,args.dataset_name))
    print('Common '+node_objects+'s')
    plt.show()
    
    working_edge_set = set(wdf[edge_objects].unique())
    gt_edge_set = set(true_df[edge_objects].unique())
    venn2([working_edge_set ,gt_edge_set], set_labels=(working_file_name,args.dataset_name))
    print('Common '+edge_objects+'s')
    plt.show()