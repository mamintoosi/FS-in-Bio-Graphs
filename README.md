Feature Selection in Bio Graphs
==========
 [![repo size](https://img.shields.io/github/repo-size/mamintoosi/FS-in-Bio-Graphs.svg)](https://github.com/mamintoosi/FS-in-Bio-Graphs/archive/master.zip)
 [![GitHub forks](https://img.shields.io/github/forks/mamintoosi/FS-in-Bio-Graphs)](https://github.com/mamintoosi/FS-in-Bio-Graphs/network)
[![GitHub issues](https://img.shields.io/github/issues/mamintoosi/FS-in-Bio-Graphs)](https://github.com/mamintoosi/FS-in-Bio-Graphs/issues)
[![GitHub license](https://img.shields.io/github/license/mamintoosi/FS-in-Bio-Graphs)](https://github.com/mamintoosi/FS-in-Bio-Graphs/blob/main/LICENSE)
 
 
A Python implementation of "Feature Selection in Bio Graphs" 
<p align="center">
  <img width="600" src="doc/header.png">
</p>

### Abstract

<p align="justify">
Motivation: Every year tremendous experimental analysis has been done for evaluation of anti-cancer properties of plants. A good ranked list of potential anti-cancer plants which rised out of veri-fied anti-cancer metabolites, reduces the time and cost for evaluating plants; otherwise, we charged for testing unrelated plants. Ranked list produced by analyzing plant-metabolite biological graphs are candidate for such situation. Graph nodes are ranked according to some graph features. A prob-lem with this approach is how to select the good features of graphs. In this paper a metric used in information retrieval and recommender systems was employed for comparing two different ranked list. In an information retrieval system such as search engines, a good system should show the top results first. A metric named Average Precision (AP) is used here for discriminating different lists, resulted from different features. The AP is a measure that takes in a ranked list of N recommenda-tions and compares it to a list of the true set of "correct" or "relevant" items. AP rewards for having a lot of "correct" (relevant) recommendations in the list, and rewards for putting the most likely cor-rect recommendations at the top. We build a network of similarity of plants according to their com-mon metabolites. After that, with various combinations of the graph features, the plants are ranked. The subset of features which produces the ranked list with higher AP score is considered as the best features for anti-cancer plant recommendation.
Results:   The proposed method could be employed to select the best graph features in screening of anti-cancer plants from an unverified plants list. So that, the plant with higher score in the list have higher chance to have anti-cancer properties. Although, different graph features are applicable for anti-cancer plant recommendation in each cancer type.
</p>

This repository provides a Python implementation of FS-in-Bio-Graphs as described in the draft paper:

> Feature Selection in Bio-Graphs,
> Mahmood Amintoosi, Eisa Kohan
> 2022


### Requirements
The codebase is implemented in Python 3.7.11 on Google colab. 

### Run on Google Colab
https://colab.research.google.com/github/mamintoosi/FS-in-Bio-Graphs/blob/master/FS_in_Bio_Graphs.ipynb

### Datasets
<p align="justify">
All of the datasets used here are accessible from <em>data</em> folder.
</p>

### Results
The best graph features for Breast anti-cancer plant recommendation
![Stomach](results%5CFS_AC_St_Plant_Met_mc1_k1-9_apk.png)

The best graph features for Stomach anti-cancer plant recommendation
![Breast](results%5CFS_AC_Br_Plant_Met_mc1_k1-9_apk.png)

[![Github All Releases](https://img.shields.io/github/downloads/mamintoosi/FS-in-Bio-Graphs/total.svg)]()
