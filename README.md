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
Cancer is one of the most important health problems around the world.  Recently, the identifying and introducing of effective metabolites of plants in the prevention and treatment of cancer has achieved much more attention. However, there is no theoretical method to determine the efficiency of a plant or metabolite in the treatment of cancer without the need of laboratory studies. One of the graphs applications in life sciences is to show the connectivity between different elements to discover their relationships. In this study, graph theory and network analysis approaches are employed to introduce a list of anti-cancer plants and metabolites. First, two separate lists of plants and metabolites whose anti-cancer properties have been approved are prepared. Then, the metabolites in anti-cancer plants and plants containing anti-cancer metabolites are analyzed in the form of networks of plants/metabolites. Herein, in addition to the introduction of the best and most effective new anti-breast cancer plants and metabolites, a new method -inspired form recommender systems- is proposed to rank the anti-cancer properties of plants based on their metabolites. A similar approach is used for metabolites’ recommendation.</p>

This repository provides a Python implementation of FS-in-Bio-Graphs as described in the draft paper:

> Feature Selection in Bio-Graphs,
> Mahmood Amintoosi, Eisa Kohan
> 2021

### Requirements
The codebase is implemented in Python 3.7.11 on Google colab. package versions used for development are just below.
```
torch-scatter 		2.0.8
torch-sparse		0.6.11
torch-geometric		1.7.2
texttable		1.6.4
karateclub		1.2.1
```

### Run on Google Colab
https://colab.research.google.com/github/mamintoosi/FS-in-Bio-Graphs/blob/master/FS_in_Bio_Graphs.ipynb

### Datasets
<p align="justify">
All of the datasets used here are accessible from <em>data</em> folder.
</p>


[![Github All Releases](https://img.shields.io/github/downloads/mamintoosi/FS-in-Bio-Graphs/total.svg)]()
