import sys, os
import pandas as pd
import numpy as np
import itertools
import networkx as nx
# embeddings
#from sklearn import preprocessing
#from sklearn.decomposition import PCA
#import umap

# own code
import preprocessing_utils as utils
import gcnPreprocessing
import gcnIO

use_quantile_norm = False # quantile or MinMax normalization
ppi_network_to_use ='nor_feature_30'
remove_blood_cancer_genes = False
cna_as_separate_omics = True
use_mutsigcv_scores = False
use_cnas = True
label_source = 'NCG'
minimum_degree_negatives = 2
patient_wise = False
all_omics_normed = []
# Load PPI network
ppi_network = pd.read_csv('F:\\KICH数据\\sadppi.tsv', sep='\t')
ppi_graph = nx.from_pandas_edgelist(df=ppi_network, source='partner1', target='partner2', edge_attr="confidence")
ppi_network = nx.to_pandas_adjacency(G=ppi_graph)
#print(ppi_network)
nodes = utils.get_ensembl_from_symbol(ppi_network.index)
nodes.columns = ['Name']
#nodes.to_csv('F:\\KICH数据\\node.csv')
nodes=nodes['Name'].str.split('_',expand=True)
#nodes['ID'] = nodes.index
nodes.columns= ['ID', 'Name']
#nodes.to_csv('D:\\EMOGI\\EMOGI\\EMOGI-master\\EMOGI-master\\node.csv')
#print(nodes)

known_cancer_genes_innet = utils.get_positive_labels(nodes,strategy=label_source)
negatives = utils.get_negative_labels(nodes, known_cancer_genes_innet, ppi_network, min_degree=minimum_degree_negatives, verbose=True)
#print(known_cancer_genes_innet)

## Do Training and Test Split
y = nodes.Name.isin(known_cancer_genes_innet).values.reshape(-1, 1)
mask = nodes.Name.isin(negatives.Name) | nodes.Name.isin(known_cancer_genes_innet)
#print(mask)
y_train, train_mask, y_test, test_mask = gcnPreprocessing.train_test_split(y, mask, 0.2)
y_train, train_mask, y_val, val_mask = gcnPreprocessing.train_test_split(y_train, train_mask, 0.1)
y_train.sum(), train_mask.sum(), y_test.sum(), test_mask.sum(), y_val.sum(), val_mask.sum()
## Write back everything to container


#feat_names_all=['MF: KIRC','MF: BRCA','MF: READ','MF: PRAD','MF: STAD','MF: HNSC','MF: LUAD','MF: THCA','MF: BLCA','MF: ESCA','MF: LIHC','MF: UCEC','MF: COAD','MF: LUSC','MF: CESC','MF: KIRP','METH: KIRC','METH: BRCA','METH: READ','METH: PRAD','METH: STAD','METH: HNSC','METH: LUAD','METH: THCA','METH: BLCA','METH: ESCA','METH: LIHC','METH: UCEC','METH: COAD','METH: LUSC','METH: CESC','METH: KIRP','GE: KIRC','GE: BRCA','GE: READ','GE: PRAD','GE: STAD','GE: HNSC','GE: LUAD','GE: THCA','GE: BLCA','GE: ESCA','GE: LIHC','GE: UCEC','GE: COAD','GE: LUSC','GE: CESC','GE: KIRP','CNA: KIRC','CNA: BRCA','CNA: READ','CNA: PRAD','CNA: STAD','CNA: HNSC','CNA: LUAD','CNA: THCA','CNA: BLCA','CNA: ESCA','CNA: LIHC','CNA: UCEC','CNA: COAD','CNA: LUSC','CNA: CESC','CNA: KIRP']
#feat_names_all=['MF: KIRC','MF: BRCA','MF: READ','MF: PRAD','MF: STAD','MF: HNSC','MF: LUAD','MF: THCA','MF: BLCA','MF: ESCA','MF: LIHC','MF: UCEC','MF: COAD','MF: LUSC','MF: CESC','MF: KIRP']
#feat_names_all=['muta','expr','copy','methy','length_RefSeq','domains_InterPro','duplicability','ohnolog','essentiality_percentage','expressed_tissues_rnaseq','expressed_tissues_protein','ppin_degree','ppin_betweenness','ppin_clustering','complexes','mirna','multiDomain','ppin_hub','ppin_central','old','metazoans','vertebrates','mammalsPrimates','essentiality_oneCellLine','rnaseq_never','rnaseq_selective','rnaseq_middle','rnaseq_ubiquitous','protein_low','protein_high']
feat_names_all=['muta','expr','copy','methy','length_RefSeq','duplicability','ohnolog','essentiality_percentage','complexes','mirna','multiDomain','old','metazoans','vertebrates','mammalsPrimates','essentiality_oneCellLine','rnaseq_never','rnaseq_selective','rnaseq_middle','rnaseq_ubiquitous','protein_low','protein_high','node1','node2','node3','node4','node5','node6','node7','node8','node9','node10','node11','node12','node13','node14','node15','node16','node17','node18','node19','node20','node21','node22','node23','node24','node25','node26','node27','node28','node29','node30']
#feat_names_all=['muta','expr','copy','methy','length_RefSeq','duplicability','ohnolog','essentiality_percentage','complexes','mirna','multiDomain','old','metazoans','vertebrates','mammalsPrimates','essentiality_oneCellLine','rnaseq_never','rnaseq_selective','rnaseq_middle','rnaseq_ubiquitous','protein_low','protein_high','node1','node2','node3','node4','node5','node6','node7','node8','node9','node10']
multi_features= pd.read_csv("F:\\KICH数据\\samples\\nor_kich_feature_node30.csv")
data= multi_features.iloc[:,1:]
multi_features = np.array(data)
#print(multi_features)

fname = "F:\\KICH数据\\{}.h5".format(ppi_network_to_use.upper())
utils.write_hdf5_container(fname, ppi_network.values, multi_features, nodes, y_train,y_val, y_test, train_mask, val_mask, test_mask,feat_names_all)
