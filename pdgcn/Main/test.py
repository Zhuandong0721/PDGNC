import pandas as pd
import  numpy as np
import networkx as nx
from sklearn import preprocessing

import preprocessing_utils as utils
import gcnPreprocessing
import gcnIO
import sys, os, h5py
''''
all_omics_normed = []
multi_features= pd.read_csv("multiomics_features.tsv")
feat_names_all=['MF: KIRC','MF: BRCA','MF: READ','MF: PRAD','MF: STAD','MF: HNSC','MF: LUAD','MF: THCA','MF: BLCA','MF: ESCA','MF: LIHC','MF: UCEC','MF: COAD','MF: LUSC','MF: CESC','MF: KIRP','METH: KIRC','METH: BRCA','METH: READ','METH: PRAD','METH: STAD','METH: HNSC','METH: LUAD','METH: THCA','METH: BLCA','METH: ESCA','METH: LIHC','METH: UCEC','METH: COAD','METH: LUSC','METH: CESC','METH: KIRP','GE: KIRC','GE: BRCA','GE: READ','GE: PRAD','GE: STAD','GE: HNSC','GE: LUAD','GE: THCA','GE: BLCA','GE: ESCA','GE: LIHC','GE: UCEC','GE: COAD','GE: LUSC','GE: CESC','GE: KIRP','CNA: KIRC','CNA: BRCA','CNA: READ','CNA: PRAD','CNA: STAD','CNA: HNSC','CNA: LUAD','CNA: THCA','CNA: BLCA','CNA: ESCA','CNA: LIHC','CNA: UCEC','CNA: COAD','CNA: LUSC','CNA: CESC','CNA: KIRP']
features_df = pd.DataFrame(multi_features, index=multi_features.iloc[:, 0],columns=feat_names_all)

names = features_df.index.values
features_df.set_index(names, inplace=True)
features_df.isnull().all(axis=1).sum()
print(features_df.head())
'''
#multi_features.columns = ['symbol']  + [i.upper() for i in multi_features.columns[1:]]
##multi_features.set_index('symbol', inplace=True)
#print(multi_features.head())

#scaler = preprocessing.MinMaxScaler()
#features = scaler.fit_transform(multi_features)
#all_omics_normed.append(multi_features)
#multi_omics_features = np.array(all_omics_normed)
#multi_omics_features = np.transpose(multi_omics_features, (1, 2, 0))
#np.save('D:\\EMOGI\\EMOGI-master\\EMOGI-master\\data\\pancancer\\features.npy', multi_omics_features)
data = gcnIO.load_hdf_data('../data/pancancer/STRINGdb_multiomics.h5')
network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data
features_df = pd.DataFrame(features, index=node_names[:, 1], columns=feat_names)

y_train.shape, y_test.shape, train_mask.shape, test_mask.shape

## Get the Simulated Network

#toy_network = nx.read_edgelist('D:\\EMOGI\\EMOGI-master\\EMOGI-master\\example\\network.edgelist')

## Get Features & Labels

# select features from cancer and non-cancer genes
#features_cancergenes = features_df[y_train]
#features_noncancergenes = features_df[np.logical_and(train_mask, np.logical_not(y_train.flatten()))]
#print(features_cancergenes)
#print(features_noncancergenes)
# start with the negatives and then update the DF
#features_toynetwork = features_noncancergenes.sample(n=toy_network.number_of_nodes())
#pos_features = features_cancergenes.sample(n=len(clique_nodes))
# workaround required to fool pandas indexing (otherwise it sets NaN values because indices don't match)
#names = features_toynetwork.index.values
#names[clique_nodes] = pos_features.index.values
#features_toynetwork.set_index(names, inplace=True)
#features_toynetwork.iloc[clique_nodes] = pos_features

# Load PPI network
ppi_network = pd.read_csv('string_SYMBOLS_highconf.tsv', sep='\t')
ppi_graph = nx.from_pandas_edgelist(df=ppi_network, source='partner1', target='partner2', edge_attr="confidence")
ppi_network = nx.to_pandas_adjacency(G=ppi_graph)

nodes = utils.get_ensembl_from_symbol(ppi_network.index)
nodes.columns = ['Name']

known_cancer_genes_innet = utils.get_positive_labels(nodes,strategy='NCG')
negatives = utils.get_negative_labels(nodes, known_cancer_genes_innet, ppi_network, min_degree=1, verbose=True)

## Do Training and Test Split
y = nodes.Name.isin(known_cancer_genes_innet).values.reshape(-1, 1)
mask = nodes.Name.isin(negatives.Name) | nodes.Name.isin(known_cancer_genes_innet)

y_train, train_mask, y_test, test_mask = gcnPreprocessing.train_test_split(y, mask, 0.25)
y_train, train_mask, y_val, val_mask = gcnPreprocessing.train_test_split(y_train, train_mask, 0.1)
y_train.sum(), train_mask.sum(), y_test.sum(), test_mask.sum(), y_val.sum(), val_mask.sum()

feat_names_all=['MF: KIRC','MF: BRCA','MF: READ','MF: PRAD','MF: STAD','MF: HNSC','MF: LUAD','MF: THCA','MF: BLCA','MF: ESCA','MF: LIHC','MF: UCEC','MF: COAD','MF: LUSC','MF: CESC','MF: KIRP','METH: KIRC','METH: BRCA','METH: READ','METH: PRAD','METH: STAD','METH: HNSC','METH: LUAD','METH: THCA','METH: BLCA','METH: ESCA','METH: LIHC','METH: UCEC','METH: COAD','METH: LUSC','METH: CESC','METH: KIRP','GE: KIRC','GE: BRCA','GE: READ','GE: PRAD','GE: STAD','GE: HNSC','GE: LUAD','GE: THCA','GE: BLCA','GE: ESCA','GE: LIHC','GE: UCEC','GE: COAD','GE: LUSC','GE: CESC','GE: KIRP','CNA: KIRC','CNA: BRCA','CNA: READ','CNA: PRAD','CNA: STAD','CNA: HNSC','CNA: LUAD','CNA: THCA','CNA: BLCA','CNA: ESCA','CNA: LIHC','CNA: UCEC','CNA: COAD','CNA: LUSC','CNA: CESC','CNA: KIRP']

features_df = pd.DataFrame(features_df, index=features_df.iloc[:, 0],columns=feat_names_all)
names = features_df.index.values
features_df.set_index(names, inplace=True)
features_df.isnull().all(axis=1).sum()
## Write to HDF5 Container

#%%

f = h5py.File('toy_example.h5', 'w')
string_dt = h5py.special_dtype(vlen=str)
#names = features_toynetwork.index.values.reshape(-1, 1)
#node_names_toy = np.hstack([names, names])
#toy_adj = nx.to_numpy_array(toy_network)
f.create_dataset('network', data=ppi_network.values, shape=ppi_network.values.shape)
f.create_dataset('features', data=features_df, shape=features_df.shape)
f.create_dataset('gene_names', data=nodes.astype('object'), dtype=string_dt)
f.create_dataset('y_train', data=y_train, shape=y_train.shape)
f.create_dataset('y_val', data=y_val, shape=y_val.shape)
f.create_dataset('y_test', data=y_test, shape=y_test.shape)
f.create_dataset('mask_train', data=train_mask, shape=train_mask.shape)
f.create_dataset('mask_val', data=val_mask, shape=val_mask.shape)
f.create_dataset('mask_test', data=test_mask, shape=test_mask.shape)
f.create_dataset('feature_names', data=np.array(feat_names_all, dtype=object), dtype=string_dt)
#f.create_dataset('features_raw', data=features_toynetwork, shape=features_toynetwork.shape)
f.close()
