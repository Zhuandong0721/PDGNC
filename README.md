# Personalized driver gene prediction using graph convolutional networks with conditional random fields

A Graph Convolutional Neural Network Method for Predicting Driver Genes on Individuals

![figure](https://github.com/night-changes/PDGNC/assets/51049985/b7a43b46-464a-4b3a-bc8d-51a3f14e1e35)

## Prerequisites

The code is written in Python 3 and was mainly tested on Python 3.6 and a Linux OS but should run on any OS that supports python and pip. Training is faster on a GPU (which requires the tensorflow-gpu instead of the normal tensorflow package) but works also on a standard computer.

PDGCN has the following dependencies:
* gcn
* Numpy
* Pandas
* Tensorflow
* h5py
* Networkx
* mygene


## Training PDGCN with Your Own Data

To train PDGCN with your own data, you have to provide a HDF5 container containing the graph, features and labels.  In general, a valid container for PDGCN has to contain a graph ( a numpy matrix of shape N x N), a feature matrix for the nodes ( shape of N x p), the gene names and IDs ( a numpy array), the training set ,the test set of the same shape, training and test masks  and the names of the features (an array of length p).

After you obtained a valid HDF5 container, you can simply train EMOGI with
```
python train_cv.py -d <path-to-hdf5-container> -hd <n_filters_layer1> <n_filters_layer2>
```

### Additional datasets

[Network of Cancer Genes (NCG) version 6.0] (http://ncg.kcl.ac.uk/download_file.php?file=cancergenes_list.txt)

[COSMIC Cancer Gene Census (CGC)] (https://cancer.sanger.ac.uk/cosmic/download)

[STRING Datebase] (https://cn.string-db.org/)

