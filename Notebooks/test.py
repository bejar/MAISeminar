"""
.. module:: test

test
******

:Description: test

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  06/05/2016
"""

__author__ = 'bejar'

from Code.STData import STData
from Code.Constants import homepath, cityparams
from Code.Clustering import cluster_events, cluster_cache
from Code.Transactions import DailyDiscretizedTransactions, DailyClusteredTransactions
from Code.TimeDiscretizer import TimeDiscretizer
import folium

data = STData('../', cityparams['bcn'], 'instagram')

data.read_data()
data.info()
data = data.select_heavy_hitters(9900, 10000)
data.info()

cluster = cluster_cache(data, alg= 'kmeans', radius=0.005, nclusters=50)
if cluster is None:
    print 'Computing Clustering'
    cluster, _ = cluster_events(data, alg= 'kmeans', radius=0.005, nclusters=50)


timedis = [6, 18] # Time discretization
trans = DailyClusteredTransactions(data, cluster=cluster, timeres=TimeDiscretizer(timedis))
trans.info()

# Minimum number of events
minloc = 5
# Attribute types 'bin'=[0,1] ; 'binidf'=[0,1]/IDF
mode = 'bin'
datamat, users = trans.generate_data_matrix(minloc=minloc, mode=mode)

from Code.Clustering import cluster_colapsed_events

# Clustering Algorithms 'kmeans', 'spectral', 'affinity'
calg = 'kmeans'
# affinity damping parmeter 0.1 - 1
damping=0.5
# number of clusters for kmeans and spectral clustering
nclust = 5
# Minimum number of elements in a cluster

cls = cluster_colapsed_events(datamat, users, alg=calg, damping=damping, nclust=nclust, minsize=1)

print [(c, len(cls[c])) for c in cls]

cluster_name = 'c0'
print cls[cluster_name]

dataclus = data.select_data_users(cls[cluster_name],pr=True)
dataclus.info()
mymap = dataclus.plot_events_cluster(cluster=cluster, dataname=cluster_name)