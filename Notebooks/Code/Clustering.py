"""
.. module:: Clustering

Clustering
*************

:Description: Clustering

    Generate clusterings from  user transactions

:Authors: bejar
    

:Version: 0.1

:Created on: 24/02/2014 9:27 

"""

__author__ = 'bejar'

import time
import os.path
import pickle

import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN, SpectralClustering
from numpy import savetxt
import folium
from sklearn.metrics import silhouette_score
from kemlglearn.cluster.Leader import Leader

from Constants import homepath


circlesize = 15000


def cluster_colapsed_events(data, users, minloc=20, nclust=10, mode='nf', alg='affinity', damping=None, minsize=0):
    """
     Generates a clustering of the users by colapsing the transactions of the user events
     the users have to have at least minloc different locations in their transactions

     :arg   trans: Transaction object
     :arg minloc: Minimum number of locations
     :arg nclust: Number of clusters, for clustering algorithms that need this parameter
     :arg mode:
      * nf = location normalized frequency frequency for the user
      * af = location absolute frequency for the user
      * bin = presence/non presence of the location for the user
      * adding idf used the inverse document frequency

    """
    # Generates a sparse matrix for the transactions and a list of users
    # data, users = trans.generate_data_matrix(minloc=minloc, mode=mode)

    print "Clustering Transactions ... ", alg

    if alg == 'affinity':
        ap = AffinityPropagation(damping=damping)
        ap.fit(data)

        ap_labels = ap.labels_
        ap_labels_unique = len(np.unique(ap_labels))
        cclass = np.zeros(ap_labels_unique)

        clusters = {}
        for v in ap_labels:
            cclass[v] += 1

        for v in range(cclass.shape[0]):
            if cclass[v] > minsize:
                clusters['c' + str(v)] = []

        for v, u in zip(ap_labels, users):
            if cclass[v] > minsize:
                clusters['c' + str(v)].append(u)
        print len(clusters)

        for c in clusters:
            print c, len(clusters[c])
    elif alg == 'kmeans':
        k_means = KMeans(init='k-means++', n_clusters=nclust, n_init=10, n_jobs=-1)
        k_means.fit(data)
        k_means_labels = k_means.labels_
        #k_means_cluster_centers = k_means.cluster_centers_
        k_means_labels_unique = len(np.unique(k_means_labels))
        cclass = np.zeros(k_means_labels_unique)
        clusters = {}

        for v in k_means_labels:
            cclass[v] += 1
        for v in range(cclass.shape[0]):
            if cclass[v] > minsize:
                clusters['c' + str(v)] = []

        for v, u in zip(k_means_labels, users):
            if cclass[v] > minsize:
                clusters['c' + str(v)].append(u)

        print len(clusters)
        for c in clusters:
            print c, len(clusters[c])
    elif alg == 'spectral':
        spectral = SpectralClustering(n_clusters=nclust,
                                      assign_labels='discretize', affinity='nearest_neighbors')
        spectral.fit(data)
        spectral_labels = spectral.labels_
        spectral_labels_unique = len(np.unique(spectral_labels))
        cclass = np.zeros(spectral_labels_unique)
        clusters = {}

        for v in spectral_labels:
            cclass[v] += 1
        for v in range(cclass.shape[0]):
            if cclass[v] > minsize:
                clusters['c' + str(v)] = []

        for v, u in zip(spectral_labels, users):
            if cclass[v] > minsize:
                clusters['c' + str(v)].append(u)

                # print len(clusters)
                # for c in clusters:
                #     print c, len(clusters[c])

    return clusters


def cluster_colapsed_events_simple(trans, minloc=20, nclust=10, mode='nf', alg='affinity', damping=None):
    """
     Generates a clustering of the users by colapsing the transactions of the user events
     the users have to have at least minloc different locations in their transactions
     Returns the clustering object

     :arg   trans: Transaction object
     :arg minloc: Minimum number of locations
     :arg nclust: Number of clusters, for clustering algorithms that need this parameter
     :arg mode:
      * nf = location normalized frequency frequency for the user
      * af = location absolute frequency for the user
      * bin = presence/non presence of the location for the user
      * adding idf used the inverse document frequency

    """
    # Generates a sparse matrix for the transactions and a list of users
    data, users = trans.generate_data_matrix(minloc=minloc, mode=mode)

    print "Clustering Transactions ..."

    if alg == 'affinity':
        ap = AffinityPropagation(damping=damping)
        ap.fit(data)
        return []
    elif alg == 'kmeans':
        lvals = []
        ic, fc = nclust
        for i in range(ic, fc):
            k_means = KMeans(init='k-means++', n_clusters=i, n_init=10, n_jobs=-1)
            k_means.fit(data)
            labels = k_means.labels_
            ssc = silhouette_score(data, labels, metric='euclidean')
            lvals.append((i, ssc))
            print i, ssc
        return lvals


def cluster_cache(data, mxhh=0, mnhh=0, radius=0.01, mins=100, size=100, alg='Leader', lhours=None):
    if alg == 'Leader':
        nfile = homepath + 'Clusters/' + data.city[2] + data.get_app_name() + '-' + 'nusr' + str(mxhh) + '+' + str(mnhh) \
                + '-' + 'Leader-crd' + str(radius) + '-mex' + str(size)
    elif alg == 'DBSCAN':
        nfile = homepath + 'Clusters/' + data.city[2] + data.get_app_name() + '-' + 'nusr' + str(mxhh) + '+' + str(mnhh) \
                + '-' + 'DBSCAN-crd' + str(radius) + '-mins' + str(mins) + '-mex' + str(size)
    if lhours is not None:
        nfile += '-hrs' + str(lhours)
    if os.path.isfile(nfile + '.pkl'):
        pfile = open(nfile + '.pkl', 'r')
        return pickle.load(pfile)
    else:
        return None


def cluster_events(data, radius=0.01, mins=100, size=100, alg='Leader', sizeprop=0):
    """
    Cluster geographical events and returns the clusters

    @param nclust:
    @return:
    """
    print "Clustering ..."

    coord = data.getDataCoordinates()

    if alg == 'Leader':
        dbs = Leader(radius=radius)
    elif alg == 'DBSCAN':
        dbs = DBSCAN(eps=radius, min_samples=mins, algorithm='kd_tree')
    else:
        dbs = None

    dbs.fit(coord)

    if alg == 'Leader':
        sizes = dbs.cluster_sizes_

        map = plot_clusters(data, dbs.cluster_centers_[sizes > size],
                      sizes[sizes > size],
                      sizeprop=250,
                      dataname='leader-crd' + str(radius) + '-mex' + str(size))
        nfile = data.wpath + 'Clusters/' + data.city[2] + data.get_app_name() + '-' + 'Leader-crd' + str(radius) + '-mex' + str(size)
        pkfile = open(nfile + '.pkl', 'w')
        pickle.dump(dbs, pkfile)
        pkfile.close()
    elif alg == 'DBSCAN':
        labset = set(dbs.labels_)
        if -1 in labset:
            dim = len(labset) - 1
        else:
            dim = len(set)
        centers = np.zeros((dim, 2))
        sizes = np.zeros(dim)
        dataset = data.dataset
        clres = np.zeros((dataset.shape[0], 3))
        for i in range(dataset.shape[0]):
            clres[i][0] = dataset[i][0]
            clres[i][1] = dataset[i][1]
            clres[i][2] = dbs.labels_[i]
            if dbs.labels_[i] != -1:
                sizes[int(dbs.labels_[i])] += 1
                centers[int(dbs.labels_[i])][0] += dataset[i][0]
                centers[int(dbs.labels_[i])][1] += dataset[i][1]

        for i in range(sizes.shape[0]):
            centers[i][0] /= sizes[i]
            centers[i][1] /= sizes[i]
            print sizes[i]
        nfile = data.wpath + 'Results/' + data.city[2] + data.get_app_name() + '-'
        savetxt(nfile + 'DBSCAN-crd' + str(radius) + '-mins' + str(mins) + '-mex' + str(size) + '.csv',
                clres, delimiter=';')
        map = plot_clusters(data, centers[sizes > size], sizes[sizes > size], dataname='dbscan-crd' + str(radius)
                                            + '-mins' + str(mins) + '-mex' + str(size), sizeprop=sizeprop)

    return dbs, map


def plot_clusters(data, centroids, csizes, sizeprop=1000, dataname=''):
    """
    Generates an scale x scale plot of the events
    Every event is represented by a point in the graph
    the ouput is an html file that uses open street maps

    :param string dataname: Name to append to the filename
    """

    print 'Generating the events plot ...'

    today = time.strftime('%Y%m%d%H%M%S', time.localtime())
    minLat, maxLat, minLon, maxLon = data.city[1]
    mymap = folium.Map(location=[(minLat + maxLat) / 2.0, (minLon + maxLon) / 2.0], zoom_start=12, width=1400,
                       height=1400)

    maxsize = np.max(csizes) / sizeprop

    for i in range(centroids.shape[0]):
        if sizeprop != 0:
            plotsize = csizes[i] / maxsize
        else:
            plotsize = 10
        mymap.circle_marker(location=[centroids[i][0], centroids[i][1]],
                            radius=plotsize,
                            line_color='#FF0000',
                            fill_color='#110000')

    nfile = data.get_app_name() + '-' + dataname

    mymap.create_map(path=data.wpath + 'Results/' + data.city[2] + nfile + '.html')
    return mymap

