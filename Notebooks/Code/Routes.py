# -*- coding: utf-8 -*-
"""
.. module:: Routes

Routes
******

:Description: Routes

    Routines that compute routes

:Authors:
    bejar

:Version: 1.0

:File: Routes

:Created on: 20/02/2014 15:17
r

"""

__author__ = 'bejar'

# from fp_growth import find_frequent_itemsets
from fim import fpgrowth
import networkx as nx
import time
import operator

import numpy as np
import matplotlib.pyplot as plt

from Transactions import DailyDiscretizedTransactions, DailyClusteredTransactions
from Util import item_key_sort, diff_items


import folium
from geojson import LineString, FeatureCollection, Feature
import geojson
from TimeDiscretizer import TimeDiscretizer


def transaction_routes_clustering(data, nfile, cluster=None, supp=30, timeres=4, colapsed=False):
    """
    Generates a diagram of the routes obtained by the frequent itemsets fp-growth algorithm

    :param: dataclean:
    :param: application:
    :param: mxhh:
    :param: mnhh:
    :param: scale:
    :param: supp:
    :param: timeres:
    """
    strtimeres = ""
    for tm in timeres:
        strtimeres += str(tm)

    nfile = data.city[2] +  data.get_app_name() + nfile + '-tr' + strtimeres + '-sp' + str(supp)
    if colapsed:
        nfile += '-c'

    # File for the textual results
    rfile = open(data.wpath + 'Routes/' + nfile + '.txt', 'w')
    userEvents = DailyClusteredTransactions(data, cluster=cluster, timeres=TimeDiscretizer(timeres))

    print 'Serializing the transactions'
    if not colapsed:
        trans = userEvents.serialize()
    else:
        trans = userEvents.colapse()
    print 'Transactions', len(trans)
    ltrans = []
    print 'Applying fp-growth'
    for itemset, sval in fpgrowth(trans, supp=-supp, zmin=2, target='m'):
        if diff_items(itemset) > 1:
            ltrans.append(itemset)
            rfile.write(str(sorted(itemset, key=item_key_sort)) + ' ' + str(sval) + '\n')
    rfile.close()

    print 'Routes', len(ltrans)

    print 'Generating plot'
    minLat, maxLat, minLon, maxLon = data.city[1]

    mymap = folium.Map(location=[(minLat + maxLat) / 2.0, (minLon + maxLon) / 2.0], zoom_start=12, width=1500,
                       height=1000)

    lgeo = []
    for t in ltrans:
        seq = []
        for i in t:
            x, y, h = i.split('#')
            seq.append((x, y, h))
        seqs = sorted(seq, key=operator.itemgetter(2))
        for p1 in range(len(seqs) - 1):
            x1, y1, _ = seqs[p1]
            x2, y2, _ = seqs[p1 + 1]
            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)
            lgeo.append(Feature(geometry=LineString([(y1, x1), (y2, x2)])))

    # Saving the plot
    geoc = FeatureCollection(lgeo)
    dump = geojson.dumps(geoc)
    jsfile = open(data.wpath + 'Results/' + nfile + '.json', 'w')
    jsfile.write(dump)
    jsfile.close()
    mymap.geo_json(geo_path=data.wpath + 'Results/' + nfile + '.json', fill_color='Black', line_color='Black',
                   line_weight=2)
    mymap.create_map(path=data.wpath + 'Results/' + nfile + '.html')
    return mymap

