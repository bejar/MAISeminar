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

import networkx as nx
import folium

from Code.Constants import  cityparams
from Code.Routes import draw_graph
city = 'bcn'
params = cityparams[city]

fname = '../Routes/bcntwitterkmeans-tr618-sp5.txt'
coord = params[1]

rfile = open(fname, 'r')

gr = nx.Graph()



maplines = []
for lines in rfile:
    lines = lines[:lines.find(']')]
    vals = lines.replace('[', '').replace(']','').replace('\n','').replace('\'','').replace(' ','').split(',')
    print vals
    for v1 in vals:
        for v2 in vals:
            if v1 != v2:
                gr.add_edge(v1,v2)
                # x1, y1, _= v1.split('#')
                # x2, y2, _= v2.split('#')

mymap = draw_graph(gr, params, '../', city+'-routes')

mymap