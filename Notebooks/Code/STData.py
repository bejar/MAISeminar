# -*- coding: utf-8 -*-
"""
.. module:: Data
.. moduleauthor:: Javier BÃ©jar

STData
************

:Description: SuperHub STData class

    Representation for Spatio Temporal data, basically latitude, longitude and time events with the user that
    generated the event

    Performs different processings to the data matrix

:Authors:
    bejar

:Version: 1.0

:File: Data

:Created on: 18/02/2014 10:09

"""

__author__ = 'bejar'

import operator
import time

from numpy import loadtxt, savetxt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Constants import homepath
import folium
from geojson import FeatureCollection, Feature, Polygon
import geojson

pltcolors = ['#F00000', '#00F000', '#0000F0', '#0F0000', '#000F00', '#00000F']


class STData:
    """
    Class for a superhub dataset:

    :arg path: Sets the path of the file
    :arg application: Sets the application of the dataset

    """
    dataset = None
    application = None
    wpath = None
    mxhh = None
    mnhh = None
    lhh = None
    datasethh = None
    city = None

    def get_app_name(self):
        if type(self.application) == list:
            nm = ''
            for ap in self.application:
                nm += ap
        else:
            nm = self.application
        return nm

    def __init__(self, path, city, application):
        """
        Just sets the path and application for the dataset


         :arg path: Sets the path of the file
         :arg application: Sets the application of the dataset

        """
        self.application = application
        self.wpath = path
        self.city = city

    def read_data(self):
        """
        Loads the data from the csv file

        """
        print 'Reading Data ...'
        fname = self.wpath + 'Data/' + self.city[2] + '-' + self.application + '.csv.bz2'
        self.dataset = loadtxt(fname, skiprows=0,
                               dtype=[('lat', 'f8'), ('lng', 'f8'), ('time', 'i4'), ('user', 'S20')],
                               usecols=(0, 1, 2, 3), delimiter=',', comments='#')


    def save_data(self, filename, coordinates=True):
        """
        Saves some columns of the data to a csv file

        @param filename:
        @return:
        """

        if not coordinates:
            savetxt(self.wpath+filename, self.dataset, delimiter=',')
        else:
            print self.wpath+filename
            savetxt(self.wpath+filename, self.getDataCoordinates(), delimiter=',')

    def info(self):
        """
        Dumps some info about the dataset

        @return:
        """
        print 'A= ', self.application
        print 'C= ', self.city
        print 'D= ', self.dataset.shape

    def getDataCoordinates(self):
        """
        Returns an array with the coordinates of all the examples
        @return:
        """
        coord = np.zeros((self.dataset.shape[0], 2))
        for i in range(len(self.dataset)):
            coord[i, 0] = self.dataset[i][0]
            coord[i, 1] = self.dataset[i][1]
        return coord

    def compute_heavy_hitters(self, mxhh, mnhh, out=False):
        """
        Computes the list of the number of events
        and returns a list with the users between the
        positions mxhh and mnhh in the descendent order

        If the list heavy hitters have already been computed it is reused

        :param int mxhh: initial position of the heavy hitters list
        :param int mnhh: final position of the heavy hitters list

        :returns: list with the list of users ordered (desc) by number of events
        :rtype: list
        """
        print 'Computing Heavy Hitters ...'
        if self.lhh is not None:
            mnhht = min(mnhh, len(self.lhh))
            hhitters = [x for x, y in self.lhh[mxhh:mnhht]]
        else:
            usercount = {}
            for i in range(self.dataset.shape[0]):
                if self.dataset[i][3] in usercount:
                    usercount[self.dataset[i][3]] += 1
                else:
                    usercount[self.dataset[i][3]] = 1
            # we memorize the list of users so it can be reused
            sorted_x = sorted(usercount.iteritems(), key=operator.itemgetter(1), reverse=True)
            self.lhh = sorted_x
            mnhht = min(mnhh, len(sorted_x))
            hhitters = [x for x, y in sorted_x[mxhh:mnhht]]
            # print usercount[hhitters[0]], usercount[hhitters[-1]]
        return hhitters

    def select_heavy_hitters(self, mxhh, mnhh):
        """
        Deletes all the events that are not from the heavy hitters
        Returns a new data object only with the heavy hitters


        :param int mxhh: initial position of the heavy hitters list
        :param int mnhh: final position of the heavy hitters list

        :retuns:
         A list of the most active users in the indicated range
        """
        self.mxhh = mxhh
        self.mnhh = mnhh
        lhh = self.compute_heavy_hitters(mxhh, mnhh)
        print 'Selecting Heavy Hitters ...'
        return self.select_data_users(lhh)

    def select_data_users(self, users):
        """
        Selects only the events from the list of users

        :arg list users: List of users to select
        :returns:  Returns a new object with the selected users
        """
        print 'Selecting Users ...'
        # First transforms the list of users to a set to be efficient
        susers = set(users)
        # computes the boolean array for the selection
        sel = [self.dataset[i][3] in susers for i in range(self.dataset.shape[0])]
        asel = np.array(sel)
        data = STData(self.wpath, self.city, self.application)
        data.dataset = self.dataset[asel]
        data.mxhh = self.mxhh
        data.mnhh = self.mnhh
        return data

    def select_hours(self, lhours):
        """
        Selects only events inside an specific range of hours

        @param ihour:
        @param fhour:
        @return:
        """
        sel = []
        for i in range(self.dataset.shape[0]):
            stime = time.localtime(np.int32(self.dataset[i][2]))
            hour = stime[3]
            for ih in lhours:
                ihour, fhour = ih
                if ihour <= hour < fhour:
                    sel.append(i)
        data = STData(self.wpath, self.city, self.application)
        data.dataset = self.dataset[sel]
        return data


    def hourly_table(self):
        """
        Computes the accumulated events by hour for the data table

        :returns:
         A list with the accumulated number of events for each hour of the day
        """
        htable = [0 for i in range(24)]
        for i in range(self.dataset.shape[0]):
            stime = time.localtime(np.int32(self.dataset[i][2]))
            evtime = stime[3]
            htable[evtime] += 1
        return htable

    def daily_table(self):
        """
        Computes the accumulated events by day for the data table

        :returns:
         A list with the accumulated number of events for each day of the week
        """
        htable = [0 for i in range(7)]
        for i in range(self.dataset.shape[0]):
            stime = time.localtime(np.int32(self.dataset[i][2]))
            evtime = stime[6]
            htable[evtime] += 1
        return htable

    def monthly_table(self):
        """
        Computes the accumulated events by month

        :returns:
         A list with the accumulated number of events for each mont of the year
        """
        htable = [0 for i in range(12)]
        for i in range(self.dataset.shape[0]):
            stime = time.localtime(np.int32(self.dataset[i][2]))
            evtime = stime[1]
            htable[evtime - 1] += 1
        return htable


    def plot_events_cluster(self, cluster, distrib=True, dataname='', timeres=None):
        """
        Generates an scale x scale plot of the events
        Every event is represented by a point in the graph
        the ouput is a pdf file and an html file that uses open street maps

        :param int scale: Scale of the spatial discretization
        :param bool distrib: If returns the frequency or the accumulated events
        :param string dataname: Name to append to the filename
        """

        def plot_notimeres():
            """
            All time colapsed
            @return:
            """
            cont = np.zeros(cluster.num_clusters())
            for i in range(self.dataset.shape[0]):
                posy = self.dataset[i][0]
                posx = self.dataset[i][1]
                ejem = np.array([[posy, posx]]).reshape(1,-1)
                ncl = cluster.predict(ejem)
                ncl = ncl[0]

                if distrib:
                    cont[ncl] += 1
                else:
                    cont[ncl] = 1

            if distrib:
                cont = cont / np.max(cont)
                for i in range(cont.shape[0]):
                    if cont[i] > 0.01:
                        cx = cluster.cluster_centers_[i][0]
                        cy = cluster.cluster_centers_[i][1]
                        mymap.circle_marker(location=[cx, cy],
                                            radius=cont[i] * circlesize,
                                            line_color='#000000',
                                            fill_color='#110000',
                                            popup=str(cont[i]), fill_opacity=0.4)
            else:
                for i in range(cont.shape[0]):
                    if cont[i] > 0.01:
                        cx = cluster.cluster_centers_[i][0]
                        cy = cluster.cluster_centers_[i][1]

                        mymap.circle_marker(location=[cx, cy],
                                            radius=30,
                                            line_color='#000000',
                                            fill_color='#110000',
                                            popup=str(cont[i]), fill_opacity=0.4)

        def plot_timeres(timeres):
            """
            Geo points separated by the time resolution zones
            @return:
            """
            tint = 24 / len(timeres.intervals)
            step = 255 / (tint + 1)

            cont = np.zeros(cluster.num_clusters())
            for i in range(self.dataset.shape[0]):
                posy = self.dataset[i][0]
                posx = self.dataset[i][1]
                ejem = np.array([[posy, posx]]).reshape(1,-1)
                ncl = cluster.predict(ejem)
                ncl = ncl[0]

                if distrib:
                    cont[ncl] += 1
                else:
                    cont[ncl] = 1

            for t in range(tint):
                color = '#' + (str(hex((t + 1) * step))[2:]) + \
                        (str(hex((t + 1) * step))[2:]) + 'FF'  # (str(hex((t+1)*step))[2:])
                cont = np.zeros(cluster.num_clusters())
                for i in range(self.dataset.shape[0]):
                    posy = self.dataset[i][0]
                    posx = self.dataset[i][1]
                    ejem = np.array([[posy, posx]]).reshape(1,-1)
                    ncl = cluster.predict(ejem)
                    ncl = ncl[0]

                    _, evtime = timeres.discretize(self.dataset[i][2])

                    if evtime == t:
                        if distrib:
                            cont[ncl] += 1
                        else:
                            cont[ncl] = 1
                if distrib:
                    cont = cont / np.max(cont)
                    for i in range(cont.shape[0]):
                        if cont[i] > 0.01:
                            cx = cluster.cluster_centers_[i][0]
                            cy = cluster.cluster_centers_[i][1]
                            mymap.circle_marker(location=[cx, cy],
                                                radius=cont[i] * circlesize,
                                                line_color=pltcolors[t],
                                                fill_color=pltcolors[t],  #'#110000',
                                                popup=str(t) + '-' + str(cont[i]), fill_opacity=0.2)
                else:
                    for i in range(cont.shape[0]):
                        for j in range(cont.shape[1]):
                            if cont[i] > 0.01:
                                cx = cluster.cluster_centers_[i][0]
                                cy = cluster.cluster_centers_[i][1]
                                mymap.circle_marker(location=[cx, cy],
                                                    radius=30,
                                                    line_color=pltcolors[t],
                                                    fill_color=pltcolors[t],  #'#110000',
                                                    popup=str(t) + '-' + str(cont[i]), fill_opacity=0.2)

        print 'Generating the events plot ...'
        circlesize = 60000 * cluster.radius

        # if timeres == 0:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)

        minLat, maxLat, minLon, maxLon = self.city[1]
        # normLat = scale / (maxLat - minLat)
        # normLon = scale / (maxLon - minLon)
        mymap = folium.Map(location=[(minLat + maxLat) / 2.0, (minLon + maxLon) / 2.0], zoom_start=12, width=1200,
                           height=1000)

        if timeres is None:
            plot_notimeres()
        else:
            plot_timeres(timeres)

        #today = time.strftime('%Y%m%d%H%M%S', time.localtime())
        nfile = self.application + '-' + dataname + '-Clust'
        if self.mnhh is not None and self.mnhh is not None:
            nfile += '-nusr' + str(self.mxhh) + '#' + str(self.mnhh)

        if timeres is not None:
            nfile += '-tr' + str(timeres.intervals)
        nfile += '-s' + str(cluster.radius)

        # if timeres == 0:
        #     fig.savefig(homepath + 'Results/' + self.city[2] + '-' +
        #                 nfile + '.pdf', orientation='landscape', format='pdf')
        #     plt.close()
        mymap.create_map(path=homepath + 'Results/clusters/' + self.city[2] + nfile + '.html')

    def generate_user_dict(self):
        res = {}
        for i in range(self.dataset.shape[0]):
            if self.dataset[i][3].strip() not in res:
                res[self.dataset[i][3].strip()] = 1
            else:
                res[self.dataset[i][3].strip()] += 1
        return res

if __name__ == '__main__':

    from Constants import homepath, cityparams
    from Clustering import cluster_events
    data = STData('../../', cityparams['bcn'], 'twitter')

    data.read_data()
    data.info()
    mymap = cluster_events(data, radius=0.005)
