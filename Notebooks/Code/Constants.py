# -*- coding: utf-8 -*-
"""
.. module:: SuperHubConstants

Constants
************

:Description: SuperHub constants,

    The coordinates of the region of interest and the path to the data files
    And the information of the mongo database

:Authors:
    bejar

:Version: 1.0


"""

__author__ = 'bejar'


import numpy as np
homepath = '../'

bcncoord = (41.20, 41.65, 1.90, 2.40)
milancoord = (45.33, 45.59, 9.03, 9.37)
pariscoord = (48.52, 49.05, 1.97, 2.68)
londoncoord = (51.23, 51.8, -0.50, 0.37)
berlincoord = (52.32, 52.62, 13.11, 13.60)
romecoord = (41.78, 42.0, 12.33, 12.62)

bcnparam = (None, bcncoord, 'bcn', None, 120, None)
milanparam = (None, milancoord, 'milan', None, 120, None)
parisparam = (None, pariscoord, 'paris', None, 60, None)
londonparam = (None, londoncoord, 'london', None, 60, None)
berlinparam = (None, berlincoord, 'berlin', None, 120, None)
romeparam = (None, romecoord, 'rome', None, 120, None)

cityparams = {
    'bcn': bcnparam,
    'milan': milanparam,
    'paris': parisparam,
    'london': londonparam,
    'berlin': berlinparam,
    'rome': romeparam,
}
