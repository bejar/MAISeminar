"""
.. module:: Util

Util
******

:Description: Util

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 1.0

:File: Util

:Created on: 20/02/2014 14:12
"""

__author__ = 'bejar'

import string
import time


def now():
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def item_key_sort(v):
    """
    auxiliary function for sorting geo-time events using the t

    :param v: item
    :return: time discretization of the item
    """
    _, _, h = v.split('#')
    return h


def diff_items(seq):
    """
    Number of different geo point in a sequence

    :param seq: sequence of ST items
    :return: number of different items in the sequence
    """
    tset = set()
    for s in seq:
        x1, y1, _ = s.split('#')
        tset.add(str(x1) + '#' + str(y1))
    return len(tset)


def item_to_column(item, scale):
    """
    Transforms an item to a column number given the scale of the discretization
    an item is a string with the format posx#posy#time

    :param string item: string corresponding to two coordinates and a time dicretization
    :param int scale: value used in the scaling
    :returns: Integer corresponding to the column of the item
    :rtype: int
    """
    x, y, t = item.split('#')
    return (int(t) * scale * scale) + (int(y) * scale) + int(x)


def strip_nl(val):
    """
    Deletes the CR of the string and deletes leading and trailing spaces
    @param val:
    @return:
    """
    return (string.lstrip(string.rstrip(val[:len(val) - 1])))
