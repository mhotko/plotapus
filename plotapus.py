"""This module helps with plotting and parsing (even BioLogic mpr files) simple data"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
import re
import csv
import os
from os import SEEK_SET
import time
from datetime import date, datetime, timedelta
from collections import defaultdict, OrderedDict
from dataclasses import dataclass


import numpy as np


def _is_notebook():
    """
    The function checks if we are in the jupyter environment (notebook/labs).
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


if _is_notebook():
    from IPython.display import display

# YOU CAN MODIFY THIS FOR YOUR USE CASE
default_values = {
    "data_pos": (3, 2),
    "slc": (0, -1, 1),
    "norm_to": 1000,
    "area": 0.07065,
    "x_lbl": "E vs RHE / V",
    "y_lbl": "I / A",
    "y_normalized_lbl": "I / mA/cm$^2$",
    "legend": True,
    "grid": True,
    "minorticks_on": True,
    'dpi': 400,
}


class DataHelper:
    """
        A simple class to plot simple data. Simply import it as
        'from plotapus import DataHelper as dh'
        and you are simply ready to go. It's simple.
    """

    def __init__(self, file_path,
                 data_pos=default_values["data_pos"],
                 slc=default_values['slc'],
                 subplot=None,
                 **kwargs):
        """
        REQUIRED ARGUMENTS:
        file_path: the path to the file, must contain the name of the file and extension './blablabla.csv'

        DEFAULT ARGUMENTS:
        data_pos: the columns in which the data is located (x, y) ---> default look at the default_values dict
        slc: what slice of data should the module use (start, end, step) -> default (0, -1, 1) where the -1 for stop means to the end of file
                                                                           example: (10, -1, 2) means start at the 10th point, -1 means go to the end, 
                                                                                    2 means take every 2nd point

        OPTIONAL ARGUMENTS:
        subplot: the individual plot you want to add data to

        OTHER ARGUMENTS:
        delimiter: the type of delimiter for csv or txt files
        header: on which line in a csv or txt file is the header located, usually 0
        sep: the separator that separates the data
        decimal: the decimal point separator

        USE:
            dh(arguments)
        """
        self.data_pos = data_pos
        self.file_path = file_path
        self.slc = slc
        self.file_name, self.file_ext = os.path.splitext(file_path)
        self.subplot = subplot
        if self.file_ext == '.xlsx':
            self.df = pd.DataFrame(pd.read_excel(self.file_path))

        elif self.file_ext == '.csv' or self.file_ext == '.txt':
            self.df = pd.DataFrame(pd.read_csv(self.file_path, **kwargs))
        elif self.file_ext == '.mpr':
            self.df = pd.DataFrame(MPRfile(self.file_path).data)
        else:
            raise InvalidFileTypeException

        self.keys = self.df.keys()

    def get_data(self, disp=False, scan_col=None, scan_num=None,):
        """
        get data for x and y columns in 2D array [[x], [y]]

        DEFAULT ARGUMENTS:
        disp: display the data, but don't return it. !!! If you want to save the data in a variable make this False !!!

        OPTIONAL ARGUMENTS:
        scan_col: if your data has a scan column you can specify it
        scan_num: specify the number of scan you want to get

        USE:
            dh(...).get_data(arguments)
        """
        t = [self.get_data_x(scan_col=scan_col, scan_num=scan_num),
             self.get_data_y(scan_col=scan_col, scan_num=scan_num)]
        if disp:
            display(t)
        else:
            return t

    def get_data_x(self, scan_col=None, scan_num=None, disp=False):
        """
        get data for x column

        DEFAULT ARGUMENTS:
        disp: display the data, but don't return it. !!! If you want to save the data in a variable make this False !!!

        OPTIONAL ARGUMENTS:
        scan_col: if your data has a scan column you can specify it
        scan_num: specify the number of scan you want to get

        USE:
            dh(...).get_data_x(arguments)
        """
        if scan_col:
            if scan_num == None:
                msn = max(self.df[self.keys[scan_col]])
                x = [self.df[self.keys[self.data_pos[0]]][x]
                     for x in range(len(self.df))
                     if self.df[self.keys[scan_col]][x] == msn]
            else:
                x = [self.df[self.keys[self.data_pos[0]]][x]
                     for x in range(len(self.df))
                     if self.df[self.keys[scan_col]][x] == scan_num]
        else:
            x = [self.df[self.keys[self.data_pos[0]]][x]
                 for x in range(len(self.df))]
        if disp:
            display(x)
        else:
            return x[self.slc[0]:(len(x) if self.slc[1] == -1 else self.slc[1]):self.slc[2]]

    def get_data_y(self, scan_col=None, scan_num=None, disp=False):
        """
        get data for y column

        DEFAULT ARGUMENTS:
        disp: display the data, but don't return it. !!! If you want to save the data in a variable make this False !!!

        OPTIONAL ARGUMENTS:
        scan_col: if your data has a scan column you can specify it
        scan_num: specify the number of scan you want to get

        USE:
            dh(...).get_data_y(arguments)
        """
        if scan_col:
            if scan_num == None:
                msn = max(self.df[self.keys[scan_col]])
                y = [self.df[self.keys[self.data_pos[1]]][x]
                     for x in range(len(self.df))
                     if self.df[self.keys[scan_col]][x] == msn]
            else:
                y = [self.df[self.keys[self.data_pos[1]]][x]
                     for x in range(len(self.df))
                     if self.df[self.keys[scan_col]][x] == scan_num]
        else:
            y = [self.df[self.keys[self.data_pos[1]]][x]
                 for x in range(len(self.df))]
        if disp:
            display(y)
        else:
            return y[self.slc[0]:(len(y) if self.slc[1] == -1 else self.slc[1]):self.slc[2]]

    def normalize(self, norm_to=default_values['norm_to'], area=default_values['area'], scan_col=None, scan_num=None):
        """
        get data for y column in a list

        DEFAULT ARGUMENTS:
        norm_to: normalize to a specific SI unit -> default 1000 to normalize to mA
        area: the area of the sample in cm^2 -> current default 0.07065 cm^2

        OPTIONAL ARGUMENTS:
        scan_col: if your data has a scan column you can specify it
        scan_num: specify the number of scan you want to get

        USE:
            dh(...).normalize(arguments)
        """
        data = self.get_data_y(scan_col=scan_col, scan_num=scan_num)
        return [y*norm_to/area for y in data]

    def plot(self,
             title="",
             x_lbl=default_values['x_lbl'],
             y_lbl=default_values['y_lbl'],
             legend=default_values['legend'],
             grid=default_values['grid'],
             minorticks_on=default_values['minorticks_on'],
             scan_col=None,
             scan_num=None,
             **kwargs):
        """
        plot the data

        DEFAULT ARGUMENTS:
        x_lbl: the label matplotlib uses for the x axis label
        y_lbl: the label matplotlib uses for the y axis label
        legend: should matplotlib show the legend, requires the label to be set
        grid: should matplotlib show the grid
        minorticks_on: should matplotlib show the minorticks

        OPTIONAL ARGUMENTS:
        scan_col: if your data has a scan column you can specify it
        scan_num: specify the number of scan you want to get

        USE:
            dh(...).plot(arguments)
        """
        x = self.get_data_x(scan_col=scan_col, scan_num=scan_num)
        y = self.get_data_y(scan_col=scan_col, scan_num=scan_num)
        self._general_plot(x, y, title, x_lbl, y_lbl, legend,
                           grid, minorticks_on, **kwargs)

    def plot_normalized(self,
                        title="",
                        x_lbl=default_values['x_lbl'],
                        y_lbl=default_values['y_normalized_lbl'],
                        legend=default_values['legend'],
                        grid=default_values['grid'],
                        minorticks_on=default_values['minorticks_on'],
                        norm_to=default_values['norm_to'],
                        area=default_values['area'],
                        scan_col=None,
                        scan_num=None,
                        **kwargs):
        """
        plot the normalized data

        DEFAULT ARGUMENTS:
        x_lbl: the label matplotlib uses for the x axis label
        y_lbl: the label matplotlib uses for the y axis label
        legend: should matplotlib show the legend, requires the label to be set
        grid: should matplotlib show the grid
        minorticks_on: should matplotlib show the minorticks
        norm_to: normalize to a specific SI unit -> default 1000 to normalize to mA
        area: the area of the sample in cm^2 -> current default 0.07065 cm^2

        OPTIONAL ARGUMENTS:
        scan_col: if your data has a scan column you can specify it
        scan_num: specify the number of scan you want to get

        USE:
            dh(...).plot_normalized(arguments)
        """
        x = self.get_data_x(scan_col=scan_col, scan_num=scan_num)
        if self.file_ext != '.mpr':
            y = self.normalize(norm_to=norm_to, area=area,
                               scan_col=scan_col, scan_num=scan_num)
        else:
            # mpr files already in in mA
            y = self.normalize(norm_to=1, area=area,
                               scan_col=scan_col, scan_num=scan_num)
        self._general_plot(x, y, title, x_lbl, y_lbl, legend,
                           grid, minorticks_on, **kwargs)

    def pprint(self):
        """
        Pretty print function works only in jupyter notebook, where we 
        have access to the display function from the IPython module

        USE:
            dh(...).pprint()
        """
        try:
            return display(self.df)
        except Exception as exc:
            raise NotInJupyterEnvirnomentException from exc

    @staticmethod
    def make():
        """
        Call this function when:
            1. You're not in the jupyter environment and want to view your plot
            2. If you don't want to plot on the same graph anymore
        USE:
            dh.make()
        """
        plt.show()

    @staticmethod
    def xlim(*args, **kwargs):
        """
        set the xlim of the plot, pass in numbers

        USE:
            dh.xlim()
        """
        plt.xlim(*args, **kwargs)

    @staticmethod
    def ylim(*args, **kwargs):
        """
        set the ylim of the plot, pass in numbers

        USE:
            dh.ylim()
        """
        plt.ylim(*args, **kwargs)

    @staticmethod
    def save_plot(*args, **kwargs):
        """
        save the plot

        DEFAULT ARGUMENTS:
        dpi: the dpi to save as

        USE:
            dh.save_plot(arguments)
        """
        plt.savefig(*args, dpi=default_values['dpi'], **kwargs)

    @staticmethod
    def zoom(region, bounds, subplot=None):
        """
        Zoom in part of plot

        REQUIRED ARGUMENTS:
        region: where do you want to zoom to [x0, x1, y0, y1]
        bounds: Lower-left corner of inset Axes, and its width and height. [x0, y0, width, height]

        OPTIONAL ARGUMENTS:
        subplot: the individual plot you want to add data to

        USE:
            dh.zoom(arguments)
        """
        if subplot:
            ax = subplot
        else:
            ax = plt.gca()
        lines = ax.get_lines()
        axins = ax.inset_axes(bounds)
        for t in lines:
            xdata, ydata = t.get_data()
            linewidth = t.get_linewidth()
            linestyle = t.get_linestyle()
            color = t.get_color()
            marker = t.get_marker()
            markersize = t.get_markersize()
            markeredgewidth = t.get_markeredgewidth()
            markeredgecolor = t.get_markeredgecolor()
            markerfacecolor = t.get_markerfacecolor()
            markerfacecoloralt = t.get_markerfacecoloralt()
            fillstyle = t.get_fillstyle()
            antialiased = t.get_antialiased()
            dash_capstyle = t.get_dash_capstyle()
            solid_capstyle = t.get_solid_capstyle()
            dash_joinstyle = t.get_dash_joinstyle()
            solid_joinstyle = t.get_solid_joinstyle()
            pickradius = t.get_pickradius()
            drawstyle = t.get_drawstyle()
            markevery = t.get_markevery()
            axins.plot(xdata, ydata, linewidth=linewidth, linestyle=linestyle, color=color, marker=marker, markersize=markersize,
                       markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor, markerfacecoloralt=markerfacecoloralt,
                       fillstyle=fillstyle, antialiased=antialiased, dash_capstyle=dash_capstyle, solid_capstyle=solid_capstyle, dash_joinstyle=dash_joinstyle,
                       solid_joinstyle=solid_joinstyle, pickradius=pickradius, drawstyle=drawstyle, markevery=markevery)
        x1, x2, y1, y2 = region[0], region[1], region[2], region[3]
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.xaxis.set_minor_locator(AutoMinorLocator())
        axins.yaxis.set_minor_locator(AutoMinorLocator())
        ax.indicate_inset_zoom(axins, edgecolor="black")

    @staticmethod
    def create_multiple(cols, rows, shape, left=None, right=None, top=None, bottom=None, hspace=None, wspace=None, figsize=None):
        """
        create multiple subplots based on the give shape and return the subplots in a list

        REQUIRED ARGUMENTS:
        cols: the number of columns
        rows: the number of rows
        shape: the shape of the plots given with a 2D list, it has to be the same shape as cols and rows
               the options are:
                                1. # for single plot
                                2. - for expanding the plot horizontally
                                3. | for expanding the plot vertically
                                4. ! to get one big plot
               EXAMPLE:
                        [["!", "!", "|"],
                         ["!", "!", "|"],
                         ["-","-", "#"]]
                        creates a plot that looks like this:
                        ------------- ----
                        |     2x2   | |  | 1x2
                        |           | |  |
                        ------------- ----
                        ------------- ----
                        |    2x1    | |  | 1x1
                        ------------- ----

        OPTIONAL ARGUMENTS:
        left, right, top, bottom:
            Extent of the subplots as a fraction of figure width or height. Left cannot be larger than right, and bottom cannot be larger than top.
        wspace: The amount of width reserved for space between subplots, expressed as a fraction of the average axis width.
        hspace: The amount of height reserved for space between subplots, expressed as a fraction of the average axis height.
        figsize: Width, height in inches. 1 inch is about 80px. figsize=(x,y)

        USE:
            dh.create_multiple(arguments)
        """
        fig = plt.figure(figsize=figsize)
        return_arr = []
        gs = GridSpec(rows, cols, left=left, right=right, top=top,
                      bottom=bottom, hspace=hspace, wspace=wspace)
        pipes = {}
        pipes_rows = []
        checked_dashes = {}
        bangs = {}
        for x in range(len(shape)):
            for y in range(len(shape[x])):
                if shape[x][y] == "#":
                    return_arr.append(fig.add_subplot(gs[x, y]))
                if shape[x][y] == "-":
                    if x not in checked_dashes:
                        checked_dashes[x] = [y]
                    else:
                        checked_dashes[x].append(y)
                elif shape[x][y] == "|":
                    pipes_rows.append(y)
                    if y not in pipes:
                        pipes[y] = [x]
                    else:
                        if x not in pipes[y]:
                            pipes[y].append(x)
                elif shape[x][y] == "!":
                    if y not in bangs:
                        bangs[y] = [x]
                    else:
                        bangs[y].append(x)

        if bool(checked_dashes):
            for k, v in checked_dashes.items():
                t = v[len(v)-1] + 1
                return_arr.append(fig.add_subplot(gs[k, v[0]:t]))
        if bool(pipes):
            i = 0
            data = {}
            for d in pipes_rows:

                t = "#".join(map(str, pipes[d]))
                if t not in data:
                    data[t] = [d]
                else:
                    if d not in data[t]:
                        data[t].append(d)
            for k, v in data.items():
                a = list(map(int, k.split("#")))
                for t in v:
                    return_arr.append(fig.add_subplot(
                        gs[min(a):max(a)+1, t:t+1]))
        if bool(bangs):
            i = 0
            data = {}
            for i in range(len(bangs.keys())):
                t = "#".join(map(str, bangs[i]))
                if t not in data:
                    data[t] = [i]
                else:
                    data[t].append(i)
            for k, v in data.items():
                a = list(map(int, k.split("#")))
                return_arr.append(fig.add_subplot(
                    gs[min(a):max(a)+1, min(v):max(v)+1]))
        # fig.tight_layout()
        return return_arr

    def __repr__(self):
        """
        if you print the object you get the string representation of the dataframe, use it as 'print(dh)', can be used anywhere

        USE:
            print(dh(...))
        """
        return repr(self.df)

    def _general_plot(self,
                      x,
                      y,
                      title,
                      x_lbl,
                      y_lbl,
                      legend,
                      grid,
                      minorticks_on,
                      **kwargs):
        """
        This is the general plotting function that all the other plot functions derive from.
        To see what the parameters do look up the actual plotting function you want to call,
        there the arguments are defined.
        """
        if self.subplot:
            if len(self.subplot.get_lines()) > 0:
                self.subplot.plot(x, y, **kwargs)
            else:
                self.subplot.plot(x, y, **kwargs)
                self.subplot.set_title(title)
                self.subplot.set_xlabel(x_lbl)
                self.subplot.set_ylabel(y_lbl)
                if grid:
                    self.subplot.grid()
                if minorticks_on:
                    self.subplot.minorticks_on()
            if legend:
                self.subplot.legend()

        else:
            ax = plt.gca()
            if len(ax.get_lines()) > 0:
                plt.plot(x, y, **kwargs)
                if legend:
                    handles, labels = ax.get_legend_handles_labels()
                    plt.legend(handles, labels)
            else:
                plt.plot(x, y, **kwargs)
                plt.xlabel(x_lbl)
                plt.ylabel(y_lbl)
                plt.title(title)
                if legend:
                    plt.legend()
                if grid:
                    plt.grid()
                if minorticks_on:
                    plt.minorticks_on()


class InvalidFileTypeException(Exception):
    def __init__(self):
        super().__init__("The file type in the plot module is not of type excel, csv, txt, mpr")


class NotInJupyterEnvirnomentException(Exception):
    def __init__(self):
        super().__init__("You are not in the Jupyter environment, so you cannot use this function. Open the file in Jupyter notebook/lab to see the output of the function. Otherwise you can use 'print(dh)' to see the output.")


############################################################
#                                                          #
#                                                          #
#           https://github.com/echemdata/galvani           #
#                                                          #
#                    MPR file parser code                  #
#                                                          #
#         DON'T TOUCH IF YOU DON'T KNOW CODING WELL        #
#        YOU COULD BREAK THE READING OF THE MPR FILE       #
#      AND IT IS GONNA BE A PAIN IN THE ASS TO FIX!!!!!    #
#                                                          #
#                                                          #
############################################################


# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2013-2020 Christopher Kerr, "bcolsen"
#
# SPDX-License-Identifier: GPL-3.0-or-later


def fieldname_to_dtype(fieldname):
    """Converts a column header from the MPT file into a tuple of
    canonical name and appropriate numpy dtype"""

    if fieldname == 'mode':
        return ('mode', np.uint8)
    elif fieldname in ("ox/red", "error", "control changes", "Ns changes",
                       "counter inc.", "Ramp upwards"):
        return (fieldname, np.bool_)
    elif fieldname in ("time/s", "P/W", "(Q-Qo)/mA.h", "x", "control/V",
                       "control/mA", "control/V/mA", "(Q-Qo)/C", "dQ/C",
                       "freq/Hz", "|Ewe|/V", "|I|/A", "Phase(Z)/deg",
                       "|Z|/Ohm", "Re(Z)/Ohm", "-Im(Z)/Ohm"):
        return (fieldname, np.float_)
    elif fieldname in ("Q charge/discharge/mA.h", "step time/s",
                       "Q charge/mA.h", "Q discharge/mA.h",
                       "Temperature/°C", "Efficiency/%", "Capacity/mA.h"):
        return (fieldname, np.float_)
    elif fieldname in ("cycle number", "I Range", "Ns", "half cycle"):
        return (fieldname, np.int_)
    elif fieldname in ("dq/mA.h", "dQ/mA.h"):
        return ("dQ/mA.h", np.float_)
    elif fieldname in ("I/mA", "<I>/mA"):
        return ("I/mA", np.float_)
    elif fieldname in ("Ewe/V", "<Ewe>/V", "Ecell/V"):
        return ("Ewe/V", np.float_)
    elif fieldname.endswith(("/s", "/Hz", "/deg",
                             "/W", "/mW", "/W.h", "/mW.h",
                             "/A", "/mA", "/A.h", "/mA.h",
                             "/V", "/mV",
                             "/F", "/mF", "/uF",
                             "/C", "/Ohm",)):
        return (fieldname, np.float_)
    else:
        raise ValueError("Invalid column header: %s" % fieldname)


def comma_converter(float_text):
    """Convert text to float whether the decimal point is '.' or ','"""
    trans_table = bytes.maketrans(b',', b'.')
    return float(float_text.translate(trans_table))


def MPTfile(file_or_path, encoding='ascii'):
    """Opens .mpt files as numpy record arrays

    Checks for the correct headings, skips any comments and returns a
    numpy record array object and a list of comments
    """

    if isinstance(file_or_path, str):
        mpt_file = open(file_or_path, 'rb')
    else:
        mpt_file = file_or_path

    magic = next(mpt_file)
    if magic not in (b'EC-Lab ASCII FILE\r\n', b'BT-Lab ASCII FILE\r\n'):
        raise ValueError("Bad first line for EC-Lab file: '%s'" % magic)

    nb_headers_match = re.match(rb'Nb header lines : (\d+)\s*$',
                                next(mpt_file))
    nb_headers = int(nb_headers_match.group(1))
    if nb_headers < 3:
        raise ValueError("Too few header lines: %d" % nb_headers)

    # The 'magic number' line, the 'Nb headers' line and the column headers
    # make three lines. Every additional line is a comment line.
    comments = [next(mpt_file) for i in range(nb_headers - 3)]

    fieldnames = next(mpt_file).decode(encoding).strip().split('\t')
    record_type = np.dtype(list(map(fieldname_to_dtype, fieldnames)))

    # Must be able to parse files where commas are used for decimal points
    converter_dict = dict(((i, comma_converter)
                           for i in range(len(fieldnames))))
    mpt_array = np.loadtxt(mpt_file, dtype=record_type,
                           converters=converter_dict)

    return mpt_array, comments


def MPTfileCSV(file_or_path):
    """Simple function to open MPT files as csv.DictReader objects

    Checks for the correct headings, skips any comments and returns a
    csv.DictReader object and a list of comments
    """

    if isinstance(file_or_path, str):
        mpt_file = open(file_or_path, 'r')
    else:
        mpt_file = file_or_path

    magic = next(mpt_file)
    if magic.rstrip() != 'EC-Lab ASCII FILE':
        raise ValueError("Bad first line for EC-Lab file: '%s'" % magic)

    nb_headers_match = re.match(r'Nb header lines : (\d+)\s*$', next(mpt_file))
    nb_headers = int(nb_headers_match.group(1))
    if nb_headers < 3:
        raise ValueError("Too few header lines: %d" % nb_headers)

    # The 'magic number' line, the 'Nb headers' line and the column headers
    # make three lines. Every additional line is a comment line.
    comments = [next(mpt_file) for i in range(nb_headers - 3)]

    mpt_csv = csv.DictReader(mpt_file, dialect='excel-tab')

    expected_fieldnames = (
        ["mode", "ox/red", "error", "control changes", "Ns changes",
         "counter inc.", "time/s", "control/V/mA", "Ewe/V", "dq/mA.h",
         "P/W", "<I>/mA", "(Q-Qo)/mA.h", "x", 'Ramp upwards'],

        ['mode', 'ox/red', 'error', 'control changes', 'Ns changes',
         'counter inc.', 'time/s', 'control/V', 'Ewe/V', 'dq/mA.h',
         '<I>/mA', '(Q-Qo)/mA.h', 'x', "P/W", 'Ramp upwards'],

        ["mode", "ox/red", "error", "control changes", "Ns changes",
         "counter inc.", "time/s", "control/V", "Ewe/V", "I/mA",
         "dQ/mA.h", "P/W", 'Ramp upwards'],

        ["mode", "ox/red", "error", "control changes", "Ns changes",
         "counter inc.", "time/s", "control/V", "Ewe/V", "<I>/mA",
         "dQ/mA.h", "P/W", 'Ramp upwards'])
    if mpt_csv.fieldnames not in expected_fieldnames:
        raise ValueError("Unrecognised headers for MPT file format")

    return mpt_csv, comments


VMPmodule_hdr = np.dtype([('shortname', 'S10'),
                          ('longname', 'S25'),
                          ('length', '<u4'),
                          ('version', '<u4'),
                          ('date', 'S8')])

# Maps from colID to a tuple defining a numpy dtype
VMPdata_colID_dtype_map = {
    4: ('time/s', '<f8'),
    5: ('control/V/mA', '<f4'),
    6: ('Ewe/V', '<f4'),
    7: ('dQ/mA.h', '<f8'),
    8: ('I/mA', '<f4'),  # 8 is either I or <I> ??
    9: ('Ece/V', '<f4'),
    11: ('I/mA', '<f8'),
    13: ('(Q-Qo)/mA.h', '<f8'),
    16: ('Analog IN 1/V', '<f4'),
    19: ('control/V', '<f4'),
    20: ('control/mA', '<f4'),
    23: ('dQ/mA.h', '<f8'),  # Same as 7?
    24: ('cycle number', '<f8'),
    26: ('Rapp/Ohm', '<f4'),
    32: ('freq/Hz', '<f4'),
    33: ('|Ewe|/V', '<f4'),
    34: ('|I|/A', '<f4'),
    35: ('Phase(Z)/deg', '<f4'),
    36: ('|Z|/Ohm', '<f4'),
    37: ('Re(Z)/Ohm', '<f4'),
    38: ('-Im(Z)/Ohm', '<f4'),
    39: ('I Range', '<u2'),
    69: ('R/Ohm', '<f4'),
    70: ('P/W', '<f4'),
    74: ('Energy/W.h', '<f8'),
    75: ('Analog OUT/V', '<f4'),
    76: ('<I>/mA', '<f4'),
    77: ('<Ewe>/V', '<f4'),
    78: ('Cs-2/µF-2', '<f4'),
    96: ('|Ece|/V', '<f4'),
    98: ('Phase(Zce)/deg', '<f4'),
    99: ('|Zce|/Ohm', '<f4'),
    100: ('Re(Zce)/Ohm', '<f4'),
    101: ('-Im(Zce)/Ohm', '<f4'),
    123: ('Energy charge/W.h', '<f8'),
    124: ('Energy discharge/W.h', '<f8'),
    125: ('Capacitance charge/µF', '<f8'),
    126: ('Capacitance discharge/µF', '<f8'),
    131: ('Ns', '<u2'),
    163: ('|Estack|/V', '<f4'),
    168: ('Rcmp/Ohm', '<f4'),
    169: ('Cs/µF', '<f4'),
    172: ('Cp/µF', '<f4'),
    173: ('Cp-2/µF-2', '<f4'),
    174: ('Ewe/V', '<f4'),
    241: ('|E1|/V', '<f4'),
    242: ('|E2|/V', '<f4'),
    271: ('Phase(Z1) / deg', '<f4'),
    272: ('Phase(Z2) / deg', '<f4'),
    301: ('|Z1|/Ohm', '<f4'),
    302: ('|Z2|/Ohm', '<f4'),
    331: ('Re(Z1)/Ohm', '<f4'),
    332: ('Re(Z2)/Ohm', '<f4'),
    361: ('-Im(Z1)/Ohm', '<f4'),
    362: ('-Im(Z2)/Ohm', '<f4'),
    391: ('<E1>/V', '<f4'),
    392: ('<E2>/V', '<f4'),
    422: ('Phase(Zstack)/deg', '<f4'),
    423: ('|Zstack|/Ohm', '<f4'),
    424: ('Re(Zstack)/Ohm', '<f4'),
    425: ('-Im(Zstack)/Ohm', '<f4'),
    426: ('<Estack>/V', '<f4'),
    430: ('Phase(Zwe-ce)/deg', '<f4'),
    431: ('|Zwe-ce|/Ohm', '<f4'),
    432: ('Re(Zwe-ce)/Ohm', '<f4'),
    433: ('-Im(Zwe-ce)/Ohm', '<f4'),
    434: ('(Q-Qo)/C', '<f4'),
    435: ('dQ/C', '<f4'),
    438: ('step time/s', '<f8'),
    441: ('<Ecv>/V', '<f4'),
    462: ('Temperature/°C', '<f4'),
    467: ('Q charge/discharge/mA.h', '<f8'),
    468: ('half cycle', '<u4'),
    469: ('z cycle', '<u4'),
    471: ('<Ece>/V', '<f4'),
    473: ('THD Ewe/%', '<f4'),
    474: ('THD I/%', '<f4'),
    476: ('NSD Ewe/%', '<f4'),
    477: ('NSD I/%', '<f4'),
    479: ('NSR Ewe/%', '<f4'),
    480: ('NSR I/%', '<f4'),
    486: ('|Ewe h2|/V', '<f4'),
    487: ('|Ewe h3|/V', '<f4'),
    488: ('|Ewe h4|/V', '<f4'),
    489: ('|Ewe h5|/V', '<f4'),
    490: ('|Ewe h6|/V', '<f4'),
    491: ('|Ewe h7|/V', '<f4'),
    492: ('|I h2|/A', '<f4'),
    493: ('|I h3|/A', '<f4'),
    494: ('|I h4|/A', '<f4'),
    495: ('|I h5|/A', '<f4'),
    496: ('|I h6|/A', '<f4'),
    497: ('|I h7|/A', '<f4'),
    498: ('Q charge/mA.h', '<f8'),
    499: ('Q discharge/mA.h', '<f8'),
    500: ('step time/s', '<f8'),
    501: ('Efficiency/%', '<f8'),
    502: ('Capacity/mA.h', '<f8'),
}

# These column IDs define flags which are all stored packed in a single byte
# The values in the map are (name, bitmask, dtype)
VMPdata_colID_flag_map = {
    1: ('mode', 0x03, np.uint8),
    2: ('ox/red', 0x04, np.bool_),
    3: ('error', 0x08, np.bool_),
    21: ('control changes', 0x10, np.bool_),
    31: ('Ns changes', 0x20, np.bool_),
    65: ('counter inc.', 0x80, np.bool_),
    463: ('Ramp upwards', 0x40, np.bool_),
}


def parse_BioLogic_date(date_text):
    """Parse a date from one of the various formats used by Bio-Logic files."""
    date_formats = ['%m/%d/%y', '%m-%d-%y', '%m.%d.%y']
    if isinstance(date_text, bytes):
        date_string = date_text.decode('ascii')
    else:
        date_string = date_text
    for date_format in date_formats:
        try:
            tm = time.strptime(date_string, date_format)
        except ValueError:
            continue
        else:
            break
    else:
        raise ValueError(f'Could not parse timestamp {date_string!r}'
                         f' with any of the formats {date_formats}')
    return date(tm.tm_year, tm.tm_mon, tm.tm_mday)


def VMPdata_dtype_from_colIDs(colIDs):
    """Get a numpy record type from a list of column ID numbers.

    The binary layout of the data in the MPR file is described by the sequence
    of column ID numbers in the file header. This function converts that
    sequence into a numpy dtype which can then be used to load data from the
    file with np.frombuffer().

    Some column IDs refer to small values which are packed into a single byte.
    The second return value is a dict describing the bit masks with which to
    extract these columns from the flags byte.

    """
    type_list = []
    field_name_counts = defaultdict(int)
    flags_dict = OrderedDict()
    for colID in colIDs:
        # print(colID)
        if colID in VMPdata_colID_flag_map:
            # Some column IDs represent boolean flags or small integers
            # These are all packed into a single 'flags' byte whose position
            # in the overall record is determined by the position of the first
            # column ID of flag type. If there are several flags present,
            # there is still only one 'flags' int
            if 'flags' not in field_name_counts:
                type_list.append(('flags', 'u1'))
                field_name_counts['flags'] = 1
            flag_name, flag_mask, flag_type = VMPdata_colID_flag_map[colID]
            # TODO what happens if a flag colID has already been seen
            # i.e. if flag_name is already present in flags_dict?
            # Does it create a second 'flags' byte in the record?
            flags_dict[flag_name] = (np.uint8(flag_mask), flag_type)
        elif colID in VMPdata_colID_dtype_map:
            field_name, field_type = VMPdata_colID_dtype_map[colID]
            field_name_counts[field_name] += 1
            count = field_name_counts[field_name]
            if count > 1:
                unique_field_name = '%s %d' % (field_name, count)
            else:
                unique_field_name = field_name
            type_list.append((unique_field_name, field_type))
        else:
            raise NotImplementedError("Column ID {cid} after column {prev} "
                                      "is unknown"
                                      .format(cid=colID,
                                              prev=type_list[-1][0]))
    return np.dtype(type_list), flags_dict


def read_VMP_modules(fileobj, read_module_data=True):
    """Reads in module headers in the VMPmodule_hdr format. Yields a dict with
    the headers and offset for each module.

    N.B. the offset yielded is the offset to the start of the data i.e. after
    the end of the header. The data runs from (offset) to (offset+length)"""

    while True:
        module_magic = fileobj.read(len(b'MODULE'))
        if len(module_magic) == 0:  # end of file
            break
        elif module_magic != b'MODULE':
            raise ValueError("Found %r, expecting start of new VMP MODULE"
                             % module_magic)

        hdr_bytes = fileobj.read(VMPmodule_hdr.itemsize)
        if len(hdr_bytes) < VMPmodule_hdr.itemsize:
            raise IOError("Unexpected end of file while reading module header")

        hdr = np.frombuffer(hdr_bytes, dtype=VMPmodule_hdr, count=1)
        hdr_dict = dict(((n, hdr[n][0]) for n in VMPmodule_hdr.names))
        hdr_dict['offset'] = fileobj.tell()
        if read_module_data:
            hdr_dict['data'] = fileobj.read(hdr_dict['length'])
            if len(hdr_dict['data']) != hdr_dict['length']:
                raise IOError("""Unexpected end of file while reading data
                    current module: %s
                    length read: %d
                    length expected: %d""" % (hdr_dict['longname'],
                                              len(hdr_dict['data']),
                                              hdr_dict['length']))
            yield hdr_dict
        else:
            yield hdr_dict
            fileobj.seek(hdr_dict['offset'] + hdr_dict['length'], SEEK_SET)


MPR_MAGIC = b'BIO-LOGIC MODULAR FILE\x1a'.ljust(48) + b'\x00\x00\x00\x00'


class MPRfile:
    """Bio-Logic .mpr file

    The file format is not specified anywhere and has therefore been reverse
    engineered. Not all the fields are known.

    Attributes
    ==========
    modules - A list of dicts containing basic information about the 'modules'
              of which the file is composed.
    data - numpy record array of type VMPdata_dtype containing the main data
           array of the file.
    startdate - The date when the experiment started
    enddate - The date when the experiment finished
    """

    def __init__(self, file_or_path):
        self.loop_index = None
        if isinstance(file_or_path, str):
            mpr_file = open(file_or_path, 'rb')
        else:
            mpr_file = file_or_path
        magic = mpr_file.read(len(MPR_MAGIC))
        if magic != MPR_MAGIC:
            raise ValueError('Invalid magic for .mpr file: %s' % magic)

        modules = list(read_VMP_modules(mpr_file))
        self.modules = modules
        settings_mod, = (m for m in modules if m['shortname'] == b'VMP Set   ')
        data_module, = (m for m in modules if m['shortname'] == b'VMP data  ')
        maybe_loop_module = [
            m for m in modules if m['shortname'] == b'VMP loop  ']
        maybe_log_module = [
            m for m in modules if m['shortname'] == b'VMP LOG   ']

        n_data_points = np.frombuffer(data_module['data'][:4], dtype='<u4')
        n_columns = np.frombuffer(data_module['data'][4:5], dtype='u1').item()
        # print(data_module['data'][:4])
        # print(data_module['data'][4:5])
        # print(data_module['data'])

        if data_module['version'] == 0:
            column_types = np.frombuffer(data_module['data'][5:], dtype='u1',
                                         count=n_columns)
            remaining_headers = data_module['data'][5 + n_columns:100]
            main_data = data_module['data'][100:]
        elif data_module['version'] in [2, 3]:
            column_types = np.frombuffer(data_module['data'][5:], dtype='<u2',
                                         count=n_columns)
            # There are bytes of data before the main array starts
            # print(column_types)
            if data_module['version'] == 3:
                num_bytes_before = 406  # version 3 added `\x01` to the start
            else:
                num_bytes_before = 405
            remaining_headers = data_module['data'][5 + 2 * n_columns:405]
            main_data = data_module['data'][num_bytes_before:]
            # print(main_data)
        else:
            raise ValueError("Unrecognised version for data module: %d" %
                             data_module['version'])

        assert(not any(remaining_headers))
        self.dtype, self.flags_dict = VMPdata_dtype_from_colIDs(column_types)
        self.data = np.frombuffer(main_data, dtype=self.dtype)
        assert(self.data.shape[0] == n_data_points)

        # No idea what these 'column types' mean or even if they are actually
        # column types at all
        self.version = int(data_module['version'])
        self.cols = column_types
        self.npts = n_data_points
        self.startdate = parse_BioLogic_date(settings_mod['date'])

        if maybe_loop_module:
            loop_module, = maybe_loop_module
            if loop_module['version'] == 0:
                self.loop_index = np.fromstring(loop_module['data'][4:],
                                                dtype='<u4')
                self.loop_index = np.trim_zeros(self.loop_index, 'b')
            else:
                raise ValueError("Unrecognised version for data module: %d" %
                                 data_module['version'])

        if maybe_log_module:
            log_module, = maybe_log_module
            self.enddate = parse_BioLogic_date(log_module['date'])

            # There is a timestamp at either 465 or 469 bytes
            # I can't find any reason why it is one or the other in any
            # given file
            ole_timestamp1 = np.frombuffer(log_module['data'][465:],
                                           dtype='<f8', count=1)
            ole_timestamp2 = np.frombuffer(log_module['data'][469:],
                                           dtype='<f8', count=1)
            ole_timestamp3 = np.frombuffer(log_module['data'][473:],
                                           dtype='<f8', count=1)
            ole_timestamp4 = np.frombuffer(log_module['data'][585:],
                                           dtype='<f8', count=1)

            if ole_timestamp1 > 40000 and ole_timestamp1 < 50000:
                ole_timestamp = ole_timestamp1
            elif ole_timestamp2 > 40000 and ole_timestamp2 < 50000:
                ole_timestamp = ole_timestamp2
            elif ole_timestamp3 > 40000 and ole_timestamp3 < 50000:
                ole_timestamp = ole_timestamp3
            elif ole_timestamp4 > 40000 and ole_timestamp4 < 50000:
                ole_timestamp = ole_timestamp4

            else:
                raise ValueError("Could not find timestamp in the LOG module")

            ole_base = datetime(1899, 12, 30, tzinfo=None)
            ole_timedelta = timedelta(days=ole_timestamp[0])
            self.timestamp = ole_base + ole_timedelta
            if self.startdate != self.timestamp.date():
                raise ValueError("Date mismatch:\n"
                                 + "    Start date: %s\n" % self.startdate
                                 + "    End date: %s\n" % self.enddate
                                 + "    Timestamp: %s\n" % self.timestamp)

    def get_flag(self, flagname):
        if flagname in self.flags_dict:
            mask, dtype = self.flags_dict[flagname]
            return np.array(self.data['flags'] & mask, dtype=dtype)
        else:
            raise AttributeError("Flag '%s' not present" % flagname)
