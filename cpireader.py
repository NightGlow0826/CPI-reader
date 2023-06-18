#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : csvreader.py
@Author  : Gan Yuyang
@Time    : 2023/6/4 19:59
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import pyecharts
import pyecharts.options as opts
from pyecharts.charts import Line
import ydata_profiling as pp
import streamlit as st
from streamlit_echarts import st_pyecharts
import streamlit.components.v1 as components

matplotlib.get_backend()
df = pd.read_csv('cpi.csv', encoding='utf-8', index_col=0)
print(df.columns)

t = df['REPORT_DATE'].tolist()


base_ = ['NATIONAL_BASE','CITY_BASE', 'RURAL_BASE']
seq_ = ['NATIONAL_SEQUENTIAL', 'CITY_SEQUENTIAL', 'RURAL_SEQUENTIAL']
same_ = ['NATIONAL_SAME', 'CITY_SAME', 'RURAL_SAME']





def linechart(lst):
    c = (
        Line()
            .set_global_opts(
            datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),

            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
                is_scale=True,
            ),
        )
            .add_xaxis(xaxis_data=t)
        # .render("a.html")
    )
    for i in lst:
        c.add_yaxis(
            series_name=i,
            y_axis=df[i].tolist(),
            is_symbol_show=True,
            label_opts=opts.LabelOpts(is_show=False),
        )

    return c

# c.render('a.html')

# city_seq = np.array(df['CITY_SEQUENTIAL'].tolist())/100
# city_same = np.array(df['CITY_SAME'].tolist())/100
# rural_seq = np.array(df['RURAL_SEQUENTIAL'].tolist())/100
# rural_same = np.array(df['RURAL_SAME'].tolist())/100
# print(city)

def report_generate():
    report = pp.ProfileReport(df)
    preview_path = 'preview.html'
    report.to_file(preview_path)
    webbrowser.open_new_tab(preview_path)

# report_generate()

def charts():
    for type_ in [base_, seq_, same_]:
        base_chart = components.html(linechart(type_).render_embed(), height=500, width=1000,)

if __name__ == '__main__':

    charts()