#!/opt/conda/envs/rapids/bin/python
"""
Pre-reqs:
---------
/opt/conda/envs/rapids/bin/pip install \
    dash \
    jupyter-dash \
    dash_bootstrap_components \
    dash_core_components \
    dash_html_components
"""

import os
import cudf
import cupy

from bokeh.plotting import figure
from bokeh.io import output_notebook, push_notebook, show

output_notebook()


def show_qq_plot(df, x_axis, y_axis):

    x_values = cupy.fromDlpack(df[x_axis].to_dlpack())
    y_values = cupy.fromDlpack(df[y_axis].to_dlpack())

    x_max = cupy.max(x_values).tolist()
    y_max = cupy.max(y_values).tolist()

    qq_fig = figure(x_range=(0, x_max), 
                    y_range=(0, y_max))
    qq_fig.circle(-cupy.log10(x_values+1e-10).get(), -cupy.log10(y_values).get())
    qq_fig.line([0, x_max], [0, y_max])

    qq_handle = show(qq_fig, notebook_handle=True)
    push_notebook(handle=qq_handle)
    return qq_fig


def show_manhattan_plot(df, group_by, x_axis, y_axis):
    chroms = df[group_by].unique().to_array()

    manhattan_fig = figure()

    start_position = -0.5
    for chrom in chroms:
        query = '%s == %s' % (group_by, chrom)
        cdf = df.query(query)

        x_array = cupy.fromDlpack(cdf[x_axis].to_dlpack())  + start_position
        y_array = cupy.fromDlpack(cdf[y_axis].to_dlpack())

        manhattan_fig.circle(
            x_array.get(),
            y_array.get(),
            size=2, color='orange' if (start_position - 0.5) % 2 == 0 else 'gray', alpha=0.5)

        start_position += 1

    manhattan_handle = show(manhattan_fig, notebook_handle=True)
    push_notebook(handle=manhattan_handle)
    return manhattan_fig