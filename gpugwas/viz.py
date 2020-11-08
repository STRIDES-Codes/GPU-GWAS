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

import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

# from dash.dependencies import Input, Output, State, ALL
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected = True)


EXT_STYLES = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]


class ManhattanPlot:

    def __init__(self, qq_spec, manhattan_spec, fig_path=None):
        self.app = dash.Dash( __name__, external_stylesheets=EXT_STYLES)

        self.qq_spec = qq_spec
        self.manhattan_spec = manhattan_spec

        self.fig_path = fig_path

        self.app.layout, self.manhattan_figure = self._construct()

    def start(self, host=None, port=5000):
        return self.app.run_server(
            debug=False, use_reloader=False, host=host, port=port)

    def _construct_qq(self):

        x_values = self.qq_spec['df'][self.qq_spec['x_axis']]
        y_values = self.qq_spec['df'][self.qq_spec['y_axis']]

        x_max = float(x_values.max())
        y_max = float(y_values.max())

        scatter_marker = go.Scattergl({
                'x': self.qq_spec['df'][self.qq_spec['x_axis']].to_array(),
                'y': y_values.to_array(),
                'mode': 'markers',
                'marker': {
                    'size': 2,
                    'color': '#406278',
                },
            })

        scatter_line = go.Scattergl({
                'x': [0, x_max],
                'y': [0, y_max],
                'mode': 'lines',
                'line': {
                    'width': 2,
                    'color': 'orange',
                },
            })

        scatter_fig = go.Figure(
            data = [scatter_marker, scatter_line],
            layout = {
                'title': 'Q-Q Plot',
                'showlegend': False,
                'grid': {
                    'columns' : 1,
                },
            })
        return scatter_fig

    def _construct_manhatten(self):
        chroms = self.manhattan_spec['df'][self.manhattan_spec['group_by']].unique().to_array()

        start_position = -0.5
        scatter_traces = []

        for chrom in chroms:
            query = '%s == %s' % (self.manhattan_spec['group_by'], chrom)
            cdf = self.manhattan_spec['df'].query(query)

            x_array = cdf[self.manhattan_spec['x_axis']] + start_position 
            scatter_trace = go.Scattergl({
                    'x': x_array.to_array(),
                    'y': cdf[self.manhattan_spec['y_axis']].to_array(),
                    'name': 'Chromosome ' + str(chrom),
                    'mode': 'markers',
                    'marker': {
                        'size': 2,
                        'color': '#406278' if (start_position - 0.5) % 2 == 0 else '#e32636',
                    },
                })
            scatter_traces.append(scatter_trace)
            start_position += 1

        manhattan_fig = go.Figure(
            data = scatter_traces, 
            layout = {
                'title': 'GWAS Manhattan Plot',
                'showlegend': False,
                'grid': {
                    'columns' : 1,
                },
                'xaxis': {
                    'showgrid': False, 
                    'gridwidth': 1, 
                    'ticks': 'outside',
                    'zeroline': False,
                    'tickvals': [t for t in range(int(start_position + 0.5))],
                    'ticktext': [str(t) for t in chroms],
                }})
        # plotly.offline.iplot({ "data": manhattan_fig, "layout": go.Layout(title="Sine wave")})
        return manhattan_fig

    def _construct(self):
        manhattan_fig = self._construct_manhatten()
        # qq_plot_fig = self._construct_qq()

        if self.fig_path:
            manhattan_fig.write_html(
                os.path.join(self.fig_path, "manhattan.html"))
            #qq_plot_fig.write_html(
            #    os.path.join(self.fig_path, "qq_plot.html"))

        layout = html.Div([
            html.Div(
                children=[
                    dcc.Markdown(
                        """
                        **GWAS**
                        """), 
                    # html.Div([dcc.Graph(id='qq_plot_fig', figure=qq_plot_fig),]),
                    html.Div([dcc.Graph(id='manhattan-figure', figure=manhattan_fig),]),
                ]),
            ])

        return layout, manhattan_fig#, qq_plot_fig


def main():
    df = cudf.read_csv('./data/data.csv')

    qq_spec = {}
    qq_spec['df'] = df
    qq_spec['x_axis'] = 'P'
    qq_spec['y_axis'] = 'ZSCORE'

    manhattan_spec = {}
    manhattan_spec['df'] = df
    manhattan_spec['group_by'] = 'CHR'
    manhattan_spec['x_axis'] = 'P'
    manhattan_spec['y_axis'] = 'ZSCORE'

    fig_path = None

    plot = ManhattanPlot(qq_spec, manhattan_spec, fig_path)
    plot.start()


if __name__=='__main__':
    main()
