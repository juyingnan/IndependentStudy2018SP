import csv
import os
import pandas as pd
from collections import OrderedDict
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import HoverTool


def read_csv(path):
    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        _headers = next(reader, None)

        # get column
        _columns = {}
        for h in _headers:
            _columns[h] = []
        for row in reader:
            for h, v in zip(_headers, row):
                _columns[h].append(v)

        for item in _headers:
            print(item)

        return _headers, _columns


tools_list = "pan," \
             "box_select," \
             "lasso_select," \
             "box_zoom, " \
             "wheel_zoom," \
             "reset," \
             "save," \
             "help"
# "hover," \

root_path = r'../csv'
nuclei_file_name = 'nuclei.csv'
vessel_file_name = 'vessel.csv'
nuclei_file_path = os.path.join(root_path, nuclei_file_name)
vessel_file_path = os.path.join(root_path, vessel_file_name)

output_file("result/distance.html")

drug_color = OrderedDict([
    ("Item Category", "#0d33ff"),
    ("Item metadata", "#c64737"),
    ("User profile", "black"),
])

gram_color = OrderedDict([
    ("negative", "#e69584"),
    ("positive", "#aeaeb8"),
])

p = figure(plot_width=1000, plot_height=1000, title='nuclei/vessel distance', match_aspect=True,
           tools=tools_list,

           )

v_headers, v_columns = read_csv(vessel_file_path)
for i in range(0, len(v_headers)):
    v_columns[v_headers[i]] = [float(value) for value in v_columns[v_headers[i]]]
v_columns['y'] = [(1400 - float(value)) for value in v_columns['y']]

data = dict()
for header in v_headers:
    data[header] = v_columns[header]
v_df = pd.DataFrame(data)

p.circle(x='x', y='y', source=v_df, color='orange', size=1, alpha=0.1)

n_headers, n_columns = read_csv(nuclei_file_path)
for i in range(1, len(n_headers)):
    n_columns[n_headers[i]] = [float(value) for value in n_columns[n_headers[i]]]
n_columns['y'] = [(1400 - float(value)) for value in n_columns['y']]
n_columns['vy'] = [(1400 - float(value)) for value in n_columns['vy']]
n_columns['alpha'] = [(float(value) + 10) / 100 for value in n_columns['distance']]

data = dict()
for header in n_headers:
    data[header] = n_columns[header]
    data['alpha'] = n_columns['alpha']
n_df = pd.DataFrame(data)

p.segment(x0='x', y0='y', x1='vx', source=n_df, y1='vy', color="#CAB2D6", line_width=1)
circle = p.circle(x='x', y='y', source=n_df, color='blue', alpha='alpha', size=3)
g1_hover = HoverTool(renderers=[circle], tooltips=[('X', "@x"), ('Y', "@y"), ('distance', "@distance")])
p.add_tools(g1_hover)

# p.legend.location = "bottom_right"

show(p)
