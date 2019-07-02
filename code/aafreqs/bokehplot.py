import numpy as np
import pandas as pd

from bokeh.plotting import figure, show, output_file

df = pd.read_csv('data.csv', index_col=0)
df['fold'] = df['freq_pathogen']/df['freq_human']

#df = df[df['fold']>10.0]

df['freq_human'] = np.log10(df['freq_human'])
df['freq_pathogen'] = np.log10(df['freq_pathogen'])

TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

TOOLTIPS = [
    ("(f_human, f_pathogen)", "($x, $y)"),
    ("peptide", "@seq"),
]

p = figure(tools=TOOLS, tooltips=TOOLTIPS)

p.scatter('freq_human', 'freq_pathogen', source=df, radius=0.02,
          fill_alpha=0.6,
          line_color=None)

output_file("color_scatter.html", title="color_scatter.py example")

show(p)  # open a browser

