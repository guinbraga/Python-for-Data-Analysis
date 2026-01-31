---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
---

# 9 - Plotting and Visualization



```python
from matplotlib.lines import lineStyles
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Increase default figure size (width, height in inches)
plt.rcParams['figure.figsize'] = [12, 8] 
# Increase DPI (dots per inch) for sharper, larger images on high-res screens
plt.rcParams['figure.dpi'] = 150
plt.rc('axes', titlesize=20, titleweight='bold', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
```

## 9.1 A Brief matplotlib API Primer

```python
data = np.arange(10)
data
```

```python
plt.plot(data)
```

### Figures and Subplots

Plots reside within figures. A figure is a blank space
until a subplot is added:

```python
fig = plt.figure()
fig
```

`.add_subplot()` divides our figure in the first 2 arguments
and selects the third. Below, we're creating a 2x2 figure and 
assigning the variable to the first panel. 

```python
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
```

```python
fig
```
We usually create plots within the axis objects, as it is 
preferable to do so than using the top-level methods such
as `plt.plot()`. Here we'll create a line plot with the 
`axis.plot()` method:

```python
ax3.plot(np.random.standard_normal(50).cumsum(), linestyle='dashed', color='black')
fig
```

The objects returned by the `add_subplot()` method are 
`AxesSubplot` objects, on which we can plot on the other
empty subplots:

```python
ax1.hist(np.random.standard_normal(100), bins=20, color='black', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.standard_normal(30))
fig
```

`alpha=0.3` sets the transparency of the plot.
We can create lots of subplots with a single code line
with `fig, axes = plt.subplots(n, m)`, with n = rows and m = columns.
The object return is an array which can be indexed with
`array[0, 1]`

We can indicate that the subplots in the array must share
the same x or y axis with `sharex` and `sharey`, respectively.

#### Adjusting the spacing around subplots

We can adjust the spacing with the figure method `subplots_adjust(
left = None, bottom = None, right = None, top = None, wspace=None, 
hspace=None)`. 

`wspace` and `hspace` refer to the width and height, respectively, 
between subplots.

```python
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
  for j in range(2):
    axes[i, j].hist(np.random.standard_normal(500), bins=50,
                    color='black', alpha=0.5)
fig.subplots_adjust(wspace=0, hspace=0)
fig
```

### Colors, Markers, and Line Styles

The line `plot()` function can plot x and y arrays as
coordinates, and accepts arguments for line style and
color. To plot x and y with a dashed green line we can
write:

```python
ax.plot(x, y, linestyle='--', color='green')
```

Colors can be specified with their HexCode also, and
linestyles can be found in the online documentation 
or in `plt.plot?`.

We can use **markers** to highlight datapoints:

```python
fig = plt.figure()
ax = fig.add_subplot()

ax.plot(np.random.standard_normal(30).cumsum(), linestyle='--',
        color='green', marker='o')
fig
```

With `drawstyle` we can change the way the line is drawn
between points, instead of the linear default. Look at the
comparison with `steps-post`:

```python
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.random.standard_normal(30).cumsum(), color='green',
        linestyle='--', label='Default')
ax.plot(np.random.standard_normal(30).cumsum(), color='black',
        linestyle='--', drawstyle='steps-post', label='steps-post')
ax.legend()
fig
```

Here, because we passed a `label` argument, we can call
`ax.legend()` to create a legend for our plot. Which leads
us to...

### Ticks, Labels and Legends

Most plot decorations can be achieved with the axis methods.
Methods such as `xlim()`, `xticks()`, and `xticklabels()` allow
you to get the current value (when called with no parameters) or
set new values (when called with parameters)

To illustrate decorations, let's use this graph of a stock price
(and definitely not a random walk):

```python
fig, ax = plt.subplots()

ax.plot(50+(np.random.standard_normal(1000)).cumsum())
```

We'll set the ticks and ticklabels with `set_xticks()` and
`set_xticklabels`. By default whatever we put in `set_xticks`
becomes the labels, but we'll do a custom example:

```python
ticks = ax.set_xticks([0, 250, 500, 750, 1000])

labels = ax.set_xticklabels(['01/24', '04/24', '07/24', '10/24', '01/25'],
                   rotation=30, fontsize=12)
fig
```

Rotation rotates in *n* degrees. Lastly, we set an x label with
`set_xlabel` and a title with `set_title`.

```python
ax.set_xlabel('Quarters', fontsize=16)
ax.set_title('Not a Stock plot', fontsize=24)
fig
```
Modifying the y axis is the same process. The axes object has
a `set()` method that allows us to batch-set all this stuff:

```python
ax.set(ylabel='Dollars', title='not a Stock plot')
fig
```

#### Adding Legends

There are a couple of ways to add a legend. One, which we've
already seen, is to pass a `label` parameter to each plot piece
we create:

```python
fig, ax = plt.subplots()
ax.plot(np.random.randn(1000).cumsum(), color='black', label='one')
ax.plot(np.random.randn(1000).cumsum(), color='green', linestyle='dashed', label='two')
ax.plot(np.random.randn(1000).cumsum(), color='red', linestyle='dotted', label='three')
```

```python
ax.legend()
fig
```

The legend location is defined by the `loc=` parameter, which
by default is 'best', the location most out of the way.

If we don't want a particular plot to have a legend, we either
pass no label or `label=_nolegend_`

### Annotations and Drawing on a Subplot

We can add text, arrows, and other elements to plots as a way
to annotate important information. For example, we can add text
at a x, y coordinate with `ax.text(x, y, 'Hello World!', family=monospace, fontsize=12)`

We'll do an example with a S&P 500 dataset starting in 2007 
(or is it?):

```python
from datetime import datetime
fig, ax = plt.subplots()

data = pd.read_csv('../pydata-book/examples/spx.csv', index_col=0, parse_dates=True)
spx = data['SPX']

spx.plot(ax=ax, color='black')

crisis_data = [
  (datetime(2007, 10, 11), "Peak of bull market"),
  (datetime(2008, 3, 12), "Bear Stearns Fails"),
  (datetime(2008, 9, 15), "Lehman Bankruptcy")
]

for date, label in crisis_data:
  ax.annotate(label, xy=(date, spx.asof(date) + 75),
              xytext=(date, spx.asof(date) + 225),
              arrowprops=dict(facecolor='black', headwidth=8, width=2, headlength=8),
              horizontalalignment='left', verticalalignment='top', fontsize=16)

  # Zoom in on 2007-2010
  ax.set_xlim(['1/1/2007', '1/1/2011'])
  ax.set_ylim([600, 1800])

  ax.set_title("Important dates in the 2008-2009 financial crisis", fontsize=24)
```

To note from this plot: 

The `annotate(label, xy)` can annotate labels on xy coordinates.

We can also add shapes, mostly known as `patches`. we do so by 
creating the patches with `plt.Circle()` or `plt.Rectangle()`, 
for example, and adding them to the plot with `ax.add_patch(patch)`:

```python
fig, ax = plt.subplots() 
rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color="black", alpha=0.3) 
circ = plt.Circle((0.7, 0.2), 0.15, color="blue", alpha=0.3) 
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],  
                   color="green", alpha=0.5) 
ax.add_patch(rect) 
ax.add_patch(circ) 
ax.add_patch(pgon)
```
### Saving Plots to a File

We can save figures to a file with `fig.savefig('file.ext')`
The file type is inferred from the extension, and can be
SVG, PNG, PDF... We can specify a dpi with `dpi` argument.

The `facecolor`, `edgecolor` parameters define the color for
the background and edges of the plots.
