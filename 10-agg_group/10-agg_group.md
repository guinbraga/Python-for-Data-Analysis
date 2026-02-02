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

# 10 - Data Aggregation and Group Operations



```python
import pandas as pd
import numpy as np
```

## 10.1 - How to think about group operations

.

### Introduction

groupby operations are usually described as dividing
by a key, applying a function, and then combining the 
result of such function. We'll get started with this 
tabular dataset:

```python
df = pd.DataFrame({"key1" : ["a", "a", None, "b", "b", "a", None],
                    "key2" : pd.Series([1, 2, 1, 2, 1, None, 1], dtype="Int64"), 
                    "data1" : np.random.standard_normal(7),  
                    "data2" : np.random.standard_normal(7)})
df
```

Suppose we want to compute some statistics over data1
column for key1 groups. One way to do so is to call
`groupby()` method over that column:

```python
grouped = df['data1'].groupby(df['key1'])
grouped
```

This `SeriesGroupBy` object can now be used to 
calculate some statistics, such as the mean:

```python
grouped.mean()
```

If we instead pass multiple keys to `groupby()`, we'll
end up with a hierarchical index:

```python
means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means
```

```python
means.unstack()
```

The key arrays here are Series, but they need not be so. We
could use numpy arrays, lists, etc, as long as they have the
right length.

If the grouping array is in the same DataFrame we're working on, 
we can use it's column names as groupby keys:

```python
df.groupby('key1').mean()
```

```python
df.groupby('key2').mean(numeric_only=True)
```

```python
df.groupby(['key1', 'key2']).mean()
```

One useful groupby method is `size()`, which returns the
size of each group:

```python
df.groupby(['key1', 'key2']).size()
```

Note that NA values were dropped by default, but can be included:

```python
df.groupby(['key1', 'key2'], dropna=False).size()
```

A similar function is `count()`, but it only
counts non-null values:

```python
df.groupby('key1', dropna=False).count()
```

### Iterating over groups

We can iterate over the object return by groupby.
Each iteration contains a 2-tuple with the group
name and the chunk of data:

```python
for name, group in df.groupby('key1'):
  print(name)
  print(group)
  print('\n')
```

In the case of multiple keys, the first element
in each tuple is itself a tuple of key values:

```python
for keys, group in df.groupby(['key1', 'key2']):
  print(keys)
  print(group)
  print('\n')
```

One useful recipe here is to compute a dictionary of
labels, data with a one-liner:

```python
data_dict = {name: data for name, data in df.groupby('key1')}
data_dict['b']
```

To groupby the columns axis, it is standard practice to transpose
the array:
group with a dictionary based on columns that start with 'key'
and columns that start with 'data':

```python
grouped = df.T.groupby({'key1':'key', 'key2':'key',
                      'data1':'data', 'data2': 'data'})
                     
for key, value in grouped:
  print(key)
  print(value.T)
```

### Selecting a Column or a Subset of Columns

We can select only a subset of columns to
aggregate and do groupby operations. It is 
common to do so by indexing the groupby object:

```python
df.groupby('key1')[['data1', 'data2']].mean()
```

The object returned is a Series if only a column
is selected, or a DataFrame if a list of columns
(that could be len == 1) is passed.

### Grouping with Dictionaries and Series

Consider the following DataFrame:

```python
people = pd.DataFrame(np.random.standard_normal((5, 5)),
                      columns=["a", "b", "c", "d", "e"],
                      index=["Joe", "Steve", "Wanda", "Jill", "Trey"])
people.iloc[2, [1,2]] = np.nan
people
```

Suppose we have a correspondence for each column, e.g.
they are related to the colors red and blue, and we want
to compute statistics for red and blue groups. We could use
a dictionary for correspondence and use it in the groupby:

```python
mapping = {'a':'red', 'b':'red', 'c':'blue', 'd':'blue',
           'e':'red', 'f':'orange'} 

by_column = people.T.groupby(mapping)

by_column.sum().T
```

We can also do that with a Series, which could
be seen as a fixed-sized mapping:

```python
map_series = pd.Series(mapping)
map_series
```

```python
people.T.groupby(map_series).count().T 
```

### Grouping with Functions

We can also pass a function to groupby, which will
apply itself to each key value and use its return
as grouby key. Suppose we want to compute the minimum
value for each first name initial:

```python
people.groupby(lambda x: x[0]).min()
```

As everything is converted to arrays internally, it's
okay to pass a function and an array or series or dict
in the array of keys:

```python
group_number = ['one', 'one', 'two', 'two', 'one']
people.groupby([lambda x: x[0], group_number]).min()
```

### Grouping by Index Levels

In hierarchically indexed objects, we can group by a 
certain index level. To do so, we pass the `level=`
argument with the level name. Consider the array:

```python
columns = pd.MultiIndex.from_arrays([["US", "US", "US", "JP", "JP"], 
                                     [1, 3, 5, 1, 3]],  
                                     names=["cty", "tenor"])
hier_df = pd.DataFrame(np.random.standard_normal((4, 5)),
                       columns=columns)
hier_df
```

```python
hier_df.T.groupby(level='cty').count().T
```

### Partial Summary:

- `df.groupby([keys])` returns a groupby object which can compute statistics about such groups
    - To compute the mean, remember to use `numeric_only=True`
    - NA values are dropped by default
- We can iterate over groups. Each iteration contains the group name and data in a tuple.
- Selecting a column or subset is done by indexing the `groupby` object.
- We can groupby passing `dicts` or `series` with correspondence to be grouped by.
- Functions will also work, as long as they return the key label.
- We can also group by index level with `groupby(level=levelname)`

## 10.2 Data Aggregation
. 
### Introduction 

Data Aggregation refers simply to the process of producing
a scalar value from an array of values, such as when we used
`mean()`, `sum()`, and many others.

Those used before and some other aggregations are optimized
for groupby operations, but we can use some others supported
by our object of interest, even though they will by unoptimized.
For example, `nsmallest(n)` method for series can be called on
each groupby piece:

```python
df
```

```python
grouped = df.groupby('key1')
grouped['data1'].nsmallest(2)
```

We can pass any custom function that returns a value from an
array as an groupby aggregation in the `.agg(function)` method:

```python
def square_sum(array):
  sum = 0
  for item in array:
    sum += item**2
  return sum
grouped['data2'].agg(square_sum)
```

Some methods work even though they aren't technically aggregations,
like `describe()`:

```python
grouped.describe()
```

### Column-Wise and Multiple Function Application

Let's bring back the tipping dataset and add that 
`tip_pct` column from earlier:

```python
tips = pd.read_csv('../pydata-book/examples/tips.csv')
tips.head(3)
```

```python
tips['tip_pct'] = tips['tip']/tips['total_bill'] 
```

The lesson here will be that we can use different
aggregations for each columns or multiple aggregation
functions at once. This will be illustrated in the 
following examples:

First, we'll aggregate by 'day' and 'smoker':

```python
grouped = tips.groupby(['day', 'smoker'])
```

Then, we'll select the tip_pct column:

```python
grouped_pct = grouped['tip_pct']
```

We can call a standard aggregation with its method
or its name as a string within `.agg()`:

```python
grouped_pct.agg('mean')
```

If we pass a list of functions or function names, we'll
instead get a DataFrame with column names taken from the
functions:

```python
grouped_pct.agg(['mean', 'std', 'max', square_sum])
```

If instead we pass tuples of `(name, function)`, the
name will be used for the column names:

```python
grouped_pct.agg([('average', 'mean'), ('sdeviation', 'std')])
```

When working with a DataFrame we have more options. First,
we'll explore passing the same functions for different
column:

```python
functions = ['mean', 'count', 'max']

result = grouped[['total_bill', 'tip_pct']].agg(functions)
result
```

To apply different functions to each column, we pass a dict
`column_name: function`:

```python
grouped.agg({'total_bill': 'mean', 'tip': 'sum'})
```

```python
grouped.agg({'tip_pct': ['min', 'max', 'mean', 'std'], 'size': 'mean'})
```

Hierarchical columns will be created only if at least one group
has more than one aggregation function.

### Returning Aggregated Data Without Row Indexes

If we want the row indexes as columns instead, and want
to avoid unnecessary computations of calling `reset_index()`,
we can just use the parameter `as_index=False`:

```python
tips.groupby(['day', 'smoker'], as_index=False).mean(numeric_only=True)
```

### Partial Summary

- Aggregation of any function can be made on `groupby` objects with `.agg(function)`
    - We can aggregate with a list of functions or a dict `{'column_name': [functions]}`
    - We can use custom column names by passing `('name', function)` tuple
    - To avoid row indexing and lessening computing, we can pass `as_index=False` to `groupby()`

## 10.3 Apply: General split-apply-combine

This section concerns itself with the `apply()` method,
considered the most general one of the `groupby` object.
It splits the object being handled into pieces, invokes
the passed function on each piece, and then attempts to
concatenate the pieces.

As an example, we'll return to the tipping dataset from
before, and try to select the top five `tip_pct` of each group.

### Apply Presentation

First, we'll write a function that selects the largest values
in a given column:
```python
def top(df, n=5, column='tip_pct'):
  return df.sort_values(column, ascending=False)[:n]

top(tips)
```

Now if we group by smoker and call apply with this function:

```python
tips.groupby('smoker').apply(top)
```

The apply function applies and combines.
The inner indexes are from the original index values.

If our function has more parameters, we can pass then 
comma-separated after the function name:

```python
tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')
```

What we can do with `apply` is pretty diverse and limited by
our creativity. The function must only return a scalar or 
pandas object.

The rest of the chapter will consist mainly of examples of
problems solved by groupby:

### Suppressing the Group Keys

On a quick note, we can `reset_index()` of the group
with `groupby(group_keys=False)` 

### Quantile and Bucket Analysis

Using pandas `cut` and `qcut` with groupby makes it convenient
to do bucket analysis on different categories of a dataset.
Consider a simple random dataset and an equal-length bucket 
categorization using pandas.cut:

```python
frame = pd.DataFrame({'data1': np.random.standard_normal(1000),
                      'data2': np.random.standard_normal(1000)})

frame.head()
```

```python
quartiles = pd.cut(frame['data1'], 4)
quartiles.head(10)
```

This Categorical object created can be passed to groupby.
Once we've grouped by quartile, we can apply functions to it:

```python
grouped = frame.groupby(quartiles)

def get_stats(group):
  return pd.DataFrame(
    {'min': group.min(), 'max': group.max(),
     'count': group.count(), 'mean': group.mean()}
  )

grouped.apply(get_stats)
```

Dealing with equal-sized buckets with `qcut` may be
cleaner if we pass `(labels=false)` to qcut:

```python
quatiles_samp = pd.qcut(frame['data1'], 4, labels=False)

frame.groupby(quatiles_samp).apply(get_stats)
```

### Example: Filling Missing Values with Group-Specific Values:

When cleaning missing data, if we wish to fill NA
values, `fillna()` is the method to use. But we may
want to use a different value or function with `fillna`
depending on the group we're filling for. Consider the
DataFrame:

```python
states = ["Ohio", "New York", "Vermont", "Florida", 
          "Oregon", "Nevada", "California", "Idaho"] 

group_key = ["East", "East", "East", "East", 
            "West", "West", "West", "West"] 

data = pd.Series(np.random.standard_normal(8), index=states)

data
```

```python
data[['Vermont', 'Nevada', 'Idaho']] = np.nan
data
```

```python
data.groupby(group_key).size()
```

Suppose we want to fill NA values for the East
with the 0 and to the West with the 1.
We could have a dict indicating that:

```python
fill_funcs = {'East': 0, 'West': 1}

def fill_group(group, fill_funcs):
  return group.fillna(fill_funcs[group.name])
  
data.groupby(group_key).apply(fill_group, fill_funcs)
```

### Example: Random Sampling and Permutation

Suppose we want to draw a random sample from a large
dataset for any purpose. There are many ways to do 'draws'.
The `sample` method for a Series will be used for the example.

First, we'll construct a deck of playing cards:

```python
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2, 11)) + ['J', 'Q', 'K']
cards = []
for suit in suits:
  cards.extend(str(num) + suit for num in base_names)
deck = pd.Series(card_val, index=cards)
```

This results in a Series with all combinations of cards:

```python
deck.head(13)
```

Drawing a hand of 5 cards is as simple as deck.sample(5)

```python
deck.sample(5)
```

But we'll create a function to apply with n cards
and draw 2 cards per suit. To do that, we'll extract
the suit by the last letter:

```python
def draw(deck, n=5):
  return deck.sample(n)

def get_suit(card):
  return card[-1]

deck.groupby(get_suit, group_keys=False).apply(draw, n=2)
```

### Example: Group Weighted Average and Correlation

Under the split-apply-combine paradigm of groupby,
operations between two different columns or arrays are
possible. Consider the weighted average problem, where
we have an array of weights and data:

```python
df = pd.DataFrame({"category": ["a", "a", "a", "a",
                                "b", "b", "b", "b"],
                   "data": np.random.standard_normal(8),
                   "weights": np.random.uniform(size=8)})
df
```

```python
def get_wavg(frame):
  return np.average(frame['data'], weights=frame['weights'])

df.groupby('category').apply(get_wavg)
```

As another example, the author uses a financial dataset
obtained from Yahoo! Finance:

```python
close_px = pd.read_csv('../pydata-book/examples/stock_px.csv', parse_dates=True,
                       index_col=0)
close_px.tail(4)
```

First it defines a function to compute the correlation
of the stock with the correlation to SPX:

```python
def spx_corr(group):
  return group.corrwith(group['SPX'])
```

Then it computes percent change:

```python
rets = close_px.pct_change().dropna()
```

Then it defines a function to extract the year
from the datetime label, so it can group per year:

```python
def get_year(x):
  return x.year

by_year = rets.groupby(get_year)

by_year.apply(spx_corr)
```

It also computes intercolumn correlations:

```python
def corr_apple_msft(group):
  return group['AAPL'].corr(group['MSFT'])

by_year.apply(corr_apple_msft)
```

```python
close_px.info()
```


### Example: Group-Wise Linear Regresion

Here the author uses the `statsmodels.api` to define
a function that returns the beta and intercept 
values for each group:

```python
import statsmodels.api as sm 
def regress(data, yvar=None, xvars=None):
  Y = data[yvar]  
  X = data[xvars] 
  X["intercept"] = 1.
  result = sm.OLS(Y, X).fit()  
  return result.params
```

```python
by_year.apply(regress, yvar='AAPL', xvars=['SPX'])
```

### Summary

- `apply` is a generalization of `agg` and can apply and then concat any function possible we create or use on each group. It can:
    - return n values for each group
    - pass function parameters after a comma
- we can groupby `Categoricals`, such as those returned by `pd.cut` and `pd.qcut`
- We can fill with different values for different groups by defining a function that uses a dict mapping
- We can sample by group and return the values sampled, as `apply` will concatenate them;
- We can compute statistics across different columns, all we need is to write the function

## 10.4 Group Transforms and "Unwrapped" GroupBys

Transforms are a more constrained version of `apply`.
They return objects of the same shape of the input, 
and do not alter the input. Here's an example:

```python
n = 100002
df = pd.DataFrame({'keys': ['a','b','c'] * int(n/3),
                   'values': np.arange(n)})
df
```

Suppose we want to create a new array with the
same size of this one, but with the `mean()` of
each group. We can do that with `transform()`:

```python
def get_mean(group):
  return group.mean()

by_key = df.groupby('keys')
by_key.transform(get_mean)
```

For built-in aggregation, we can use the function
name:

```python
by_key.transform('median')
```

For a more complex example, we can create the rank
of each value within it's group in descending order:

```python
def get_rank(group):
  return group.rank(ascending=False)

by_key.transform(get_rank)
```

Consider this normalize function:

```python
def normalize(x):
  return (x-x.mean())/x.std()

by_key.transform(normalize)
```

```python
%%timeit
by_key.transform(normalize)
```

```python
%%timeit
by_key.apply(normalize)
```

```python
df = df.reset_index(drop=True)
```

```python
%%timeit
(df['values'] - by_key['values'].transform('mean')) / by_key['values'].transform('std')
```

In general, although the syntax is more convoluted, it
is more efficient to do arithmetic between different
groupby operation outputs than to apply the functions 
to each value using `apply` or `transform`.

### Summary:

- `groupby.transform(func)` returns a pandas object of the same shape
- `arithmetic` between groupbys outputs is more efficient than `applying` or `transforming` each element. 

## 10.5 Pivot Tables and Cross-Tabulation

.

### Pivot Tables

Pivot tables are a top-level `pd.pivot_table` function
and dataframe method for creating tables of aggregations
for comparing aggvalues between different groups. Take the
tips dataset as example:

```python
tips.pivot_table(index=['time', 'smoker'], columns='day',
                 values=['tip_pct', 'total_bill'], aggfunc='sum', margins=True)
```

```python
tips.pivot_table(index=['time', 'smoker'], columns='day',
                 values=['tip_pct', 'total_bill'], aggfunc='sum', 
                 fill_value=0, margins=True, margins_name='Total')
```

As demonstrated, we can also use fill_values.

### Cross-Tabulations: Crosstab

`pd.crosstab(index, column)` creates a crosstab similar
to `pivot_table`, with counts as default `aggfunc`.
