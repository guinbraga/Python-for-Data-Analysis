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

# 8 - Data Wrangling: Join, Combine, and Reshape


```python
import pandas as pd
import numpy as np
```
## 8.1 Hierarchical Indexing

### MultiIndexing

Hierarchical indexing refers to pandas ability to have objects
indexed in more than one level: Here's some ways of accessing these indices:

```python
import pandas as pd
import numpy as np
```

```python
data = pd.Series(np.random.uniform(size=9),
                index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'], [1, 2, 3, 1, 3, 1, 2, 2, 3]])
data
```

```python
data.iloc[1]
```

```python
data.loc['a', 2]
```

```python
data.loc['b':'d']
```

```python
data.loc[['b', 'd']]
```

```python
data.loc[:, 2]
```

Hierarchical indexed data can be rearranged into more dimensions
when doing operations. This example can be `unstack()`ed in a DataFrame:

```python
data.unstack()
```

```python
data.unstack().stack()
```

DataFrames also can have multilevel indexing in columns and index:

```python
frame = pd.DataFrame(np.arange(12).reshape(4,3),
                     index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                     columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
frame
```

```python
frame.loc[('a', 2), ('Ohio', 'Red')]
```

We can check the levels of an index with the `nlevel` attribute:

```python
frame.index.nlevels
```

```python
frame.columns.nlevels
```

We can name the indices:

```python
frame.columns.names = ['states', 'colors']
frame.index.names=['key1', 'key2']
frame
```

Finally, we can create the MultiIndex by itself and then reuse it:

```python
pd.MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']], names=['states', 'colors'])
```

### Reordering and Sorting Levels

```python
frame.swaplevel('key1', 'key2')
```

```python
frame.swaplevel(0, 1).sort_index(level=0)
```

```python
frame.columns.swaplevel(0, 1)
```

Data selection is much better on hierarchically sorted indexed objects if the index is lexicographically sorted with the outer most level 
That is, use `sort_index()`!


### Summary statistics by level

We can compute summary statistics by level, for example, with the
`groupby(level=n)` method:

```python
frame.groupby(level='key2').sum()
```

```python
frame.groupby(level='colors', axis='columns').mean()
```

```python
dataset = pd.DataFrame({
    'Sexo': ['H', 'M', 'M', 'M', 'M', 'H', 'H', 'H', 'M', 'M'],
    'Idade': [53, 72, 54, 27, 30, 40, 58, 32, 44, 51]
})
dataset.groupby('Sexo').mean()
```

```python
df = pd.DataFrame(data = {'Fulano': [8, 10, 4, 8, 6, 10, 8],
                          'Sicrano': [7.5, 8, 7, 8, 8, 8.5, 7]}, 
                  index = ['Matemática', 
                           'Português', 
                           'Inglês', 
                           'Geografia', 
                           'História', 
                           'Física', 
                           'Química'])
df.rename_axis('Matérias', axis = 'columns', inplace = True)
(df - df.mean()).abs().mean()
```

### Indexing with a DataFrame's columns

Pandas allows us to "move" columns to indices with
`df.set_index([columns])`

```python
frame = pd.DataFrame({"a": range(7), "b": range(7, 0, -1), "c": ["one", "one", "one", "two", "two", "two", "two"], "d": [0, 1, 2, 0, 1, 2, 3]})
frame
```

```python
frame2 = frame.set_index(['c', 'd'])
frame2
```

We can also do the opposite: move hierarchical
index levels to columns with `df.reset_index()`

```python
frame2.reset_index()
```

## 8.2 Combining and Merging Datasets

There are a number of different combining and merging
operations in pandas. We'll start with the famous *join*
operations from databases, here implemented in the 
`pd.merge(df1, df2)`

### Database-Style DataFrame Joins

```python
df1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "a", "b"], "data1": pd.Series(range(7), dtype="Int64")})
df2 = pd.DataFrame({"key": ["a", "b", "d"], "data2": pd.Series(range(3), dtype="Int64")})

df1
```

```python
df2
```

Here we'll do a *many-to-one* join operation, which I
recall from mapping N:1 relations in databases. In that
case, the '1' table is simply added as a column in
the N table:

```python
pd.merge(df1, df2)
```

Although we didn't specify which key to merge on, it is
good practice to do so:

```python
pd.merge(df1, df2, on='key')
```

If we need to specify different keys for different
tables, we do that with `left_on=` and `right_on=`:

```python
df3 = pd.DataFrame({"lkey": ["b", "b", "a", "c", "a", "a", "b"], "data1": pd.Series(range(7), dtype="Int64")})
df4 = pd.DataFrame({"rkey": ["a", "b", "d"], "data2": pd.Series(range(3), dtype="Int64")})

pd.merge(df3, df4, left_on='lkey', right_on='rkey')
```

The default join is an inner join, which is an intersection
and drops missing keys in both tables. We can also do outer,
left and right joins with the `how=` keyword:

```python
pd.merge(df1, df2, how='outer')
```

**Many to many** joins form the cartesian product of the
tables joined. This means that for n keys 'a' found in
table 1, and m keys 'b' found in table 2, there will be
n*m keys 'a' in the resulting table, minimum of 1.

```python
df1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"], "data1": pd.Series(range(6), dtype="Int64")})
df2 = pd.DataFrame({"key": ["a", "b", "a", "b", "d"], "data2": pd.Series(range(5), dtype="Int64")})

df1
```

```python
df2
```

```python
pd.merge(df1, df2, on='key', how='right')
```

We can merge on multiple keys with a list of column names:

```python
left = pd.DataFrame({"key1": ["foo", "foo", "bar"], 
                     "key2": ["one", "two", "one"], 
                     "lval": pd.Series([1, 2, 3], dtype='Int64')})

right = pd.DataFrame({"key1": ["foo", "foo", "bar", "bar"], 
                      "key2": ["one", "one", "one", "two"], 
                      "rval": pd.Series([4, 5, 6, 7], dtype='Int64')})

pd.merge(left, right, on=['key1', 'key2'], how='outer')
```

When merging with multiple keys, to think about which keys
will be present in the resulting DataFrame, we can think about
the pairs of keys from each table as being one single key
consisting of a tuple that's being matched against other tuples.

Lastly, when merging DataFrames that have columns with overlapping
names that are NOT the keys being merged on, pandas treats this
by adding suffixes to each dataframe (`_x` to the right, `_y` to the
left one). We can override the suffixes names with the `suffix=['_left',
'_right']` parameter.

### Merging on Index

We can merge on indexes instead of columns with the
`left_index=True` and `right_index=True` parameters.

DataFrames have a `.join(df)` method to simplify joining
on index. This is by default a `left` join, and we can
specify a column from the passed df to join onto.

```python
left1 = pd.DataFrame({"key": ["a", "b", "a", "a", "b", "c"], "value": pd.Series(range(6), dtype="Int64")})

right1 = pd.DataFrame({"group_val": [3.5, 7]}, index=["a", "b"])

left1
```

```python
right1
```

```python
left1.join(right1)
```

```python
left1.join(right1, on='key')
```

Lastly, we can pass a list of DataFrames to `join()`
as an alternative to concating lots of times.

```python
left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=["a", "c", "e"], columns=["Ohio", "Nevada"]).astype("Int64")

right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]], index=["b", "c", "d", "e"], columns=["Missouri", "Alabama"]).astype("Int64")

another = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]], index=["a", "c", "e", "f"], columns=["New York", "Oregon"])
```

```python
left2
```

```python
right2
```

```python
another
```

```python
left2.join([right2, another])
```

```python
left2.join([right2, another], how='outer')
```

### Concatenating Along an Axis

This is inspired from numpy's `concatenate()` method:

```python
arr = np.arange(12).reshape((3, 4))
arr
```

```python
np.concatenate([arr, arr])
```

```python
np.concatenate([arr, arr], axis=1)
```

In pandas we implement this functionality as well,
but considering that the data is labeled and we may
or may not want to use only values with common labels,
or identify the concatenated data in the resulting frame, 
and preserving data. Consider these Series:

```python
s1 = pd.Series([0, 1], index=["a", "b"], dtype="Int64") 
s2 = pd.Series([2, 3, 4], index=["c", "d", "e"], dtype="Int64")
s3 = pd.Series([5, 6], index=["f", "g"], dtype="Int64")
```

```python
s1
```

```python
s2
```

```python
s3
```

```python
pd.concat([s1, s2, s3])
```

We can concat over the 'columns' axis as well.
Concat is by default an outer join:

```python
pd.concat([s1, s2, s3], axis=1)
```

We choose the type of concatenation with the
`join=` method.

```python
pd.concat([s1, s2], axis=1, join='inner')
```

```python
s4 = pd.concat([s1, s3])
pd.concat([s1, s4], axis=1, join='inner')
```

We can identify which object each label came from
with the `keys=[]` parameter, which will result in
an hierarchically indexed object:

```python
result = pd.concat([s1, s1, s4], keys=['one', 'two', 'three'])
result
```

```python
result.unstack()
```

Concating over the columns with keys specified turns 
the keys into column labels:

```python
result2 = pd.concat([s1, s1, s4], axis=1, keys=['one', 'two', 'three'])
result2
```

Note that the above dataframe is equivalent to the 
unstacked version of concating over the index:

```python
result2.T
```

The result is similar when concating dataframes.
The keys will become higher level column labels,
as a way to identify from which dataframe each 
column came from:

```python
df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=["a", "b", "c"], columns=["one", "two"])
df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=["a", "c"], columns=["three", "four"])
df1
```

```python
df2
```

```python
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])
```

We can achieve the same result (but with arguably
more precision in the naming if each level) by passing
a dictionary to the concat argument. Each dict key will
be a higher level label to identify the dataframe value:

```python
pd.concat({'level1':df1, 'level2':df2}, axis=1)
```

We can name the column levels created with the 
`names=` parameter:

```python
pd.concat({'level1':df1, 'level2':df2}, axis=1, names=['higher', 'lower'])
```

Lastly, if the index from the rows does not contain
any useful data, we may discard it with the `ignore_index=True`
argument, which will reset the indexes in the resulting DataFrame

```python
df1 = pd.DataFrame(np.random.standard_normal((3, 4)),
                   columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.random.standard_normal((2, 3)),
                   columns=['b', 'd', 'a'])
df1
```

```python
df2
```

```python
pd.concat([df1, df2])
```

```python
pd.concat([df1, df2], ignore_index=True)
```

Lastly, the `verify_integrity=True` parameter
will make the concatenation fail if there are
any duplicates in the objects' indices:

```python
pd.concat([s1, s1], verify_integrity=True, axis=0)
```

```python
pd.concat([s1, s1], verify_integrity=True, axis=1)
```


### Combining Data with Overlap

Pandas has a way of using data from a DataFrame
to patch missing values from another. This is done
by the `combine_first()`, which will result in the
union of both dataframes

```python
df1 = pd.DataFrame({"a": [1., np.nan, 5., np.nan], "b": [np.nan, 2., np.nan, 6.], "c": range(2, 18, 4)})

df2 = pd.DataFrame({"a": [5., 4., np.nan, 3., 7.], "b": [np.nan, 3., 4., 6., 8.]})

df1
```

```python
df2
```

```python
df1.combine_first(df2)
```

## 8.3 Reshaping and Pivoting

These are operations for rearranging tabular data.

### Reshaping with Hierarchical Indexing

There are two primary actions for reshaping data
arranged in hierarchical indexing (or not!):
`stack()` and `unstack()`

```python
data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(["Ohio", "Colorado"], name="state"),
                    columns=pd.Index(["one", "two", "three"],
                    name="number"))
data
```

Stacking this will turn the number columns into
the inner-most layer of the index:

```python
result = data.stack()
result
```

We can unstack it:

```python
result.unstack()
```

Although by default the inner-most level is
unstacked into a column, we can choose which
level is with the `(level=)` keyword and the
int or name corresponding to the unstacked level:

```python
result.unstack(level=0)
```

```python
result.unstack(level='state')
```

Unstacking might introduce missing data if
the unstacked label aren't found in both parent
indices:

```python
s1 = pd.Series([0, 1, 2, 3], index=["a", "b", "c", "d"], dtype="Int64")
s2 = pd.Series([4, 5, 6], index=["c", "d", "e"], dtype="Int64")
data2 = pd.concat([s1, s2], keys=["one", "two"]) 

data2
```

```python
data2.unstack()
```

By default, stacking filters missing data to
make the operation more easily reversible. This
can be overridden with `stack(dropna=false)`

```python
data2.unstack().stack()
```

```python
data2.unstack().stack(dropna=False)
```

The level `unstacked` is always the lowest
level in the resulting object:

```python
result
```

```python
df = pd.DataFrame({"left": result, "right": result + 5}, columns=pd.Index(["left", "right"], name="side"))
df
```

```python
df.unstack(level='state')
```

We can also specify the level to be stacked:

```python
df.unstack(level='state').stack(future_stack=True,level='side').sort_index(axis=1)
```

### Pivoting 'Long' to 'Wide' Format

This is how to turn data stacked in a long
format into a wider format, while showing off
the author's ability to deal with time series data:

```python
data = pd.read_csv("../pydata-book/examples/macrodata.csv")
data= data.loc[:, ['year', 'quarter', 'realgdp', 'infl', 'unemp']]
data.head()
```

First he turns 'year' and 'quarter' into a 
PeriodIndex, which will be later discussed,
with `datetime` values at the end of each quarter:

```python
periods = pd.PeriodIndex(year=data.pop('year'),
                         quarter=data.pop('quarter'),
                         name='date')
periods
```

```python
data.index = periods.to_timestamp("D")
data.head()
```

Then, he selects a subset of columns and give them a name 'item'

```python
data = data.reindex(columns=['realgdp', 'infl', 'unemp'])
```

```python
data.columns.name='item'
data.head()
```

Finally, he makes the data long by stacking it,
resetting indices and renaming the column containing
the data to 'value'

```python
long_data = data.stack()
long_data
```

```python
long_data = long_data.reset_index()
long_data
```

```python
long_data = long_data.rename(columns={0:'values'})
long_data
```

Although this method of table is frequently used
to store data in databases, we main want to untie
this mess with the `.pivot(index=, columns=, values=)` 
method:

```python
pivoted = long_data.pivot(index='date', columns='item', values='values')
pivoted.head()
```

If we had two values columns and omitted the
`value` keyword, the result would be a hierarchical
column dataframe:

```python
long_data['value2'] = np.random.standard_normal(len(long_data))
long_data.head()
```

```python
long_data.pivot(index='date', columns='item')
```

*pivot* is equivalent to creating a hierarchical index 
using *set_index* followed by an unstack:

```python
unstacked = long_data.set_index(['date', 'item'])
unstacked
```

```python
unstacked = unstacked.unstack(level='item')
unstacked
```

### Pivoting "Wide" to "Long" Format

The opposite method from `pivot()` is `melt()`

## Summary:

- We learned multilevel indexing and manipulation by giving names and stacks/unstacking
    - We can `swaplevel(key1, key2)` and `frame.columns.swaplevel(n, m)`
    - If we `groupby(level)`, we can compute summary statistics from a certain level;
    - We can set columns to indices with `set_index([column_list])` and turn them to columns with `reset_index()`
- We learned to combine datasets with `merge()` (on key), `concat()` and `df.join()` (add datasets), `combine_first`(fill holes in datasets)
    - We can merge on indexes too (`right_index=True`)
    - Concating can be done over axis=0 or axis=1
- We reshape with hierarchical indexes by stacking and unstacking selecting the level we want to move;
    - A simpler way and pretty used is to `pivot()` to a wide format and `melt()` into a long format;
