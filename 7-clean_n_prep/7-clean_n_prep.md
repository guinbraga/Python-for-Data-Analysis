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

# 7 - Data Cleaning and Preparation

In this chapter we discuss tools for handling missing data,
duplicate data, string manipulation, and some other analytical
data transformations. The next chapter is then focused on 
combining and rearranging datasets in various ways.

## 7.1 Handling Missing Data

Pandas has plenty of ways of handling missing data. Some of
the statistical built-in methods of pandas objects already 
exclude missing data for default, for example, but we may
be interested in handling missing data in different ways:

First, it is important we remember that we can set which 
values are considered NA when importing a dataset with 
`read_csv()` or other read functions by using the `na_values`
parameter.

Second, if the DataFrame is already loaded, we can treat it 
with functions such as `replace()` or `map()`.

### Filtering Out Missing Data

There are different ways of filtering missing data, depending
on whether we want to drop rows, columns, and the missing data
threshold we consider for being dropped.

Although we could use boolean indexing with the `notna()` method,
`dropna()` allows us to customize all these options above-mentioned.


```python
import numpy as np
import pandas as pd

data = pd.Series([1, np.nan, 3.5, np.nan, 7])
data.dropna()
```

Note that these return copies of the object by default. 
To modify the original dataset, we use the `inplace=` 
parameter.

To present some of the different ways to drop NA values:

```python
data = pd.DataFrame([[1., 6.5, 3.], [1., np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, 6.5, 3.]])
data
```

`dropna()` drops rows that have any missing value by default:

```python
data.dropna() 
```

Passing `how='all'` drops only rows **that have all values
missing**

```python
data.dropna(how='all') 
```

We can **drop columns** instead with the `axis` parameter:

```python
data[4] = np.nan
data
```

```python
data.dropna(axis=1, how='all')
```

To drop only past a certain **threshold** of missing values, 
we use the `thresh` parameter:

```python
df = pd.DataFrame(np.random.standard_normal((7,3)))
df.iloc[:4, 1] = np.nan
df.iloc[:2, 2] = np.nan
df
```

```python
df.dropna()
```

```python
df.dropna(thresh=2)
```

### Filling In Missing Data

Rather than discarding missing data, we may
want to fill it with some value, such as an
integer, the mean for that columns, of the median.
`fillna()` will do that for us:

```python
df.fillna(0) 
```

To **use different fill values for different
columns**, we can pass a dictionary as parameter
to the method:

```python
df.fillna({1:0, 2:2})
```

We can also **fill forwards** or **fill backwards**
with the `ffill()` and `bfill()` methods:

```python
df = pd.DataFrame(np.random.standard_normal((6,3)))
df.iloc[2:, 1] = np.nan
df.iloc[4:, 2] = np.nan
df
```

```python
df.ffill()
```

```python
df.ffill(limit=2)
```

With `fillna()` we may also fill with the mean
or median of a column:

```python
df.fillna(df.mean())
```

## 7.2 Data Transformation
