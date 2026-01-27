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

Asides from dealing with missing data, filtering, cleaning and
transforming are also essential parts of the data wrangling job:

### Removing duplicates

Consider the following example of a DataFrame that contains duplicates:

```python
data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'], 'k2': [1, 1, 2, 3, 3, 4, 4,]})
data
```

The `duplicated()` method returns a boolean array indicating 
if any row is a duplicate of a previously iterated row:

```python
data.duplicated()
```

The `drop_duplicates()` method returns a DataFrame with only
the rows indicated as `False` by `duplicated()`:

```python
data.drop_duplicates()
```

These methods by default consider all columns, but 
suppose we want to restrict the duplicate checking 
and dropping to only a subset of columns. We do that
with the `subset` parameter:

```python
data['v1'] = range(7)
data
```

```python
data.duplicated()
```

```python
data.duplicated(subset=['k1'])
```

By default, `drop_duplicates()` keeps the first values
it encounters. Passing `keep='last'` will keep the last
ones instead.

```python
data.drop_duplicates(subset=['k1', 'k2'], keep='last')
```

### Transforming Data Using a Function Or Mapping

Frequently we'll want to do some transformation depending
on the values present in the current array. Consider this
hypothetical data collected about kinds of meat:

```python
data = pd.DataFrame({"food": ["bacon", "pulled pork", "bacon", "pastrami", "corned beef", "bacon", "pastrami", "honey ham", "nova lox"], "ounces": [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data
```

Suppose we want to add a new column indicating the 
type of animal the meat came from. We can create a 
dict to map each meat to an animal:

```python
meat_to_animal = {
  'bacon' : 'pig',
  'pulled pork': 'pig',
  'pastrami' : 'cow',
  'corned beef' : 'cow',
  'honey ham' : 'pig',
  'nova lox' : 'salmon'
}
```

The `map` method accepts a function or a dictionary-like
object to perform a function or mapping into a series of
values:

```python
data['animal'] = data['food'].map(meat_to_animal)
data
```

Passing a function that returns the value from the dict
would also have done the trick. Lets try it with a lambda
function:

```python
data['food'].map(lambda x: meat_to_animal[x])
```

### Replacing Values

While `map()` can be seen as a way to replace values,
`replace()` offers a simpler and more flexible way to 
do so. Consider the Series:

```python
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data
```

The `-999` values might be a sentinel for missing data.
To replace them with a value that pandas understands as 
NA, we can use `replace()`:

```python
data.replace(-999, np.nan)
```

We can also **replace multiple values at once** with 
by passing a list of values to the first argument:

```python
values_to_replace = [-999, -1000]
data.replace(values_to_replace, np.nan)
```

**Replacing different values with different
replacements** can be done by passing a list
to the second argument (of equal length to the
first list) or by passing a dict as argument:

```python
data.replace({-999: 0, -1000: np.nan})
```

### Renaming Axis Indexes

We can both create new objects with different labels
than the first one, or modify labels in place.
To modify it in place, we can use the `index.map()` 
method assigning it to the index of the DataFrame:

```python
data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                    index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data
```

```python
data.index = data.index.map(lambda x: x[:4].upper())
data
```

To return a new object without modifying the original, 
we use the `rename()` method:

```python
data.rename(index=str.title, columns=str.upper)
```

Rename can also be used with a dictionary-like object
to rename specific values to specific replacements:

```python
data.rename(index={'OHIO': 'Indiana'}, columns={'three': 'peekaboo'})
```
