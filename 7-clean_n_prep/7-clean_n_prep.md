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

### Discretization and Binning

We can organize continuous data into discrete
groups (or bins) for analysis. Suppose this data
to be categorized in different bins:

```python
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
```

To divide it into bins, we first determine the
limits of our bins and then use `pd.cut` with
our data and bins as parameters:

```python
bins = [18, 25, 35, 60, 100]

age_categories = pd.cut(ages, bins)

age_categories
```

This is a special Categorical object. Each bin
is identified by this special pandas interval value
containing upper and lower limits. Let's explore this 
object:

```python
age_categories.codes
```

```python
age_categories.categories
```

```python
age_categories.categories[0]
```

```python
age_categories.value_counts()
```

Here, the interval openness is indicated by the
parenthesis and brackets. The parenthesis means the
interval is open (exclusive) and the bracket means 
closed (inclusive). We can change the side that's
inclusive with the `right=False` parameter:

```python
pd.cut(ages, bins, right=False).value_counts()
```

We may with to use other names for labels instead
of the interval-based ones with the `labels` parameter:

```python
age_groups = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=age_groups)
```

If we pass an integer as the bins argument, pandas
will automatically create equally spaced intervals
to bin our data based on the maximum and minimum 
values:

```python
data = np.random.uniform(size=20)
pd.cut(data, 4, precision=2) # the precision limits the precision of the decimal points of the bins to two digits
```

A closely related function is `pd.qcut()`, which 
will separate the data in equally populated quantiles:

```python
data = np.random.standard_normal(size=1000)
quartiles = pd.qcut(data, 4, precision=2)
quartiles
```

```python
quartiles.value_counts()
```

We can also pass our own intervals for the quantiles,
as long as they are number between 0 and 1, inclusive:

```python
pd.qcut(data, [0, 0.1, 0.5, 0.7, 1]).value_counts()
```

### Detecting and Filtering Outliers

Filtering or transforming outliers is usually a matter
of applying array operations. Consider this DataFrame
with some normally distributed data:

```python
data = pd.DataFrame(np.random.standard_normal((1000, 4)))
data.describe()
```

Suppose we want to find all values in a given column
with absolute value > 3:

```python
col = data[2]
col[col.abs() >3]
```

To find all rows with any absolute value bigger than
three, we can use the `any` method on a Boolean DataFrame:

```python
data[(data.abs()>3).any(axis='columns')]
```

We can use the boolean indexing of `data.abs() > 3`
to set a new value, for example, 3 or -3. Here, 
we'll use the `np.sign()` method to determine the 
sign of the data, and therefore if it should have
3 or -3 attributed:

```python
data[data.abs() > 3] = np.sign(data) * 3
data.describe()
```

### Permutation and Random Sampling

So, there's this numpy method that returns a permutation
the size we give it in the parameter:

```python
sampler = np.random.permutation(5)
sampler
```

We can use this with `iloc[]` to return a random sample
of rows. We can also use it with the `take()` method,
that accepts an `axis` argument to sample columns as well.

```python
df = pd.DataFrame(np.arange(5*7).reshape(5, 7))
df
```

```python
df.iloc[sampler]
```

```python
df.take(sampler, axis=1)
```

Alternatively, there's the `sample()` df method,
which also allows for `replace=True` parameter
to allow for replacements:

```python
df.sample(n=3, axis=1, replace=True)
```

### Computing Indicator/Dummy Variables

A dummy or indicator matrix is a matrix indicating
if a value belongs to a certain category or not.
Take the example:

```python
df = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"], "data1": range(6)})
df
```

```python
pd.get_dummies(df['key'])
```

We can also add a prefix to our dummy columns
to make it easier to join them to the original 
DataFrame later:

```python
pd.get_dummies(df['key'], prefix='key')
```

If a row in a dataframe belongs to multiple categories,
we'll have to use a different approach to create the 
dummies, using the `str.get_dummies(<separator>)` method:

```python
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('../pydata-book/datasets/movielens/movies.dat', sep='::', 
                       header=None, names=mnames, engine='python')
movies[:10]
```

```python
dummies = movies['genres'].str.get_dummies('|')
dummies.iloc[:10, :6]
```

Then we could add a prefix with the `add_prefix()` method:

```python
dummies.add_prefix('genre_')
```

A useful statistical application is to use `get_dummies()` with
binning to create a matrix indicating belonging to a certain bin:

```python
np.random.seed(12345)
values=np.random.uniform(size=10)
values
```

```python
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))
```

### Summary

We've learned plenty of data transformation techniques:

- Removing and checking for duplicates with `duplicated()` and `drop_duplicates()`
- Transforming data with `.map` and a function or dictionary-like object;
- Replacing values with `replace()` and lists of values or dict-like objects;
- Renaming Axis Indexes with `rename()` and `index=`, `column=` parameters;
- Discretization and Binning with `pd.cut` and `pd.qcut()`, accepting different quantiles;
- Detecting and Filtering outliers with assigning with boolean arrays and `np.sign(data)`;
- Permutation and Random Sampling with `np.random.permutation(n)` and `iloc[]` or `take()`,
plus `df.sample(n, axis, replace)`;
- Computing dummies and indicators with `pd.get_dummies(col)` and `df[col].str.get_dummies(sep)`
- Combining dummies with binnings with `pd.get_dumies(pd.cut(values, bins))`;
