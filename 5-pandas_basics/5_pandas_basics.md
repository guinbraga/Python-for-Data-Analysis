# 5 - Pandas Basics

## 5.1 Pandas Structures

### Series

series are array-like structures that can hold values of the same type and have an associated
index array to them

```python
import pandas as pd
import numpy as np
```

```python
obj = Series([4, 2, 1, 7])
obj
```

```python
obj.array
```

```python
obj.index
```

### DataFrame

the DataFrame is a collection of columns containing some type of data, with an index both
for the rows and for the columns. it can be created by a dictionaary, such as below:

```python
data = {
    "state": ["Parana", "Goias", "Bahia", "Goias", "Parana"],
    "year": [1992, 2001, 2013, 2002, 2003],
    "pop": [1.3, 2.5, 3.1, 4.6, 3.0],
}
frame = DataFrame(data)
frame
```

```python
# we can see its head and tails
frame.head()
```

```python
frame.tail()
```

```python
# we can set its columns to a ceartin order:
pd.DataFrame(data, columns=["pop", "year"])
```

```python
# passing a column with no correspondence creates a column with missing values
frame2 = pd.DataFrame(data, columns=["pop", "year", "debt"])
```

```python
# columns can be acessed as indexes, returning a Series object, or as parameters
frame["state"]
```

```python
frame.state
```

```python
# we can return rows with loc and iloc methods.
frame2.iloc[0]
```

```python
frame.loc[1]
```

```python
# we can assign values to columns
frame2["debt"] = np.arange(5.0)
frame2
```

```python
# when assigning a series, its indexes are alighned to the dataframe ones. values without indexes are missing values.
# assiging a column that doesnt exist creates the column.
# columns can be deleted with the del keyword. in this example, we first add a boolean column of states == Parana
frame["Boolean"] = frame["state"] == "Parana"
frame
```

```python
del frame["Boolean"]
```

```python
frame
```

```python
# We can transpose a dataframe with the .T attribute. it should be noted, however, that if the columns do not have all the same type, the type is discarded and a object is left.
```

```python
# when creating a dataframe from a nested dictionary, the outer dictionary  keys are considered the columns and the inner dictionary keys are considered row indices.
populations = {
    "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
    "Nevada": {2001: 2.4, 2002: 2.9},
}
DataFrame(populations)
```

```python
# if indexes are explicitally called, then the inner keys might not be the indexes of the DataFrame
frame3 = DataFrame(populations, index=[2001, 2002, 2003])
```

```python
frame3.columns.name = "states"
```

```python
frame3.columns
```

```python
frame3
```

```python
frame4 = frame3.loc[[2003, 2001], ["Ohio"]]
frame4.loc[[2003], ["Ohio"]] = 4.7
frame3
```

```python
frame4
```

```python
indexes_to_keep = frame3.index.difference([2002])
frame5 = frame3.reindex(indexes_to_keep)
frame5
```

### Dropping Entries from an Axis example

```python
data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=["Ohio", "Colorado", "Utah", "New York"],
    columns=["one", "two", "three", "four"],
)
data
```

```python
data.drop(index=["Colorado", "Ohio"])
```

```python
data.drop(columns=["two"])
```

```python
data
```

### Indexing, selection, filtering

Tests with a dataframe

```python
data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=["Ohio", "Colorado", "Utah", "New York"],
    columns=["one", "two", "three", "four"],
)
data
```

```python
data["two"]
```

```python
data.iloc[[1, 2],[3, 0]]
```

```python
data.iloc[:, :3][data.three > 5]
```

```python
data.loc[data.three >= 3]
```

```python

```
