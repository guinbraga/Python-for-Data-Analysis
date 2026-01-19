---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python for Data Analysis
    language: python
    name: pydata_book
---

# Reading and Writing Data in Text Format 

There are lots of functions in pandas for reading data in  a 
text format. These functions also have lots of arguments, as
particular tasks require particular reading setups.

Let's start with a small csv:

```python
!cat ../pydata-book/examples/ex1.csv
```

As it is a csv, we can use `pandas.read_csv`:

```python
import pandas as pd
import numpy as np
```

```python
df = pd.read_csv("../pydata-book/examples/ex1.csv")
df
```

Not all files have a header. Consider this file:

```python
!cat ../pydata-book/examples/ex2.csv
```

Here, we have two options: let Pandas assign column names
with `header=None`, or add them ourselves with `names=[names]`

```python
pd.read_csv("../pydata-book/examples/ex2.csv", header=None)
```

```python
pd.read_csv("../pydata-book/examples/ex2.csv", names=["a", 'b', 'c', 'd', 'message'])
```

Suppose we **want a specif column, such as message, as the index**.
We can do that indicating either the integer 4 or the string "message"
in the `index_col` argument:

```python
names=["a", 'b', 'c', 'd', 'message']
pd.read_csv("../pydata-book/examples/ex2.csv", names=names, index_col="message")
```

The `index_col` argument accepts a list of indexes, as to make
a hierarchical index dataset: 

```python
!cat ../pydata-book/examples/csv_mindex.csv
```

```python
parsed = pd.read_csv("../pydata-book/examples/csv_mindex.csv", index_col=["key1", "key2"])
parsed
```

Sometimes the delimiter is different than a comma. Consider
the following file:

```python
!cat ../pydata-book/examples/ex3.txt
```

Although one could try editing it by hand, we can pass an expression,
such as `\s+`, which will account for any number of whitespaces:

```python
result = pd.read_csv("../pydata-book/examples/ex3.txt", sep="\s+")
result
```

... as the first line of this dataset has one fewer values than the
rest, pandas infers the first column to be the index.

One example of useful parsing argument is the `skiprows`, with accepts
a list of rows to skip:

```python
!cat ../pydata-book/examples/ex4.csv
```

```python
pd.read_csv("../pydata-book/examples/ex4.csv", skiprows=[0, 2, 3])
```

*Handling missing data* is a point of interest. Missing data is usually 
represented in the csv file as a placeholder, such as NULL or NONE, or 
simply an empty string. Pandas uses a default set of common ocurring 
sentinels to handle missing data:

```python
!cat ../pydata-book/examples/ex5.csv
```

this output has the NA and empty string as non-present values.

```python
result = pd.read_csv("../pydata-book/examples/ex5.csv")
result
```

We can disable the default values set as NA with the `keep_default_na=false`
argument: 

```python
result2 = pd.read_csv("../pydata-book/examples/ex5.csv", keep_default_na=False)
result2
```

```python
result2.isna()
```

```python
result3 = pd.read_csv("../pydata-book/examples/ex5.csv", keep_default_na=False, na_values=["NA"])
result3
```

At last, we can specify different sentinels for each column:

```python
sentinels = {"message" : ["foo", 'NA'], 'something' : ['two']}
pd.read_csv('../pydata-book/examples/ex5.csv', keep_default_na=False, na_values=sentinels)
```

## Reading Text Files in Pieces

To process very large files, like perhaps those related 
to biological data, we may wish to read smaller parts of 
the file or to iterate through it in smaller chunks.

To do so, we may make pandas display settings more compact:

```python
pd.options.display.max_rows = 10
```

Now we have:

```python
result = pd.read_csv("../pydata-book/examples/ex6.csv")
result
```

To read a smaller number of lines, we may indicate it with
`nrows` argument: 

```python
pd.read_csv("../pydata-book/examples/ex6.csv", nrows=5)
```

To read iterate a file in pieces, we use the `chunksize` argument:

```python
chunker = result = pd.read_csv("../pydata-book/examples/ex6.csv", chunksize=1000)
type(chunker)
```

This is an iterable. Let's say we want to iterate through it counting
the amount of times any given key appears:

```python
tot = pd.Series([], dtype='int64')

for piece in chunker:
  tot = tot.add(piece['key'].value_counts(), fill_value=0)

tot = tot.sort_values(ascending=False)
tot[:10]
```

The `TextFileReader` object also has a `get_chunk` method that enables you
to read pieces of an arbitrary size.

## Writing Data to Text Format

This is how we export data to a delimited format. Let's consider one of the files
previously used in our examples:


```python
data = pd.read_csv("../pydata-book/examples/ex5.csv")
data
```

To export it with a custom delimiter:

```python
data.to_csv('../pydata-book/examples/out.csv', sep='|')
```

```python
!cat '../pydata-book/examples/out.csv'
```

By default, empty values appear as empty strings. If
we wish to use another sentinel value, `na_rep` comes 
to the rescue!

```python
import sys # we'll write to sys.stdout to avoiding writing in a file
data.to_csv(sys.stdout, na_rep='NULL')
```

By default, column and row labels are written. We
can also disable this feature:

```python
data.to_csv(sys.stdout, index=False, header=False) 
```

You may also write a subset of the columns in a chosen order:

```python
data.to_csv(sys.stdout, index=False, columns=['c', 'message', 'd']) 
```

## JSON data

Python has a json library built-in which we will use to read an
file example:

```python
obj = """ {"name": "Wes",  "cities_lived": ["Akron", "Nashville", "New York", "San Francisco"],  "pet": null,  "siblings": [{"name": "Scott", "age": 34, "hobbies": ["guitars", "soccer"]},  {"name": "Katie", "age": 42, "hobbies": ["diving", "art"]}] } """
```

```python
import json

result = json.loads(obj)

result
```

`json.dumps()` is the method for converting from python object
to json.

how to convert a json to DataFrame is up to the user. One convenient
way would be to pass a list of dictionaries (which where JSON) to the
DataFrame constructor and a subset of data fields:

```python
siblings = pd.DataFrame(result['siblings'], columns=['name', 'hobbies'])
siblings
```

pandas has a `read_json` method for automatically converting JSON 
datasets, although it should be a "well-behaved" json dataset. In the
following example we can see that the `read_json` method assumes each 
json object is a row in a dataset:

```python
!cat ../pydata-book/examples/example.json
```

```python
data = pd.read_json("../pydata-book/examples/example.json")
data
```

To convert back to json, pandas also has a `to_json()` method:

```python
data.to_json(sys.stdout)
```

```python
data.to_json(sys.stdout, orient='records')
```

## XML and HTML: Web Scraping

Python has many libraries for reading and writing data in HTML 
and XML formats. Pandas leverages these libraries to make it 
possible to efficiently extract data from these formats with 
the `read_html()` method. It has many options, but it's default
behavior is to search and fetch data inside `<table>` tags.

Let's analyse an example provided by the book:

```python
tables = pd.read_html("../pydata-book/examples/fdic_failed_bank_list.html")
len(tables)
```

```python
failures = tables[0]
failures.head()
```

As we will learn in the next chapters, from here it is possible to do some 
data cleaning and analysis, for example, finding the count of bank failures
per year:

```python
close_timestamps = pd.to_datetime(failures["Closing Date"])
close_timestamps.dt.year.value_counts()
```

Pandas also has the `read_xml()` function, which makes reading xml files
a lot easier than the alternative with the `lxml` library. The example
table is present in page 192 of the book.


## Summary

Today we learned the main methods for reading and writing text files from
and to the disk. These include:

- The `read_csv()` function, with arguments to:
    - label columns with `names` and indexes with `index_col`;
    - skiprows, skipfooter, header presence;
    - read by chunks in an iterable with `chunksize` and by pieces with `nrows`;
    - set values to consider as `NaN` with (na_values) and override the default values; 
    - set delimiter characters, including "any amount of whitespace" (`\s+`)
    - exporting with `to_csv` setting delimeters and NaN expression with `na_rep`
- The `read_json()` function and `to_json`
- The `read_html` and `read_xml` functions

**upnext**: dealing with binary formats of data, dealing with web APIs
and Databases!
    
