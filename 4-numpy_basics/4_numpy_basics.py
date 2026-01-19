# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python for Data Analysis
#     language: python
#     name: pydata_book
# ---

# %%
# can we create np arrays from dictionaries?
import numpy as np

# %%
dict_test = [{1: 1, 2: 2, 3: 3}, {4: 4, 5: 5, 6: 6}]
dict_array = np.array(dict_test)
dict_array.shape

# %%
# can we create array with strings of different sizes?
string_list = ["two", "three"]
np.array(string_list, dtype="U10")

# %%
# casting types
arr = np.array(np.arange(10))
arr.dtype

# %%
arr.astype(float).dtype

# %%
arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
arr

# %%
arr * arr

# %%
arr - arr

# %%
(arr**arr)[1][2]

# %% jupyter={"outputs_hidden": true}
1 / arr

# %%
arr / 2

# %%
arr2 = np.array([[0.0, 4.0, 1.0], [7.0, 2.0, 12.0]])

# %%
arr2 == arr

# %%
arr2 > arr

# %% [markdown]
# ## Array Indexing
# ### 1-dimensional arrays

# %%
arr = np.arange(10)
arr

# %%
arr[5:8]

# %%
arr[5:8] = 12
arr
# as we can note, array slices are views of the array, not copies. follow the next cell's example:

# %%
arr_slice = arr[5:8]
arr_slice -= 1
arr

# %%
arr_slice[:] = 64
arr

# %%
arr[:] = 32
arr

# %% [markdown]
# ## higher dimension arrays indexing and slicing

# %%
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]

# %%
# these are equivalent
print(arr2d[0][2])
print(arr2d[0, 2])

# %%
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d.shape

# %%
arr3d[0]

# %%
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d

# %%
arr3d[0] = old_values
arr3d

# %%
arr3d[0, 1]

# %% [markdown]
# ## slicing n-dimension arrays

# %%
print(arr2d)
arr2d[:2, 1:]
# returns the first two rows, from their second element

# %%
lower_dim_slice = arr2d[1, :2]
# returns the second row, first 2 columns
print(lower_dim_slice)

# %%
lower_dim_slice.shape

# %% [markdown]
# ## Boolean Indexing
#
# Where we index an array with another array of booleans, refering to True values

# %%
# consider this array of names with some duplicates:
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])

# now we create an array of booleans, such as true for "Bob"
booleans = names == "Bob"
booleans

# %%
# suppose we have a different array of some data:
data = np.array([[4, 7], [0, 2], [-5, 6], [0, 0], [1, 2], [-12, -4], [3, 4]])
data.shape

# %%
# as both arrays have the same number of columns, we can index the columns of the data array with the boolean array:
data[booleans]  # should return [4, 7] and [0, 0]

# %%
# we can mix this with slicing, too. for selecting only the second value of these selected columns:
data[booleans, 1:]

# %%
# the ~ operator inverts the boolean values. May be used to "anti-filter" the array:
~(names == "Bob")

# %%
data[~booleans]

# %%
(names == "Bob") | (names == "Will")

# %% [markdown]
# ## Fancy Indexing
#
# Using arrays as indexer to select a subset of an array in a particular order

# %%
# consider this array:
arr = np.zeros((8, 4))
for i in range(8):
    arr[i] = i
arr

# %%
# if we create an array of integers in a certain order...
indexer = np.array([4, 0, 7, 1])
# we can index our array with this indexer!
arr[indexer]


# %% [markdown]
# ### Multiple arrays for indexing
#
# We can pass multiple arrays for indexing too. Consider:

# %%
arr = np.arange(32).reshape(8, 4)
arr

# %%
indexer1 = [7, 4, 1]
indexer2 = [3, 1, 2]
arr[indexer1, indexer2]

# %%
# we can also create a region with the combination of the indices passed as so:
arr[indexer1][:, indexer2]

# %%
# assigning the fancy indexed array to a variable creates a copy;
# assigning a value to the fancy indexed array WITHOUT a variable modifies the array.
arr[indexer1, indexer2] = 0
arr

# %%
#  transposing arrays
# We can transpose arrays with the transpose method and the T attribute
arr = np.arange(15).reshape((3, 5))
arr
#
# %%

# %%
arr.T
# %%

# %%

# %%
# we can do matrix multiplication in two ways:
# np.dot way:
np.dot(arr, arr.T)

# %%

# %%
# @ operator
arr @ arr.T

# %%

# %% [markdown]
# ## Swapaxes
#
# the T attribute is a special case of the swapaxes function, which swaps two axes. This function can be more generally used for n dimension arrays.
# It does not make a copy.

# %%

# %%
arr

# %%
arr.swapaxes(0, 1)

# %%

# %%
arr

# %%
# %% [markdown]
# ## array oriented programming
# Instead of using loops, we can we array functions to express some complicated or big operations
# Suppose we have an two arrays (x and y) and we wish to evaluate sqrt(x² + y²)

# %%
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
ys
# %%

# %%
z = np.sqrt(xs**2 + ys**2)
z

# %%

# %%
import matplotlib.pyplot as plt

plt.imshow(z, cmap=plt.cm.gray, extent=[-5, 5, -5, 5])
plt.colorbar()
plt.title("image plot of $\\sqrt{x² + y²} for a grid of value")

# %%

# %%
rng = np.random.default_rng(12345)
arr = rng.standard_normal((5, 4))
arr

# %%

# %%
arr.mean()
# %%

# %%
np.mean(arr)

# [markdown]
# ## Boolean arrays and their methods

# %%
# boolean arrays and their methods
arr = rng.standard_normal(100)
(arr > 0).sum()

# %%
(arr < -0).sum()

# %%

# %%
# any checks if any is true, all checksif all are true
bools = arr > 0
bools.any()

# %%

# %%
bools.all()

# %%

# [markdown]
# ## Random walk with numpy

# %%
nsteps = 1000
rng = np.random.default_rng(seed=12345)
draws = rng.integers(0, 2, size=nsteps)
steps = np.where(draws == 0, 1, -1)
walk = steps.cumsum()

# %%

# %%
walk.min()
# %%

# %%
walk.max()
# %%

# %%
(np.abs(walk) >= 10).argmax()
# %%

# %%
nwalks = 5000
nsteps = 1000
draws = rng.integers(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(axis=1)
walks
# %%

# %%
walks.max()
# %%

# %%
walks.min()
# %%

# %%

# %%
hits30 = (np.abs(walks) >= 30).any(axis=1)
hits30

# %%

# %%
hits30.sum()
# %%

# %%
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(axis=1)
crossing_times

# %%

# %%
crossing_times.mean()
# %%
