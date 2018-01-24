# ARASH NEMATI HAYATI
# Looking at the Skulls dataset with Python
# IBM Machine Learning with Python
# Big Data University ML0101EN

# Remove the column containing the target name since it doesn't contain numeric values.
# Also remove the column that contains the row number
# axis=1 means we are removing columns instead of rows.
# Function takes in a pandas array and column numbers and returns a numpy array without
# the stated columns
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values

def targetAndtargetNames(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    target_names = list()
    count = -1
    for i in range(len(my_data.values)):
        if my_data.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[my_data.values[i][targetColumnIndex]] = count
        target.append(target_dict[my_data.values[i][targetColumnIndex]])
    # Since a dictionary is not ordered, we need to order it and output it to a list so the
    # target names will match the target.
    for targetName in sorted(target_dict, key=target_dict.get):
        target_names.append(targetName)
    return np.asarray(target), target_names

## Main Code ##
import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier
my_data = pandas.read_csv("https://ibm.box.com/shared/static/u8orgfc65zmoo3i0gpt9l27un4o0cuvn.csv", delimiter=",")
new_data = removeColumns(my_data) # numpy array without headers
target, target_names = targetAndtargetNames(my_data, 1)
print target
print target_names
print new_data
print type(new_data)
print new_data.shape
new_data[...,1] = target
X = new_data # data for machine elarning
y = target # target category as a number 
neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(X,y)
print('Prediction: '), neigh.predict(new_data[10])
print('Actual:'), y[10]

