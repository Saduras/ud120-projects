#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

my_list = []
for key in data_dict.keys():
    if data_dict[key]["salary"] != "NaN" and data_dict[key]["bonus"] != "NaN":
        my_list.append((key, data_dict[key]["salary"], data_dict[key]["bonus"]))

my_list.sort(key=lambda tub : tub[2], reverse = True)
print "Outlier:",my_list[0]

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

