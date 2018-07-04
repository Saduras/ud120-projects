#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 
                # Financial features
                'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 
                # Email features
                'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 

# Drop non-numeric features
drop_features = ["email_address","other"]

for feature in drop_features:
    features_list.remove(feature)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Create dataframe from dataset
data = []
for key, value in data_dict.iteritems():
    value["name"] = key
    data.append(value)

import pandas as pd
df = pd.DataFrame(data)
df[features_list] = df[features_list].astype("float")

### Task 2: Remove outliers

## Inspect highest values per feature
# for feature in features_list:
#     sorted_list = df[feature].sort_values(ascending=False)
#     print sorted_list.head()

# Index 104 name=TOTAL contains obvious outlier data
# I leave all other extreme values in there since they might indicators for POIs
df.drop(104, inplace=True)

### Task 3: Create new feature(s)
features_list.append("from_poi_msgs_relative")
features_list.append("to_poi_msgs_relative")

df["from_poi_msgs_relative"] = df["from_poi_to_this_person"] / df["to_messages"]
df["to_poi_msgs_relative"] = df["from_this_person_to_poi"] / df["from_messages"]

# Drop source features
drop_features = ["from_poi_to_this_person", "to_messages",
                "from_this_person_to_poi", "from_messages"]
for feature in drop_features:
    features_list.remove(feature)

## Use correlation matrix to detect possible irrelevant features
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.heatmap(data=df[features_list].corr(), center=0, annot=True)
# plt.show()

# Correlation matrix indicated weak correlation between 
# poi and [deferral_payments, restricted_stock_deferred, from_messages]
# these are candidates to drop
drop_features = ["deferral_payments", "restricted_stock_deferred"]
for feature in drop_features:
    features_list.remove(feature)

# Convert dataframe back to dictionary
df.set_index("name")
df = df[features_list]
df.fillna(0, inplace=True)
data_dict = df.to_dict("index")

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
classifier = [
    {"name":"GaussianNB", "clf":GaussianNB()},
    {"name":"DecisionTree", "clf":DecisionTreeClassifier()},
    {"name":"AdaBoost", "clf":AdaBoostClassifier(DecisionTreeClassifier())},
    {"name":"GradientBoosting", "clf":GradientBoostingClassifier()},
    {"name":"RandomForest", "clf":RandomForestClassifier()},
    {"name":"SVC", "clf":SVC()},
    {"name":"KNeighbors", "clf":KNeighborsClassifier()},
]

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.metrics import f1_score, accuracy_score
for clf in classifier:
    print "evalutate",clf["name"],"..."
    clf["clf"].fit(features_train, labels_train)
    pred = clf["clf"].predict(features_test)
    clf["accuracy"] = accuracy_score(labels_test, pred)
    clf["f1"] = f1_score(labels_test, pred)

classifier.sort(key=lambda c:c["f1"])
for clf in classifier:
    print clf["name"],":",clf["f1"],":",clf["accuracy"]

clf = classifier[0]["clf"]

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)