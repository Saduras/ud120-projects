#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print "persons:",len(enron_data)
print "feature count:",len(enron_data.values()[0])
print "features",enron_data.values()[0].keys()

poi_count = 0
for key in enron_data.keys():
    poi_count = poi_count + enron_data[key]['poi']

print "person of interests", poi_count

print "James Prentice total stock value:",enron_data["PRENTICE JAMES"]["total_stock_value"]
print "Wesley Colwell message to POIs:", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "Jeffrey K Skilling excercied stock value:",enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]