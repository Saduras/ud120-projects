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
salary_count = 0
email_count = 0
payment_count = 0
poi_payment_count = 0
for key in enron_data.keys():
    poi_count = poi_count + enron_data[key]["poi"]
    if enron_data[key]["poi"]:
        poi_payment_count = poi_payment_count + (enron_data[key]["total_payments"] != "NaN")    
    salary_count = salary_count + (enron_data[key]["salary"] != "NaN")
    email_count = email_count + (enron_data[key]["email_address"] != "NaN")
    payment_count = payment_count + (enron_data[key]["total_payments"] != "NaN")

print "person of interests", poi_count
print "poi payment count", poi_payment_count, (poi_payment_count/float(poi_count))
print "salary count", salary_count
print "email address count", email_count
print "payment count", payment_count, (payment_count/float(len(enron_data)))

print "James Prentice total stock value:",enron_data["PRENTICE JAMES"]["total_stock_value"]
print "Wesley Colwell message to POIs:", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "Jeffrey K Skilling excercied stock value:",enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print "Skilling payment", enron_data["SKILLING JEFFREY K"]["total_payments"]
print "Lay payment", enron_data["LAY KENNETH L"]["total_payments"]
print "Fastow payment", enron_data["FASTOW ANDREW S"]["total_payments"]