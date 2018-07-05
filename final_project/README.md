# Enron Submission Free-Response Questions

[Question Source](https://docs.google.com/document/d/1NDgi1PrNJP7WTbfSUuRUnz8yzs5nGVTSzpO7oeNTEWA/edit)

[Rubrics](https://review.udacity.com/#!/rubrics/27/view)

1) **Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those? [relevant rubric items: “data exploration”, “outlier investigation”]**

The goal of this project is to identify person of interest (POI) in the Enron fraud scandal from 2001. The dataset is financial data and emails from Enron employees. It's the biggest public email dataset. I am analysis the connections between employees through emails and the financial data to predict POIs.

The financial data contains one outlier entry "TOTAL" which contains only the sum of all other entries and therefore can be safely removed. Some other data points sticked out with unusual high financial values. I decided to keep those since they might related to the some of the few POIs and therefore might be crutial for the classification.

2) **What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]**

The final feature list is:

'salary', 'total_payments', 'loan_advances', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees', 'shared_receipt_with_poi', 'from_poi_msgs_relative', 'to_poi_msgs_relative'

I started out with the full feature list and just removed non-numeric features 'email_address' and 'other' since they don't contain any quantifiable information. All other features I scaled with a MinMaxScaler to ensure that their values have equal impact on the classifier even though the original values varied in magnitute.

Further more I engineer two new features:

'from_poi_msgs_relative' = 'from_poi_to_this_person' / 'to_messages'

'to_poi_msgs_relative' = 'from_this_person_to_poi' / 'from_messages'

These feature should help to analyse the relation of employees to POIs. For this relation I thinks more relevant which fraction of emails was sent to/recived from POIs than the total number. After adding the features I dropped the source features.

Finally I inspected the correlation of all features with POIs and decied to drop 'deferral_payments' and 'restricted_stock_deferred' since they seem to have a very low correlation to POIs and there fore have little use in predicting POIs.

3) **What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms? [relevant rubric item: “pick an algorithm”]**

In the end DecisionTreeClassifier was used, choosen based on their cross evaluation F1 score. It was tested against AdaBoost, GradientBoosting, RandomForest, SVC, KNeighbors and GaussianNB. 

4) **What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]**

Parameter tuning is important to prevent overfitting and reduce bias. Also the training time is affected by the parameters. With the correct parameters the algorithm can perform better (low bias) while still generalising well to unseen data (not overfitting). The parameters where evaluated with GridSearchCV choosing a range of parameters around the default values. GridSearchCV then test all combinations of parameters against each other and gives back the best estimator by a given scoring.

5) **What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis? [relevant rubric items: “discuss validation”, “validation strategy”]**

Validation means test the classifier on unseen data by predicting the labels and scoring them against the actual labels. Since this data set has very few data points for POIs compared to non-POIs it can easily come to an uneven distribution of these data points between training and test set. This should be taken into account for example by using StratifiedShuffleSplit to devide the data for validation.

6) **Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metric**

Two relevant evaluation metrics for this project are recall and precision. The final classifier achieved a precision of 48.25% and a recall of 46.75%. 

Precision means here how accurate does the classifier label a POI as such. Given that the current data point represents a POIs the classifier labels it as POI 48.25% of the time. It is the false negative rate.

Recall means in this context how many non-POIs get labeled as non-POIs. The classifier labels non-POIs correctly as non-POI 46.75% of the time. These are also called false positive.

Accuracy it not a good metric for this project since the POIs count is very small and just by classifying everyone as non-POIs one could achive over 80% accuracy.
