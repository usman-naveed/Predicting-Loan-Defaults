import numpy as np
import pandas as pd
from FeatureEngineering import featureEngineering
from CleaningData import cleanMyData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression



#Import datasets, train and test have already been defined for us
#testData = pd.read_csv("./application_test.csv")
trainDataRaw = pd.read_csv("./application_train.csv")
trainDataFeaturesFinal, trainDataTargetResampled = cleanMyData(trainDataRaw)

featureImportanceScore = featureEngineering(trainDataFeaturesFinal, trainDataTargetResampled)
colNames = trainDataFeaturesFinal.columns
print('Number of features after cleaning: ', len(colNames))
featureArray = colNames[featureImportanceScore]
print('Number of features after feature reduction: ', len(featureArray))
print('New features: ', featureArray)
featureReducedTrainData = trainDataFeaturesFinal[featureArray]

#create a scorer for evaluation
scoring = ['f1_macro', 'precision_macro', 'recall_macro']
scoring2 = ['f1_macro', 'precision_macro', 'recall_macro']

knn = KNeighborsClassifier(n_neighbors=2).fit(featureReducedTrainData, trainDataTargetResampled)
knn_score = cross_validate(knn, featureReducedTrainData, trainDataTargetResampled, cv=5, scoring=scoring)
print("KNN average precision score: {}".format(np.mean(knn_score['test_precision_macro'])))
print("KNN average f1 score: {}".format(np.mean(knn_score['test_f1_macro'])))
print("KNN average recall score: {}".format(np.mean(knn_score['test_recall_macro'])))


logistic = LogisticRegression(random_state=42, solver='liblinear', max_iter=500).fit(featureReducedTrainData, trainDataTargetResampled)
logisticScore = cross_validate(logistic, featureReducedTrainData, trainDataTargetResampled, cv=5, scoring=scoring2)
print("Logistic Regression average precision score: {}".format(np.mean(logisticScore['test_precision_macro'])))
print("Logistic Regression average f1 score: {}".format(np.mean(logisticScore['test_f1_macro'])))
print("Logistic Regression average recall score: {}".format(np.mean(logisticScore['test_recall_macro'])))
