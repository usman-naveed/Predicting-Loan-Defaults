import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from imblearn.over_sampling import SMOTE


def cleanMyData(trainDataRaw):
    # to show imbalance data (0: 282,686 || 1: 24825)
    print(trainDataRaw['TARGET'].value_counts())

    # reducing the data to 10,000 rows just for local testing
    trainData = trainDataRaw.loc[1:10000, ]
    # print(trainData.describe(include='all'))
    # print(trainData.info())

    # splitting categorical and numerical features
    numericCols = trainData._get_numeric_data().columns  # getting all numerical columns
    cols = trainData.columns
    categoricalFeatures = trainData[list(set(cols) - set(numericCols))]  # difference between all columns and numerical columns = categorical columns
    numericalFeatures = trainData[numericCols]

    # impute missing categorical data: using the mode of each feature to impute the missing data
    categoricalFeatures = categoricalFeatures.fillna(categoricalFeatures.mode().iloc[0])
    categoricalFeaturesNames = list(categoricalFeatures)
    # print(categoricalFeatures.head())

    # impute missing numerical data: using the mean of each feature to impute the missing data
    numericalFeatures = numericalFeatures.fillna(numericalFeatures.mean())
    numericalFeaturesNames = list(numericalFeatures)
    # print(numericalFeatures.head())

    # now that the missing values are dealt with, lets combine both
    # categorical and numerical dataframes
    trainDataNoMissing = numericalFeatures.join(categoricalFeatures)
    trainDataNoMissing.columns = [numericalFeaturesNames + categoricalFeaturesNames]
    # print(trainData.head())

    # one hot encoding to get rid with categorical features
    trainDataEncoded = pd.get_dummies(trainDataNoMissing, columns=list(categoricalFeaturesNames))
    #print(list(trainDataEncoded))
    #print(trainDataEncoded.iloc[:, 0])

    # Upsampling using SMOTE
    trainDataTarget = trainDataEncoded.iloc[:, 1]
    trainDataFeatures = trainDataEncoded.iloc[:, 2:]
    sm = SMOTE(random_state=42)
    trainDataFeaturesResampled, trainDataTargetResampled = sm.fit_resample(trainDataFeatures, trainDataTarget)
    print('Value counts of the upsampled data: ', np.count_nonzero(trainDataTargetResampled))
    print('Length of upsampled: ', len(trainDataTargetResampled))
    print('Value counts of the normal data: ', np.count_nonzero(trainDataTarget))

    # scale the data
    trainDataFeaturesFinal = pd.DataFrame(scale(trainDataFeaturesResampled), columns=list(trainDataFeatures))
    #print(trainDataFeaturesFinal.head())
    #print(trainDataFeaturesFinal.columns)

    return trainDataFeaturesFinal, trainDataTargetResampled
