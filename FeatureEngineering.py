from sklearn import tree


def featureEngineering(trainDataFeatures, trainDataTarget):
    dt = tree.DecisionTreeClassifier()
    dt.fit(trainDataFeatures, trainDataTarget)
    # dt_score = cross_val_score(dt,trainDataFeaturesFinal,trainDataTarget,cv=5)
    importantFeatures = dt.feature_importances_ > 0.002
    print(importantFeatures)

    return importantFeatures
