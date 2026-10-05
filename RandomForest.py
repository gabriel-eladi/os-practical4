import pandas as pd
from sklearn import tree
import numpy as np

class RandomForest:

    def __init__(self, trainingDataset: pd.DataFrame, testingDataset: pd.DataFrame, treeCount=10000, featureSetCount=3, sampleSize=1000):
        self.treeCount = treeCount
        self.featureSetCount = featureSetCount
        self.sampleSize = sampleSize
        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset 

        self.featureSets: pd.DataFrame
        self.trainSample = pd.DataFrame()
        self.treeSampleList = []
        self.beanResults = []
        self.beanPredictions = []

    def createForest(self):
        features = self.trainingDataset.drop(columns="Class")

        # For each new tree
        for i in range(self.treeCount):
            cols = features.sample(n=self.featureSetCount, axis="columns", replace=True).columns.tolist()
            cols = list(dict.fromkeys(cols))  
            sample = self.trainingDataset.sample(n=self.sampleSize, replace=True)
            self.treeSampleList.append((cols, sample[cols + ["Class"]]))

    def getForestResult(self, dataset=None):
        if dataset is None:
            dataset = self.testingDataset
        # Clear previous results (in case we want the test fold results after getting the training fold results)
        self.beanResults = []
        # Data set could be ither the training fold or the testing fold
        X = X = dataset.drop(columns="Class", errors="ignore")
        # Predict
        for cols, sample in self.treeSampleList:
            x = sample[cols].values.tolist()
            y = sample["Class"]
            clf = tree.DecisionTreeClassifier().fit(x, y)
            self.beanPredictions = clf.predict(X[cols].values.tolist()).tolist()
            self.beanResults.append(self.beanPredictions)