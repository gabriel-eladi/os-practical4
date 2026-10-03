import pandas as pd
from sklearn import tree
import numpy as np

class RandomForest:

    def init(self, treeCount=10000, featureSetCount=3, sampleSize=100, sampleCount=5, trainingDataset=None, testingDataset=None):
        self.treeCount = treeCount
        self.featureSetCount = featureSetCount
        self.sampleSize = sampleSize
        self.sampleCount = sampleCount
        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset 

        self.featureSets: pd.DataFrame
        self.trainSample = pd.DataFrame()
        self.treeSampleList = []
        self.beanResults = []

    def createForest(self):
        for i in range(self.treeCount): #for each new tree
            self.featureSets = self.trainingDataset.drop(columns="Class").sample(n=self.featureSetCount, axis='columns', replace=True) #remove class from random selection of columns
            self.featureSets["Class"] = self.trainingDataset["Class"] #add class back on
            for j in range(self.sampleCount):
                start = np.random.randint(0, len(self.featureSets) - self.sampleSize + 1)
                self.trainSample = pd.concat([self.trainSample, self.featureSets.iloc[start:start + self.sampleSize]], axis=0)

            self.treeSampleList.append(self.trainSample)
            self.trainSample = pd.DataFrame()

    def getForestResult(self):
        for sample in self.treeSampleList:
            x = self.sample.drop(columns="Class").values.tolist()
            y = self.sample["Class"].astype("category").cat.codes.tolist()
            bean_types = dict(enumerate(self.trainSample["Class"].astype("category").cat.categories))
            clf = tree.DecisionTreeClassifier().fit(x, y)

            beans = []
            for bean in clf.predict(self.testingDataset):
                beans.append(bean_types[bean])
            self.beanResults.append(beans)