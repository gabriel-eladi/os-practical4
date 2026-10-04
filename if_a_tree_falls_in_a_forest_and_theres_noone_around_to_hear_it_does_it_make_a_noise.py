import graphviz
from typing import List
import pandas as pd
import numpy as np
from sklearn import tree        
import RandomForest
from collections import Counter

training_df = pd.read_csv("dry_bean_train.csv")
test_df = pd.read_csv("dry_bean_test.csv").values.tolist()

df = training_df
x = df.drop(columns="Class").values.tolist()
y = df["Class"].astype("category").cat.codes.tolist()
beans_by_class = df["Class"].values
unique_bean_classes = np.unique(df["Class"])
bean_types = dict(enumerate(df["Class"].astype("category").cat.categories))

clf = tree.DecisionTreeClassifier().fit(x, y)

#print(unique_bean_classes)

def stratified_kfold(k: int) -> List[pd.DataFrame]:
    folds: List[List[int]] = [[] for _ in range(k)]
    seed = 0
    rng = np.random.default_rng(seed)
    for bean_class in unique_bean_classes:
        # Beans that matches the current class
        filtered_beans = np.where(beans_by_class == bean_class)[0]

        # Shuffle the filtered beans to ensure randomness
        rng.shuffle(filtered_beans)

        # Split the filtered beans into k folds and stratifying them
        for i, chunk in enumerate(np.array_split(filtered_beans, k)):
            folds[i].extend(chunk.tolist())

    #Test if the folds are stratified correctly
    #for i in folds:
    #    bean_types_in_fold = [beans_by_class[idx] for idx in i]
    #    print(f"fold {folds.index(i)}: {bean_types_in_fold}")
    return [df.iloc[fold] for fold in folds]

def forest_accuracy(bean_predictions, true_classes):
    predictions = []
    # The forests democratic voting
    # zip(*...) goes down 1 column at a time and count the number of time a bean shows up
    for votes in zip(*bean_predictions):
        # Get the most common (highest vote) bean  
        winning_class = Counter(votes).most_common(1)[0][0]
        # Take the most voted class as the prediction      
        predictions.append(winning_class)

    correct = 0
    for predicted, actual in zip(predictions, true_classes):
        if predicted == actual:
            correct += 1
    return correct / len(true_classes)

# Generate folds
folds = stratified_kfold(k=5)
scores = []

# Cross-validation loop, 1 forrest per fold
for i in range(len(folds)):
    train_df = pd.concat([fold for j, fold in enumerate(folds) if j != i])
    # Each forrest gets a random sample
    forest = RandomForest.RandomForest(trainingDataset=train_df, testingDataset=folds[i], treeCount=100, featureSetCount=4, sampleSize=1000) 
    forest.createForest()
    # Predict the training folds
    forest.getForestResult(train_df)                       
    train_acc = forest_accuracy(forest.beanResults, train_df["Class"].tolist())

    # predict the testing fold
    forest.getForestResult()                               
    test_acc = forest_accuracy(forest.beanResults, folds[i]["Class"].tolist())

    print(f"Fold {i + 1}: train = {train_acc:.4f}, test = {test_acc:.4f}")

    predictions = []
    # Beans by classes for the current fold
    true_classes = forest.testingDataset["Class"].tolist()

    # Calculate accuracy
    scores.append(test_acc)

print(f"Average testing accuracy: {np.average(scores):.4f}")

# Tree visualization (optional)
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph = graph.render("beans")