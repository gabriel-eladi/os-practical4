import graphviz
import pandas as pd
import numpy as np
from sklearn import tree        
import RandomForest
from collections import Counter
from typing import List

training_df = pd.read_csv("dry_bean_train.csv")
test_df = pd.read_csv("dry_bean_test.csv")

x = training_df.drop(columns="Class").values.tolist()
y = training_df["Class"].astype("category").cat.codes.tolist()
beans_by_class = training_df["Class"].values
unique_bean_classes = np.unique(training_df["Class"])
bean_types = dict(enumerate(training_df["Class"].astype("category").cat.categories))

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
    return [training_df.iloc[fold] for fold in folds]


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
folds = stratified_kfold(k=10)
scores = []

# Cross-validation loop, 1 forrest per fold
for i in range(len(folds)):
    train_df = pd.concat([fold for j, fold in enumerate(folds) if j != i])
    # Each forrest gets a random sample
    forest = RandomForest.RandomForest(trainingDataset=train_df, testingDataset=folds[i], treeCount=100, featureSetCount=12, sampleSize=100) 
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

final_forest = RandomForest.RandomForest(trainingDataset=training_df,
                                         testingDataset=test_df,
                                         treeCount=100, featureSetCount=12, sampleSize=1000)
final_forest.createForest()
final_forest.getForestResult(test_df)           

# Drop old prediction column
output_df = test_df.drop(columns="Class", errors="ignore").copy()
# Add new class column
output_df["Class"] = final_forest.beanPredictions                    
output_df.to_csv("forest.csv", index=False)
print("forest.csv complete!")
print(output_df["Class"].value_counts())
# Tree visualization (optional)
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph = graph.render("beans")

#forest = RandomForest.RandomForest(treeCount=10000, featureSetCount=3, sampleSize=100, trainingDataset=training_df, testingDataset=None)
#forest.createForest()