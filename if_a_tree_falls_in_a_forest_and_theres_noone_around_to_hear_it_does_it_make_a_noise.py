import graphviz
from typing import List
import pandas as pd
import numpy as np
from sklearn import tree        

training_df = pd.read_csv("dry_bean_train.csv")
test_df = pd.read_csv("dry_bean_train.csv").drop(columns="Class").values.tolist()

df = training_df
x = df.drop(columns="Class").values.tolist()
y = df["Class"].astype("category").cat.codes.tolist()
beans_by_class = df["Class"].values
unique_bean_classes = np.unique(df["Class"])
bean_types = dict(enumerate(df["Class"].astype("category").cat.categories))

# print(classifiers)
# print(features)

# le = LabelEncoder()

# print(f"number of rows = {len(rows)}, number of features = {len(fields)}")
# print("field names: [" + ', '.join(fields) + ']')

clf = tree.DecisionTreeClassifier().fit(x, y)

beans = []
for bean in clf.predict(test_df):
    beans.append(bean_types[bean])

print(unique_bean_classes)
print(type(y))

def stratified_kfold(k: int) -> List[List[int]]:
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
    
    #for i in folds:
    #    bean_types_in_fold = [beans_by_class[idx] for idx in i]
    #    print(f"fold {folds.index(i)}: {bean_types_in_fold}")
    return folds
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph = graph.render("beans")

for fold in stratified_kfold(5):
    print(f"Fold {stratified_kfold(5).index(fold)}: {fold}")