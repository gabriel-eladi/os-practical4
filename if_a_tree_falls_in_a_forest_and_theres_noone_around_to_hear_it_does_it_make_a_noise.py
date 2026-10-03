import graphviz
import pandas as pd
from sklearn import tree

training_df = pd.read_csv("dry_bean_train.csv")
test_df = pd.read_csv("dry_bean_train.csv").drop(columns="Class").values.tolist()

x = df.drop(columns="Class").values.tolist()
y = df["Class"].astype("category").cat.codes.tolist()
bean_types = dict(enumerate(df["Class"].astype("category").cat.categories))

# print(classifiers)
# print(features)

# le = LabelEncoder()

# print(f"number of rows = {len(rows)}, number of features = {len(fields)}")
# print("field names: [" + ', '.join(fields) + ']')

clf = tree.DecisionTreeClassifier().fit(x, y)

beans = []
for bean in clf.predict(test):
    beans.append(bean_types[bean])


print(beans[0:10])

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph = graph.render("beans")