import numpy as np
import pandas as pd

pulsar_stars_df = pd.read_csv('pulsar_stars.csv')

feature_names = [i for i in list(pulsar_stars_df) if i != 'target_class']

X = pulsar_stars_df[list(feature_names)]
y = pulsar_stars_df['target_class']
# print(feature_names)



from sklearn.model_selection import train_test_split

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=4)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


for k in range(1,10):
	knn_pulsar = KNeighborsClassifier(n_neighbors=k) 
	knn_pulsar.fit(X_train, y_train)
	y_predict_knn = knn_pulsar.predict(X_test)
	knn_acc_list = cross_val_score(knn_pulsar, X, y, cv=10, scoring='precision')
	print("K =",k, "\n", knn_acc_list, "\n Mean Accuracy: ", knn_acc_list.mean(), end="\n\n")


