import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn . ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


data= pd.read_csv("mail.csv")


#devide the data to train and test data, using around 75% as training data 
X=data["Message"]
y=data["Category"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=48)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
X_train = feature_extraction.fit_transform(X_train)

X_test = feature_extraction.transform(X_test)



# Inisilasing Adaptive Boosting with base_estimator = base , n_estimators =100 , learning_rate =1.2

base = DecisionTreeClassifier(max_depth=1)
model = RandomForestClassifier(max_depth=50, n_estimators=90, max_samples=800)

model.fit(X_train,Y_train)
 


prediction = model.predict(X_test)





print(pd.crosstab(prediction,Y_test), '\n')
print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")