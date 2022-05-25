import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#read the data and changes spam to a 0 and not spam to a 1
data= pd.read_csv("mail.csv")


#devide the data to train and test data, using around 75% as training data 
X=data["Message"]
y=data["Category"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=48)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
X_train = feature_extraction.fit_transform(X_train)

X_test = feature_extraction.transform(X_test)

X_train=X_train.toarray()
X_test=X_test.toarray()

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


classrate=[]

model = skl_nb.KNeighborsClassifier(n_neighbors=1)
model.fit(X_train,Y_train)

prediction = model.predict(X_test)
classrate.append(np.mean(prediction == Y_test))




print(pd.crosstab(prediction,Y_test), '\n')
print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")
