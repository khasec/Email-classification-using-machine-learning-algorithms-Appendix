from unicodedata import category
import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


#read the data and changes spam to a 0 and not spam to a 1
data= pd.read_csv("mail.csv")

X=data["Message"]
Y=data["Category"]


#inisilasing a 10 kfold cross validation 
kf = KFold(n_splits=100)
KFold(n_splits=100, random_state=None, shuffle=False)
classification=[]
save_classification=[]
index=[]


test1=[1,10,20,30,40,50,60]
test2=[50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]
test3=[1,10,50,100,150,200,400,600,800,990]

testrun=[]
for i in test2:
    print(i)
    for train_index, test_index in kf.split(X):


        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Changes the strings in the data into numerical data
        feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test)
        Y_train = Y_train.astype('int')
        Y_test = Y_test.astype('int')

        # Inisilasing Adaptive Boosting with base_estimator = None , n_estimators =100 , learning_rate =1.4
        # define base model
    
        model = RandomForestClassifier(max_depth=50, n_estimators=i, max_samples=990)
        model.fit(X_train_features,Y_train)
        prediction = model.predict(X_test_features)
        classification.append(np.mean(prediction == Y_test))

    testrun.append(np.mean(classification))
 
    
#plot the result 

plt.plot(test2,testrun)
plt.suptitle('Mean accuracy for different values of n_estimators')

plt.ylabel("Mean accuracy")
plt.xlabel("n_estimators")
plt.show() 


#Prints the result for 10 k-fold cross validation
print("_______________Ada________________")
print(f"Mean accuracy: {np.mean(classification)}")
print(f"Mean std: {np.std(classification)}")
    
    