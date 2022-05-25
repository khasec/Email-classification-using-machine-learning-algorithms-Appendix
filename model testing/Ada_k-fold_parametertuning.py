from unicodedata import category
import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt



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


test1=[1,2,3]
test2=[50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]
test3=[0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7]
testrun=[]
for i in test3:
    print(i)
    for train_index, test_index in kf.split(X):


        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Changes the strings in the data into numerical data
        feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='False')
        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test)
        Y_train = Y_train.astype('int')
        Y_test = Y_test.astype('int')

        # Inisilasing Adaptive Boosting with base_estimator = None , n_estimators =100 , learning_rate =1.4
        # define base model
        base = DecisionTreeClassifier(max_depth=1)
        model = AdaBoostClassifier ( base_estimator = base , n_estimators =70 , learning_rate =i)
        model.fit(X_train_features,Y_train)
        prediction = model.predict(X_test_features)
        classification.append(np.mean(prediction == Y_test))

    testrun.append(np.mean(classification))
 
    
#plot the result 

plt.plot(test3,testrun)
plt.suptitle('Mean accuracy for different values of learning_rate')

plt.ylabel("Mean accuracy")
plt.xlabel("learning_rate")
plt.show() 



#Prints the result for 10 k-fold cross validation
print("_______________Ada________________")
print(f"Mean accuracy: {np.mean(classification)}")
print(f"Mean std: {np.std(classification)}")
    
    
