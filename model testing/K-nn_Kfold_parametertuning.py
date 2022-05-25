from unicodedata import category
import sklearn.neighbors as skl_nb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


#read the data and changes spam to a 0 and not spam to a 1
data= pd.read_csv("mail.csv")

X=data["Message"]
Y=data["Category"]

#inisilasing a 10 kfold cross validation for diffrent k
kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=False)

save_classification=[]
index=[]


for i in range(100):
    classification=[]
    for train_index, test_index in kf.split(X):


        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Changes the strings in the data into numerical data
        feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test)
        Y_train = Y_train.astype('int')
        Y_test = Y_test.astype('int')

        # Inisilasing k-nn method and using diffrent k
        model = skl_nb.KNeighborsClassifier(n_neighbors=i+1)
        model.fit(X_train_features,Y_train)
        prediction = model.predict(X_test_features)
        classification.append(np.mean(prediction == Y_test))

    #displaying the result for diffrent k after kfold
    save_classification.append(np.mean(classification))
    index.append(i)
    print("_______________K-nn________________")
    print(f"Mean accuracy: {np.mean(classification)}")
    print(f"Mean std: {np.std(classification)}")
    print(f"for k = {i+1}")


plt.plot(index,save_classification)
plt.suptitle('Mean accuracy for different values of k')
plt.ylabel("Mean accuracy")
plt.xlabel("variable k")
plt.show() 

#desplaying the best result
print("______Best run______")
print(f"best mean accuracy {max(save_classification)}")
print(f"for k = {save_classification.index(max(save_classification))+1}")