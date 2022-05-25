#imported libraries

from fileinput import filename
import os.path

from google.oauth2.credentials import Credentials

from googleapiclient.discovery import build

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt



#formatting the data and prepering it for the model
data= pd.read_csv("mail.csv")

X=data["Message"]
y=data["Category"]
z=data["real"]

X_train = X[:1000]
Y_train = y[:1000]

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
X_train = feature_extraction.fit_transform(X_train)



X_train=X_train.toarray()


Y_train = Y_train.astype('int')


# define the keras model and building the neural network model
model = Sequential()
model.add(Dense(100, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X_train, Y_train, shuffle=True, epochs=20, batch_size=10)



Xt_train = X[1000:-1]
Y_test =y[1000:-1]
z_test = z[1000:-1]
Input_test = feature_extraction.transform(Xt_train)
Input_test=Input_test.toarray()

#test the model on the new data and adding it to the data file. 

predictions = model.predict(Input_test)

hold=np.round_(predictions)
z_test2=z_test.to_list()
k=0
count=0
for i in hold:
    if not i == z_test2[k]:
        count = 1 + count
    k = k + 1

print(f'total number of errors {count}')



x = [count,74]
num_bins = 2
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
plt.show()



        
