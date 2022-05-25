from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer

#read the data and setting it up for the nureal network
data= pd.read_csv("mail.csv")






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

# define the keras model
model = Sequential()
model.add(Dense(2, input_dim=X_train.shape[1], activation='sigmoid'))

model.add(Dense(10, activation='sigmoid'))


model.add(Dense(1, activation='sigmoid'))

model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['accuracy'])

#run the model

history=model.fit(X_train, Y_train, shuffle=True, epochs=20, batch_size=10)

#plor the loss and accuracy

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

print(model.evaluate(X_test, Y_test))

predict=model.predict(X_test)
rounded_predict=np.around(predict)
test=[]
for i in rounded_predict:
  test.append(int(i[0]))

Y_test2=Y_test.tolist()


Y_test3 = np.array(Y_test2)
test2 = np.array(test)


print(pd.crosstab(test2,Y_test3), '\n')
#print confusion matrix