import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data= pd.read_csv("mail.csv")


#devide the data to train and test data, using around 75% as training data 
X=data["Message"]
y=data["Category"]
z=data["real"]

hold = 0

k=0

for i in z:
    if i != y[k]:
        hold=hold+1
    k=k+1

hold0 = 0
hold1 = 0
k=0

print(f"total number of errors {hold}")
print(hold/(len(X)))

for i in z:
    if i != y[k] and i == 0:
        hold0=hold0+1
    elif i != y[k] and i == 1:
        hold1=hold1+1
    k=k+1


print(f"total number of errors r/gaming {hold0}")
print(f"total number of errors r/worldnews {hold1}")

hold0 = []
hold1 = 0
k=0



for i in z:
    if i != y[k]:
        hold1 = hold1+1
    hold0.append(hold1)
    k=k+1



index = list(range(0, len(hold0)))

plt.plot(index,hold0)
plt.suptitle('Miss classifications over iterations')
plt.ylabel("Miss classification")
plt.xlabel("Number of iterations")
plt.show() 



