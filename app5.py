import numpy as np
from sklearn.linear_model import LinearRegression

x_train = np.array([ [1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
y_train = np.array([6,15,24,33,42])


model = LinearRegression()

model.fit(x_train,y_train)

x_test = np.array([[100,100,100], [4,2,5], [125,5,10 ] ])
y_pred = model.predict(x_test)


print(y_pred)