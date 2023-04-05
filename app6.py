import numpy as np
from sklearn.linear_model import LinearRegression

x_train = np.array([ [1,2],[6,4],[7,10],[20,10],[11,31]])
y_train = np.array([-1,2,-3,10,-20])


model = LinearRegression()

model.fit(x_train,y_train)

x_test = np.array([[90,100], [250,50], [25,-50] ])
y_pred = model.predict(x_test)


print(y_pred)