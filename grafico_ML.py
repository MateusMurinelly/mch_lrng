import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

areas = np.array([80,100,120,140,160,180,200,220,240,260])
precos = np.array([400000,500000,600000,700000,800000,900000,1000000,1100000,1200000,1300000])

reg = LinearRegression()
reg.fit(areas.reshape(-1,1),precos)


nova_area = np.array([100]).reshape(-1,1)
preco_previsto = reg.predict(nova_area)


plt.scatter(areas,precos)
plt.plot(areas, reg.predict(areas.reshape(-1,1)),color='red')
plt.xlabel('area')
plt.ylabel('preco')
plt.title('regressão linear')
plt.show()

print(f'Preço final:{preco_previsto[0]:,.2f}')