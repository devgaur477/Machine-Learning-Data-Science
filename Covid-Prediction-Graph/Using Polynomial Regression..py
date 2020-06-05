# This is prediction made by using the help of polynomial Regression technique.Input can be changed through y_pred method. For an example
#i have take what will be the number of cases in next month that is 6. The prediction value if 204576.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('final_data.csv')
x = dataset.iloc[: , 2].values
y = dataset.iloc[: ,-1].values
x = x.reshape(-1 , 1)

from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()

lin_regressor.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 8)

x_poly= poly_regressor.fit_transform(x)

lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(x_poly, y)


plt.plot(x , y , color='red')
plt.plot(x , lin_regressor_2.predict(x_poly) , color= 'blue')
plt.title('COVID graph for india')
plt.xlabel('Months')
plt.ylabel('Number of cases')
plt.show()


poly_pred = lin_regressor_2.predict( poly_regressor.fit_transform([[7.5]]))


print(poly_pred)



#@ Dev Gaur 2020