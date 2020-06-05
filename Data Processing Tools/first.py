import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values             
y = dataset.iloc[: ,-1].values



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan , strategy='mean')
imputer.fit(x[:,1:3])
x[: ,1:3] =  imputer.transform(x[: ,1:3])

print('This is previous value')
print(x)
print()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers =[('encoder',OneHotEncoder(),[0])] , remainder= 'passthrough')
x= np.array(ct.fit_transform(x))

print(x)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y =le.fit_transform(y)

print()
print(y)
#Splitting the dataset into the raining set and test set.
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x, y , test_size = 0.2 , random_state = 1)
print('these are training and test outputs')
print()
print()
print(x_train)
print()
print(x_test)
print()
print(y_train)
print()
print(y_test)

#Always remeber that splitting is done before feature scaling to prevent data leakag.

#Feature Scaling : To avoid dominating features. Applied for some of them only not all.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[: , 3:5] = sc.fit_transform(x_train[: , 3:5])
x_test[: , 3:5] = sc.fit_transform(x_test[: , 3:5])
print()
print(x_train)