# Импорт библиотек
import numpy as np
import pandas as pd
import os, sys
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Загрузка и фильтрование данных датасета
data=pd.read_csv("Mall_Customers.csv")
data = data.drop(['Age'], 1)
data = data.drop(['CustomerID'],1)
data = data.drop(['Annual Income (k$)'],1)
data = data.drop(['Gender'],1)
print(data)


# Загрузка набора данных и создание отложенного представления
values = DataFrame(data.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.colimns = ['t-1', 't+1']
print(dataframe.head(5))
# Создание и разделене наборов данных, 50% под оценку и 50% под прогнозирование
X = dataframe.values
train_size = int(len(X)*0.50)
train, test = X[1:train_size], X[train_size:]
train_x, train_y = train[:,0], train[:,1]
test_x, test_y = test[:,0],test[:,1]
# Определения модели постоянтсва
def model_pers(x):
    return x
# Сотставление и оценка прогноза 
prediction = list()
for x in test_x:
    y_hat = model_pers(x)
    prediction.append(y_hat)
# Создание графиков прогноза 
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in prediction])
data.plot()
pyplot.show()