# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:47:07 2018

@author: Prathyusha Mallela
"""
import pandas as pd;
import numpy as np;
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
boston=load_boston();
bos_x=pd.DataFrame(boston.data,columns=boston.feature_names);
#print(bos)
bos_y=pd.DataFrame(boston.target);
#print(bos_y)
bos_x.describe();
reg = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(bos_x, bos_y, test_size=0.2, random_state=4)
x_train.head();
reg.fit(x_train,y_train);
reg.coef_
reg.predict(x_test)
print(reg.predict(x_test))
#np.mean((reg.predict(x_test) - y_test)**2)
#print(y_test)