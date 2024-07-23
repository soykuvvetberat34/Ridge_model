import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\Hitters.csv")
datas=datas.dropna()#NaN değerine sahip verileri siler
dummies=pd.get_dummies(datas[["League","Division","NewLeague"]])#sınıflandırma verilerini binary verilere çevirir
y=datas["Salary"]
x_=datas.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
x=pd.concat([x_,dummies],axis=1)


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=99)


#ridge model
ridge_model=Ridge()
ridge_model.fit(x_train,y_train)
predict=ridge_model.predict(x_test)
RMSE=np.sqrt(np.mean(mean_squared_error(y_test,predict)))
#cross validation yapılmış tahmin
cv=cross_val_predict(ridge_model,x_train,y_train,cv=10)
#yapıyı 10 a böldü her biri için tahminlerde bulundu
cv_mean_sqrt=np.sqrt(np.mean(cv))

lambdalar1=np.random.randint(-2,10,100)#-2 ile 10 arasında 100 tane int getir
lambdalar2=np.linspace(10,-2,100)#10 ile -2 arasındaki sayıları 100 parça olacak şekilde eşit olarak ayır
lambdalar2=10**lambdalar2
#ridgecv ile optimum alpha değerini bulma
ridge_cv=RidgeCV(alphas=lambdalar2,cv=10,scoring="neg_mean_squared_error")
ridge_cv.fit(x_train,y_train)
#optimum alpha değerini almak 
optimum_alpha=ridge_cv.alpha_
ridge_tuned=Ridge(alpha=optimum_alpha)
ridge_tuned.fit(x_train,y_train)
predict_tuned=ridge_tuned.predict(x_test)
RMSE2=np.sqrt(np.mean(mean_squared_error(y_test,predict_tuned)))
print(RMSE2)
