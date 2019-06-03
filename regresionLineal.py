import pandas as pd		#Importamos panda
from sklearn.cross_validation import train_test_split	#importamos sklern
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt				#Importamos matlab

dataset = pd.read_csv('Salary_Data.csv')		#Cargamos el archivo a trabajar
X = dataset.iloc[:,:-1].values				#Tomamos la primera columna
y = dataset.iloc[:,1].values				#Tomamos la segunda columna

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
regressor = LinearRegression()			#Aplicamos la funcion linear regresion
regressor.fit(X_train, y_train)			#Llamamos al metodo fit con sus respectivos parametros

y_pred = regressor.predict(X_test)		
plt.scatter(X_train, y_train, color = 'red')	#Graficamos los valores obtenidos
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()					#Mostramos la grafica