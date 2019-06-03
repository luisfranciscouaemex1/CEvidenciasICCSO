from sklearn.linear_model import LinearRegression #importamos sklearn
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt			#Importamos matplot
import pandas as pd				#Importamos pandas

dataset = pd.read_csv("Examen_Segundo_Parcial_ICCSO/exam_A_dataset.csv") #Cargamos el archivo
X = dataset.iloc[:,:-1].values			#Cargamos primera columna
y = dataset.iloc[:,1].values			#Cargamos segunda columna

lin_reg = LinearRegression()			#Llamamos la funcion LinearRegression
poly_reg = PolynomialFeatures(degree=4)		#Ajustamos el polinomio a grado 4

X_poly = poly_reg.fit_transform(X)		#Transformamos la columna x
poly_reg.fit(X_poly,y)				#Llamamos la funcion de regresion
lin_reg.fit(X_poly,y)			

plt.scatter(X,y)				#Graficamos x
plt.scatter(X,lin_reg.predict(poly_reg.fit_transform(X)))
plt.show()					#Mostramos la grafica