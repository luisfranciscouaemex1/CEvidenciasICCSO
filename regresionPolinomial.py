import numpy as np		#Importamos numpy
import pandas as pd		#Importamos pandas
import matplotlib.pyplot as plt	#Importamos matplolib
import sympy as S		#Importamos sympy

file = "Examen_Segundo_Parcial_ICCSO/exam_A_dataset.csv"	#Cargamos el archivo de datos
data = np.loadtxt(file, delimiter = ',', skiprows = 1, usecols=[0,1])
X = np.loadtxt(file, delimiter = ',', skiprows = 1, usecols=[0])	#Tomamos la colmna 1
Y = np.loadtxt(file, delimiter = ',', skiprows = 1, usecols=[1])	#Tomamos la columna 2



def polyfit2(x,y,n):		#Definimos la funcion de grado 2

    def inv(A):			#Tomamos valores de A
        return np.linalg.inv(A)
    def trans(A):
        return A.getT()		#Retornamos valor de A
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B):
        return np.dot(A,B)

    xlen = len(x)		#Obtenemos longitud de x
    ylen = len(y)		#Obtenemos longitud de y
    one = np.ones((xlen,n+1),dtype=int)	#Aplicamos operacions
    c1=one[:,[1]]		#Polinomio de grado 1
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)		#Elevamos al cuadrado
    A=np.hstack([c1,c2,c3])	#Obtenemos 3 resultados

    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT)) #Regresamos resultado final

#print(polyfit2(X,Y,2))

def polyfit3(x,y,n):

    def inv(A):
        return np.linalg.inv(A)
    def trans(A):
        return A.getT()
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B):
        return np.dot(A,B)

    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    c4=np.power(c2,3)
    A=np.hstack([c1,c2,c3,c4])

    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))

def polyfit4(x,y,n):

    def inv(A):
        return np.linalg.inv(A)
    def trans(A):
        return A.getT()
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B):
        return np.dot(A,B)

    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    c4=np.power(c2,3)
    c5=np.power(c2,4)
    A=np.hstack([c1,c2,c3,c4,c5])

    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))
print(polyfit4(X,Y,2))

x = S.symbols('x')
y = S.symbols('y')


y = -3.26599160e+00 + 5.53685798e-01 * x -1.96329350e+00 * pow(x,2) + 5.21537693e-01*pow(x,3) + -2.39372653e-03* pow(x,4)

f= S.lambdify(x,y,'math')
yen= f(X)
print (yen)
yout=yen.astype(list)

plt.scatter(X,Y,color = 'blue')
plt.scatter(X,yout,color = 'yellow')
plt.show()