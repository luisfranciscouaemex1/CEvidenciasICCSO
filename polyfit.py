import numpy as np 		#importa la libreria numpy
def polyfit1(x,y,n):		#Define la funcion polyfit1 recibiendo 3 parametros
    
    def inv(A):			#Define la funcion inv que recibe A como parametro
        return np.linalg.inv(A)	#Retorna el valor de la funcion linalg
    def trans(A):		#Define la funcion trans que recibe A como parametro
        return A.getT()
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B):		#Define la funcion prod que recibe A y B como parametro
        return np.dot(A,B)
    
    xlen = len(x)		#La variable xlen almacena la longitud de x
    ylen = len(y)		#La variable ylen almacena la longitud de y
    one = np.ones((xlen,n+1),dtype=int)	
    c1=one[:,[1]]		
    xT=np.matrix(x)		#Almacena el valor de la matriz
    yT=np.matrix(y)		#Almacena el valor de la matriz
    A=np.hstack([c1,xT.getT()])
    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT)) #Retorna el resultado de la ecuacion
    
x=[1,1.6,3.4,4,5.2]
y=[1.2,2,2.4,3.5,3.5]

polyfit1(x,y,1)			#Llama la funcion polyfit de grado uno

def polyfit2(x,y,n):		#definimos la funcion de grado 2
    
    def inv(A):
        return np.linalg.inv(A)	#Mismo procedimiento de la funcion de grado 1
    def trans(A):
        return A.getT()
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B):
        return np.dot(A,B)
    
    xlen = len(x)		#La variable xlen almacena la longitud de x
    ylen = len(y)		#La variable ylen almacena la longitud de y
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)		#Elevamos a la potencia cuadrada
    A=np.hstack([c1,c2,c3])
    print(A)			#imprimimos el valor de A
    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))

x=[1,2,3]
y=[1,4,9]

polyfit2(x,y,2)			#Llamamos a la funcion polyfit2 de segundo grado
