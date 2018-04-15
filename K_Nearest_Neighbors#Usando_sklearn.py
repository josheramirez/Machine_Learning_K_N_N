import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df=pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1,inplace=True)

#guardo en x el dataframe sin la columna class
X=np.array(df.drop(['class'],1))
#guardo en y la columna class
y=np.array(df['class'])


#creo la relacion entra el vector resultados y los vectores de datos y creo las variables de prueba y testeo
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

#creo el objeto con el algoritmo KNN
clf=neighbors.KNeighborsClassifier()
#entrno el algoritmo
clf.fit(X_train, y_train)
#exactitud de clf (entrenamiento)
exactitud=clf.score(X_test,y_test)
print(exactitud)

#probando el algoritmo
#creo un arreglo numpy para predecir (cada elemento de la lista es una columna)
""" 
    Parametro                  valor posible
    
    Clump Thickness               1 - 10
    Uniformity of Cell Size       1 - 10
    Uniformity of Cell Shape      1 - 10
    Marginal Adhesion             1 - 10
    Single Epithelial Cell Size   1 - 10
    Bare Nuclei                   1 - 10
    Bland Chromatin               1 - 10
    Normal Nucleoli               1 - 10
    Mitoses                       1 - 10
    
    lista de 9 elementos
""" 

medidas_ejemplo=np.array([8,10,10,8,7,10,9,7,1])
# reshape cambia la forma de la matriz a 1 fila -1 desconocidas columnas
medidas_ejemplo=medidas_ejemplo.reshape(1,-1)
#calculo el 
prediccion=clf.predict(medidas_ejemplo)
#print("Prediccion es : ",type(prediccion))
if prediccion[0]==2:
    print("Cancer beningno")
else:
    print("Cancer maligno")