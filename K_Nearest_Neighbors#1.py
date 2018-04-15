import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data,predict,k):
    if len(data)>=k:
        warnings.warn('K es menor que el total de grupos')
    distances=[]
#recorro  la matriz con datos "puntos" (data), para calcular distancias 
    for group in data:
        for features in data[group]:
            #distancia euclidea cuadratica basica solo sirve para 2 dimensiones
            #euclidean_distance=sqrt((feature[0]-predict[0])**2+(features[1]-predict[1])**2)
            
            #uso numpy para hacer calculos multidimensional
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
            #guarda cada distancia [distancia,grupo al que pertenece]
            distances.append([euclidean_distance,group])
    
    #guardo el 2º elemento del vector distancia(grupo) , solo los 3 primeros elementos [:K] 
    votes=[i[1] for i in sorted(distances)[:k]]
    #Counter devulve la cantidad de veces q se repite un elemento en cada lista
    #metodo .most_common(n) devuelve los n conjuntos de datos mas repetidos
    #n=1, el mas repetido (dato,cantidad de veces)
    #n=2, el mas repetido y el 2º mas repetido (dato, cantidad de veces)
    
            # print(Counter(votes).most_common(1))
    # .most_common(n)[0][0] -> devuelve solo el dato mas repetido
    # .most.commom(n)[0][1] -> devuleve la cantidad de veces q se repite 
    
    vote_result=Counter(votes).most_common(1)[0][0]
    # la confianza indica que grado de certeza tiene el el resultado vote_result
    # si bien encuentra el grupo al cual mas se acerca el dato, la confianza indica que tanto se acerca al grupo ese dato
    # si tengo un dato comparado con 5 vecinos y ese datos tiene mayor cantidad de cercania al grupoA (3 vecinos cerca) contra a grupo B (2)
    # se dice que el grado de confianza en la obtencion de este dato es de 5/3=(0.6)=60%
    
    confianza=Counter(votes).most_common(1)[0][1]/k
    return vote_result,confianza


exactitud=[]

#iteracion para calcular el promedio de confianza
for i in range(1):
    df=pd.read_csv("breast-cancer-wisconsin.data")
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)
    
    #.astype retorna un array nuevo con todos sus elementos (float) 
    #value to list pasa info del diccionario a una lista
    
    #diccionario es un tipo de dato que almacena datos de forma
    #diccionario={'key_1':valor_1,'key_2':valor_2}
    #se maneja distintos a list o array
    
    #array en python guarda datos homogeneos
    #list es un tipo de array que gusrada datos heterogeneos
    #creo nueva lista con datos tipo float 
    full_data=df.astype(float).values.tolist()
    #print(full_data)
            #print(full_data[-1])
            #print(full_data[-3:])
    #random.shuffle reordena de forma aleatoria la lista
    random.shuffle(full_data)
    #tamaño test (0.2, es un 20%)
    test_size=0.2
    #se declara un diccionario con 2 elementos (llamados 2 y 4)
    # 2:[] (key es 2 y valor es [], lista vacia)
    # 4:[] (key es 4 y valor es [], lista vacia)
    train_set={2:[], 4:[]}
    # el 2 corresponde a la clase beningno
    # el 4 corresponde a la clase maligno
    test_set={2:[],4:[]}
    #creo lista basado en full_data descontando el 20% final
    #[-1] -> es el ultimo elemento de la lista
    #[-2] -> es el penultimo elemento de la lista
    #[:5] -> considera desde el primer elemento hasta el 5ª
    #[5:] -> considera desde el 5ª elemento hasta el ultimo
    #[-3:] -> considera desde 3ª ultimo elemento hasta el final
    #[:-3] -> considera desde el primer elemento hasta el 3ª ultimo elemento
    
    # toma desde el 1º elemento hasta el 20º ultim elemento
    train_data=full_data[:-int(test_size*len(full_data))]
    # toma desde el 20º ultimo elemento hasta el ultimo elemento 
    test_data=full_data[-int(test_size*len(full_data)):]
    
    # por cada fila de train_data, agrego en la columna [i[-1]] (que puede ser columna 2 o 4) de train set
    # la lista de elementos desde el 1º hasta el -1 elemento final, es decir guardo todo menos la clase
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    
    correct=0
    total=0
    
    
    # aqui se recorrec cada grupo de test_set, cada grupo es una de las lista dentro de test_set[2 o 4]
    # data es cada lista contenida dentro de lista de grupo (filas con elementos) [[fila1],[fila...n]]
    # aloritmo k_nearest_neighbors toma el la lista dentro de la lista (fila con elementos), train_set, y k que es la cantidad de vecinos
    # el algoritmo calcula la distancia entre data y el total de de elementos de de train_set
    # en trega como respuesta el grupo al que esta mas cercano el data (entre los k vecinos mas cercanos)
    # si vote (respuesta) coincide con con el grupo que se esta analizando se considera como prediccion correcta y aumnenta en 1 correct
    #por cada iteracion de listas de ambos grupos se aumenta el contador total
    # se calcula la exactitud como correct/total
    for group in test_set:
        for data in test_set[group]:
            vote,confianza=k_nearest_neighbors(train_set, data, k=5)
            if group==vote:
                correct+=1
           # else:
                #imprimo aquellos que son erroneos
                #print(confianza)
            total+=1
            
    #print('Exactitud del algoritmo: ', correct/total )
    exactitud.append(correct/total)

#exactitud del algoritmo al predecir
print('Promedio de exactitud: ',sum(exactitud)/len(exactitud))

#(x,y) x es el tipo de cancer(4,maligno o 2,beningno)
#y es el grado de certeza de esta prediccion 

print(k_nearest_neighbors(train_set,[4,2,1,8,7,2,3,1,1],k=3))