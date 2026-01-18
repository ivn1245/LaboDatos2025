#!/usr/bin/env python
# coding: utf-8
"""

Alumnos: Camila Lamblot, Nicole Lamblot e Iván Pérez
Materia: Laboratorio de Datos

El siguiente archivo se encuentro dividido en 3 secciones:
    -Análisis exploratorio   
    -Clasificación binaria 
    -Clasificación multiclase

"""
# Visualizar imágenes


#%% Import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

#%%

#Función auxiliar que nos servirá para el punto 2
def lista_subconjuntos(l):
    res = []
    for i in range(1, len(l) + 1):
        sublista = l[-i:]
        res.append(sublista[::-1])
    return res

#Devuelve una lista de subconjuntos a partir de la lista original en el orden contrario.
#Por ejemplo, si tomamos la lista [1,2,3,4] la función devuelve [[4],[4,3],[4,3,2],[4,3,2,1]]

#%%
#################################################################################################################


#Analisis exploratorio


#################################################################################################################
np.random.seed(123) #para que sea siempre el mismo resultado cuando usamos cosas randoms
#%% Load dataset 

data_df = pd.read_csv("Fashion-MNIST.csv", index_col=0)
print(data_df.head())


#%% Select single image and convert to 28x28 array

img_nbr = 3

# keep label out
img = np.array(data_df.iloc[img_nbr,:-1]).reshape(28,28)


#%% Plot image

plt.imshow(img, cmap = "gray")


#%%

#1)
#a)

clases = [0,1,2,3,4,5,6,7,8,9]
matriz_promedios = []

#Hacemos un gráfico promedio de cada clase

for clase in clases:
    df_filtrado = data_df[data_df['label']==clase].drop('label',axis=1) #Armamos un df con todas las imágenes de esta clase
    promedios = df_filtrado.mean()
    img_promedio = promedios.values.reshape(28, 28)
    matriz_promedios.append(promedios)
    
    plt.imshow(img_promedio, cmap='gray')
    plt.title(f'Imagen promedio clase {clase}')
    plt.axis('off')
    plt.show()

matriz_promedios = np.array(matriz_promedios)

#Calculamos la desviación estándar de cada píxel entre clases.
std_entre_clases = matriz_promedios.std(axis=0) 

#Nos dice cuánto varía cada píxel del promedio entre las clases: si su desviación es baja, entonces no nos sirve para diferenciar las clases entre sí.

std_matriz = std_entre_clases.reshape(28,28)

#Gráfico de desviación estandar
plt.imshow(std_matriz, cmap='hot')
plt.title('Desviación estándar entre clases (por píxel)')
plt.colorbar()
plt.axis('off')
plt.show()

#%% 

#Hacemos un gráfico del promedio de todas las clases
promedio_total = data_df.drop('label',axis=1).mean()
img_promedio_total = promedio_total.values.reshape(28, 28)

plt.imshow(img_promedio_total, cmap='gray')
plt.title('Imagen promedio de todas las clases')
plt.axis('off')
plt.show()
#%%
#################################################################################################################


#Clasificacion binaria
# punto 2

#################################################################################################################

#%%

#Hacemos un gráfico que muestre la diferencia entre el promedio de la clase 0 y el de la 8
X = data_df.drop(columns='label')
y = data_df['label']

prom_0 = X[y == 0].mean().values
prom_8 = X[y == 8].mean().values
diff = np.abs(prom_0 - prom_8)
img_diff_prom=diff.reshape(28,28)
plt.imshow(img_diff_prom, cmap='hot')
plt.title("Diferencia promedio Clase 0 vs. 8")
plt.colorbar()
plt.show()

#%%
#hacemos un grafico para ver cuales son los 30 pixeles de mayor varianza

top_indices = np.argsort(diff)[-30:]
top_valores = diff[top_indices]
nro_pixel = data_df.columns[top_indices]
nro_pixel = list(nro_pixel)

plt.figure(figsize=(15,10))
plt.bar(range(30), top_valores,color="lightsalmon")
plt.xticks(range(30), nro_pixel, rotation=90,fontsize=18)
plt.xlabel("Pixel",fontsize=18)
plt.ylabel("Diferencia absoluta",fontsize=18)
plt.title("30 píxeles con mayor diferencia absoluta entre clase 0 y 8",fontsize=28)
plt.tight_layout()
plt.show()

#%%

# viendo la diferencia promedio entre las clases 0 y 8 podemos detectar 3 zonas de pixeles con mayor variacion, la superior centrica, la central izquierda y la central derecha
clase0=data_df[y == 0]
clase8=data_df[y == 8]

print(f'Vemos que hay {len(clase0)} imagenes del grupo 0 y {len(clase8)} de la clase 8')

#hay 7000 fotos en ambas categorias, por lo que el df esta balanceado

#%% 

#armamos el df solo con imagenes de las clases 0 y 8

clases0y8_df = data_df[(y == 0) | (y == 8)]

X = clases0y8_df.drop(columns=["label"])
Y = clases0y8_df["label"]

#%%

#buscamos cual es el k mas robusto para nuestro modelo

Nrep = 3
valores_n = range(1, 4)

resultados_test = np.zeros((Nrep, len(valores_n)))
resultados_train = np.zeros((Nrep, len(valores_n)))


for i in range(Nrep):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify=Y, random_state=42)  # 70% para train y 30% para test de forma balanceada
    for k in valores_n:
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train, Y_train) 
        Y_pred = model.predict(X_test)
        Y_pred_train = model.predict(X_train)
        acc_test = metrics.accuracy_score(Y_test, Y_pred)
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
        resultados_test[i, k-1] = acc_test
        resultados_train[i, k-1] = acc_train

#%%

promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test = np.mean(resultados_test, axis = 0) 


plt.plot(valores_n, promedios_train, label='Train', marker='o')
plt.plot(valores_n, promedios_test, label='Test', marker='s')

plt.legend()
plt.grid(True)
plt.title('Exactitud del modelo de kNN')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Exactitud (accuracy)')
plt.show()

#vemos que el k mas estable es el 3 si bien es mas alto que el 1 la efectividad con el test es practicamente la misma y a diferencia de k=1, no hace overfitting

#%%
#sabiendo que lo mas estable es usar el k=3 ahora resta probar distintos subconjuntos para entrenar al modelo
# con varios subconjuntos randoms de 5 pixeles de mayor variacion


#%%

valores_n = range(1, 5)

resultados_test = []
resultados_train = []
dfs=[]

#creamos las listas de subconjuntos de pixeles a testear
for i in range (len(valores_n)):
    muestra = np.random.choice(nro_pixel, size=5, replace=False)  # sin repetición
    lista=list(muestra)
    dfs.append(lista)

Y_prueba = Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify=Y, random_state=42)  # 70% para train y 30% para test de forma balanceada
#%%
#evaluamos para cada subconjunto de pixels con el mismo conjunto de datos
resultados_test = np.zeros((Nrep, len(valores_n)))
resultados_train = np.zeros((Nrep, len(valores_n)))

for r in range(Nrep):
    for i in range(len(dfs)):
        X_train0 = X_train[dfs[i]]
        X_test0 = X_test[dfs[i]]
        model = KNeighborsClassifier(n_neighbors = 3)
        model.fit(X_train0, Y_train) 
        Y_pred = model.predict(X_test0)
        Y_pred_train = model.predict(X_train0)
        acc_test = metrics.accuracy_score(Y_test, Y_pred)
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
        resultados_test[r, i-1] = acc_test
        resultados_train[r, i-1] = acc_train

#%%

promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test = np.mean(resultados_test, axis = 0) 

x = [1, 2, 3, 4]

plt.scatter(x, promedios_train, label='Train')
plt.scatter(x, promedios_test, label='Test')
plt.legend()
plt.grid(True)
plt.title('Prueba para distintos subconjuntos de 5 pixeles')
plt.xlabel('Número de subconjunto')
plt.ylabel('Exactitud (accuracy)')
plt.show()

#%%
#Ahora agarramos los 30 píxeles que hallamos con mayor varianza y tomamos los primeros 10.
#Con eso, creamos 10 subconjuntos que contengan de 1 a 10 atributos, respetando el orden de mayor varianza.


nro_pixel_acortado = nro_pixel[-10:]

#print(nro_pixel_acortado)

#Evaluamos para cada subconjunto de pixels con el mismo conjunto de datos con k fijo k = 3
Nrep = 3
lista_col_subsets = lista_subconjuntos(nro_pixel_acortado)
lista_cant_atributos = []
lista_exactitudes_train = []
lista_exactitudes_test = []

for subset in lista_col_subsets:
    acc_train_reps = []
    acc_test_reps = []
    for _ in range(Nrep):
        # Entrenamiento y evaluación con los atributos del subset
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[subset], Y_train)
        Y_pred_train = model.predict(X_train[subset])
        Y_pred_test = model.predict(X_test[subset])
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
        acc_test = metrics.accuracy_score(Y_test, Y_pred_test)
        acc_train_reps.append(acc_train)
        acc_test_reps.append(acc_test)

    # Promediamos exactitudes para este subset
    lista_cant_atributos.append(len(subset))
    lista_exactitudes_train.append(np.mean(acc_train_reps))
    lista_exactitudes_test.append(np.mean(acc_test_reps))

# Gráfico
plt.figure(figsize=(8,5))
plt.plot(lista_cant_atributos, lista_exactitudes_train, marker='o', label='Train')
plt.plot(lista_cant_atributos, lista_exactitudes_test, marker='s', label='Test')
plt.legend()
plt.title('Exactitud del modelo kNN vs Cantidad de atributos')
plt.xlabel('Cantidad de atributos')
plt.ylabel('Exactitud')
plt.grid(True)
plt.show()

#%%

#hacemos lo mismo que antes, pero ahora variamos el k

Nrep = 3
lista_col_subsets = lista_subconjuntos(nro_pixel_acortado)
lista_cant_atributos = []
lista_exactitudes_train = []
lista_exactitudes_test = []

k_valor = list(range(1,11))
exactitudes_test_matriz = np.zeros((len(lista_col_subsets), len(k_valor)))
exactitudes_train_matriz = np.zeros((len(lista_col_subsets), len(k_valor)))

for k in k_valor:
    for n in range(10):
        acc_train_reps = []
        acc_test_reps = []
        for _ in range(Nrep):
            subset = lista_col_subsets[n]
            # Entrenamiento y evaluación con los atributos del subset
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train[subset], Y_train)
            Y_pred_train = model.predict(X_train[subset])
            Y_pred_test = model.predict(X_test[subset])
            acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
            acc_test = metrics.accuracy_score(Y_test, Y_pred_test)
            acc_train_reps.append(acc_train)
            acc_test_reps.append(acc_test)
    
        # Promediamos exactitudes para este subset
        lista_cant_atributos.append(len(subset))
        lista_exactitudes_train.append(np.mean(acc_train_reps))
        lista_exactitudes_test.append(np.mean(acc_test_reps))
        
        exactitudes_test_matriz[n,k-1] = np.mean(acc_test_reps)
        exactitudes_train_matriz[n,k-1] = np.mean(acc_train_reps)

#%%

# Creamos un DataFrame de test
df_heatmap_test = pd.DataFrame(exactitudes_test_matriz, 
                          index=[len(subset) for subset in lista_col_subsets], 
                          columns=k_valor)

df_heatmap_test.index.name = 'Cantidad de atributos'
df_heatmap_test.columns.name = 'Valor de k'


# Creamos un DataFrame de train
df_heatmap_train = pd.DataFrame(exactitudes_train_matriz, 
                          index=[len(subset) for subset in lista_col_subsets], 
                          columns=k_valor)

df_heatmap_train.index.name = 'Cantidad de atributos'
df_heatmap_train.columns.name = 'Valor de k'


#buscamos el maximo y el minimo de ambos dfs para graficar con una misma escala
vmin = min(df_heatmap_test.min().min(), df_heatmap_train.min().min())
vmax = max(df_heatmap_test.max().max(), df_heatmap_train.max().max())

#Heatmap train
plt.figure(figsize=(10, 6))
sns.heatmap(df_heatmap_train, annot=True, fmt=".2f", cmap='viridis', cbar_kws={'label': 'Exactitud'}, vmin=vmin, vmax=vmax)
plt.title('Exactitud en train según cantidad de atributos y valor de k (kNN)')
plt.ylabel('Cantidad de atributos')
plt.xlabel('Valor de k')
plt.tight_layout()
plt.show()

#Heatmap test
plt.figure(figsize=(10, 6))
sns.heatmap(df_heatmap_test, annot=True, fmt=".2f", cmap='viridis', cbar_kws={'label': 'Exactitud'}, vmin=vmin, vmax=vmax)
plt.title('Exactitud en test según cantidad de atributos y valor de k (kNN)')
plt.ylabel('Cantidad de atributos')
plt.xlabel('Valor de k')
plt.tight_layout()
plt.show()
#%%
#################################################################################################################


#Clasificacion multiclase
# punto 3

#################################################################################################################
#%% 

#traemos nuevamente nuestro dataset

X = data_df.drop(columns='label')
Y = data_df['label']
#3)a)
# separamos el df en dev y held-out

X_dev, X_heldout, Y_dev, Y_heldout = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=42)
#%%
# definimos los hiperparametros que vamos a testear para ajustar el mejor arbol de decision
#criterios=["entropy"]
criterios=["entropy","gini"]

#3)b) y c)
#usamos profundidades pares para que tarde menos en ejecutar que si probamos todas 
profundidades=[2,4,6,8,10]

# hacemos el diccionario para guardar las listas de resultados de exactitud segun la condicion
resultados = {}

for criterio in criterios:
    exactitudes = []
    for profundidad in profundidades:
        modelo = tree.DecisionTreeClassifier(criterion=criterio, max_depth=profundidad, random_state=42)
        # lista de exactitud con el cross validation 
        scores = cross_val_score(modelo, X_dev, Y_dev, cv=5, scoring='accuracy', n_jobs=5)
        exactitud_promedio = scores.mean()
        exactitudes.append(exactitud_promedio)
    #agregamos al diccionario el criterio con la lista de exactitudes promedios
    resultados[criterio] = exactitudes

# Graficamos ambas series a la vez con el diccionario
plt.figure(figsize=(8,5))
for criterio in criterios:
    plt.plot(profundidades, resultados[criterio], marker='o', label=f"Criterion: {criterio}")
plt.xlabel("Profundidad del árbol")
plt.ylabel("Exactitud promedio")
plt.title("Comparación de árboles de decisión")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()


# %%

'''Vemos que la exactitud promedio crece a medida que crece la profundidad del arbol pero
como no varia demasiado a partir de max depth de 8, utilizaremos esa profundidad,
 por otro lado, ambos criterios mantienen una exactitud similar, por lo que elegiremos 
 entropy para seguir'''

# 3)d)

#creamos el modelo final
modelo_final = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=42)
modelo_final.fit(X_dev, Y_dev)

#%%
# Predecimos sobre el conjunto held-out
y_pred = modelo_final.predict(X_heldout)

# calculamos la exactitud del modelo
exactitud = metrics.accuracy_score(Y_heldout, y_pred)
print(f"Exactitud sobre el conjunto held-out: {exactitud}")

# Matriz de confusión
matriz_confusion = metrics.confusion_matrix(Y_heldout, y_pred)
print("Matriz de confusión:")

# graficamos la matriz de confusion
plt.figure(figsize=(8,6))
sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Oranges")
plt.title("Matriz de Confusión")
plt.show()
# %%
