# Este código es una manera de limpiar tu base de datos 
# para no tener datos faltantes
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# Importamos los datos
    # si importamos los datos de la siguiente manera
    # f=open("automobiledataset.txt","r")
    # print(f.read())
    # estaremos importando los datos como un archivo y no
    # podremos leerlos como un DataFrame. Para ello, lo haremos 
    # de la siguiente manera

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

f=pd.read_csv("automobiledataset.txt", header=None ,names=headers)

# usamos el método head() para mostrar las 1ras 5 filas
print(f.head())

    # Como puedes ver, varios signos de interrogación aparecieron en el marco de datos
    # esos valores que faltan pueden obstaculizar un análisis posterior.

    # Entonces, ¿cómo identificamos todos esos valores faltantes y lidiamos con ellos?

    # ¿Cómo trabajar con datos faltantes?

    # 1.-Identificar datos faltantes
    # 2.-Lidiar con datos faltantes
    # 3.-Corregir el formato de los datos

# reemplazaremos los ? por NaN
f.replace("?", np.nan , inplace=True)
print(f.head())

# Los valores faltantes se convierten por default. 
# Se puede usar dos métodos para detectar datos faltantes:

# .isnull()
# .notnull()

# El resultado es un valor booleano que indica si el valor que se pasa 
# como argumento es, de hecho, un dato faltante.

df=f.isnull()
df.head(5)

#Ahora contaremos cuantos valores faltantes hay por columna
for columna in df.columns.values.tolist():
    print(columna)
    print(df[columna].value_counts())
    print(" ")
    

# Aquí se recorre cada columna del DataFrame df para contar cuántos valores 
# faltantes hay en cada una.

# df[columna].value_counts(): Cuenta los valores True (faltantes) y False (no faltantes) 
# en cada columna. Esto muestra cuántos datos faltantes tiene cada columna.
# print(columna): Muestra el nombre de la columna actual que se está procesando.
# print(" "): Imprime una línea en blanco para separar la salida de cada columna

# Basado en el resumen anterior, cada columna tiene 205 filas de datos y siete de las columnas 
# contienen datos faltantes.

# "normalized-losses": 41 datos faltantes
# "num-of-doors": 2 datos faltantes
# "bore": 4 datos faltantes
# "stroke" : 4 datos faltantes
# "horsepower": 2 datos faltantes
# "peak-rpm": 2 datos faltantes
# "price": 4 datos faltantes

# ¿Cómo deberías tratar los datos faltantes?
# Eliminar datos
# a. Eliminar toda la fila
# b. Eliminar toda la columna
# Reemplazar datos
# a. Reemplazarlo por la media
# b. Reemplazarlo por la frecuencia
# c. Reemplazarlo basado en otras funciones

# Solo se debería eliminar columnas completas si la mayoría de las 
# entradas en la columna están vacías. En el conjunto de datos, 
# ninguna de las columnas está lo suficientemente vacía como para 
# eliminarla completamente.


# calcular el promedio para la columna "normalized-losses"
prom_norm_loss=f["normalized-losses"].astype(float).mean(axis=0)
print("promedio de normalized-losses:",prom_norm_loss)

# reemplazar "NaN" con el valor del promedio en la columna "normalized-losses"
f["normalized-losses"].replace(np.nan,prom_norm_loss,inplace=True)

# hacemos lo mismo para la columna "bore" , "horsepower" ,"peak-rpm",
prom_bore=f["bore"].astype(float).mean(axis=0)
print("promedio de bore:",prom_bore)
f["bore"].replace(np.nan,prom_bore,inplace=True)
prom_horsepower = f['horsepower'].astype('float').mean(axis=0)
print("promedio horsepower:", prom_horsepower)
f['horsepower'].replace(np.nan, prom_horsepower, inplace=True)
prom_peakrpm=f['peak-rpm'].astype('float').mean(axis=0)
print("promedio peak rpm:", prom_peakrpm)
f['peak-rpm'].replace(np.nan, prom_peakrpm, inplace=True)

# para ver que valores estan presentes en cada columna podemos usar .value_counts()
# para ademas ver el valor más comuún podemos usar .idmax()
f["num-of-doors"].value_counts().idxmax()

# remplazamos los valores de "num-of-doors" faltantes por el valor mas frecuente
f["num-of-doors"].replace(np.nan,"four",inplace=True)

# Por último,eliminamos todas las filas que no tengan datos de precios
# eliminamos toda la fila que tenga NaN en la columna precio
f.dropna(subset=["price"], axis=0, inplace=True)

# reiniciamos el conteo del indice por las filas eliminadas
f.reset_index(drop=True, inplace=True)

# nos muestra informacion general sobre nuestro dataframe
f.info()

# Grafico
# 1ro convertimos los datos al formato correcto
f["horsepower"]=f["horsepower"].astype(int, copy=True)

# realizamos un histograma de "horsepower" para ver la distribución de los datos
plt.hist(f["horsepower"])
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

# binning es un proceso de transformar variables numéricas continuas en categorías 
# discretas o "bins" para un análisis agrupado
# nosotros queremos incluir el valor mínimo y máximo de horsepower construyendo 3 bins
# de misma longitud, por ello se ocuparán 4 divisiones
bins = np.linspace(min(f["horsepower"]), max(f["horsepower"]), 4)
bins
nombres = ['bajo', 'medio', 'alto']
# aplicamos la función "cut" para determinar que valor de f["horsepower"] 
# corresponde a cada una de las secciones anteriores
f['horsepower-binned'] = pd.cut(f['horsepower'], bins, labels=nombres, include_lowest=True )
f[['horsepower','horsepower-binned']].head(20)
# valor de vehiculos en cada bin
f["horsepower-binned"].value_counts()

# histograma para visualizar la distribucion de los bins
plt.hist(f["horsepower"], bins = 3)
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

# Resumen
# 1.- Carga y limpieza de datos:El código maneja valores faltantes, reemplazando ? por NaN y luego rellenando esos valores con la media o la moda (valor más frecuente)
# 2.- Visualización de datos: Utiliza gráficos para mostrar la distribución de datos en la columna "horsepower" y su clasificación en categorías (bins)
# 3.- Binning:  Divide los valores continuos de "horsepower" en categorías discretas para un análisis más sencillo