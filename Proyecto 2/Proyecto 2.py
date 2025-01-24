# como escoger el mejor método de visualización?
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# el archivo CSV lo tengo en la misma carpeta donde se instaló python
f=pd.read_csv("automobileEDA.csv")

# Al visualizar variables individuales, es importante primero entender 
# qué tipo de variable estás tratando. Esto nos ayudará a encontrar el método 
# de visualización adecuado para esa variable.
print(f.dtypes)
sns.regplot(x="engine-size",y="price",data=f) #figura 1
plt.ylim(0,)
plt.show()
# conforme engine-size sube , price también lo hace lo cuál
# indica una correlación directa positiva entre estas dos variables
print(f[["engine-size", "price"]].corr())

sns.regplot(x="highway-mpg", y="price", data=f)#figura 2
plt.show()
# conforme highway-mpg sube ,price baja lo cuál
# indica una relación inversa/negativa entre estas dos variables
print(f[['highway-mpg', 'price']].corr())

# Ahora veamos si "peak-rpm" es una buena variable predictora de "price"
sns.regplot(x="peak-rpm", y="price", data=f)#figura 3
plt.show()
# No parece en absoluto un buen predictor de price ya que la línea de 
# regresión es casi horizontal. Además, los puntos de datos están muy 
# dispersos y alejados de la línea ajustada, lo que muestra mucha variabilidad. 
# Por lo tanto, no es una variable confiable.
print(f[['peak-rpm','price']].corr())

# veamos la relación entre "body-style" y "price"
sns.boxplot(x="body-style", y="price", data=f)#figura 4
plt.show()
# vemos que las distribuciones de precios entre las diferentes categorías de 
# body-style tienen una superposición significativa, por lo que body-style no 
# sería un buen predictor del precio.En cambio:
sns.boxplot(x="engine-location", y="price", data=f)#figura 5
plt.show()
# Vemos que la distribución de precio entre estas dos categorías de ubicación 
# del motor (engine-location), delantera y trasera, son lo suficientemente distintas 
# como para considerar la ubicación del motor como un buen predictor potencial del precio.

# Calculemos el Coeficiente de correlación de Pearson y el p-valor de Wheel-Base vs. Price
pearson_coef, p_value = stats.pearsonr(f['wheel-base'], f['price'])
print("el Coeficiente de correlación de Pearson", pearson_coef, " con un p-valor de P =", p_value)
# La correlación de Pearson mide la dependencia lineal entre dos variables X e Y.
# El coeficiente resultante es un valor entre -1 y 1 , donde:
# 1: Correlación lineal positiva perfecta.
# 0: Sin correlación lineal, lo más probable es que las dos variables no se afecten entre sí.
# -1: Correlación lineal negativa perfecta

# El p-valor es el valor de probabilidad de que la correlación entre estas dos variables sea 
# estadísticamente significativa. Normalmente, elegimos un nivel de significancia de 0,05, lo que 
# significa que tenemos un 95% de confianza en que la correlación entre las variables es significativa.
# Por convención, cuando el p-valor es <0.001: decimos que hay pruebas sólidas de que la correlación es significativa


# En nuestro caso,un valor de 0.5846 sugiere una correlación moderada a fuerte y positiva entre wheel-base 
# y price, es decir, cuando la distancia entre ejes aumenta, es probable que también lo haga el precio del vehículo.
# Además,el p-valor que se obtuvó es extremadamente bajo (8.08e-20), lo que indica que la relación entre las dos 
# variables es estadísticamente significativa.