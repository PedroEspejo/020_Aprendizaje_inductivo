from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Generar datos de entrenamiento simulados
np.random.seed(0)
num_muestras = 100
datos_de_entrenamiento = np.random.rand(num_muestras, 2)  # Ejemplo de características
etiquetas = np.random.randint(2, size=num_muestras)  # Ejemplo de etiquetas (0 o 1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(datos_de_entrenamiento, etiquetas, test_size=0.2)

# Crear un modelo de Regresión Logística
modelo = LogisticRegression()

# Entrenar el modelo en los datos de entrenamiento
modelo.fit(X_entrenamiento, y_entrenamiento)

# Realizar predicciones en los datos de prueba
predicciones = modelo.predict(X_prueba)

# Calcular la precisión del modelo
precision = accuracy_score(y_prueba, predicciones)
print("Precisión del modelo:", precision)
