from sklearn.cluster import KMeans
import numpy as np

# Generar datos de clientes simulados
np.random.seed(0)
n_clientes = 200
X = np.random.rand(n_clientes, 2)  # Datos de ejemplo (2 características)

# Crear un modelo de K-Means para clusterizar en 3 segmentos
n_segmentos = 3
modelo = KMeans(n_clusters=n_segmentos, random_state=0)

# Entrenar el modelo en los datos
modelo.fit(X)

# Obtener las etiquetas de cluster para cada cliente
etiquetas = modelo.labels_

# Visualizar los segmentos
for i in range(n_segmentos):
    clientes_en_segmento_i = X[etiquetas == i]
    print(f"Segmento {i + 1}: {len(clientes_en_segmento_i)} clientes")
