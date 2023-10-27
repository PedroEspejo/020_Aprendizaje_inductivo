import numpy as np

# Definir el entorno de cuadrícula
# 'S' representa el estado inicial
# 'G' representa la meta
# 'X' representa obstáculos
# 'E' representa estados vacíos
entorno = np.array([
    ['S', 'E', 'X', 'E', 'E'],
    ['E', 'X', 'E', 'X', 'E'],
    ['E', 'X', 'E', 'X', 'E'],
    ['E', 'E', 'E', 'X', 'E'],
    ['X', 'X', 'X', 'E', 'G']
])

# Inicializar parámetros
n_estados = entorno.size  # Número de estados
n_acciones = 4  # Cuatro acciones posibles: arriba, abajo, izquierda, derecha
factor_aprendizaje = 0.1
factor_descuento = 0.9
num_episodios = 1000

# Inicializar la matriz Q con ceros
Q = np.zeros((n_estados, n_acciones))

# Función para obtener las acciones posibles en un estado
def acciones_posibles(estado):
    i, j = divmod(estado, entorno.shape[1])
    acciones = []
    if i > 0 and entorno[i - 1, j] != 'X':
        acciones.append(0)  # Arriba
    if i < entorno.shape[0] - 1 and entorno[i + 1, j] != 'X':
        acciones.append(1)  # Abajo
    if j > 0 and entorno[i, j - 1] != 'X':
        acciones.append(2)  # Izquierda
    if j < entorno.shape[1] - 1 and entorno[i, j + 1] != 'X':
        acciones.append(3)  # Derecha
    return acciones

# Ciclo de episodios
for episodio in range(num_episodios):
    estado = 0  # Iniciar en el estado inicial (0)
    while estado != n_estados - 1:  # Continuar hasta alcanzar la meta (n_estados - 1)
        acciones_posibles_estado = acciones_posibles(estado)
        accion = np.random.choice(acciones_posibles_estado)  # Seleccionar una acción al azar
        nuevo_estado = estado + 1 if accion == 3 else (estado - 1 if accion == 2 else (estado - entorno.shape[1] if accion == 0 else estado + entorno.shape[1]))
        recompensa = 1 if nuevo_estado == n_estados - 1 else 0  # Recompensa 1 al alcanzar la meta
        Q[estado, accion] = Q[estado, accion] + factor_aprendizaje * (recompensa + factor_descuento * np.max(Q[nuevo_estado, :]) - Q[estado, accion])
        estado = nuevo_estado

# Mostrar la matriz Q
print("Matriz Q aprendida:")
print(Q)