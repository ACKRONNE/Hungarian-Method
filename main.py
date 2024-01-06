import numpy as np
from scipy.optimize import linear_sum_assignment

def pad_matrix(cost_matrix, pad_value=0):
    """Añade filas o columnas ficticias con un valor de padding para hacer la matriz cuadrada."""
    rows, cols = cost_matrix.shape
    if rows == cols:
        return cost_matrix
    elif rows > cols:
        padding = pad_value * np.ones((rows, rows - cols))
        return np.hstack([cost_matrix, padding])
    else:
        padding = pad_value * np.ones((cols - rows, cols))
        return np.vstack([cost_matrix, padding])

def hungarian_method(num_operarios, num_tareas, costos):
    """Aplica el método húngaro para encontrar la asignación de costo mínimo."""
    # Crear la matriz de costos a partir de los costos de entrada
    cost_matrix = np.array(costos).reshape((num_operarios, num_tareas))
    # Asegurar que la matriz de costos sea cuadrada
    cost_matrix = pad_matrix(cost_matrix)
    # Aplicar el método húngaro para encontrar la asignación óptima
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Calcular el costo total de la asignación
    total_cost = cost_matrix[row_ind, col_ind].sum()
    # Devolver los índices de asignación y el costo total
    return row_ind, col_ind, total_cost

# Ejemplo de uso
num_operarios = 4
num_tareas = 3
costos = [50, 130, 190, 130, 100, 150, 110, 150, 270, 150, 90, 60]  # Ejemplo de costos

# Obtener la asignación y el costo total
row_ind, col_ind, total_cost = hungarian_method(num_operarios, num_tareas, costos)

print("Asignación óptima:", list(zip(row_ind, col_ind)))
print("Costo total:", total_cost)
