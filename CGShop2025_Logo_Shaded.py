import json
import numpy as np
from cgshop2025_pyutils import DelaunayBasedSolver, verify
import matplotlib.pyplot as plt
import os
import networkx as nx
import random

ENABLE_PLOT = True

# File path and instance name
file_name = 'meslatt_logo_cgshop_instance.json'
directory = 'challenge_instances_cgshop25'
file_path = os.path.join(directory, file_name)

# Load data from JSON
with open(file_path, "r") as file:
    data = json.load(file)

class JsonInstance:
    def __init__(self, data):
        self.points_x = np.array(data["points_x"])
        self.points_y = np.array(data["points_y"])
        self.region_boundary = data["region_boundary"]
        self.additional_constraints = data["additional_constraints"]
        self.instance_uid = data["instance_uid"]

instance = JsonInstance(data)

# Solve using DelaunayBasedSolver
print("Resolviendo triangulacion de Delaunay...")
solver = DelaunayBasedSolver(instance)
solution = solver.solve()

# Verify solution
print("Verificando solucion...")
result = verify(instance, solution)
print(f"Solution errors: {result.errors}")

# Prepare points and boundary for visualization
points = np.array(list(zip(instance.points_x, instance.points_y)), dtype=float)
boundary = np.array([points[i] for i in instance.region_boundary])
boundary = np.append(boundary, [boundary[0]], axis=0)

print("Calculando sombreado (Independent Set)...")
# 1. Encontrar triangulos
graph = nx.Graph()
graph.add_edges_from(solution.edges)
triangles = [c for c in nx.enumerate_all_cliques(graph) if len(c) == 3]

# 2. Construir adyacencia (comparten una arista = 2 vertices)
def share_edge(t1, t2):
    return len(set(t1).intersection(set(t2))) == 2

adj = {i: [] for i in range(len(triangles))}
for i in range(len(triangles)):
    for j in range(i+1, len(triangles)):
        if share_edge(triangles[i], triangles[j]):
            adj[i].append(j)
            adj[j].append(i)

# 3. Independent Set Algoritmo
random.seed(1337) # Semilla fija para que el diseño sea consistente
indices = list(range(len(triangles)))
random.shuffle(indices)

filled_set = set()
filled_triangles = []

# Probabilidad de relleno para controlar densidad
FILL_PROBABILITY = 0.55

for i in indices:
    if random.random() < FILL_PROBABILITY:
        can_fill = True
        for neighbor in adj[i]:
            if neighbor in filled_set:
                can_fill = False
                break
        
        if can_fill:
            filled_set.add(i)
            filled_triangles.append(triangles[i])

print(f"Rellenando {len(filled_triangles)} triangulos de {len(triangles)} totales.")

THEME_COLOR = '#2b2b2b'

if ENABLE_PLOT:
    plt.figure(figsize=(9, 9))
    plt.title("Meslatt Logo Triangulation - Low Poly Shaded")
    
    # Draw original boundary
    plt.plot(boundary[:, 0], boundary[:, 1], color=THEME_COLOR, linewidth=2.0, label='Silhouette', zorder=10)
    
    # Draw nodes
    plt.scatter(points[:, 0], points[:, 1], color=THEME_COLOR, s=15, zorder=11)

    # Draw additional constraints
    for constraint in instance.additional_constraints:
        x_coords = [instance.points_x[constraint[0]], instance.points_x[constraint[1]]]
        y_coords = [instance.points_y[constraint[0]], instance.points_y[constraint[1]]]
        plt.plot(x_coords, y_coords, color=THEME_COLOR, linewidth=1.5, zorder=5)

    # Draw triangulation edges
    for edge in solution.edges:
        x_coords = [instance.points_x[edge[0]], instance.points_x[edge[1]]]
        y_coords = [instance.points_y[edge[0]], instance.points_y[edge[1]]]
        plt.plot(x_coords, y_coords, color=THEME_COLOR, linewidth=0.8, alpha=0.7, zorder=1)
        
    # Rellenar los triangulos seleccionados
    for tri in filled_triangles:
        x_coords = [instance.points_x[v] for v in tri]
        y_coords = [instance.points_y[v] for v in tri]
        plt.fill(x_coords, y_coords, color=THEME_COLOR, alpha=1.0, zorder=2)

    plt.axis('equal')
    plt.axis('off')
    plt.show()

print("Proceso terminado.")
