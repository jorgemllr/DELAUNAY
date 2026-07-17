import json
import numpy as np
from cgshop2025_pyutils import DelaunayBasedSolver, verify
import matplotlib.pyplot as plt
import networkx as nx
import os
import random

ENABLE_PLOT = True

file_name = 'meslatt_logo_cgshop_instance.json'
directory = 'challenge_instances_cgshop25'
file_path = os.path.join(directory, file_name)

with open(file_path, "r") as file:
    data = json.load(file)

class JsonInstance:
    def __init__(self, pts_x, pts_y, boundary, constraints, uid):
        self.points_x = np.array(pts_x)
        self.points_y = np.array(pts_y)
        self.region_boundary = boundary
        self.additional_constraints = constraints
        self.instance_uid = uid

# 1. Cargar datos originales
orig_pts_x = data["points_x"]
orig_pts_y = data["points_y"]
orig_boundary = data["region_boundary"]
orig_constraints = data["additional_constraints"]

random.seed(42) # Semilla fija

# 2. Identificar y limpiar restricciones internas en la trompa
filtered_constraints = []
for c in orig_constraints:
    x1, x2 = orig_pts_x[c[0]], orig_pts_x[c[1]]
    if x1 > 600 and x2 > 600:
        if random.random() > 0.8: # Conserva el 20%
            filtered_constraints.append(c)
    else:
        filtered_constraints.append(c)

# 3. Limpiar nodos Steiner
required_indices = set(orig_boundary)
for c in filtered_constraints:
    required_indices.add(c[0])
    required_indices.add(c[1])

all_indices = set(range(len(orig_pts_x)))
steiner_indices = list(all_indices - required_indices)

DROP_PERCENTAGE = 1.0 # Eliminar el 100% de los nodos internos para limpieza
num_to_drop = int(len(steiner_indices) * DROP_PERCENTAGE)
dropped_indices = set(random.sample(steiner_indices, num_to_drop))

keep_indices = [i for i in all_indices if i not in dropped_indices]
keep_indices.sort()
old_to_new = {old: new for new, old in enumerate(keep_indices)}

# 4. Construir la nueva instancia limpia
new_pts_x = [orig_pts_x[i] for i in keep_indices]
new_pts_y = [orig_pts_y[i] for i in keep_indices]
new_boundary = [old_to_new[i] for i in orig_boundary]
new_constraints = [[old_to_new[c[0]], old_to_new[c[1]]] for c in filtered_constraints]

instance = JsonInstance(new_pts_x, new_pts_y, new_boundary, new_constraints, data["instance_uid"])

# 5. Resolver Triangulacion
print("Resolviendo triangulacion de Delaunay...")
solver = DelaunayBasedSolver(instance)
solution = solver.solve()

points = np.array(list(zip(instance.points_x, instance.points_y)), dtype=float)
boundary = np.array([points[i] for i in instance.region_boundary])
boundary = np.append(boundary, [boundary[0]], axis=0)

# 6. Algoritmo de Sombreado (Relaxed Independent Set)
print("Calculando sombreado (Relaxed Independent Set)...")
graph = nx.Graph()
graph.add_edges_from(solution.edges)
triangles = [c for c in nx.enumerate_all_cliques(graph) if len(c) == 3]

def share_edge(t1, t2):
    return len(set(t1).intersection(set(t2))) == 2

adj = {i: [] for i in range(len(triangles))}
for i in range(len(triangles)):
    for j in range(i+1, len(triangles)):
        if share_edge(triangles[i], triangles[j]):
            adj[i].append(j)
            adj[j].append(i)

indices = list(range(len(triangles)))
random.shuffle(indices)

filled_set = set()
filled_triangles = []
FILL_PROBABILITY = 0.70

for i in indices:
    if random.random() < FILL_PROBABILITY:
        black_neighbors = sum(1 for neighbor in adj[i] if neighbor in filled_set)
        if black_neighbors < 2:
            filled_set.add(i)
            filled_triangles.append(triangles[i])

# 7. Renderizado Soft Geometry
THEME_COLOR = '#303F9F' # Indigo vibrante y amigable
LINE_THICKNESS = 4.0

if ENABLE_PLOT:
    plt.figure(figsize=(9, 9))
    plt.title("Meslatt Logo - Soft Geometry + Shaded (v5)")
    
    line_kwargs = {
        'color': THEME_COLOR,
        'linewidth': LINE_THICKNESS,
        'solid_capstyle': 'round',
        'solid_joinstyle': 'round',
        'zorder': 10
    }
    
    plt.plot(boundary[:, 0], boundary[:, 1], **line_kwargs)

    for constraint in instance.additional_constraints:
        x_coords = [instance.points_x[constraint[0]], instance.points_x[constraint[1]]]
        y_coords = [instance.points_y[constraint[0]], instance.points_y[constraint[1]]]
        plt.plot(x_coords, y_coords, **line_kwargs)

    for edge in solution.edges:
        x_coords = [instance.points_x[edge[0]], instance.points_x[edge[1]]]
        y_coords = [instance.points_y[edge[0]], instance.points_y[edge[1]]]
        plt.plot(x_coords, y_coords, **line_kwargs)
        
    # Nodos suaves (esquinas redondas)
    plt.scatter(points[:, 0], points[:, 1], color=THEME_COLOR, s=LINE_THICKNESS*15, zorder=11, edgecolors='none')

    # Relleno de triangulos sombreados
    for tri in filled_triangles:
        x_coords = [instance.points_x[v] for v in tri]
        y_coords = [instance.points_y[v] for v in tri]
        plt.fill(x_coords, y_coords, color=THEME_COLOR, alpha=1.0, zorder=2)

    plt.axis('equal')
    plt.axis('off')
    plt.show()

print("Proceso terminado.")
