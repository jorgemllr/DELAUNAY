import json
import numpy as np
from cgshop2025_pyutils import DelaunayBasedSolver, verify
import matplotlib.pyplot as plt
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

# 2. Identificar y limpiar restricciones internas en la trompa
# Para quitar densidad en la trompa (X alto), ignoraremos algunas de las lineas internas que dibujaste ahi.
# Al ignorarlas, sus puntos se vuelven "Steiner" y seran eliminados.
filtered_constraints = []
for c in orig_constraints:
    x1, x2 = orig_pts_x[c[0]], orig_pts_x[c[1]]
    # Si la linea esta en la trompa (X > 600), la eliminamos con 80% de probabilidad
    if x1 > 600 and x2 > 600:
        if random.random() > 0.8: # Solo conserva el 20%
            filtered_constraints.append(c)
    else:
        filtered_constraints.append(c)

# 2.1 Identificar nodos esenciales vs nodos Steiner (internos)
required_indices = set(orig_boundary)
for c in filtered_constraints:
    required_indices.add(c[0])
    required_indices.add(c[1])

all_indices = set(range(len(orig_pts_x)))
steiner_indices = list(all_indices - required_indices)

# 3. Eliminar nodos Steiner aleatoriamente
random.seed(42) # Semilla fija para consistencia visual
DROP_PERCENTAGE = 1.0 # Eliminar el 100% de los nodos internos para que quede ultra limpio
num_to_drop = int(len(steiner_indices) * DROP_PERCENTAGE)
dropped_indices = set(random.sample(steiner_indices, num_to_drop))

keep_indices = [i for i in all_indices if i not in dropped_indices]
keep_indices.sort()

# Crear mapa de indices viejos a nuevos
old_to_new = {old: new for new, old in enumerate(keep_indices)}

# 4. Construir la nueva instancia limpia
new_pts_x = [orig_pts_x[i] for i in keep_indices]
new_pts_y = [orig_pts_y[i] for i in keep_indices]
new_boundary = [old_to_new[i] for i in orig_boundary]
new_constraints = [[old_to_new[c[0]], old_to_new[c[1]]] for c in filtered_constraints]

print(f"Instancia Original: {len(orig_pts_x)} puntos.")
print(f"Instancia Limpia (v6): {len(new_pts_x)} puntos. (Eliminados {num_to_drop} nodos Steiner).")

instance = JsonInstance(new_pts_x, new_pts_y, new_boundary, new_constraints, data["instance_uid"])

# 5. Resolver Triangulacion
print("Resolviendo triangulacion de Delaunay...")
solver = DelaunayBasedSolver(instance)
solution = solver.solve()

points = np.array(list(zip(instance.points_x, instance.points_y)), dtype=float)
boundary = np.array([points[i] for i in instance.region_boundary])
boundary = np.append(boundary, [boundary[0]], axis=0)

THEME_COLOR = '#0070DF' # Color solicitado por el usuario
LINE_THICKNESS = 4.0

if ENABLE_PLOT:
    plt.figure(figsize=(9, 9))
    # Se elimino el plt.title() para que el vector SVG salga perfectamente limpio sin texto
    
    line_kwargs = {
        'color': THEME_COLOR,
        'linewidth': LINE_THICKNESS,
        'solid_capstyle': 'round',
        'solid_joinstyle': 'round',
        'zorder': 10
    }
    
    # Dibujar Boundary
    plt.plot(boundary[:, 0], boundary[:, 1], **line_kwargs)

    # Dibujar restricciones adicionales
    for constraint in instance.additional_constraints:
        x_coords = [instance.points_x[constraint[0]], instance.points_x[constraint[1]]]
        y_coords = [instance.points_y[constraint[0]], instance.points_y[constraint[1]]]
        plt.plot(x_coords, y_coords, **line_kwargs)

    # Dibujar triangulacion de Delaunay
    for edge in solution.edges:
        x_coords = [instance.points_x[edge[0]], instance.points_x[edge[1]]]
        y_coords = [instance.points_y[edge[0]], instance.points_y[edge[1]]]
        plt.plot(x_coords, y_coords, **line_kwargs)
        
    # NOTA: En esta version V6 NO dibujamos los nodos (plt.scatter) a peticion del usuario.
    # Solo quedaran las lineas desnudas conectandose entre si.

    plt.axis('equal')
    plt.axis('off')
    
    # Exportar el vector SVG automaticamente
    plt.savefig("meslatt_logo_final.svg", format="svg", transparent=True, bbox_inches='tight')
    print("Logotipo guardado como vector en 'meslatt_logo_final.svg'")
    
    plt.show()

print("Proceso terminado.")
