import json
import numpy as np
from cgshop2025_pyutils import DelaunayBasedSolver, verify
import matplotlib.pyplot as plt
import os

ENABLE_PLOT = True

file_name = 'meslatt_logo_cgshop_instance.json'
directory = 'challenge_instances_cgshop25'
file_path = os.path.join(directory, file_name)

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

print("Resolviendo triangulacion de Delaunay...")
solver = DelaunayBasedSolver(instance)
solution = solver.solve()

points = np.array(list(zip(instance.points_x, instance.points_y)), dtype=float)
boundary = np.array([points[i] for i in instance.region_boundary])
boundary = np.append(boundary, [boundary[0]], axis=0)

THEME_COLOR = '#303F9F' # Indigo vibrante y amigable
LINE_THICKNESS = 4.0 # Trazos gruesos uniformes

if ENABLE_PLOT:
    plt.figure(figsize=(9, 9))
    plt.title("Meslatt Logo - Soft Geometry (v3)")
    
    # Configuracion de bordes redondeados para Matplotlib
    # solid_capstyle='round' hace que la punta de la linea sea redonda
    # solid_joinstyle='round' hace que las esquinas sean redondas
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
        
    # Agregamos los nodos del mismo color y grosor que la linea 
    # para asegurar que las intersecciones multiples se vean como un solo punto suave
    plt.scatter(points[:, 0], points[:, 1], color=THEME_COLOR, s=LINE_THICKNESS*15, zorder=11, edgecolors='none')

    plt.axis('equal')
    plt.axis('off') # Sin ejes para que se vea limpio
    plt.show()

print("Proceso terminado.")
