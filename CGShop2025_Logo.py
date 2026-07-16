import json
import numpy as np
from cgshop2025_pyutils import DelaunayBasedSolver, verify
import matplotlib.pyplot as plt
import os

ENABLE_PLOT = True

# File path and instance name
file_name = 'meslatt_logo_cgshop_instance.json'  # Instance filename para el logo
directory = 'challenge_instances_cgshop25'
file_path = os.path.join(directory, file_name)

# Load data from JSON
with open(file_path, "r") as file:
    data = json.load(file)

# Creating an instance compatible with the library
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

# Thematic Color
THEME_COLOR = '#2b2b2b'

if ENABLE_PLOT:
    # White background by default (do not use dark_background)
    plt.figure(figsize=(9, 9))
    plt.title("Meslatt Logo Triangulation")
    
    # Draw original boundary
    plt.plot(boundary[:, 0], boundary[:, 1], color=THEME_COLOR, linewidth=2.0, label='Silhouette Boundary', zorder=10)
    
    # Draw nodes, slightly smaller (s=15)
    plt.scatter(points[:, 0], points[:, 1], color=THEME_COLOR, s=15, label='Original Points', zorder=11)

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

    plt.axis('equal')
    # Remove axis borders and ticks for a cleaner logo look
    plt.axis('off')
    plt.show()

print("Guardando solucion maestra (edges)...")
export_path = os.path.join(directory, "meslatt_logo_triangulated_masterpiece.json")
with open(export_path, "w") as out_f:
    json.dump({"edges": solution.edges}, out_f, indent=4)
print(f"Solucion exportada a {export_path}")