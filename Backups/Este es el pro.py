import json
import numpy as np
from cgshop2025_pyutils import DelaunayBasedSolver, verify
import matplotlib.pyplot as plt
import os
from shapely.geometry import Polygon
import networkx as nx
import math
from collections import defaultdict

def calculate_angle(a, b, c):
    """Calcula el ángulo en el vértice B (entre lados a y c) usando Ley de Cosenos."""
    angle_rad = math.acos((a**2 + c**2 - b**2) / (2 * a * c))
    return math.degrees(angle_rad)

def calculate_triangle_angles(triangle_points):
    """Calcula los 3 ángulos de un triángulo dado sus puntos."""
    A, B, C = triangle_points
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    
    angle_A = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2 * b * c)))
    angle_B = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))
    angle_C = np.degrees(np.arccos((a**2 + b**2 - c**2) / (2 * a * b)))
    
    if not np.isclose(angle_A + angle_B + angle_C, 180, atol=1.0):
        print(f"¡Advertencia! Ángulos no suman 180°: {angle_A:.1f} + {angle_B:.1f} + {angle_C:.1f} = {angle_A + angle_B + angle_C:.1f}")
    
    return angle_A, angle_B, angle_C

# File path and instance name
file_name = 'ortho_60_c423f527.instance.json'
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
solver = DelaunayBasedSolver(instance)
solution = solver.solve()

# Verify solution
result = verify(instance, solution)
print(f"Solution errors: {result.errors}")

# Prepare points and boundary for visualization
points = np.array(list(zip(instance.points_x, instance.points_y)))
boundary = np.array([points[i] for i in instance.region_boundary])
boundary = np.append(boundary, [boundary[0]], axis=0)
boundary_polygon = Polygon(boundary)

# Set dark mode theme
plt.style.use('dark_background')
plt.figure(figsize=(9, 9))
plt.plot(boundary[:, 0], boundary[:, 1], 'white', linewidth=2, label='Boundary')
plt.scatter(points[:, 0], points[:, 1], color='lightsteelblue', label='Original Points')

# Draw edges
for edge in solution.edges:
    x_coords = [instance.points_x[edge[0]], instance.points_x[edge[1]]]
    y_coords = [instance.points_y[edge[0]], instance.points_y[edge[1]]]
    plt.plot(x_coords, y_coords, 'steelblue', linewidth=1.2)

# Reconstruct triangles using networkx
graph = nx.Graph()
graph.add_edges_from(solution.edges)
triangles = [c for c in nx.enumerate_all_cliques(graph) if len(c) == 3]

# --- Triangle processing functions ---
def calculate_incenter(triangle):
    a = np.linalg.norm(triangle[1] - triangle[2])
    b = np.linalg.norm(triangle[0] - triangle[2])
    c = np.linalg.norm(triangle[0] - triangle[1])
    perimeter = a + b + c
    incenter = (a * triangle[0] + b * triangle[1] + c * triangle[2]) / perimeter
    return incenter

def calculate_inradius(triangle):
    a = np.linalg.norm(triangle[1] - triangle[2])
    b = np.linalg.norm(triangle[0] - triangle[2])
    c = np.linalg.norm(triangle[0] - triangle[1])
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    inradius = area / s
    return inradius

def calculate_tangency_points(triangle, incenter):
    tangency_points = []
    for i in range(3):
        p1 = triangle[i]
        p2 = triangle[(i + 1) % 3]
        edge = p2 - p1
        edge_unit = edge / np.linalg.norm(edge)
        proj = np.dot(incenter - p1, edge_unit)
        tangency_point = p1 + proj * edge_unit
        tangency_points.append(tangency_point)
    return tangency_points

# --- Data structures for alpha-edge-triangle relationships ---
triangle_info = {}  # Almacena información completa de cada triángulo
vertex_to_alphas = defaultdict(list)  # Mapea vértices a sus ángulos α
edge_to_alphas = defaultdict(list)  # Mapea aristas a sus ángulos α

for triangle in triangles:
    tri_points = points[triangle]
    sorted_triangle = tuple(sorted(triangle))
    if boundary_polygon.contains(Polygon(tri_points)):
        # Calcular propiedades del triángulo
        angle_A, angle_B, angle_C = calculate_triangle_angles(tri_points)
        incenter = calculate_incenter(tri_points)
        inradius = calculate_inradius(tri_points)
        tangency_points = np.array(calculate_tangency_points(tri_points, incenter))
        
        # Calcular ángulos α para cada vértice
        alpha_A = (180 - angle_A) / 2
        alpha_B = (180 - angle_B) / 2
        alpha_C = (180 - angle_C) / 2
        
        # Definir las aristas del triángulo
        edges = [
            tuple(sorted([triangle[0], triangle[1]])),
            tuple(sorted([triangle[1], triangle[2]])),
            tuple(sorted([triangle[2], triangle[0]]))
        ]
        
        # Almacenar información completa del triángulo
        triangle_info[sorted_triangle] = {
            'vertices': triangle,
            'points': tri_points,
            'angles': {
                'A': {'value': angle_A, 'alpha': alpha_A, 'vertex': triangle[0], 'edges': [edges[0], edges[2]]},
                'B': {'value': angle_B, 'alpha': alpha_B, 'vertex': triangle[1], 'edges': [edges[0], edges[1]]},
                'C': {'value': angle_C, 'alpha': alpha_C, 'vertex': triangle[2], 'edges': [edges[1], edges[2]]}
            },
            'edges': edges,
            'incenter': incenter,
            'tangency_points': tangency_points
        }
        
        # Mapear vértices y aristas a sus ángulos α
        vertex_to_alphas[triangle[0]].append({'alpha': alpha_A, 'triangle': sorted_triangle, 'edges': [edges[0], edges[2]]})
        vertex_to_alphas[triangle[1]].append({'alpha': alpha_B, 'triangle': sorted_triangle, 'edges': [edges[0], edges[1]]})
        vertex_to_alphas[triangle[2]].append({'alpha': alpha_C, 'triangle': sorted_triangle, 'edges': [edges[1], edges[2]]})
        
        for edge in edges:
            edge_to_alphas[edge].append({
                'alpha_A': alpha_A if edge in [edges[0], edges[2]] else None,
                'alpha_B': alpha_B if edge in [edges[0], edges[1]] else None,
                'alpha_C': alpha_C if edge in [edges[1], edges[2]] else None,
                'triangle': sorted_triangle
            })

# --- Visualization with alpha labels ---
for tri_data in triangle_info.values():
    # Dibujar el triángulo y sus elementos
    for angle_key, angle_data in tri_data['angles'].items():
        vertex_idx = angle_data['vertex']
        vertex_point = tri_data['points'][list(tri_data['vertices']).index(vertex_idx)]
        alpha = angle_data['alpha']
        
        # Posición del texto (40% hacia el incentro)
        direction = tri_data['incenter'] - vertex_point
        label_pos = vertex_point + 0.4 * direction
        
        # Mostrar α en blanco
#         plt.text(
#             label_pos[0], label_pos[1], 
#             f"α={alpha:.1f}°", 
#             color='white', fontsize=6, ha='center', va='center',
#             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'),
#             zorder=10
#         )
    
    # Dibujar el triángulo interno (slategray)
    plt.fill(
        tri_data['tangency_points'][:, 0],
        tri_data['tangency_points'][:, 1],
        facecolor='slategray',
        edgecolor='white',
        linewidth=1.2,
        zorder=7
    )

# --- Edge processing for tangent lines ---
red_targets = defaultdict(list)
edge_to_triangles = defaultdict(list)
edge_to_tangency = defaultdict(list)

for triangle in triangles:
    tri_points = points[triangle]
    sorted_triangle = tuple(sorted(triangle))
    if boundary_polygon.contains(Polygon(tri_points)):
        edges = [
            tuple(sorted([triangle[0], triangle[1]])),
            tuple(sorted([triangle[1], triangle[2]])),
            tuple(sorted([triangle[2], triangle[0]]))
        ]
        for edge in edges:
            edge_to_triangles[edge].append(sorted_triangle)
            if edge not in edge_to_tangency:
                edge_to_tangency[edge] = []
            # Agregar puntos de tangencia correspondientes
            tangents = triangle_info[sorted_triangle]['tangency_points']
            if edge == edges[0]:
                edge_to_tangency[edge].append((sorted_triangle, tangents[0]))  # Punto entre v0 y v1
            elif edge == edges[1]:
                edge_to_tangency[edge].append((sorted_triangle, tangents[1]))  # Punto entre v1 y v2
            else:
                edge_to_tangency[edge].append((sorted_triangle, tangents[2]))  # Punto entre v2 y v0

# Proyectar puntos y mostrar líneas de tangencia
for edge, tris in edge_to_triangles.items():
    if len(tris) != 2 or edge not in edge_to_tangency:
        continue

    tangency_info = edge_to_tangency[edge]
    if len(tangency_info) != 2:
        continue

    (tri_a, tg_a), (tri_b, tg_b) = tangency_info

    for vertex in edge:
        pt_vertex = points[vertex]
        d_a = np.linalg.norm(pt_vertex - tg_a)
        d_b = np.linalg.norm(pt_vertex - tg_b)

        if d_a < d_b:
            closest_tg = tg_a
            source_tri = tri_a
            target_tri = tri_b
        else:
            closest_tg = tg_b
            source_tri = tri_b
            target_tri = tri_a

        target_verts = list(triangle_info[target_tri]['vertices'])
        candidate_edges = [
            tuple(sorted([vertex, v]))
            for v in target_verts if v != vertex
        ]
        valid_edges = [e for e in candidate_edges if e != edge]
        if len(valid_edges) != 1:
            continue

        correct_edge = valid_edges[0]
        v_other = correct_edge[0] if correct_edge[1] == vertex else correct_edge[1]

        a = points[vertex].astype(float)
        b = points[v_other].astype(float)
        dir_vec = b - a
        norm = np.linalg.norm(dir_vec)
        if norm == 0:
            continue
        dir_vec /= norm

        dist = np.linalg.norm(pt_vertex - closest_tg)
        proj_point = a + dist * dir_vec

        # Dibujar línea blanca de tangencia (la primera que se genera)
        plt.plot([closest_tg[0], proj_point[0]], [closest_tg[1], proj_point[1]], color='white', linewidth=1.2, zorder=7)
        
        # Puntos rojo (extremo) y verde (tangencia)
        plt.scatter(proj_point[0], proj_point[1], color='red', s=10, zorder=8)
        plt.scatter(closest_tg[0], closest_tg[1], color='lime', s=10, zorder=8)
        
        # Registrar el nuevo punto rojo como target en la arista proyectada
        red_targets[tuple(sorted(correct_edge))].append({
            'point': proj_point,
            'from_triangle': target_tri
        })

# --- Helper functions to query relationships ---
def get_alphas_for_vertex(vertex_id):
    """Obtiene todos los ángulos α asociados a un vértice"""
    return vertex_to_alphas.get(vertex_id, [])

def get_vertex_alpha_info(vertex_id):
    """Obtiene información detallada de todos los ángulos α de un vértice"""
    alphas = vertex_to_alphas.get(vertex_id, [])
    result = []
    for alpha_info in alphas:
        tri_data = triangle_info[alpha_info['triangle']]
        angle_data = None
        # Encontrar el ángulo correspondiente en triangle_info
        for angle_key, angle_val in tri_data['angles'].items():
            if angle_val['vertex'] == vertex_id and abs(angle_val['alpha'] - alpha_info['alpha']) < 0.1:
                angle_data = angle_val
                break
        if angle_data:
            result.append({
                'alpha': alpha_info['alpha'],
                'triangle': alpha_info['triangle'],
                'edges': angle_data['edges'],
                'triangle_points': tri_data['points'],
                'adjacent_vertices': [v for v in alpha_info['triangle'] if v != vertex_id]
            })
    return result

def get_alphas_for_edge(edge):
    """Obtiene los ángulos α asociados a una arista"""
    return edge_to_alphas.get(tuple(sorted(edge)), [])

def get_triangles_for_alpha(alpha_value, tolerance=0.1):
    """Encuentra todos los triángulos que contienen un α específico"""
    result = []
    for tri, data in triangle_info.items():
        for angle_data in data['angles'].values():
            if abs(angle_data['alpha'] - alpha_value) < tolerance:
                result.append((tri, angle_data))
                break
    return result

def is_edge_on_boundary(edge):
    return len(edge_to_triangles[edge]) == 1

def propagate_red_targets():
    new_targets = []

    for tri_key, tri_data in triangle_info.items():
        for edge in tri_data['edges']:
            red_pts = red_targets.get(edge, [])
            # Buscar los puntos de tangencia entre el triángulo actual y la arista
            tg_points = [tp for t, tp in edge_to_tangency[edge] if t == tri_key]
            if not tg_points:
                continue

            # Obtener vértices de la arista
            v1, v2 = edge
            pt_v1 = points[v1]
            pt_v2 = points[v2]

            # Calcular distancias desde cada vértice a su punto de tangencia
            # Nota: Suponemos que el punto verde (tg_points[0]) está más cerca del punto medio de la arista.
            # Así que usamos la distancia al punto medio como criterio de asociación (aproximación razonable)
            tg_point = tg_points[0]
            d_tg_v1 = np.linalg.norm(pt_v1 - tg_point)
            d_tg_v2 = np.linalg.norm(pt_v2 - tg_point)

            # Obtener vértices de la arista
            v1, v2 = edge
            d_v1 = np.linalg.norm(points[v1] - tg_point)
            d_v2 = np.linalg.norm(points[v2] - tg_point)

            for pt_data in red_pts:
                pt = pt_data['point']
                d1 = np.linalg.norm(points[v1] - pt)
                d2 = np.linalg.norm(points[v2] - pt)

                if d1 < d_tg_v1:
                    origin_vertex = v1
                    dist = d1
                elif d2 < d_tg_v2:
                    origin_vertex = v2
                    dist = d2
                else:
                    continue  # No propaga

                # Buscar otra arista que contiene origin_vertex pero no edge
                candidate_edges = [e for e in tri_data['edges'] if origin_vertex in e and e != edge]
                if not candidate_edges:
                    continue

                new_edge = candidate_edges[0]
                if is_edge_on_boundary(new_edge):
                    continue  # Evita propagar fuera del dominio
                v_other = new_edge[0] if new_edge[1] == origin_vertex else new_edge[1]
                a = points[origin_vertex].astype(float)
                b = points[v_other].astype(float)
                dir_vec = b - a
                norm = np.linalg.norm(dir_vec)
                if norm == 0:
                    continue
                dir_vec /= norm
                new_pt = a + dist * dir_vec

                # Dibujar línea blanca
                plt.plot([pt[0], new_pt[0]], [pt[1], new_pt[1]], color='white', linewidth=1.2, zorder=7)
                plt.scatter(new_pt[0], new_pt[1], color='purple', s=2, zorder=9)

                # Guardar como nuevo target
                new_targets.append((tuple(sorted(new_edge)), {
                    'point': new_pt,
                    'from_triangle': tri_key
                }))

    for edge_key, red_data in new_targets:
        red_targets[edge_key].append(red_data)
        
# Ejecutar la propagación de puntos rojos
max_iterations = 3
for _ in range(max_iterations):
    num_before = sum(len(v) for v in red_targets.values())
    propagate_red_targets()
    num_after = sum(len(v) for v in red_targets.values())
    if num_after == num_before:
        break  # No hubo nuevos targets, detener propagación
    
# Final plot settings
plt.title('Triangulation with Alpha Angles and Projected Tangents')
plt.axis('equal')
plt.legend()
plt.show()

# Ejemplo de uso de las funciones de consulta
if triangles:
    ejemplo_vertice = triangles[0][0]  # Primer vértice del primer triángulo
    print(f"\nInformación detallada de ángulos α para el vértice {ejemplo_vertice}:")
    for alpha_info in get_vertex_alpha_info(ejemplo_vertice):
        print(f"  α={alpha_info['alpha']:.1f}°")
        print(f"  Triángulo: {alpha_info['triangle']}")
        print(f"  Aristas asociadas: {alpha_info['edges']}")
        print(f"  Vértices adyacentes: {alpha_info['adjacent_vertices']}\n")
    
    ejemplo_arista = tuple(sorted([triangles[0][0], triangles[0][1]]))
    print(f"\nÁngulos α para la arista {ejemplo_arista}:")
    for alpha_info in get_alphas_for_edge(ejemplo_arista):
        print(f"  Triángulo {alpha_info['triangle']}:")
        if alpha_info['alpha_A'] is not None:
            print(f"    α_A={alpha_info['alpha_A']:.1f}°")
        if alpha_info['alpha_B'] is not None:
            print(f"    α_B={alpha_info['alpha_B']:.1f}°")
        if alpha_info['alpha_C'] is not None:
            print(f"    α_C={alpha_info['alpha_C']:.1f}°")