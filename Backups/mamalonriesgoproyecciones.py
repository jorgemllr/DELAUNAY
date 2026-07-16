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
file_name = 'ortho_80_06ee55d4.instance.json'
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

def register_red_point(edge, red_point, source_triangle):
    """Registra un punto rojo en la base de datos"""
    sorted_edge = tuple(sorted(edge))
    red_points_db[sorted_edge].append({
        'point': red_point,
        'source_triangle': source_triangle,
        'target_triangle': edge_to_triangles[sorted_edge][0] 
                          if edge_to_triangles[sorted_edge][1] == source_triangle 
                          else edge_to_triangles[sorted_edge][1]
    })

# --- Data structures for alpha-edge-triangle relationships ---
triangle_info = {}  # Almacena información completa de cada triángulo
vertex_to_alphas = defaultdict(list)  # Mapea vértices a sus ángulos α
edge_to_alphas = defaultdict(list)  # Mapea aristas a sus ángulos α
# --- Nuevas estructuras para proyecciones púrpuras ---
purple_lines_data = []  # Almacena todas las líneas púrpuras
red_points_db = defaultdict(list)  # Base de datos de puntos rojos por arista
red_points_targets = defaultdict(list)  # Almacena puntos rojos targets por triángulo

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
            
def process_purple_projections(vertex, reference_green_dist, current_edge):
    """Procesa las proyecciones púrpuras desde un vértice"""
    if vertex in instance.region_boundary:
        return []  # Detener en frontera
    
    purple_lines = []
    
    # 1. Encontrar el triángulo al que pertenece el vértice
    current_tri = next((t for t in triangle_info if vertex in t), None)
    if not current_tri:
        return []
    
    # 2. Encontrar la otra arista del triángulo que comparte este vértice
    other_edges = [e for e in triangle_info[current_tri]['edges'] 
                  if vertex in e and e != current_edge]
    if not other_edges:
        return []
    
    other_edge = other_edges[0]
    v_other = other_edge[0] if other_edge[1] == vertex else other_edge[1]
    
    # 3. Buscar todos los puntos rojos targets en el triángulo actual
    for red_point_info in red_points_targets.get(current_tri, []):
        red_point = red_point_info['point']
        dist_to_red = np.linalg.norm(points[vertex] - red_point)
        
        # 4. Comprobar si la distancia es menor que la referencia verde
        if dist_to_red < reference_green_dist:
            # 5. Calcular proyección púrpura
            a = points[vertex].astype(float)
            b = points[v_other].astype(float)
            dir_vec = (b - a) / np.linalg.norm(b - a)
            purple_proj = a + dist_to_red * dir_vec
            
            purple_lines.append({
                'start': red_point,
                'end': purple_proj,
                'vertex': vertex,
                'distance': dist_to_red,
                'edge': other_edge
            })
            
            # Registrar nuevo punto rojo para posible propagación
            red_points_targets[current_tri].append({
                'point': purple_proj,
                'source_vertex': vertex
            })
    
    return purple_lines

# --- Nueva función para encontrar proyecciones rojas ---
def find_red_projections(edge, green_point, source_tri):
    """Encuentra los puntos rojos asociados a un punto verde en una arista"""
    red_points = []
    vertex_a, vertex_b = edge
    
    # Buscar en ambos vértices de la arista
    for vertex in [vertex_a, vertex_b]:
        pt_vertex = points[vertex]
        dist = np.linalg.norm(pt_vertex - green_point)
        
        # Obtener el triángulo opuesto
        tris = edge_to_triangles[edge]
        target_tri = tris[0] if tris[1] == source_tri else tris[1]
        
        # Obtener la otra arista del target_tri que comparte este vértice
        other_edges = [e for e in triangle_info[target_tri]['edges'] if vertex in e and e != edge]
        if other_edges:
            other_edge = other_edges[0]
            v_other = other_edge[0] if other_edge[1] == vertex else other_edge[1]
            
            # Calcular punto rojo
            a = points[vertex].astype(float)
            b = points[v_other].astype(float)
            dir_vec = (b - a) / np.linalg.norm(b - a)
            red_point = a + dist * dir_vec
            
            red_points.append(red_point)
    
    return red_points

# --- Nueva función para proyecciones púrpuras ---
def project_purple_lines(vertex, reference_green_dist, target_tri, edge_to_avoid):
    """Proyecta distancias a puntos rojos targets y dibuja líneas púrpuras"""
    # Verificar si el vértice está en la frontera
    if vertex in instance.region_boundary:
        return []
        
    purple_lines = []
    
    # Obtener todas las aristas del target_tri que contienen al vertex
    target_edges = [e for e in triangle_info[target_tri]['edges'] if vertex in e and e != edge_to_avoid]
    
    if not target_edges:
        return purple_lines
    
    target_edge = target_edges[0]  # La otra arista que comparte el vértice
    v_other = target_edge[0] if target_edge[1] == vertex else target_edge[1]
    
    # Obtener todos los puntos rojos targets asociados a este triángulo
    red_targets = []
    for edge in triangle_info[target_tri]['edges']:
        if edge != edge_to_avoid:
            # Buscar proyecciones rojas existentes en esta arista
            for e, tris in edge_to_triangles.items():
                if len(tris) == 2 and edge in e:
                    tangents = edge_to_tangency.get(e, [])
                    for tri, tg in tangents:
                        if tri == target_tri:
                            reds = find_red_projections(e, tg, tri)
                            red_targets.extend(reds)
    
    # Procesar cada punto rojo target
    for red_point in red_targets:
        dist_to_red = np.linalg.norm(points[vertex] - red_point)
        
        if dist_to_red < reference_green_dist:
            # Calcular proyección púrpura
            a = points[vertex].astype(float)
            b = points[v_other].astype(float)
            dir_vec = (b - a) / np.linalg.norm(b - a)
            purple_proj = a + dist_to_red * dir_vec
            
            purple_lines.append({
                'start': red_point,
                'end': purple_proj,
                'vertex': vertex,
                'distance': dist_to_red
            })
    
    return purple_lines

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
        plt.text(
            label_pos[0], label_pos[1], 
            f"α={alpha:.1f}°", 
            color='white', fontsize=6, ha='center', va='center',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'),
            zorder=10
        )
    
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

    # Primero registrar todos los puntos rojos de esta arista
    for vertex in edge:
        pt_vertex = points[vertex]
        d_a = np.linalg.norm(pt_vertex - tg_a)
        d_b = np.linalg.norm(pt_vertex - tg_b)

        if d_a < d_b:
            closest_tg = tg_a
            source_tri = tri_a
            target_tri = tri_b
            reference_dist = d_a
        else:
            closest_tg = tg_b
            source_tri = tri_b
            target_tri = tri_a
            reference_dist = d_b

        # Proyección amarilla original
        target_verts = list(triangle_info[target_tri]['vertices'])
        candidate_edges = [tuple(sorted([vertex, v])) for v in target_verts if v != vertex]
        valid_edges = [e for e in candidate_edges if e != edge]
        
        if len(valid_edges) == 1:
            correct_edge = valid_edges[0]
            v_other = correct_edge[0] if correct_edge[1] == vertex else correct_edge[1]
            
            a = points[vertex].astype(float)
            b = points[v_other].astype(float)
            dir_vec = (b - a) / np.linalg.norm(b - a)
            proj_point = a + reference_dist * dir_vec
            
            # Registrar punto rojo target
            red_points_targets[target_tri].append({
                'point': proj_point,
                'source_vertex': vertex
            })
            
            # Dibujar línea amarilla
            plt.plot([closest_tg[0], proj_point[0]], [closest_tg[1], proj_point[1]], 
                     color='yellow', linewidth=1.2, zorder=7)
            plt.scatter(proj_point[0], proj_point[1], color='red', s=25, zorder=8)
            plt.scatter(closest_tg[0], closest_tg[1], color='lime', s=25, zorder=8)

    # Ahora procesar proyecciones púrpuras para ambos vértices
    for vertex in edge:
        pt_vertex = points[vertex]
        d_a = np.linalg.norm(pt_vertex - tg_a)
        d_b = np.linalg.norm(pt_vertex - tg_b)
        reference_dist = min(d_a, d_b)
        
        # Procesar proyecciones púrpuras
        purple_lines = process_purple_projections(vertex, reference_dist, edge)
        
        for line in purple_lines:
            plt.plot([line['start'][0], line['end'][0]],
                     [line['start'][1], line['end'][1]],
                     color='purple', linewidth=1.2, zorder=7)
            plt.scatter(line['end'][0], line['end'][1], color='white', s=15, zorder=8)
            purple_lines_data.append(line)

        # --- Código original para proyecciones amarillas ---
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
        
        # Registrar el punto rojo
        register_red_point(edge, proj_point, source_tri)

        # Dibujar línea amarilla de tangencia
        plt.plot([closest_tg[0], proj_point[0]], [closest_tg[1], proj_point[1]], color='yellow', linewidth=1.2, zorder=7)
        
        # Puntos rojo (extremo) y verde (tangencia)
        plt.scatter(proj_point[0], proj_point[1], color='red', s=25, zorder=8)
        plt.scatter(closest_tg[0], closest_tg[1], color='lime', s=25, zorder=8)

# Final plot settings
plt.title('Triangulation with Alpha Angles and Projected Tangents')
plt.axis('equal')
plt.legend()
plt.show()

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

# --- Funciones de consulta para proyecciones púrpuras ---
def get_purple_lines_for_vertex(vertex_id):
    """Obtiene todas las líneas púrpuras asociadas a un vértice"""
    return [line for line in purple_lines_data if line['vertex'] == vertex_id]

def get_purple_lines_for_triangle(triangle):
    """Obtiene todas las líneas púrpuras asociadas a un triángulo"""
    tri_edges = triangle_info[triangle]['edges']
    return [line for line in purple_lines_data if any(edge in tri_edges for edge in edge_to_triangles)]

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

# --- Extender las funciones de consulta ---
def get_purple_lines_for_vertex(vertex_id):
    """Obtiene todas las líneas púrpuras asociadas a un vértice"""
    return [line for line in purple_lines_data if line['vertex'] == vertex_id]

def get_purple_lines_for_triangle(triangle):
    """Obtiene todas las líneas púrpuras asociadas a un triángulo"""
    tri_edges = triangle_info[triangle]['edges']
    return [line for line in purple_lines_data if any(edge in tri_edges for edge in edge_to_triangles)]

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