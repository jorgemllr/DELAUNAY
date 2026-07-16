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

# AÑADE AQUÍ la función line_intersection:
def line_intersection(p1, p2, p3, p4):
    """Calcula la intersección entre segmentos p1-p2 y p3-p4"""
    denom = (p4[1]-p3[1])*(p2[0]-p1[0]) - (p4[0]-p3[0])*(p2[1]-p1[1])
    if denom == 0:
        return None  # Líneas paralelas
        
    ua = ((p4[0]-p3[0])*(p1[1]-p3[1]) - (p4[1]-p3[1])*(p1[0]-p3[0])) / denom
    ub = ((p2[0]-p1[0])*(p1[1]-p3[1]) - (p2[1]-p1[1])*(p1[0]-p3[0])) / denom
    
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return np.array([
            p1[0] + ua*(p2[0]-p1[0]),
            p1[1] + ua*(p2[1]-p1[1])
        ])
    return None

def find_intersection(start_point, direction, current_tri, exclude_edge):
    """Encuentra intersección de un rayo con las aristas del triángulo"""
    tri_edges = triangle_info[current_tri]['edges']
    
    for edge in tri_edges:
        if edge == exclude_edge:
            continue
            
        edge_p1 = points[edge[0]]
        edge_p2 = points[edge[1]]
        
        # Calcular intersección
        denom = (edge_p2[1]-edge_p1[1])*direction[0] - (edge_p2[0]-edge_p1[0])*direction[1]
        if denom == 0:
            continue
            
        ua = ((edge_p2[0]-edge_p1[0])*(start_point[1]-edge_p1[1]) - (edge_p2[1]-edge_p1[1])*(start_point[0]-edge_p1[0])) / denom
        ub = (direction[0]*(start_point[1]-edge_p1[1]) - direction[1]*(start_point[0]-edge_p1[0])) / denom
        
        if ua >= 0 and 0 <= ub <= 1:
            intersection = start_point + ua * direction
            # Determinar vértice más cercano
            d1 = np.linalg.norm(intersection - edge_p1)
            d2 = np.linalg.norm(intersection - edge_p2)
            closest_vertex = edge[0] if d1 < d2 else edge[1]
            return intersection, edge, closest_vertex
    
    return None, None, None

def get_projection_direction(start_point, vertex, alpha_deg):
    """
    Calcula el vector dirección para proyectar desde 'start_point' usando el ángulo α.
    - 'vertex': vértice asociado al ángulo α.
    - 'alpha_deg': ángulo en grados (ej: 66°).
    """
    # Convertir α a radianes
    alpha_rad = np.radians(alpha_deg)
    
    # Vector desde el vértice al punto rojo (dirección de referencia)
    reference_dir = start_point - points[vertex]
    reference_dir /= np.linalg.norm(reference_dir)
    
    # Rotar el vector referencia por α
    direction = np.array([
        reference_dir[0] * np.cos(alpha_rad) - reference_dir[1] * np.sin(alpha_rad),
        reference_dir[0] * np.sin(alpha_rad) + reference_dir[1] * np.cos(alpha_rad)
    ])
    
    return direction

def choose_alpha(red_point, edge):
    """Elige el ángulo α basado en el punto verde más cercano al punto rojo."""
    data = edge_to_alphas[edge]
    dist1 = np.linalg.norm(red_point - data['vertex_1'][2])  # Distancia a punto verde 1
    dist2 = np.linalg.norm(red_point - data['vertex_2'][2])  # Distancia a punto verde 2
    return data['vertex_1'][1] if dist1 <= dist2 else data['vertex_2'][1]  # Alpha del más cercano

def project_from_red_point(red_point_info, max_iterations=5):
    current_info = red_point_info
    iterations = 0
    
    while iterations < max_iterations:
        target_tri = current_info['target_tri']
        tri_data = triangle_info.get(target_tri)
        
        if tri_data is None:
            break
        
        alpha_angle = choose_alpha(current_info['point'], current_info['edge'])
        
        if alpha_angle is None:
            break
        
        # Calcular dirección usando SOLO el ángulo α (sin incentro)
        direction = get_projection_direction(
            current_info['point'],
            current_info['vertex'],
            alpha_angle
        )
        
        # Proyectar hasta la siguiente arista (igual que antes)
        new_point, new_edge, new_vertex = find_intersection(
            current_info['point'], 
            direction,
            target_tri,
            current_info['edge']
        )
        
        if new_point is None:
            break
            
        # Dibujar línea roja
        plt.plot([current_info['point'][0], new_point[0]], 
        [current_info['point'][1], new_point[1]], 
        color='red', linewidth=1.2, zorder=9)
        
        # --- AÑADE ESTO ---
        if iterations > 0:  # Puntos azules desde la segunda proyección
            plt.scatter(new_point[0], new_point[1], color='blue', s=25, zorder=10)
        
        # Actualizar información para la siguiente iteración
        adjacent_tris = edge_to_triangles.get(new_edge, [])
        if len(adjacent_tris) != 2:
            break
            
        new_target_tri = adjacent_tris[0] if adjacent_tris[1] == target_tri else adjacent_tris[1]
        
        current_info = {
            'point': new_point,
            'source_tri': target_tri,
            'target_tri': new_target_tri,
            'edge': new_edge,
            'vertex': new_vertex
        }
        iterations += 1

# --- Data structures for alpha-edge-triangle relationships ---
triangle_info = {}  # Almacena información completa de cada triángulo
vertex_to_alphas = defaultdict(list)  # Mapea vértices a sus ángulos α
# --- Data structures for alpha-edge-triangle relationships ---
triangle_info = {}  # Almacena información completa de cada triángulo
vertex_to_alphas = defaultdict(list)  # Mapea vértices a sus ángulos α
edge_to_alphas = {}  # Diccionario para almacenar ambos ángulos α por arista

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
            tuple(sorted([triangle[0], triangle[1]])),  # edge 0-1
            tuple(sorted([triangle[1], triangle[2]])),  # edge 1-2
            tuple(sorted([triangle[2], triangle[0]]))   # edge 2-0
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
            'tangency_points': tangency_points  # [tangent_point0, tangent_point1, tangent_point2]
        }
        
        # Mapear vértices a sus ángulos α (se mantiene igual)
        vertex_to_alphas[triangle[0]].append({'alpha': alpha_A, 'triangle': sorted_triangle})
        vertex_to_alphas[triangle[1]].append({'alpha': alpha_B, 'triangle': sorted_triangle})
        vertex_to_alphas[triangle[2]].append({'alpha': alpha_C, 'triangle': sorted_triangle})
        
        # Mapeo NUEVO de aristas a ambos ángulos α y puntos verdes
        for i, edge in enumerate(edges):
            # Obtener los vértices y puntos verdes correspondientes a esta arista
            vertex1 = triangle[i]  # Primer vértice de la arista
            vertex2 = triangle[(i+1)%3]  # Segundo vértice de la arista
            alpha1 = [alpha_A, alpha_B, alpha_C][i]  # Ángulo α del primer vértice
            alpha2 = [alpha_A, alpha_B, alpha_C][(i+1)%3]  # Ángulo α del segundo vértice
            tangent1 = tangency_points[i]  # Punto verde asociado a vertex1
            tangent2 = tangency_points[(i+1)%3]  # Punto verde asociado a vertex2
            
            if edge not in edge_to_alphas:
                # Primer triángulo que encuentra esta arista
                edge_to_alphas[edge] = {
                    'vertex_1': (vertex1, alpha1, tangent1),
                    'vertex_2': (vertex2, alpha2, tangent2)
                }
            else:
                # Segundo triángulo que comparte esta arista (actualiza solo el segundo vértice)
                # Nota: Esto asume que el primer vértice ya fue registrado por el otro triángulo
                edge_to_alphas[edge]['vertex_2'] = (vertex2, alpha2, tangent2)
                
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
# Inicializar lista global para almacenar información de puntos rojos
red_points_info = []

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
            source_tri = tri_a  # Triángulo que origina la proyección
            target_tri = tri_b  # Triángulo que recibe la proyección
            tg_point = tg_a
        else:
            closest_tg = tg_b
            source_tri = tri_b
            target_tri = tri_a
            tg_point = tg_b

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

        # Dibujar línea amarilla de tangencia
        plt.plot([closest_tg[0], proj_point[0]], [closest_tg[1], proj_point[1]], 
                color='yellow', linewidth=1.2, zorder=7)
        
        # Dibujar puntos verde (tangencia) y rojo (proyección)
        plt.scatter(closest_tg[0], closest_tg[1], color='lime', s=25, zorder=8)
        red_point = plt.scatter(proj_point[0], proj_point[1], color='red', s=25, zorder=8)
        
        # Llamar a la proyección desde este punto rojo (AÑADE ESTO)
        project_from_red_point({
            'point': proj_point,
            'vertex': vertex,
            'edge': edge,
            'source_tri': source_tri,
            'target_tri': target_tri
        })

        # Almacenar información completa del punto rojo
        red_points_info.append({
            'point': proj_point,          # Coordenadas (x,y) del punto rojo
            'vertex': vertex,             # Vértice de la arista compartida
            'edge': edge,                 # Arista compartida
            'source_tri': source_tri,     # Triángulo que generó la proyección (contiene el punto verde)
            'target_tri': target_tri,     # Triángulo que recibe la proyección
            'tg_point': tg_point,         # Punto verde de tangencia que originó esta proyección
            'projection_line': (closest_tg, proj_point)  # Línea amarilla completa
        })

        # Añadir etiqueta identificadora al punto rojo
        plt.text(proj_point[0], proj_point[1], 
                f"R{len(red_points_info)}", 
                color='white', fontsize=6, ha='center', va='bottom',
                zorder=10)

# AÑADE ESTO JUSTO AQUÍ (antes de plt.show()):
# --------------------------------------------------
# Proyección recursiva desde puntos rojos iniciales
print(f"\nIniciando proyecciones desde {len(red_points_info)} puntos rojos...")
for i, red_point in enumerate(red_points_info):
    print(f"Proyectando desde punto rojo {i+1} en arista {red_point['edge']}")
    project_from_red_point(red_point)

# Asegúrate que esto venga después:
plt.title('Triangulación con Proyecciones Recursivas')
plt.axis('equal')
plt.legend()
plt.show()

# --- Helper functions to query relationships ---
def get_alphas_for_vertex(vertex_id):
    """Obtiene todos los ángulos α asociados a un vértice"""
    return vertex_to_alphas.get(vertex_id, [])

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

# Ejemplo de uso de las funciones de consulta
if triangles:
    ejemplo_vertice = triangles[0][0]  # Primer vértice del primer triángulo
    print(f"\nÁngulos α para el vértice {ejemplo_vertice}:")
    for alpha_info in get_alphas_for_vertex(ejemplo_vertice):
        print(f"  α={alpha_info['alpha']:.1f}° en triángulo {alpha_info['triangle']}")
    
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