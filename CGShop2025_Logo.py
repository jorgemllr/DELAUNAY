import json
import numpy as np
from cgshop2025_pyutils import DelaunayBasedSolver, verify
import matplotlib.pyplot as plt
import os
from shapely.geometry import Polygon
import networkx as nx
import math
from collections import defaultdict
from shapely import prepared

ENABLE_PLOT = True
SHOW_DEBUG_ARROWS_AND_POINTS = False
global next_point_id
next_point_id = 0

def calculate_sides(triangle_points):
    """Calculates the sides a, b, c of a triangle given its 3 points A, B, C."""
    A, B, C = triangle_points
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    return a, b, c

def calculate_triangle_angles(triangle_points):
    """Calculates the 3 angles of a triangle given its 3 points A, B, C."""
    a, b, c = calculate_sides(triangle_points)
    angle_A = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2 * b * c)))
    angle_B = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))
    angle_C = np.degrees(np.arccos((a**2 + b**2 - c**2) / (2 * a * b)))
    if not np.isclose(angle_A + angle_B + angle_C, 180, atol=1.0):
        print(f"Warning! Angles do not sum to 180°: {angle_A:.1f} + {angle_B:.1f} + {angle_C:.1f} = {angle_A + angle_B + angle_C:.1f}")
    return angle_A, angle_B, angle_C

def other_vertex(edge, vertex):
    """Given an edge (tuple of 2 vertices) and a vertex, returns the opposite vertex in that edge."""
    return edge[1] if edge[0] == vertex else edge[0]

def calculate_angle(a, b, c):
    """Calculates the angle at vertex B (between sides a and c) using the Law of Cosines."""
    angle_rad = math.acos((a**2 + c**2 - b**2) / (2 * a * c))
    return math.degrees(angle_rad)

# File path and instance name
# file_name = 'ortho_80_06ee55d4.instance.json'  # Instance filename
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
solver = DelaunayBasedSolver(instance)
solution = solver.solve()

# Verify solution
result = verify(instance, solution)
print(f"Solution errors: {result.errors}")

# Prepare points and boundary for visualization
points = np.array(list(zip(instance.points_x, instance.points_y)), dtype=float)
boundary = np.array([points[i] for i in instance.region_boundary])
boundary = np.append(boundary, [boundary[0]], axis=0)
boundary_polygon = Polygon(boundary)
prepared_boundary = prepared.prep(boundary_polygon)

# Set dark mode theme
if ENABLE_PLOT:
    plt.style.use('dark_background')
    plt.figure(figsize=(9, 9))
    plt.plot(boundary[:, 0], boundary[:, 1], 'white', linewidth=0, label='Boundary', zorder=10)
    plt.scatter(points[:, 0], points[:, 1], color='lightsteelblue', label='Original Points', zorder=11)

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
def calculate_incenter(triangle, a, b, c):
    perimeter = a + b + c
    incenter = (a * triangle[0] + b * triangle[1] + c * triangle[2]) / perimeter
    return incenter

def calculate_inradius(a, b, c):
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
triangle_info = {}  # Stores complete information about each triangle
vertex_to_alphas = defaultdict(list)  # Maps vertices to their α angles
edge_to_alphas = defaultdict(list)  # Maps edges to their α angles
triangle_edges_cache = {}  # Cache to avoid rebuilding edges

def get_triangle_edges(triangle):
    sorted_triangle = tuple(sorted(triangle))
    if sorted_triangle in triangle_edges_cache:
        return triangle_edges_cache[sorted_triangle]
    
    edges = [
        tuple(sorted([triangle[0], triangle[1]])),
        tuple(sorted([triangle[1], triangle[2]])),
        tuple(sorted([triangle[2], triangle[0]]))
    ]
    triangle_edges_cache[sorted_triangle] = edges
    return edges

# Create triangle polygons only once (cache)
tri_polygons = {tuple(sorted(tri)): Polygon(points[tri]) for tri in triangles}

for triangle in triangles:
    tri_points = points[triangle]
    sorted_triangle = tuple(sorted(triangle))
    
    # Use cached polygon instead of creating it again
    tri_polygon = tri_polygons[sorted_triangle]
    if prepared_boundary.contains(tri_polygon):
        # Calculate sides and angles with optimized functions
        a, b, c = calculate_sides(tri_points)
        angle_A, angle_B, angle_C = calculate_triangle_angles(tri_points)

        # Optional validation
        if not np.isclose(angle_A + angle_B + angle_C, 180, atol=1.0):
            print(f"Warning! Angles do not sum to 180°: {angle_A:.1f} + {angle_B:.1f} + {angle_C:.1f} = {angle_A + angle_B + angle_C:.1f}")

        # Use precomputed sides for incenter and radius
        incenter = calculate_incenter(tri_points, a, b, c)
        inradius = calculate_inradius(a, b, c)
        tangency_points = np.array(calculate_tangency_points(tri_points, incenter))

        # Calculate α angles for each vertex
        alpha_A = (180 - angle_A) / 2
        alpha_B = (180 - angle_B) / 2
        alpha_C = (180 - angle_C) / 2

        # Define the edges of the triangle
        edges = get_triangle_edges(triangle)

        # Store complete information of the triangle
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

        # Map vertices to their α angles
        vertex_to_alphas[triangle[0]].append({'alpha': alpha_A, 'triangle': sorted_triangle, 'edges': [edges[0], edges[2]]})
        vertex_to_alphas[triangle[1]].append({'alpha': alpha_B, 'triangle': sorted_triangle, 'edges': [edges[0], edges[1]]})
        vertex_to_alphas[triangle[2]].append({'alpha': alpha_C, 'triangle': sorted_triangle, 'edges': [edges[1], edges[2]]})

        # Map edges to α angles
        for edge in edges:
            edge_to_alphas[edge].append({
                'alpha_A': alpha_A if edge in [edges[0], edges[2]] else None,
                'alpha_B': alpha_B if edge in [edges[0], edges[1]] else None,
                'alpha_C': alpha_C if edge in [edges[1], edges[2]] else None,
                'triangle': sorted_triangle
            })

# --- Visualization with alpha labels ---
if ENABLE_PLOT:
    for tri_data in triangle_info.values():
        # Draw the triangle and its elements
        for angle_key, angle_data in tri_data['angles'].items():
            vertex_idx = angle_data['vertex']
            vertex_point = tri_data['points'][list(tri_data['vertices']).index(vertex_idx)]
            alpha = angle_data['alpha']
            
            # Text position (40% towards the incenter)
            direction = tri_data['incenter'] - vertex_point
            label_pos = vertex_point + 0.4 * direction
            
            # Display α in white (uncomment if you want to see it)
            # plt.text(
            #     label_pos[0], label_pos[1], 
            #     f"α={alpha:.1f}°", 
            #     color='white', fontsize=6, ha='center', va='center',
            #     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'),
            #     zorder=10
            # )

        # Draw the inner triangle (slategray)
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
        edges = get_triangle_edges(triangle)
        for edge in edges:
            edge_to_triangles[edge].append(sorted_triangle)
            # Add corresponding tangency points
            tangents = triangle_info[sorted_triangle]['tangency_points']
            if edge == edges[0]:
                edge_to_tangency[edge].append((sorted_triangle, tangents[0]))  # Point between v0 and v1
            elif edge == edges[1]:
                edge_to_tangency[edge].append((sorted_triangle, tangents[1]))  # Point between v1 and v2
            else:
                edge_to_tangency[edge].append((sorted_triangle, tangents[2]))  # Point between v2 and v0

# Project points and display tangency lines
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
        v_other = other_vertex(correct_edge, vertex)

        a = points[vertex].astype(float)
        b = points[v_other].astype(float)
        dir_vec = b - a
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-8:
            continue
        dir_vec /= norm

        dist = np.linalg.norm(pt_vertex - closest_tg)
        proj_point = a + dist * dir_vec

        if ENABLE_PLOT and SHOW_DEBUG_ARROWS_AND_POINTS:
            plt.plot([closest_tg[0], proj_point[0]], [closest_tg[1], proj_point[1]], color='orange', linewidth=1.2, zorder=7)
            plt.scatter(proj_point[0], proj_point[1], color='red', s=10, zorder=8)
            plt.scatter(closest_tg[0], closest_tg[1], color='lime', s=10, zorder=8)
            
        if ENABLE_PLOT:
            plt.plot([closest_tg[0], proj_point[0]], [closest_tg[1], proj_point[1]], color='white', linewidth=1.2, zorder=7)
        
        # Register the new red point as a target on the projected edge
        red_targets[tuple(sorted(correct_edge))].append({
            'point': proj_point,
            'from_triangle': target_tri,
            'parent': None,          # Has no parent, it's the initial point
            'generation': 0,         # First generation
            'id': next_point_id
        })
        next_point_id += 1

# --- Helper functions to query relationships ---
def get_alphas_for_vertex(vertex_id):
    """Gets all α angles associated with a vertex"""
    return vertex_to_alphas.get(vertex_id, [])

def get_vertex_alpha_info(vertex_id):
    """Gets detailed information of all α angles for a vertex"""
    alphas = vertex_to_alphas.get(vertex_id, [])
    result = []
    for alpha_info in alphas:
        tri_data = triangle_info[alpha_info['triangle']]
        angle_data = None
        # Find the corresponding angle in triangle_info
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
    """Gets the α angles associated with an edge"""
    return edge_to_alphas.get(tuple(sorted(edge)), [])

def get_triangles_for_alpha(alpha_value, tolerance=0.1):
    """Finds all triangles that contain a specific α angle"""
    result = []
    for tri, data in triangle_info.items():
        for angle_data in data['angles'].values():
            if abs(angle_data['alpha'] - alpha_value) < tolerance:
                result.append((tri, angle_data))
                break
    return result

def is_edge_on_boundary(edge):
    return len(edge_to_triangles[edge]) == 1

def propagate_red_targets(generation):
    global next_point_id
    new_targets = []
    PROCESS_SHARED_EDGES_TWICE = True  # Change to False to process edges only once
    processed_edges = set() if not PROCESS_SHARED_EDGES_TWICE else set()

    for tri_key, tri_data in triangle_info.items():
        for edge in tri_data['edges']:
            # Avoid processing the same edge twice if PROCESS_SHARED_EDGES_TWICE is False
            sorted_edge = tuple(sorted(edge))
            if not PROCESS_SHARED_EDGES_TWICE and sorted_edge in processed_edges:
                continue
            processed_edges.add(sorted_edge)

            # Red points from the previous generation on this edge
            red_pts = [
                pt for pt in red_targets.get(edge, []) 
                if pt['generation'] == generation - 1 
                and not pt.get('stop_here', False)
                and pt['from_triangle'] != tri_key
            ]  # only if it comes from another triangle
            if not red_pts:
                continue  # No red points to process

            # Get the tangency point of THIS triangle on THIS edge
            tg_points = [tp for t, tp in edge_to_tangency[edge] if t == tri_key]
            if len(tg_points) != 1:
                print(f"Warning: Expected exactly 1 tangency point for edge {edge} in triangle {tri_key}, but found {len(tg_points)}")
                continue
            tg_point = tg_points[0]  # Green tangency point

            # Get vertices of the edge
            v1, v2 = edge
            pt_v1 = points[v1]
            pt_v2 = points[v2]

            # Compute distance from green point to each vertex
            d_tg_v1 = np.linalg.norm(pt_v1 - tg_point)
            d_tg_v2 = np.linalg.norm(pt_v2 - tg_point)

            for pt_data in red_pts:
                pt = pt_data['point']

                # Compute distance from red point to each vertex
                d_pt_v1 = np.linalg.norm(pt_v1 - pt)
                d_pt_v2 = np.linalg.norm(pt_v2 - pt)

                # Select origin vertex based on distances
                if d_pt_v1 < d_tg_v1:
                    origin_vertex = v1
                    dist = d_pt_v1
                elif d_pt_v2 < d_tg_v2:
                    origin_vertex = v2
                    dist = d_pt_v2
                else:
                    continue  # Doesn't meet any condition, skip this point

                # Find another edge that contains origin_vertex but is not the current one
                candidate_edges = [e for e in tri_data['edges'] if origin_vertex in e and e != edge]
                if not candidate_edges:
                    continue

                new_edge = candidate_edges[0]
                # Check if the new edge is on the domain boundary
                # PROJECTION IS DONE EVEN IF ON BOUNDARY
                on_boundary = is_edge_on_boundary(new_edge)

                #331
                # Get the other vertex of the new edge
                v_other = new_edge[0] if new_edge[1] == origin_vertex else new_edge[1]

                # Calculate projection direction
                a = points[origin_vertex].astype(float)
                b = points[v_other].astype(float)
                dir_vec = b - a
                norm = np.linalg.norm(dir_vec)
                if norm < 1e-8:
                    continue
                dir_vec /= norm

                # Create new projected point
                new_pt = a + dist * dir_vec

                # Draw white line and purple point
                if ENABLE_PLOT and SHOW_DEBUG_ARROWS_AND_POINTS:
                    dx = new_pt[0] - pt[0]
                    dy = new_pt[1] - pt[1]

                    plt.arrow(pt[0], pt[1], dx, dy,
                            head_width=5000,
                            head_length=5000,
                            fc='white', ec='white',
                            linewidth=1.2, zorder=7,
                            length_includes_head=True)
                    
                    plt.scatter(new_pt[0], new_pt[1], color='purple', s=2, zorder=9)
                    plt.scatter(new_pt[0], new_pt[1], color='purple', s=2, zorder=9)

                # New block to show white lines without arrows or purple points
                if ENABLE_PLOT:
                    plt.plot([pt[0], new_pt[0]], [pt[1], new_pt[1]], color='white', linewidth=1.2, zorder=7)

                # Register new red target point
                new_targets.append((tuple(sorted(new_edge)), {
                    'point': new_pt,
                    'from_triangle': tri_key,
                    'parent': pt_data['id'],
                    'generation': generation,
                    'id': next_point_id,
                    'stop_here': on_boundary
                }))
                next_point_id += 1

    # Add new points to red_targets
    for edge_key, red_data in new_targets:
        red_targets[edge_key].append(red_data)
        
# Run red point propagation
max_iterations = 100
for generation in range(1, max_iterations + 1):
    num_before = sum(len(v) for v in red_targets.values())
    propagate_red_targets(generation)
    num_after = sum(len(v) for v in red_targets.values())
    if num_after == num_before:
        break  # No new points were generated, stop propagation
    
# Final plot settings
if ENABLE_PLOT:
    plt.title('Triangulation with P-Paths')
    plt.axis('equal')
    plt.legend()
    plt.show()

# Example usage of the query functions
if triangles:
    ejemplo_vertice = triangles[0][0]  # First vertex of the first triangle
    print(f"\nDetailed α angle information for vertex {ejemplo_vertice}:")
    for alpha_info in get_vertex_alpha_info(ejemplo_vertice):
        print(f"  α={alpha_info['alpha']:.1f}°")
        print(f"  Triangle: {alpha_info['triangle']}")
        print(f"  Associated edges: {alpha_info['edges']}")
        print(f"  Adjacent vertices: {alpha_info['adjacent_vertices']}\n")
    
    ejemplo_arista = tuple(sorted([triangles[0][0], triangles[0][1]]))
    print(f"\nα angles for edge {ejemplo_arista}:")
    for alpha_info in get_alphas_for_edge(ejemplo_arista):
        print(f"  Triangle {alpha_info['triangle']}:")
        if alpha_info['alpha_A'] is not None:
            print(f"    α_A={alpha_info['alpha_A']:.1f}°")
        if alpha_info['alpha_B'] is not None:
            print(f"    α_B={alpha_info['alpha_B']:.1f}°")
        if alpha_info['alpha_C'] is not None:
            print(f"    α_C={alpha_info['alpha_C']:.1f}°")

# --- EXPORTAR LA TRIANGULACIÓN FINAL PARA EL LOGO ---
export_data = {
    "points_x": list(instance.points_x),
    "points_y": list(instance.points_y),
    "edges": [list(edge) for edge in solution.edges]
}

export_filename = "meslatt_logo_triangulated_masterpiece.json"
with open(export_filename, "w") as f:
    json.dump(export_data, f, indent=4)

print(f"\n[EXITO] La triangulacion final se guardo en {export_filename}")
print("Llevate este archivo a tu Mac para vectorizar el logo.")