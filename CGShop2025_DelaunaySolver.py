# Import required libraries
import json  # For handling JSON files
import numpy as np  # For numerical operations
from cgshop2025_pyutils import DelaunayBasedSolver, verify  # CG:SHOP challenge utilities
import matplotlib.pyplot as plt  # For visualization
import os  # For file path operations
from shapely.geometry import Polygon  # For geometric operations
import networkx as nx  # For graph operations

# ======================================================================
# 1. DATA LOADING AND PREPARATION
# ======================================================================

# Define the input file path
file_name = 'ortho_80_06ee55d4.instance.json'  # Instance filename
directory = 'challenge_instances_cgshop25'  # Directory containing instances
file_path = os.path.join(directory, file_name)  # Full file path

# Load the JSON data from the instance file
with open(file_path, "r") as file:
    data = json.load(file)

# Create a class to structure the instance data in a format compatible with the solver
class JsonInstance:
    def __init__(self, data):
        # Convert points to numpy arrays for efficient computation
        self.points_x = np.array(data["points_x"])  # X coordinates of points
        self.points_y = np.array(data["points_y"])  # Y coordinates of points
        self.region_boundary = data["region_boundary"]  # Indices of boundary points
        self.additional_constraints = data["additional_constraints"]  # Any additional constraints
        self.instance_uid = data["instance_uid"]  # Unique identifier for the instance

# Create an instance of our structured data
instance = JsonInstance(data)

# ======================================================================
# 2. TRIANGULATION SOLUTION
# ======================================================================

# Initialize the Delaunay-based solver with our instance
solver = DelaunayBasedSolver(instance)
# Compute the triangulation solution
solution = solver.solve()

# Verify the solution meets all challenge requirements
result = verify(instance, solution)
print(f"Solution errors: {result.errors}")  # Print any validation errors

# ======================================================================
# 3. VISUALIZATION PREPARATION
# ======================================================================

# Prepare point coordinates as numpy array for easier manipulation
points = np.array(list(zip(instance.points_x, instance.points_y)))

# Extract boundary points and close the polygon by repeating the first point
boundary = np.array([points[i] for i in instance.region_boundary])
boundary = np.append(boundary, [boundary[0]], axis=0)
boundary_polygon = Polygon(boundary)  # Create Shapely polygon for containment checks

# Configure visualization with dark theme
plt.style.use('dark_background')
plt.figure(figsize=(9, 9))  # Create a large square figure

# Plot the region boundary
plt.plot(boundary[:, 0], boundary[:, 1], 'white', linewidth=2, label='Boundary')
# Plot the original input points
plt.scatter(points[:, 0], points[:, 1], color='lightsteelblue', label='Original Points')

# Draw all edges from the computed triangulation
for edge in solution.edges:
    x_coords = [instance.points_x[edge[0]], instance.points_x[edge[1]]]
    y_coords = [instance.points_y[edge[0]], instance.points_y[edge[1]]]
    plt.plot(x_coords, y_coords, 'steelblue', linewidth=1.2)

# ======================================================================
# 4. TRIANGLE ANALYSIS
# ======================================================================

# Reconstruct all triangles from the edge list using graph theory
graph = nx.Graph()
graph.add_edges_from(solution.edges)
# Find all 3-cliques (triangles) in the graph
triangles = [c for c in nx.enumerate_all_cliques(graph) if len(c) == 3]

# ----------------------------------------------------------------------
# Geometric calculation functions
# ----------------------------------------------------------------------

def calculate_incenter(triangle):
    """
    Calculate the incenter of a triangle (center point of incircle).
    Uses barycentric coordinates weighted by edge lengths.
    """
    # Calculate lengths of all three edges
    a = np.linalg.norm(triangle[1] - triangle[2])  # Edge opposite vertex 0
    b = np.linalg.norm(triangle[0] - triangle[2])  # Edge opposite vertex 1
    c = np.linalg.norm(triangle[0] - triangle[1])  # Edge opposite vertex 2
    
    perimeter = a + b + c
    # Compute incenter using barycentric coordinates
    incenter = (a * triangle[0] + b * triangle[1] + c * triangle[2]) / perimeter
    return incenter

def calculate_inradius(triangle):
    """
    Calculate the radius of the incircle using Heron's formula.
    """
    # Edge lengths
    a = np.linalg.norm(triangle[1] - triangle[2])
    b = np.linalg.norm(triangle[0] - triangle[2])
    c = np.linalg.norm(triangle[0] - triangle[1])
    
    # Semi-perimeter
    s = (a + b + c) / 2
    # Area using Heron's formula
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    # Inradius formula
    inradius = area / s
    return inradius

def calculate_tangency_points(triangle, incenter):
    """
    Calculate points where the incircle touches each edge (tangency points).
    Uses orthogonal projection of the incenter onto each edge.
    """
    tangency_points = []
    for i in range(3):  # For each edge of the triangle
        p1 = triangle[i]  # First vertex of edge
        p2 = triangle[(i + 1) % 3]  # Second vertex of edge
        
        # Vector along the edge
        edge = p2 - p1
        edge_unit = edge / np.linalg.norm(edge)  # Unit vector
        
        # Project incenter onto the edge
        proj = np.dot(incenter - p1, edge_unit)
        # Calculate exact tangency point
        tangency_point = p1 + proj * edge_unit
        
        tangency_points.append(tangency_point)
    return tangency_points

# ----------------------------------------------------------------------
# Process all triangles
# ----------------------------------------------------------------------

# Dictionaries to store geometric information
incenters = {}  # Maps triangles to their incenters
tangency_data = {}  # Maps triangles to their tangency points
edge_to_tangency = {}  # Maps edges to their adjacent triangles' tangency points

for triangle in triangles:
    # Get the actual points for this triangle
    tri_points = points[triangle]
    # Use sorted tuple as dictionary key for consistency
    sorted_triangle = tuple(sorted(triangle))
    
    # Only process triangles fully contained within the boundary
    if boundary_polygon.contains(Polygon(tri_points)):
        # Calculate geometric properties
        incenter = calculate_incenter(tri_points)
        inradius = calculate_inradius(tri_points)
        incenters[sorted_triangle] = incenter

        # Calculate and store tangency points
        tangency_points = np.array(calculate_tangency_points(tri_points, incenter))
        tangency_data[sorted_triangle] = tangency_points

        # Visualize the "inner triangle" formed by tangency points
        plt.fill(
            tangency_points[:, 0], tangency_points[:, 1],
            facecolor='slategray',  # Fill color
            edgecolor='white',  # Border color
            linewidth=1.2,  # Border width
            zorder=7  # Drawing order (higher = on top)
        )

        # Map each edge to its tangency points
        for i in range(3):
            edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
            if edge not in edge_to_tangency:
                edge_to_tangency[edge] = []
            edge_to_tangency[edge].append((sorted_triangle, tangency_points[i]))

# ======================================================================
# 5. EDGE AND TRIANGLE MAPPING
# ======================================================================

# Create a mapping from edges to their adjacent triangles
edge_to_triangles = {}
for triangle in triangles:
    tri_points = points[triangle]
    sorted_triangle = tuple(sorted(triangle))
    
    if boundary_polygon.contains(Polygon(tri_points)):
        # Get all three edges of the triangle (sorted for consistency)
        edges = [
            tuple(sorted([triangle[0], triangle[1]])),
            tuple(sorted([triangle[1], triangle[2]])),
            tuple(sorted([triangle[2], triangle[0]]))
        ]
        # Update the edge-to-triangles mapping
        for edge in edges:
            if edge not in edge_to_triangles:
                edge_to_triangles[edge] = []
            edge_to_triangles[edge].append(sorted_triangle)

# ======================================================================
# 6. VOID TRIANGLE PROCESSING
# ======================================================================

# Dictionaries to track void triangles and their properties
void_triangles = {}  # Maps void IDs to triangle data
tri_to_void_id = {}  # Maps triangles to their void IDs

# Process edges shared by two triangles
for edge, tris in edge_to_triangles.items():
    # Skip edges not shared by exactly two triangles or without tangency data
    if len(tris) != 2 or edge not in edge_to_tangency:
        continue

    tangency_info = edge_to_tangency[edge]
    # Skip if we don't have exactly two tangency points (one per triangle)
    if len(tangency_info) != 2:
        continue

    # Unpack the tangency information
    (tri_a, tg_a), (tri_b, tg_b) = tangency_info

    # Process both vertices of the edge
    for vertex in edge:
        pt_vertex = points[vertex]  # Current vertex coordinates
        
        # Calculate distances to both tangency points
        d_a = np.linalg.norm(pt_vertex - tg_a)
        d_b = np.linalg.norm(pt_vertex - tg_b)

        # Determine which tangency point is closer
        if d_a < d_b:
            closest_tg = tg_a
            source_tri = tri_a
            target_tri = tri_b
        else:
            closest_tg = tg_b
            source_tri = tri_b
            target_tri = tri_a

        # Get vertices of the target triangle
        target_verts = list(target_tri)

        # Find candidate edges that connect to our vertex
        candidate_edges = [
            tuple(sorted([vertex, v]))
            for v in target_verts if v != vertex
        ]
        # Exclude the current edge
        valid_edges = [e for e in candidate_edges if e != edge]
        
        # We expect exactly one valid edge (the other edge of the triangle)
        if len(valid_edges) != 1:
            continue

        correct_edge = valid_edges[0]
        # Get the other vertex of this edge
        v_other = correct_edge[0] if correct_edge[1] == vertex else correct_edge[1]

        # Prepare for projection
        a = points[vertex].astype(float)  # Current vertex
        b = points[v_other].astype(float)  # Connected vertex
        dir_vec = b - a  # Direction vector
        norm = np.linalg.norm(dir_vec)
        if norm == 0:  # Skip if points coincide
            continue
        dir_vec /= norm  # Normalize direction vector

        # Project the tangency point onto the adjacent edge
        dist = np.linalg.norm(pt_vertex - closest_tg)
        proj_point = a + dist * dir_vec

        # Visualization of the projection
        # Yellow line from tangency point to projected point
        plt.plot([closest_tg[0], proj_point[0]], [closest_tg[1], proj_point[1]], 
                color='yellow', linewidth=1.2, zorder=7)
        
        # Red point at the projected location
        plt.scatter(proj_point[0], proj_point[1], color='red', s=25, zorder=8)
        
        # Green point at the tangency point (start of projection)
        plt.scatter(closest_tg[0], closest_tg[1], color='lime', s=25, zorder=8)

        # Associate the projected point with its void triangle
        rounded_proj = tuple(np.round(proj_point, decimals=6))  # Avoid floating point issues
        sorted_target_tri = tuple(sorted(target_tri))

        # Create new void triangle entry if needed
        if sorted_target_tri not in tri_to_void_id:
            void_id = len(void_triangles) + 1
            tri_to_void_id[sorted_target_tri] = void_id
            void_triangles[void_id] = {
                "triangle": sorted_target_tri,
                "points": []
            }

        # Add this projected point to the void triangle
        void_id = tri_to_void_id[sorted_target_tri]
        void_triangles[void_id]["points"].append(rounded_proj)

# ======================================================================
# 7. OUTPUT AND FINAL VISUALIZATION
# ======================================================================

# Print information about all void triangles
for void_id, data in void_triangles.items():
    print(f"Void_Triangle {void_id} (verts: {data['triangle']}):")
    for pt in data["points"]:
        print(f"  🔴 Point: {pt}")

# Finalize the visualization
plt.title('Triangulation with Projected Lines from Tangency')
plt.axis('equal')  # Ensure proper aspect ratio
plt.legend()  # Show the legend
plt.show()  # Display the plot