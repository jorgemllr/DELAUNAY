import json
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from itertools import combinations

# --- Configuration ---
EXAMPLES_DIR = 'examples'
FILE_NAME = 'example_ps_20_nt2_pfd5_random.json'
OUTPUT_DIR = 'results'  # Output folder for results


# --- Helper Functions ---
def plot_triangulation(ax, title, points, edges):
    """
    Draws a triangulation on the given axis (subplot).
    """
    edge_set = {tuple(sorted(edge)) for edge in edges}
    for edge in edge_set:
        p1_idx, p2_idx = edge
        x_coords = [points[p1_idx, 0], points[p2_idx, 0]]
        y_coords = [points[p1_idx, 1], points[p2_idx, 1]]
        ax.plot(x_coords, y_coords, 'steelblue', linewidth=1.2, zorder=1)

    ax.scatter(points[:, 0], points[:, 1], color='lightsteelblue', zorder=2)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')


def find_convex_quadrilaterals(edges, points):
    """
    Finds all convex quadrilaterals in the current triangulation.
    Each quadrilateral is represented by its internal and external diagonals.
    """
    g = nx.Graph()
    g.add_edges_from(edges)

    # Identify all triangles in the graph
    triangles = [frozenset(c) for c in nx.enumerate_all_cliques(g) if len(c) == 3]
    convex_quads = []

    def orientation(p, q, r):
        """Check orientation of three points."""
        val = (points[q][1] - points[p][1]) * (points[r][0] - points[q][0]) - \
              (points[q][0] - points[p][0]) * (points[r][1] - points[q][1])
        if val == 0:
            return 0
        return 1 if val > 0 else -1

    # Compare all triangle pairs to find shared edges and possible quadrilaterals
    for t1, t2 in combinations(triangles, 2):
        shared_nodes = t1.intersection(t2)
        if len(shared_nodes) == 2:
            p1_idx, p2_idx = tuple(shared_nodes)
            p3_idx = list(t1.difference(shared_nodes))[0]
            p4_idx = list(t2.difference(shared_nodes))[0]

            orient1 = orientation(p1_idx, p2_idx, p3_idx)
            orient2 = orientation(p1_idx, p2_idx, p4_idx)

            if orient1 != 0 and orient2 != 0 and orient1 != orient2:
                convex_quads.append({
                    'internal_diag': tuple(sorted((p1_idx, p2_idx))),
                    'external_diag': tuple(sorted((p3_idx, p4_idx))),
                })

    return convex_quads


def backtracking_search(current_edges, target_edges_set, k_remaining, points, visited_states):
    """
    Backtracking search to transform one triangulation into another.
    Stops if the target triangulation is reached or if no flips are possible
    within the given flip budget.
    """
    current_edges_set = frozenset(current_edges)

    if current_edges_set == target_edges_set:
        return True, []

    if k_remaining <= 0 or current_edges_set in visited_states:
        return False, None

    visited_states.add(current_edges_set)
    possible_flips = find_convex_quadrilaterals(current_edges, points)

    for quad in possible_flips:
        old_diag, new_diag = quad['internal_diag'], quad['external_diag']
        next_edges = list(current_edges)
        next_edges.remove(old_diag)
        next_edges.append(new_diag)

        success, path_found = backtracking_search(
            next_edges, target_edges_set, k_remaining - 1, points, visited_states
        )
        if success:
            return True, [(old_diag, new_diag)] + path_found

    return False, None


# --- Main Script ---
if __name__ == "__main__":
    # Load example data
    file_path = os.path.join(EXAMPLES_DIR, FILE_NAME)
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Make sure results folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    points = np.array(list(zip(data['points_x'], data['points_y'])))
    t_initial_edges = [tuple(sorted(edge)) for edge in data['triangulations'][0]]
    t_final_edges_set = {tuple(sorted(edge)) for edge in data['triangulations'][-1]}

    # --- Phase 1: Greedy algorithm ---
    print("--- PHASE 1: Running greedy algorithm ---")
    t_current_edges = list(t_initial_edges)
    greedy_flips_path = []

    while True:
        found_free_diagonal_flip = False
        quads = find_convex_quadrilaterals(t_current_edges, points)

        for quad in quads:
            old_diag, new_diag = quad['internal_diag'], quad['external_diag']
            if new_diag in t_final_edges_set and old_diag not in t_final_edges_set:
                print(f"  Greedy flip found: {old_diag} -> {new_diag}")
                t_current_edges.remove(old_diag)
                t_current_edges.append(new_diag)
                greedy_flips_path.append((old_diag, new_diag))
                found_free_diagonal_flip = True
                break

        if not found_free_diagonal_flip:
            break

    print(f"Phase 1 finished. {len(greedy_flips_path)} greedy flips performed.")

    # --- Phase 2: Backtracking search if needed ---
    full_path = []
    search_path = []

    if frozenset(t_current_edges) != t_final_edges_set:
        print("\n--- PHASE 2: Starting backtracking search ---")
        remaining_flips_needed = len(t_final_edges_set.difference(t_current_edges))
        k_budget = remaining_flips_needed + 2
        print(f"Flip budget for search: {k_budget}")

        success, search_path_found = backtracking_search(
            t_current_edges, t_final_edges_set, k_budget, points, set()
        )

        if success:
            print(f"Success! Found a solution with {len(search_path_found)} additional flips.")
            search_path = search_path_found
        else:
            print("Search did not find a complete solution within the flip budget.")
    else:
        print("\nGreedy algorithm was enough to reach the target triangulation.")

    # --- Final Processing ---
    full_path = greedy_flips_path + search_path
    total_flips = len(full_path)

    t_result_edges = list(t_initial_edges)
    for old, new in full_path:
        t_result_edges.remove(old)
        t_result_edges.append(new)

    midpoint_index = total_flips // 2
    first_half_flips = full_path[:midpoint_index]

    t_central_edges = list(t_initial_edges)
    for old, new in first_half_flips:
        t_central_edges.remove(old)
        t_central_edges.append(new)

    print(f"\nProcess completed. Total flips: {total_flips}")
    print(f"Central triangulation reached after {midpoint_index} flips.")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Flip Distance Analysis for "{FILE_NAME}"', fontsize=16)

    plot_triangulation(axes[0, 0], 'Initial Triangulation', points, t_initial_edges)
    plot_triangulation(axes[0, 1], 'Target Triangulation', points, list(t_final_edges_set))
    plot_triangulation(axes[1, 0], f'Central Triangulation (after {midpoint_index} flips)', points, t_central_edges)
    plot_triangulation(axes[1, 1], f'Final Result (after {total_flips} flips)', points, t_result_edges)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure to output folder
    base_name = os.path.splitext(FILE_NAME)[0]
    output_filename = f"{base_name}_analysis.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully at: {output_path}")

    # Show figure on screen
    plt.show()