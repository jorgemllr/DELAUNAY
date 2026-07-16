import json
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from itertools import combinations

# --- Configuración ---
EXAMPLES_DIR = 'examples'
FILE_NAME = 'example_ps_20_nt2_pfd5_random.json'

# --- Funciones Auxiliares (sin cambios) ---
def plot_triangulation(ax, title, points, edges):
    """Dibuja una triangulación en un eje (subplot) específico."""
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
    """Encuentra todos los cuadriláteros CONVEXOS."""
    g = nx.Graph()
    g.add_edges_from(edges)
    triangles = [frozenset(c) for c in nx.enumerate_all_cliques(g) if len(c) == 3]
    convex_quads = []
    
    def orientation(p, q, r):
        val = (points[q][1] - points[p][1]) * (points[r][0] - points[q][0]) - \
              (points[q][0] - points[p][0]) * (points[r][1] - points[q][1])
        if val == 0: return 0
        return 1 if val > 0 else -1

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

# --- NUEVA FASE 2: BÚSQUEDA CON RETROCESO (BACKTRACKING) ---
def backtracking_search(current_edges, target_edges_set, k_remaining, points, visited_states):
    """
    Busca una secuencia de flips de forma recursiva.
    """
    # Condición de éxito: hemos llegado a la triangulación final
    current_edges_set = frozenset(current_edges)
    if current_edges_set == target_edges_set:
        return True, [] # Devuelve éxito y una lista de flips vacía (camino encontrado)

    # Condición de fallo: nos quedamos sin flips o entramos en un ciclo
    if k_remaining <= 0:
        return False, None
    
    current_state = frozenset(current_edges)
    if current_state in visited_states:
        return False, None
    
    visited_states.add(current_state)

    # Exploración: prueba todos los flips válidos desde el estado actual
    possible_flips = find_convex_quadrilaterals(current_edges, points)
    for quad in possible_flips:
        old_diag = quad['internal_diag']
        new_diag = quad['external_diag']
        
        # Crea la nueva triangulación aplicando el flip
        next_edges = list(current_edges)
        next_edges.remove(old_diag)
        next_edges.append(new_diag)
        
        # Llamada recursiva
        success, path_found = backtracking_search(next_edges, target_edges_set, k_remaining - 1, points, visited_states)
        
        # Si la llamada recursiva tuvo éxito, construye el camino y propágalo hacia arriba
        if success:
            solution_path = [(old_diag, new_diag)] + path_found
            return True, solution_path
            
    # Si ningún flip desde este estado lleva a una solución, es un callejón sin salida
    return False, None

# --- Script Principal ---
if __name__ == "__main__":
    file_path = os.path.join(EXAMPLES_DIR, FILE_NAME)
    with open(file_path, 'r') as f:
        data = json.load(f)

    points = np.array(list(zip(data['points_x'], data['points_y'])))
    t_initial_edges = [tuple(sorted(edge)) for edge in data['triangulations'][0]]
    t_final_edges_set = {tuple(sorted(edge)) for edge in data['triangulations'][-1]}
    
    # --- FASE 1: ALGORITMO VORAZ RÁPIDO ---
    print("--- FASE 1: Ejecutando algoritmo voraz rápido ---")
    t_current_edges = list(t_initial_edges)
    greedy_flips_path = []
    while True:
        found_free_diagonal_flip = False
        quads = find_convex_quadrilaterals(t_current_edges, points)
        for quad in quads:
            old_diag, new_diag = quad['internal_diag'], quad['external_diag']
            if new_diag in t_final_edges_set and old_diag not in t_final_edges_set:
                print(f"  Flip voraz encontrado! Cambiando {old_diag} -> {new_diag}")
                t_current_edges.remove(old_diag)
                t_current_edges.append(new_diag)
                greedy_flips_path.append((old_diag, new_diag))
                found_free_diagonal_flip = True
                break
        if not found_free_diagonal_flip:
            break
    
    print(f"Fase 1 completada. Se realizaron {len(greedy_flips_path)} flips voraces.")

    # --- FASE 2: BÚSQUEDA PROFUNDA SI ES NECESARIO ---
    if frozenset(t_current_edges) != t_final_edges_set:
        print("\n--- FASE 2: La solución no fue encontrada. Iniciando búsqueda con retroceso... ---")
        
        # Calculamos un presupuesto de flips razonable para la búsqueda
        # Basado en cuántas aristas todavía no coinciden.
        remaining_flips_needed = len(t_final_edges_set.difference(t_current_edges))
        k_budget = remaining_flips_needed + 2 # Damos un pequeño margen
        
        print(f"Presupuesto de flips para la búsqueda: {k_budget}")

        visited_states_cache = set()
        success, search_path = backtracking_search(
            t_current_edges, t_final_edges_set, k_budget, points, visited_states_cache
        )

        if success:
            print(f"¡Éxito! Búsqueda encontró una solución con {len(search_path)} flips adicionales.")
            # Aplicar los flips encontrados en la búsqueda
            for old_diag, new_diag in search_path:
                t_current_edges.remove(old_diag)
                t_current_edges.append(new_diag)
            
            total_flips = len(greedy_flips_path) + len(search_path)
        else:
            print("La búsqueda no encontró una solución dentro del presupuesto de flips.")
            total_flips = len(greedy_flips_path)
    else:
        print("\n¡El algoritmo voraz fue suficiente para encontrar la solución!")
        total_flips = len(greedy_flips_path)
        
    print(f"\nProceso finalizado. Total de flips realizados: {total_flips}")

    # --- Visualización ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f'Algoritmo Híbrido para "{FILE_NAME}"', fontsize=16)

    plot_triangulation(ax1, 'Triangulación Inicial', points, t_initial_edges)
    plot_triangulation(ax2, f'Resultado Tras {total_flips} Flips', points, t_current_edges)
    plot_triangulation(ax3, 'Triangulación Final (Objetivo)', points, list(t_final_edges_set))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()