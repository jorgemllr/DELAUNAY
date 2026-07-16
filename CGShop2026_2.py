import json
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from itertools import combinations

# --- Configuración ---
EXAMPLES_DIR = 'examples'
FILE_NAME = 'example_ps_20_nt2_pfd5_random.json'

# --- Funciones Auxiliares ---
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
    """
    Encuentra todos los cuadriláteros CONVEXOS formados por pares de triángulos
    que comparten una arista.
    """
    g = nx.Graph()
    g.add_edges_from(edges)
    
    triangles = [frozenset(c) for c in nx.enumerate_all_cliques(g) if len(c) == 3]
    
    convex_quads = []
    
    # Función para la prueba de orientación (basada en producto cruzado)
    def orientation(p, q, r):
        # Compara la orientación de la tupla de puntos (p, q, r)
        # Devuelve > 0 si giran a la izquierda (anti-horario), < 0 si giran a la derecha (horario), 0 si son colineales.
        val = (points[q][1] - points[p][1]) * (points[r][0] - points[q][0]) - \
              (points[q][0] - points[p][0]) * (points[r][1] - points[q][1])
        if val == 0: return 0  # Colineal
        return 1 if val > 0 else -1  # Horario o Anti-horario

    for t1, t2 in combinations(triangles, 2):
        shared_nodes = t1.intersection(t2)
        if len(shared_nodes) == 2:
            p1_idx, p2_idx = tuple(shared_nodes)
            p3_idx = list(t1.difference(shared_nodes))[0]
            p4_idx = list(t2.difference(shared_nodes))[0]

            # --- ¡LA NUEVA COMPROBACIÓN DE CONVEXIDAD! ---
            # Para que el cuadrilátero sea convexo, p3 y p4 deben estar en lados opuestos
            # de la línea formada por la diagonal compartida (p1, p2).
            # Esto ocurre si las orientaciones (p1,p2,p3) y (p1,p2,p4) son diferentes.
            orient1 = orientation(p1_idx, p2_idx, p3_idx)
            orient2 = orientation(p1_idx, p2_idx, p4_idx)
            
            if orient1 != 0 and orient2 != 0 and orient1 != orient2:
                # Es convexo, por lo tanto es un candidato válido para flip.
                convex_quads.append({
                    'internal_diag': tuple(sorted((p1_idx, p2_idx))),
                    'external_diag': tuple(sorted((p3_idx, p4_idx))),
                    'vertices': {p1_idx, p2_idx, p3_idx, p4_idx}
                })
    return convex_quads
    
# --- Script Principal ---
if __name__ == "__main__":
    file_path = os.path.join(EXAMPLES_DIR, FILE_NAME)
    
    if not os.path.exists(file_path):
        print(f"Error: El archivo '{file_path}' no fue encontrado.")
    else:
        print(f"Cargando instancia desde: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)

        points = np.array(list(zip(data['points_x'], data['points_y'])))
        t_initial_edges = [tuple(sorted(edge)) for edge in data['triangulations'][0]]
        t_final_edges = {tuple(sorted(edge)) for edge in data['triangulations'][-1]}
        t_current_edges = list(t_initial_edges)
        
        flips_realizados = 0
        print("\nIniciando el proceso de flips voraces (versión robusta)...")

        while True:
            found_free_diagonal_flip = False
            
            # 1. Identificar solo los cuadriláteros CONVEXOS
            quads = find_convex_quadrilaterals(t_current_edges, points)
            
            for quad in quads:
                old_diag = quad['internal_diag']
                new_diag = quad['external_diag']
                
                if new_diag in t_final_edges and old_diag not in t_final_edges:
                    print(f"  Flip convexo válido encontrado! Cambiando {old_diag} -> {new_diag}")
                    
                    t_current_edges.remove(old_diag)
                    t_current_edges.append(new_diag)
                    flips_realizados += 1
                    found_free_diagonal_flip = True
                    break
            
            if not found_free_diagonal_flip:
                print("\nNo se encontraron más 'diagonales libres' en cuadriláteros convexos. El algoritmo ha finalizado.")
                break
        
        print(f"\nTotal de flips realizados: {flips_realizados}")
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(f'Algoritmo Voraz Robusto para "{FILE_NAME}"', fontsize=16)

        plot_triangulation(ax1, 'Triangulación Inicial', points, t_initial_edges)
        plot_triangulation(ax2, f'Resultado Tras {flips_realizados} Flips', points, t_current_edges)
        plot_triangulation(ax3, 'Triangulación Final (Objetivo)', points, list(t_final_edges))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()