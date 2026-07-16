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

def find_quadrilaterals(edges):
    """
    Encuentra todos los cuadriláteros formados por pares de triángulos
    que comparten una arista.
    """
    g = nx.Graph()
    g.add_edges_from(edges)
    
    # Encuentra todos los triángulos (cliques de tamaño 3)
    triangles = [frozenset(c) for c in nx.enumerate_all_cliques(g) if len(c) == 3]
    
    quadrilaterals = []
    # Compara cada par de triángulos para encontrar los que comparten una arista
    for t1, t2 in combinations(triangles, 2):
        shared_nodes = t1.intersection(t2)
        if len(shared_nodes) == 2:
            # Estos dos triángulos forman un cuadrilátero
            p1, p2 = tuple(shared_nodes)
            p3 = list(t1.difference(shared_nodes))[0]
            p4 = list(t2.difference(shared_nodes))[0]
            # La diagonal interna es (p1, p2), la externa es (p3, p4)
            quadrilaterals.append({
                'internal_diag': tuple(sorted((p1, p2))),
                'external_diag': tuple(sorted((p3, p4))),
                'vertices': {p1, p2, p3, p4}
            })
    return quadrilaterals
    
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
        
        # Triangulaciones inicial y final
        t_initial_edges = [tuple(sorted(edge)) for edge in data['triangulations'][0]]
        t_final_edges = {tuple(sorted(edge)) for edge in data['triangulations'][-1]}

        # Copia de la triangulación inicial que vamos a modificar
        t_current_edges = list(t_initial_edges)
        
        flips_realizados = 0
        print("\nIniciando el proceso de flips voraces...")

        # Bucle principal: sigue intentando hacer flips mientras se encuentren mejoras
        while True:
            found_free_diagonal_flip = False
            
            # 1. Identificar todos los cuadriláteros en la triangulación actual
            quads = find_quadrilaterals(t_current_edges)
            
            for quad in quads:
                old_diag = quad['internal_diag']
                new_diag = quad['external_diag']
                
                # 2. Comprobar si el flip resulta en una diagonal de la triangulación final
                if new_diag in t_final_edges and old_diag not in t_final_edges:
                    print(f"  Flip encontrado! Cambiando {old_diag} -> {new_diag}")
                    
                    # 3. Realizar el flip
                    t_current_edges.remove(old_diag)
                    t_current_edges.append(new_diag)
                    
                    flips_realizados += 1
                    found_free_diagonal_flip = True
                    # Rompemos el bucle para re-evaluar los cuadriláteros desde el principio
                    # ya que la estructura ha cambiado.
                    break 
            
            # Si en una pasada completa no encontramos ningún flip bueno, nos detenemos.
            if not found_free_diagonal_flip:
                print("\nNo se encontraron más 'diagonales libres'. El algoritmo voraz ha finalizado.")
                break
        
        print(f"\nTotal de flips realizados: {flips_realizados}")
        
        # --- Visualización ---
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(f'Algoritmo Voraz para "{FILE_NAME}"', fontsize=16)

        plot_triangulation(ax1, 'Triangulación Inicial', points, t_initial_edges)
        plot_triangulation(ax2, f'Resultado Tras {flips_realizados} Flips', points, t_current_edges)
        plot_triangulation(ax3, 'Triangulación Final (Objetivo)', points, list(t_final_edges))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()