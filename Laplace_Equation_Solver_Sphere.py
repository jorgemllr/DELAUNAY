import numpy as np
import matplotlib.pyplot as plt

# --- PARÁMETROS DE LA SIMULACIÓN ---
L = 200  # Tamaño de la cuadrícula
dx = 1.0 / L  # Resolución espacial (1 metro dividido entre L celdas)
V = np.zeros((L, L))  # Potencial inicial
R_cells = int(0.1 / dx)  # Radio de la esfera en celdas (0.1 m)
cx, cy = L // 2, L // 2  # Centro de la esfera
V0 = 1000  # Potencial de la esfera

# --- ASIGNACIÓN DE POTENCIAL FIJO A LA ESFERA ---
for i in range(L):
    for j in range(L):
        if (i - cx)**2 + (j - cy)**2 <= R_cells**2:
            V[i, j] = V0

# --- MÉTODO DE RELAJACIÓN ---
for k in range(1500):
    for i in range(1, L - 1):
        for j in range(1, L - 1):
            if (i - cx)**2 + (j - cy)**2 > R_cells**2:
                V[i, j] = 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])

# --- CÁLCULO DEL CAMPO ELÉCTRICO ---
Ey, Ex = np.gradient(-V, dx)
E_mag = np.sqrt(Ex**2 + Ey**2)

# --- MEDIR EL CAMPO A DIFERENTES DISTANCIAS ---
distancias_m = np.array([0.12, 0.15, 0.18, 0.2, 0.25, 0.3])  # en metros
r_celdas = (distancias_m / dx).astype(int)

valores_simulados = []
valores_teoricos = []

for r in r_celdas:
    px, py = cx + r, cy  # Punto a la derecha del centro
    e_val = E_mag[py, px]
    valores_simulados.append(e_val)

    # Teoría (3D)
    E_teo = V0 * 0.1 / (distancias_m[list(r_celdas).index(r)] ** 2)
    valores_teoricos.append(E_teo)

# --- MOSTRAR TABLA DE RESULTADOS ---
print("Distancia (m) | E_sim (V/m) | E_teo (V/m) | Error (%)")
print("-" * 50)
for i in range(len(distancias_m)):
    error = abs(valores_simulados[i] - valores_teoricos[i]) / valores_teoricos[i] * 100
    print(f"{distancias_m[i]:>12.2f} | {valores_simulados[i]:>11.2f} | {valores_teoricos[i]:>11.2f} | {error:>8.2f}")

# --- GRAFICAR COMPARACIÓN ---
plt.figure(figsize=(7, 5))
plt.plot(distancias_m, valores_teoricos, 'k--', label='Teórico (3D)')
plt.plot(distancias_m, valores_simulados, 'ro-', label='Simulado (2D)')
plt.xlabel("Distancia desde el centro (m)")
plt.ylabel("Campo eléctrico |E| (V/m)")
plt.title("Comparación entre simulación 2D y teoría 3D")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- GRAFICAR EL CAMPO COMPLETO ---
plt.figure(figsize=(6, 6))
plt.contourf(V, levels=50, cmap='plasma')
plt.quiver(Ex, Ey, scale=1000, color='white', width=0.002)
plt.title("Campo eléctrico alrededor de una esfera 2D")
plt.xlabel("x (celdas)")
plt.ylabel("y (celdas)")
plt.axis('equal')
plt.colorbar(label='Potencial [V]')
plt.tight_layout()
plt.show()
