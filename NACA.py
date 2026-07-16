import numpy as np

def naca4(m, p, t, c=1.0, n=100):
    x = np.linspace(0, c, n)
    yt = 5 * t * (0.2969*np.sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)**2 + 0.2843*(x/c)**3 - 0.1015*(x/c)**4)

    yc = np.where(x < p*c,
                  m/p**2 * (2*p*(x/c) - (x/c)**2),
                  m/(1-p)**2 * ((1 - 2*p) + 2*p*(x/c) - (x/c)**2))

    dyc_dx = np.where(x < p*c,
                      2*m/p**2 * (p - x/c),
                      2*m/(1 - p)**2 * (p - x/c))
    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_all = np.append(xu, xl[::-1])
    y_all = np.append(yu, yl[::-1])

    with open("naca2412_points.csv", "w") as f:
        f.write("x,y\n")
        for x_val, y_val in zip(x_all, y_all):
            f.write(f"{x_val:.5f},{y_val:.5f}\n")

    print("Archivo 'naca2412_points.csv' generado correctamente.")

# NACA 2412: m = 0.02, p = 0.4, t = 0.12
naca4(m=0.02, p=0.4, t=0.12)
