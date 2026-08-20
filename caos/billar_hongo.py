import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime


class BillarHongo:
    def __init__(self, R=1.0, L=0.5, D=1.0):
        """
        R: Radio del sombrero
        L: Ancho del tallo
        D: Profundidad del tallo
        """
        self.R = R
        self.L = L
        self.D = D
        # Perímetro: arco + 2 bases + 2 paredes tallo + fondo tallo
        self.perimetro = np.pi * R + 2 * (R - L / 2) + 2 * D + L

        # Almacenamiento de múltiples trayectorias
        self.lotes_r = []
        self.lotes_birkhoff = []
        self.lotes_condiciones = []

    def lanzar_particula(self, x0, y0, angulo0, colisiones=1000):
        """Inicializa y evoluciona una sola trayectoria."""
        r = np.array([x0, y0], dtype=float)
        v = np.array([np.cos(angulo0), np.sin(angulo0)], dtype=float)

        historial_r = [r.copy()]
        birkhoff_s = []
        birkhoff_p = []

        for _ in range(colisiones):
            t_min, frontera = self._tiempo_interseccion(r, v)
            r_nuevo = r + v * t_min
            n = self._vector_normal(r_nuevo, frontera)

            # Reflexión
            v_nueva = v - 2 * np.dot(v, n) * n
            v_nueva = v_nueva / np.linalg.norm(v_nueva)

            # Coordenadas de fase
            s, p = self._calcular_birkhoff(r_nuevo, v_nueva, n, frontera)
            birkhoff_s.append(s)
            birkhoff_p.append(p)

            r = r_nuevo
            v = v_nueva
            historial_r.append(r.copy())

        self.lotes_r.append(np.array(historial_r))
        self.lotes_birkhoff.append((birkhoff_s, birkhoff_p))
        self.lotes_condiciones.append((x0, y0, angulo0))

    def _tiempo_interseccion(self, r, v):
        x, y = r
        vx, vy = v
        tiempos = []
        epsilon = 1e-9

        # 1. Arco del sombrero (x^2 + y^2 = R^2)
        b = vx * x + vy * y
        c = x ** 2 + y ** 2 - self.R ** 2
        disc = b ** 2 - c
        if disc >= 0:
            for t in [-b + np.sqrt(disc), -b - np.sqrt(disc)]:
                if t > epsilon and (y + vy * t) >= 0:
                    tiempos.append((t, 'arco'))

        # 2 y 3. Bases del sombrero (y = 0)
        if vy != 0:
            t = (0 - y) / vy
            if t > epsilon:
                x_imp = x + vx * t
                if -self.R <= x_imp <= -self.L / 2:
                    tiempos.append((t, 'base_izq'))
                elif self.L / 2 <= x_imp <= self.R:
                    tiempos.append((t, 'base_der'))

        # 4. Pared izquierda del tallo (x = -L/2)
        if vx != 0:
            t = (-self.L / 2 - x) / vx
            if t > epsilon and -self.D <= y + vy * t <= 0:
                tiempos.append((t, 'pared_izq'))

        # 5. Pared derecha del tallo (x = L/2)
        if vx != 0:
            t = (self.L / 2 - x) / vx
            if t > epsilon and -self.D <= y + vy * t <= 0:
                tiempos.append((t, 'pared_der'))

        # 6. Fondo del tallo (y = -D)
        if vy != 0:
            t = (-self.D - y) / vy
            if t > epsilon and -self.L / 2 <= x + vx * t <= self.L / 2:
                tiempos.append((t, 'fondo'))

        if not tiempos:
            raise ValueError(f"Fuga detectada en r=({x},{y}), v=({vx},{vy})")

        tiempos.sort(key=lambda item: item[0])
        return tiempos[0]

    def _vector_normal(self, r, frontera):
        x, y = r
        if frontera == 'arco':
            return np.array([-x / self.R, -y / self.R])
        elif frontera in ['base_izq', 'base_der', 'fondo']:
            return np.array([0.0, 1.0])
        elif frontera == 'pared_izq':
            return np.array([1.0, 0.0])
        elif frontera == 'pared_der':
            return np.array([-1.0, 0.0])

    def _calcular_birkhoff(self, r, v, n, frontera):
        x, y = r

        # Parametrización antihoraria del perímetro s
        if frontera == 'arco':
            angulo = np.arctan2(y, x)
            s = self.R * (np.pi - angulo)  # Empieza en x=R (angulo 0) y crece hacia x=-R (angulo pi)
        elif frontera == 'base_izq':
            s = np.pi * self.R + (x - (-self.R))
        elif frontera == 'pared_izq':
            s = np.pi * self.R + (self.R - self.L / 2) + (0 - y)
        elif frontera == 'fondo':
            s = np.pi * self.R + (self.R - self.L / 2) + self.D + (x - (-self.L / 2))
        elif frontera == 'pared_der':
            s = np.pi * self.R + (self.R - self.L / 2) + self.D + self.L + (y - (-self.D))
        elif frontera == 'base_der':
            s = np.pi * self.R + (self.R - self.L / 2) + 2 * self.D + self.L + (x - self.L / 2)

        # Momento p = v · t (vector tangente antihorario)
        t_vec = np.array([n[1], -n[0]])
        p = np.dot(v, t_vec)

        return s, p

    def graficar_estadio(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Fronteras geométricas
        arc = patches.Arc((0, 0), 2 * self.R, 2 * self.R, theta1=0, theta2=180, color='black', lw=2)
        ax.add_patch(arc)
        ax.plot([-self.R, -self.L / 2], [0, 0], 'k-', lw=2)
        ax.plot([self.L / 2, self.R], [0, 0], 'k-', lw=2)
        ax.plot([-self.L / 2, -self.L / 2], [0, -self.D], 'k-', lw=2)
        ax.plot([self.L / 2, self.L / 2], [0, -self.D], 'k-', lw=2)
        ax.plot([-self.L / 2, self.L / 2], [-self.D, -self.D], 'k-', lw=2)

        # Apertura (línea punteada opcional para guiar la vista)
        ax.plot([-self.L / 2, self.L / 2], [0, 0], 'k:', alpha=0.3)

        colores = plt.cm.tab10(np.linspace(0, 1, len(self.lotes_r)))

        for idx, historial in enumerate(self.lotes_r):
            x_vals = historial[:, 0]
            y_vals = historial[:, 1]
            ax.plot(x_vals, y_vals, color=colores[idx], alpha=0.5, lw=0.6)
            ax.scatter(x_vals[0], y_vals[0], color=colores[idx], s=20, zorder=5)

        ax.set_aspect('equal')
        ax.set_title(f'Trayectorias en el Billar de Hongo\n(Se simularon {len(self.lotes_r)} C.I. distintas)',
                     fontsize=12)
        ax.axis('off')

        nombre_archivo = f"hongo_coordenadas_{datetime.now().strftime('%Y-%m-%d_%H%M')}.png"
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada: {nombre_archivo}")
        plt.show()

    def graficar_espacio_fase(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        colores = plt.cm.tab10(np.linspace(0, 1, len(self.lotes_birkhoff)))

        for idx, (birkhoff_s, birkhoff_p) in enumerate(self.lotes_birkhoff):
            ax.scatter(birkhoff_s, birkhoff_p, s=0.5, color=colores[idx], alpha=0.7)

        ax.set_title(
            'Espacio de Fase Mixto (Coordenadas de Birkhoff)\nObserva las "Islas de Estabilidad" y el "Mar Caótico"',
            fontsize=12)
        ax.set_xlabel('Parámetro de arco perimetral $s$')
        ax.set_ylabel('Momento tangencial $p = \\cos(\\phi)$')
        ax.set_xlim(0, self.perimetro)
        ax.set_ylim(-1, 1)
        ax.grid(True, linestyle='--', alpha=0.4)

        # Líneas verticales divisorias sutiles para identificar las paredes
        fronteras_s = [
            np.pi * self.R,
            np.pi * self.R + (self.R - self.L / 2),
            np.pi * self.R + (self.R - self.L / 2) + self.D,
            np.pi * self.R + (self.R - self.L / 2) + self.D + self.L,
            np.pi * self.R + (self.R - self.L / 2) + 2 * self.D + self.L
        ]
        for fs in fronteras_s:
            ax.axvline(fs, color='gray', linestyle=':', alpha=0.5)

        nombre_archivo = f"hongo_fase_{datetime.now().strftime('%Y-%m-%d_%H%M')}.png"
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada: {nombre_archivo}")
        plt.show()


# --- Ejecución del Experimento ---
hongo = BillarHongo(R=1.0, L=0.5, D=1.0)

# 1. Trayectoria Caótica (Comienza en el tallo, visitará todo)
hongo.lanzar_particula(x0=0.0, y0=-0.5, angulo0= np.pi / 3.7, colisiones=2000)

# 2. Trayectoria Regular 1 (Atrapada en la izquierda del sombrero)
hongo.lanzar_particula(x0=-0.7, y0=0.3, angulo0=np.pi / 2.1, colisiones=1000)

# 3. Trayectoria Regular 2 (Atrapada en la derecha, ángulo rasante)
hongo.lanzar_particula(x0=0.8, y0=0.2, angulo0=np.pi / 1.2, colisiones=1000)

hongo.graficar_estadio()
hongo.graficar_espacio_fase()