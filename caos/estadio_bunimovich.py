import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

class EstadioBunimovich:
    def __init__(self, L=1.0, R=1.0):
        """
        Inicializa el estadio con longitud del rectángulo 2L y radio R.
        """
        self.L = L
        self.R = R
        # El perímetro total, crucial para las coordenadas de Birkhoff
        self.perimetro = 4 * L + 2 * np.pi * R

        # Historial de estados: posiciones (x, y) y velocidades justo después del rebote
        self.historial_r = []
        self.historial_v = []
        self.birkhoff_s = []
        self.birkhoff_p = []

        # Variables para almacenar condiciones iniciales
        self.x0 = None
        self.y0 = None
        self.angulo0 = None

    def estado_inicial(self, x0, y0, angulo_inicial):
        """Establece y almacena la condición inicial de la partícula."""
        self.x0 = x0
        self.y0 = y0
        self.angulo0 = angulo_inicial

        self.r = np.array([x0, y0], dtype=float)
        self.v = np.array([np.cos(angulo_inicial), np.sin(angulo_inicial)], dtype=float)

        self.historial_r = [self.r.copy()]
        self.historial_v = [self.v.copy()]

    def _tiempo_interseccion(self, r, v):
        """Calcula el tiempo mínimo para el próximo impacto."""
        x, y = r
        vx, vy = v
        tiempos = []
        epsilon = 1e-9  # Tolerancia para evitar que detecte la colisión actual como la próxima

        # 1. Techo (y = R)
        if vy > 0:
            t = (self.R - y) / vy
            if t > epsilon and -self.L <= x + vx * t <= self.L:
                tiempos.append((t, 'techo'))

        # 2. Suelo (y = -R)
        if vy < 0:
            t = (-self.R - y) / vy
            if t > epsilon and -self.L <= x + vx * t <= self.L:
                tiempos.append((t, 'suelo'))

        # 3. Semicírculo Derecho ((x-L)^2 + y^2 = R^2)
        # Ecuación cuadrática: t^2 + 2*b*t + c = 0 (ya que vx^2 + vy^2 = 1)
        b = vx * (x - self.L) + vy * y
        c = (x - self.L) ** 2 + y ** 2 - self.R ** 2
        discriminante = b ** 2 - c
        if discriminante >= 0:
            t1 = -b + np.sqrt(discriminante)
            t2 = -b - np.sqrt(discriminante)
            for t in [t1, t2]:
                if t > epsilon and (x + vx * t) >= self.L:
                    tiempos.append((t, 'der'))

        # 4. Semicírculo Izquierdo ((x+L)^2 + y^2 = R^2)
        b_izq = vx * (x + self.L) + vy * y
        c_izq = (x + self.L) ** 2 + y ** 2 - self.R ** 2
        disc_izq = b_izq ** 2 - c_izq
        if disc_izq >= 0:
            t1 = -b_izq + np.sqrt(disc_izq)
            t2 = -b_izq - np.sqrt(disc_izq)
            for t in [t1, t2]:
                if t > epsilon and (x + vx * t) <= -self.L:
                    tiempos.append((t, 'izq'))

        # Retornar el evento con el tiempo mínimo positivo
        if not tiempos:
            raise ValueError("La partícula escapó del estadio (error numérico).")

        tiempos.sort(key=lambda item: item[0])
        return tiempos[0]

    def _vector_normal(self, r_impacto, frontera):
        """Calcula el vector normal unitario apuntando hacia ADENTRO del estadio."""
        x, y = r_impacto
        if frontera == 'techo':
            return np.array([0.0, -1.0])
        elif frontera == 'suelo':
            return np.array([0.0, 1.0])
        elif frontera == 'der':
            return np.array([-(x - self.L) / self.R, -y / self.R])
        elif frontera == 'izq':
            return np.array([-(x + self.L) / self.R, -y / self.R])

    def _calcular_birkhoff(self, r, v, n):
        """Mapea el estado actual a las coordenadas de fase (s, p)."""
        x, y = r

        # Parámetro s (longitud de arco medida en sentido antihorario desde (L, -R))
        if x > self.L:  # Semicírculo derecho
            angulo = np.arctan2(y, x - self.L)
            s = self.R * (angulo + np.pi / 2)
        elif y > 0 and -self.L <= x <= self.L:  # Techo
            s = np.pi * self.R + (self.L - x)
        elif x < -self.L:  # Semicírculo izquierdo
            angulo = np.arctan2(y, x + self.L)
            s = np.pi * self.R + 2 * self.L + self.R * (angulo - np.pi / 2)
        else:  # Suelo
            s = 2 * np.pi * self.R + 2 * self.L + (x + self.L)

        # Momento p = v · t (donde t es el vector tangente antihorario)
        # Si la normal interior es (nx, ny), el tangente es (ny, -nx)
        t = np.array([n[1], -n[0]])
        p = np.dot(v, t)

        return s, p

    def evolucionar(self, colisiones):
        """Itera la dinámica del billar."""
        for _ in range(colisiones):
            t_min, frontera = self._tiempo_interseccion(self.r, self.v)

            # Actualizar posición
            r_nuevo = self.r + self.v * t_min

            # Calcular normal y reflexionar velocidad
            n = self._vector_normal(r_nuevo, frontera)
            v_nueva = self.v - 2 * np.dot(self.v, n) * n

            # Normalizar para evitar acumulación de errores de punto flotante
            v_nueva = v_nueva / np.linalg.norm(v_nueva)

            # Calcular coordenadas de Birkhoff de este impacto
            s, p = self._calcular_birkhoff(r_nuevo, v_nueva, n)
            self.birkhoff_s.append(s)
            self.birkhoff_p.append(p)

            # Actualizar estado e historial
            self.r = r_nuevo
            self.v = v_nueva
            self.historial_r.append(self.r.copy())
            self.historial_v.append(self.v.copy())

        self.historial_r = np.array(self.historial_r)

    def graficar_estadio(self):
        """Dibuja la trayectoria en el espacio de coordenadas."""
        fig, ax = plt.subplots(figsize=(10, 5))

        # Dibujar fronteras
        rect = patches.Rectangle((-self.L, -self.R), 2 * self.L, 2 * self.R, fill=False, color='black', lw=2)
        arc_der = patches.Arc((self.L, 0), 2 * self.R, 2 * self.R, theta1=-90, theta2=90, color='black', lw=2)
        arc_izq = patches.Arc((-self.L, 0), 2 * self.R, 2 * self.R, theta1=90, theta2=270, color='black', lw=2)

        ax.add_patch(rect)
        ax.add_patch(arc_der)
        ax.add_patch(arc_izq)

        # Ocultar las líneas verticales del rectángulo dentro de los semicírculos
        ax.plot([-self.L, -self.L], [-self.R, self.R], color='white', lw=3)
        ax.plot([self.L, self.L], [-self.R, self.R], color='white', lw=3)

        # Dibujar trayectoria
        x_vals = self.historial_r[:, 0]
        y_vals = self.historial_r[:, 1]
        ax.plot(x_vals, y_vals, color='blue', alpha=0.6, lw=0.8)
        ax.scatter(x_vals[0], y_vals[0], color='red', zorder=5, label='Inicio')

        # Construcción del título y subtítulo
        subtitulo = f"C.I.: $x_0={self.x0}$, $y_0={self.y0}$, $\\theta_0={self.angulo0:.3f}$ rad"
        titulo_principal = f'Trayectoria en el Estadio de Bunimóvich ({len(x_vals) - 1} colisiones)'
        ax.set_title(f'{titulo_principal}\n{subtitulo}', fontsize=12)

        ax.set_aspect('equal')
        ax.axis('off')
        plt.legend()

        # Guardar archivo con formato de fecha
        fecha_actual = datetime.now().strftime("%Y-%m-%d")
        nombre_archivo = f"estadio_coordenadas_{fecha_actual}.png"
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada exitosamente como: {nombre_archivo}")

        plt.show()

    def graficar_espacio_fase(self):
        """Dibuja el Mapa de Poincaré en coordenadas de Birkhoff."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(self.birkhoff_s, self.birkhoff_p, s=1, color='purple', alpha=0.5)

        # Construcción del título y subtítulo
        subtitulo = f"C.I.: $x_0={self.x0}$, $y_0={self.y0}$, $\\theta_0={self.angulo0:.3f}$ rad"
        ax.set_title(f'Espacio de Fase (Coordenadas de Birkhoff)\n{subtitulo}', fontsize=12)

        ax.set_xlabel('Parámetro de arco perimetral $s$')
        ax.set_ylabel('Momento tangencial $p = \\cos(\\phi)$')
        ax.set_xlim(0, self.perimetro)
        ax.set_ylim(-1, 1)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Guardar archivo con formato de fecha
        fecha_actual = datetime.now().strftime("%Y-%m-%d")
        nombre_archivo = f"estadio_fase_{fecha_actual}.png"
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada exitosamente como: {nombre_archivo}")

        plt.show()


# --- Ejecución del Experimento ---
billar = EstadioBunimovich(L=1.0, R=1.0)
# Condición inicial: arranca en (0,0) con un ángulo irracional para evitar órbitas periódicas triviales
billar.estado_inicial(x0=0.0, y0=0.0, angulo_inicial=np.pi / 3)

# Simulamos 1000 colisiones
billar.evolucionar(colisiones=5000)

# Graficamos los resultados
billar.graficar_estadio()
billar.graficar_espacio_fase()