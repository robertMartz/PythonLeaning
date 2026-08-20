import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.colors import Normalize


class CuantizacionEstadio:
    def __init__(self, L=1.0, R=1.0, h=0.04):
        """
        L: Mitad de la longitud de la región rectangular
        R: Radio de los semicírculos
        h: Paso de la malla (resolución espacial)
        """
        self.L = L
        self.R = R
        self.h = h

        # Generar la malla que cubre la caja exterior del estadio
        x_min, x_max = -L - R, L + R
        y_min, y_max = -R, R

        # Malla 1D
        self.x = np.arange(x_min, x_max + h, h)
        self.y = np.arange(y_min, y_max + h, h)

        # Malla 2D
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Ny, self.Nx = self.X.shape

        self.mascara = None
        self.indices_puntos = None
        self.puntos_interiores = 0

    def construir_geometria(self):
        """Identifica qué puntos de la malla están dentro del estadio."""
        # 1. Región rectangular
        mascara_rect = (np.abs(self.X) <= self.L) & (np.abs(self.Y) <= self.R)

        # 2. Semicírculo derecho
        mascara_der = ((self.X - self.L) ** 2 + self.Y ** 2 <= self.R ** 2) & (self.X > self.L)

        # 3. Semicírculo izquierdo
        mascara_izq = ((self.X + self.L) ** 2 + self.Y ** 2 <= self.R ** 2) & (self.X < -self.L)

        # Máscara total del estadio
        self.mascara = mascara_rect | mascara_der | mascara_izq

        # Asignar un índice único (0, 1, 2...) a cada punto interior
        self.indices_puntos = np.full((self.Ny, self.Nx), -1, dtype=int)
        interiores_y, interiores_x = np.where(self.mascara)
        self.puntos_interiores = len(interiores_y)

        # Mapeamos las coordenadas matriciales 2D a un vector 1D
        self.indices_puntos[interiores_y, interiores_x] = np.arange(self.puntos_interiores)

        print(f"Resolución de malla: {self.Nx}x{self.Ny}. Puntos efectivos en matriz: {self.puntos_interiores}")

    def ensamblar_laplaciano(self):
        """Construye la matriz dispersa del Hamiltoniano (Laplaciano negativo)."""
        row = []
        col = []
        data = []

        interiores_y, interiores_x = np.where(self.mascara)

        for idx in range(self.puntos_interiores):
            i = interiores_y[idx]
            j = interiores_x[idx]

            # Término diagonal: -4 * (-1) = 4
            row.append(idx)
            col.append(idx)
            data.append(4.0)

            # Vecinos: Arriba, Abajo, Izquierda, Derecha
            vecinos = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

            for ni, nj in vecinos:
                # Comprobar que el vecino está dentro de la caja matriz
                if 0 <= ni < self.Ny and 0 <= nj < self.Nx:
                    indice_vecino = self.indices_puntos[ni, nj]
                    # Si el vecino está dentro del estadio, aplicar acoplamiento (-1)
                    if indice_vecino != -1:
                        row.append(idx)
                        col.append(indice_vecino)
                        data.append(-1.0)

        # Crear la matriz dispersa y dividir por h^2 según la fórmula de diferencias finitas
        H = sp.coo_matrix((data, (row, col)), shape=(self.puntos_interiores, self.puntos_interiores))
        self.H_csr = H.tocsr() / (self.h ** 2)

    def resolver_modos(self, num_modos=6):
        """Encuentra los valores y vectores propios de menor energía."""
        print("Diagonalizando la matriz dispersa... esto puede tardar un momento.")
        # Usamos eigsh porque la matriz es simétrica. 'SM' busca los autovalores de Menor Magnitud.
        valores_k2, autovectores = spla.eigsh(self.H_csr, k=num_modos, which='SM')
        return valores_k2, autovectores

    def graficar_modos(self, valores_k2, autovectores):
        """Reconstruye y grafica las funciones de onda 2D."""
        num_modos = len(valores_k2)
        fig, axes = plt.subplots(int(np.ceil(num_modos / 2)), 2, figsize=(12, 3 * np.ceil(num_modos / 2)))
        axes = axes.flatten()

        interiores_y, interiores_x = np.where(self.mascara)

        for n in range(num_modos):
            # Crear matriz 2D vacía (llena de ceros, satisfaciendo Dirichlet)
            psi_2d = np.zeros((self.Ny, self.Nx))

            # Llenar los puntos interiores con los valores del autovector n
            psi_2d[interiores_y, interiores_x] = autovectores[:, n]

            # Graficar
            ax = axes[n]
            # Normalizar el color centrado en 0
            norm = Normalize(vmin=-np.max(np.abs(psi_2d)), vmax=np.max(np.abs(psi_2d)))
            cp = ax.contourf(self.X, self.Y, psi_2d, levels=50, cmap='RdBu_r', norm=norm)

            ax.set_title(f"Modo {n + 1} | Energía $\\propto$ $k^2$ = {valores_k2[n]:.2f}")
            ax.set_aspect('equal')
            ax.axis('off')

        plt.tight_layout()
        plt.show()


# --- Ejecución del Experimento ---
# Con un paso h=0.04 obtenemos una malla de ~6000 puntos internos, lo cual es rápido.
estadio_quantico = CuantizacionEstadio(L=1.0, R=1.0, h=0.04)
estadio_quantico.construir_geometria()
estadio_quantico.ensamblar_laplaciano()

# Calcular los primeros 6 estados de energía
energia, eigenestados = estadio_quantico.resolver_modos(num_modos=6)

# Visualizar
estadio_quantico.graficar_modos(energia, eigenestados)