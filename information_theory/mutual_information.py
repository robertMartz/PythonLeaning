import math
import matplotlib.pyplot as plt


# 1. Función para el Punto 5 (Dados pares) - Distribución Binomial
def calcular_entropia_pares(N):
    entropia = 0
    # Iteramos desde 0 hasta N dados pares
    for k in range(N + 1):
        # Fórmula de probabilidad binomial: C(N, k) * (0.5)^N
        prob = math.comb(N, k) / (2 ** N)
        if prob > 0:
            entropia -= prob * math.log2(prob)
    return entropia


# 2. Función para el Punto 4 (Suma de las caras) - Convolución iterativa
def calcular_entropia_suma(N):
    # p_suma guardará la distribución de probabilidades. Para N=0, la suma es 0 con prob 1.0
    p_suma = {0: 1.0}

    for _ in range(N):
        nueva_p = {}
        for valor_actual, prob_actual in p_suma.items():
            for cara in range(1, 7):
                nuevo_valor = valor_actual + cara
                # Multiplicamos por 1/6 que es la prob de cada cara
                nueva_p[nuevo_valor] = nueva_p.get(nuevo_valor, 0) + prob_actual * (1 / 6)
        p_suma = nueva_p

    entropia = 0
    for prob in p_suma.values():
        if prob > 0:
            entropia -= prob * math.log2(prob)
    return entropia


# 3. Generamos los datos para N desde 1 hasta 10
N_valores = list(range(1, 11))
I_suma = [calcular_entropia_suma(n) for n in N_valores]
I_pares = [calcular_entropia_pares(n) for n in N_valores]

# Imprimimos la tabla
print(f"{'N':<5} | {'I(X,Y) Suma':<15} | {'I(X,Y) Pares':<15}")
print("-" * 38)
for i in range(len(N_valores)):
    print(f"{N_valores[i]:<5} | {I_suma[i]:<15.4f} | {I_pares[i]:<15.4f}")

# 4. Graficamos los resultados
plt.figure(figsize=(9, 5))
plt.plot(N_valores, I_suma, marker='o', linestyle='-', linewidth=2, label='Suma de las caras (Pto 4)')
plt.plot(N_valores, I_pares, marker='s', linestyle='-', linewidth=2, color='orange', label='Dados pares (Pto 5)')

plt.title('Información Mutua vs Número de Dados', fontsize=14)
plt.xlabel('Número de dados ($N$)', fontsize=12)
plt.ylabel('Información Mutua $I(X,Y)$ [bits]', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Mostramos la gráfica
plt.show()