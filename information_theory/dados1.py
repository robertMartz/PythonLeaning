import math
from collections import Counter
from itertools import product

# 1. Generamos los posibles resultados de los 3 dados (Espacio X)
caras = [1, 2, 3, 4, 5, 6]
resultados_X = list(product(caras, caras, caras))

# 2. Calculamos la suma para cada resultado (Variable Y)
resultados_Y = [sum(dado) for dado in resultados_X]

# 3. Contamos las frecuencias
frecuencias_Y = Counter(resultados_Y)
total_combinaciones = len(resultados_Y)

# 4. Calculamos Entropía H(Y) que equivale a I(X,Y)
entropia_Y = 0
for suma, conteo in frecuencias_Y.items():
    prob = conteo / total_combinaciones
    entropia_Y -= prob * math.log2(prob)

print("--- Distribución de Y (Suma: Frecuencia) ---")
for suma, conteo in sorted(frecuencias_Y.items()):
    print(f"Suma {suma}: {conteo} combinaciones")

print("-" * 44)
print(f"Información Mutua I(X,Y) = {entropia_Y:.4f} bits")