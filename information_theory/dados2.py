import math
from collections import Counter
from itertools import product

# 1. Generamos el espacio X
caras = [1, 2, 3, 4, 5, 6]
resultados_X = list(product(caras, caras, caras))

# 2. Calculamos Y: número de dados pares en cada tirada
resultados_Y = [sum(1 for c in dado if c % 2 == 0) for dado in resultados_X]

# 3. Contamos las frecuencias
frecuencias_Y = Counter(resultados_Y)
total = len(resultados_Y)

# 4. Calculamos la Entropía H(Y)
entropia_Y = 0
for pares, conteo in frecuencias_Y.items():
    prob = conteo / total
    entropia_Y -= prob * math.log2(prob)

print("--- Distribución de Y (Dados Pares: Frecuencia) ---")
for pares, conteo in sorted(frecuencias_Y.items()):
    print(f"{pares} dados pares: {conteo} combinaciones (Prob: {conteo/total:.3f})")

print("-" * 48)
print(f"Información Mutua I(X,Y) = {entropia_Y:.4f} bits")