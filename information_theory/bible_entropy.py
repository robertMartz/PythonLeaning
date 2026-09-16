import math
from collections import Counter

def calcular_entropia_espanol(ruta_archivo):
    # 1. Definición del abecedario

    # Incluimos letras, la ñ, vocales con tilde, la diéresis y el espacio.
    alfabeto_es = set("abcdefghijklmnñopqrstuvwxyzáéíóúü ")

    # 2. Lectura del archivo

    # Usamos utf-8 para que lea correctamente los acentos y la ñ
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        texto = f.read()

    # 3. Normalización y filtrado

    texto_minusculas = texto.lower()
    # Generamos una lista solo con los caracteres que pertenecen a nuestro alfabeto
    texto_limpio = [c for c in texto_minusculas if c in alfabeto_es]

    # Total de caracteres válidos que quedaron en el texto
    N = len(texto_limpio)

    # 4. Conteo de frecuencias

    # Counter crea un diccionario donde la llave es el carácter y el valor es cuántas veces aparece
    frecuencias = Counter(texto_limpio)

    # 5. Cálculo de probabilidades y Entropía de Shannon

    entropia = 0
    for caracter, conteo in frecuencias.items():
        probabilidad = conteo / N
        entropia -= probabilidad * math.log2(probabilidad)

    # Resultados
    print("--- Resultados para Español ---")
    print(f"Tamaño del alfabeto definido: {len(alfabeto_es)} símbolos")
    print(f"Caracteres válidos procesados: {N}")
    print(f"Entropía de Shannon: {entropia:.4f} bits/símbolo\n")

    # Opcional: imprimir las 5 letras más comunes para verificar que el espacio sea la más frecuente
    print("Top 5 símbolos más frecuentes:")
    for char, freq in frecuencias.most_common(5):
        simbolo_impreso = "' '" if char == " " else char
        print(f"  {simbolo_impreso}: {freq / N:.4f} (Probabilidad)")

def analizar_entropia_ngramas(ruta_archivo, max_n):
    # 1. Definir alfabeto

    # Alfabeto corto vs alfabeto largo:

    # alfabeto_es = set("abcdefghijklmnñopqrstuvwxyzáéíóúü ")
    alfabeto_es = set("abcdefghijklmnñopqrstuvwxyzáéíóúüABCDEFGHIJKLMNÑOPQRSTUVWXYZÁÉÍÓÚÜ .,:;!¡?¿\"'-()")

    # 2. Leer y limpiar el texto
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        # texto = f.read().lower()
        texto = f.read()

    # Unimos todo en un solo string limpio
    texto_limpio = "".join([c for c in texto if c in alfabeto_es])
    N_total = len(texto_limpio)

    print(f"Total de caracteres en el texto limpio: {N_total}\n")
    print("-" * 50)

    # 3. Iterar para distintos tamaños de bloque (n)
    for n in range(1, max_n + 1):
        # Generamos los n-gramas con una ventana deslizante
        # Ejemplo: para "hola" con n=2 -> "ho", "ol", "la"
        ngramas = [texto_limpio[i:i + n] for i in range(N_total - n + 1)]
        total_ngramas = len(ngramas)

        # Contamos frecuencias
        frecuencias = Counter(ngramas)

        # Calculamos la entropía del bloque
        entropia_bloque = 0
        for ngrama, conteo in frecuencias.items():
            prob = conteo / total_ngramas
            entropia_bloque -= prob * math.log2(prob)

        # Normalizamos para tener bits por carácter y poder comparar
        entropia_por_caracter = entropia_bloque / n

        # Imprimimos resultados
        print(f"Tamaño de bloque (n) = {n}")
        print(f"  Tamaño teórico del abecedario: {len(alfabeto_es) ** n:,}")
        print(f"  Símbolos únicos realmente encontrados: {len(frecuencias):,}")
        print(f"  Entropía del bloque (H_n): {entropia_bloque:.4f} bits/bloque")
        print(f"  Entropía por carácter (H_n / n): {entropia_por_caracter:.4f} bits/símbolo")

        # Mostramos los 5 más comunes para ilustrar
        top_5 = [(k.replace(" ", "_"), f"Prob : {v/total_ngramas:.4f}" ) for k, v in frecuencias.most_common(5)]
        print(f"  Top 5 más comunes: {top_5}")
        print("-" * 50)


# Ejecutamos la función
# calcular_entropia_espanol("./files/biblia_es.txt")

# analizar_entropia_ngramas("./files/biblia_es.txt", 1) # Unigramas
# analizar_entropia_ngramas("./files/biblia_es.txt", 2) # Bigramas
# analizar_entropia_ngramas("./files/biblia_es.txt", 3) # Trigramas
analizar_entropia_ngramas("./files/biblia_es.txt", 4)


