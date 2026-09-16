import math
from collections import Counter


def analizar_entropia(ruta_archivo, idioma, max_n=4, mayusculas=False, puntuaciones=False):
    # 1. Definir bloques base
    bases = {
        "es": "abcdefghijklmnñopqrstuvwxyzáéíóúü",
        "en": "abcdefghijklmnopqrstuvwxyz",
        "fr": "abcdefghijklmnopqrstuvwxyzàâçéèêëîïôûùüÿœæ"
    }

    puntuacion_base = " .,:;!¡?¿\"'-()"

    # 2. Construir el alfabeto dinámicamente
    alfabeto = bases[idioma] + " "  # Siempre incluimos el espacio

    if mayusculas:
        alfabeto += bases[idioma].upper()

    if puntuaciones:
        # Añadimos los signos, evitando duplicar el espacio
        alfabeto += puntuacion_base.replace(" ", "")

    alfabeto_set = set(alfabeto)

    # 3. Leer y normalizar el texto
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        texto = f.read()

    if not mayusculas:
        texto = texto.lower()

    # 4. Filtrado
    texto_limpio = "".join([c for c in texto if c in alfabeto_set])
    N_total = len(texto_limpio)

    """
    print(f"--- Análisis para {ruta_archivo} (Idioma: {idioma.upper()}) ---")
    print(f"Tamaño del alfabeto base: {len(alfabeto_set)} símbolos.")
    print(f"Mayúsculas: {mayusculas} | Puntuaciones: {puntuaciones}")
    print("")
    print("-" * 50) 
    """

    # Lista para guardar los resultados de entropía por carácter para cada 'n'
    resultados_entropia = []

    # 5. Cálculo para distintos n-gramas
    for n in range(1, max_n + 1):
        ngramas = [texto_limpio[i:i + n] for i in range(N_total - n + 1)]
        total_ngramas = len(ngramas)
        frecuencias = Counter(ngramas)

        entropia_bloque = 0
        for conteo in frecuencias.values():
            prob = conteo / len(ngramas)
            entropia_bloque -= prob * math.log2(prob)

        """        
        # Imprimimos resultados
        print(f"Tamaño de bloque (n) = {n}")
        print(f"  Tamaño teórico del abecedario: {len(alfabeto) ** n:,}")
        print(f"  Símbolos únicos realmente encontrados: {len(frecuencias):,}")
        print(f"  Entropía del bloque (H_n): {entropia_bloque:.4f} bits/bloque")
        print(f"  Entropía por carácter (H_n / n): {entropia_bloque / n:.4f} bits/símbolo")
        

        # Mostramos los 5 más comunes para ilustrar
        top_5 = [(k.replace(" ", "_"), f"Prob : {v/total_ngramas:.4f}" ) for k, v in frecuencias.most_common(5)]
        print(f"  Top 5 más comunes: {top_5}")
        print("-" * 50)
        """
        entropia_por_caracter = entropia_bloque / n
        resultados_entropia.append(entropia_por_caracter)

    return resultados_entropia


# 2. función integradora (Wrapper) para comparar los idiomas
def comparar_idiomas(max_n=4, mayusculas=False, puntuaciones=False):
    # Lista de archivos e idiomas correspondientes
    archivos = [
        ("./files/biblia_es.txt", "es"),
        ("./files/biblia_en.txt", "en"),
        ("./files/biblia_fr.txt", "fr")
    ]

    # Diccionario para almacenar todos los resultados
    datos_comparativos = {}

    print(f"=== COMPARACIÓN DE ENTROPÍA (bits/símbolo) ===")
    print(f"Configuración -> Mayúsculas: {mayusculas} | Puntuaciones: {puntuaciones}\n")

    # Recopilamos los datos usando la función principal
    for ruta, idioma in archivos:
        try:
            entropias = analizar_entropia(ruta, idioma, max_n, mayusculas, puntuaciones)
            datos_comparativos[idioma.upper()] = entropias
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {ruta}. Asegúrate de que esté en la carpeta.")
            return

    # Imprimimos la tabla comparativa
    # Encabezado de la tabla
    encabezado = f"{'Idioma':<10}" + "".join([f"| n={i:<9}" for i in range(1, max_n + 1)])
    print(encabezado)
    print("-" * len(encabezado))

    # Filas con los resultados de cada idioma
    for idioma, entropias in datos_comparativos.items():
        fila = f"{idioma:<10}"
        for h in entropias:
            fila += f"| {h:<10.4f}"
        print(fila)
    print("-" * len(encabezado))


# --- EJECUCIÓN ---
# versión simple (sin mayúsculas ni puntuaciones)
comparar_idiomas(max_n=4, mayusculas=True, puntuaciones=True)

# Ejemplo de uso:
# analizar_entropia("./files/biblia_en.txt", "en", max_n=4, mayusculas=False, puntuaciones=False)