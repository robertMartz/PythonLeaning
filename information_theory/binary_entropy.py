import math
from collections import Counter
import os


def calcular_entropia_simbolos(simbolos):
    """Calcula la entropía de Shannon dada una lista de símbolos."""
    total_simbolos = len(simbolos)
    if total_simbolos == 0:
        return 0.0

    frecuencias = Counter(simbolos)
    entropia = 0.0

    for conteo in frecuencias.values():
        prob = conteo / total_simbolos
        entropia -= prob * math.log2(prob)

    return entropia


def analizar_archivo_binario(ruta_archivo):
    """Lee un archivo y devuelve su entropía para abecedarios de 1, 8 y 16 bits."""
    with open(ruta_archivo, "rb") as f:
        datos = f.read()

    if not datos:
        return None

    # 1. Abecedario de 8 bits (1 byte)
    # En Python, leer un archivo binario ya nos da una secuencia de enteros de 0 a 255.
    simbolos_8bits = list(datos)
    h_8bits = calcular_entropia_simbolos(simbolos_8bits)

    # 2. Abecedario de 1 bit
    # Convertimos cada byte en una cadena binaria de 8 caracteres
    simbolos_1bit = []
    for byte in datos:
        bits = bin(byte)[2:].zfill(8)
        simbolos_1bit.extend(list(bits))
    h_1bit = calcular_entropia_simbolos(simbolos_1bit)

    # 3. Abecedario de 16 bits (2 bytes)
    # Agrupamos los bytes de dos en dos.
    simbolos_16bits = [bytes(datos[i:i + 2]) for i in range(0, len(datos) - 1, 2)]
    h_16bits = calcular_entropia_simbolos(simbolos_16bits)

    # 4. Abecedario de 32 bits (4 bytes)
    # Agrupamos los bytes de 4 en 4.
    simbolos_32bits = [bytes(datos[i:i + 4]) for i in range(0, len(datos) - 1, 4)]
    h_32bits = calcular_entropia_simbolos(simbolos_32bits)

    return {
        "1_bit": h_1bit,
        "8_bits": h_8bits,
        "16_bits": h_16bits,
        "32_bits": h_32bits
    }


def comparar_formatos(archivos):
    """Genera una tabla comparativa para una lista de archivos."""
    print("=== COMPARACIÓN DE ENTROPÍA EN ARCHIVOS BINARIOS (bits/símbolo) ===")
    encabezado = f"{'Archivo':<15} | {'1 bit (Max:1)':<15} | {'8 bits (Max:8)':<15} | {'16 bits (Max:16)':<15} | {'32 bits (Max:32)':<15}"
    print(encabezado)
    print("-" * len(encabezado))

    for archivo in archivos:
        if os.path.exists(archivo):
            resultados = analizar_archivo_binario(archivo)
            fila = (f"{archivo[:17]:<15} | {resultados['1_bit']:<15.4f} | {resultados['8_bits']:<15.4f} "
                    f"| {resultados['16_bits']:<15.4f} | {resultados['32_bits']:<15.4f}")
            print(fila)
        else:
            print(f"{archivo[:14]:<15} | ERROR: Archivo no encontrado")

    print("-" * len(encabezado))


# --- EJECUCIÓN ---
# Para probar esto, asegúrate de tener un par de archivos en la misma carpeta.
# Por ejemplo, la misma imagen guardada como .bmp y como .jpg
lista_de_archivos = [
    "./files/imagenBMP.bmp",
    "./files/imagenJPG.jpg",
    "./files/audioWAV.wav",
    "./files/audioMP3.mp3"
]

comparar_formatos(lista_de_archivos)