class CajaRegistradora:
    def __init__(self, nombre_archivo):
        self.registros = []  # Aquí guardaremos todos los datos
        self.archivo_cargado = False

        try:
            with open(nombre_archivo, "r") as archivo:
                lineas = archivo.readlines()

                for linea in lineas[1:]:
                    datos = linea.strip().split(",")

                    # INTENTO 2: Convertir los datos a números
                    try:
                        dia = {
                            "fecha": datos[0],
                            "producto": datos[1],
                            "ventas": float(datos[2]),  # Aquí podría haber texto en vez de números
                            "gastos": float(datos[3]),
                            "metodo_pago": datos[4]
                        }
                        self.registros.append(dia)

                    # Si falla al convertir a float(), atrapamos el error aquí:
                    except ValueError:
                        print(
                            f"⚠️ Advertencia: Dato inválido en la línea -> '{linea.strip()}'. Esta línea será ignorada.")

            # Si terminamos el bloque sin que el archivo explote, cambiamos la bandera
            self.archivo_cargado = True
            print(f"✅ ¡Éxito! Archivo '{nombre_archivo}' cargado con {len(self.registros)} días válidos.")

            # Si el archivo no existe o está mal escrito, atrapamos el error aquí:
        except FileNotFoundError:
            print(f"❌ ERROR CRÍTICO: No se pudo encontrar el archivo '{nombre_archivo}'.")
            print("   Por favor, verifica que el nombre sea correcto y esté en la misma carpeta.")

            # Si ocurre CUALQUIER otro error raro que no previmos:
        except Exception as e:
            print(f"❌ Ocurrió un error inesperado: {e}")

    # 1. El Balance Real
    def calcular_ganancia_total(self):
        ganancia_total = 0
        for dia in self.registros:
            ganancia_total += (dia["ventas"] - dia["gastos"])
        return ganancia_total

    # 2. El Día de Mayor Rentabilidad
    def dia_mas_rentable(self):
        max_ganancia = 0
        fecha_max = ""

        for dia in self.registros:
            ganancia = dia["ventas"] - dia["gastos"]
            if ganancia > max_ganancia:
                max_ganancia = ganancia
                fecha_max = dia["fecha"]

        return fecha_max, max_ganancia

    # 3. Guerra de Métodos de Pago
    def resumen_metodos_pago(self):
        totales = {"Efectivo": 0, "Tarjeta": 0}
        for dia in self.registros:
            metodo = dia["metodo_pago"]
            totales[metodo] += dia["ventas"]

        return totales

    # 4. El Rey de la Barra
    def producto_estrella(self):
        conteo_productos = {}
        for dia in self.registros:
            producto = dia["producto"]
            if producto in conteo_productos:
                conteo_productos[producto] += 1
            else:
                conteo_productos[producto] = 1

        # Encontramos la llave (producto) con el valor más alto en el diccionario
        mejor_producto = max(conteo_productos, key=conteo_productos.get)
        return mejor_producto, conteo_productos[mejor_producto]

    # 5. Días Críticos
    def dias_criticos(self):
        dias_malos = []
        for dia in self.registros:
            if dia["gastos"] > 500.00 and dia["ventas"] < 1900.00:
                dias_malos.append(dia["fecha"])
        return dias_malos

    def generar_reporte(self, nombre_salida="reporte_final.txt"):
        # Verificamos si hay datos para procesar
        if not self.archivo_cargado or not self.registros:
            print("⚠️ No hay datos cargados para generar un reporte.")
            return

        try:
            with open(nombre_salida, "w") as reporte:
                # Escribimos un encabezado bonito
                reporte.write("========================================\n")
                reporte.write("      REPORTE DE CAJA REGISTRADORA      \n")
                reporte.write("========================================\n\n")

                # 1. Ganancia Total
                ganancia = self.calcular_ganancia_total()
                reporte.write(f"GANANCIA NETA TOTAL: ${ganancia:.2f} MXN\n")

                # 2. Día más rentable
                fecha, monto = self.dia_mas_rentable()
                reporte.write(f"DÍA MÁS RENTABLE: {fecha} (Ganancia: ${monto:.2f})\n\n")

                # 3. Métodos de Pago
                reporte.write("VENTAS POR MÉTODO DE PAGO:\n")
                pagos = self.resumen_metodos_pago()
                for metodo, total in pagos.items():
                    reporte.write(f" - {metodo}: ${total:.2f}\n")

                reporte.write("\n")

                # 4. Producto Estrella
                prod, cant = self.producto_estrella()
                reporte.write(f"PRODUCTO ESTRELLA: {prod} ({cant} días como top)\n")

                # 5. Días Críticos
                criticos = self.dias_criticos()
                reporte.write(f"DÍAS CRÍTICOS (Gastos > 500 y Ventas < 1900):\n")
                if criticos:
                    reporte.write(f" - {', '.join(criticos)}\n")
                else:
                    reporte.write(" - Ninguno registrado.\n")

                reporte.write("\n========================================\n")
                reporte.write("FIN DEL REPORTE\n")

            print(f"📄 Reporte generado con éxito: '{nombre_salida}'")

        except Exception as e:
            print(f"❌ Error al escribir el reporte: {e}")

# ==========================================
# CÓMO USAR LA CLASE EN LA PRÁCTICA
# ==========================================

print("--- REPORTE DE LA CAJA REGISTRADORA ---")

# 1. Creamos el objeto (esto ejecuta el __init__ y lee el archivo)
mi_caja = CajaRegistradora("sells_data.txt")

# 2. Llamamos a los métodos
# ganancia = mi_caja.calcular_ganancia_total()
print(f"1. Ganancia neta total: ${mi_caja.calcular_ganancia_total():.2f} MXN")

fecha_rentable, monto_rentable = mi_caja.dia_mas_rentable()
print(f"2. Día más rentable: {fecha_rentable} (Ganancia: ${monto_rentable:.2f})")

pagos = mi_caja.resumen_metodos_pago()
print(f"3. Ventas por método de pago: Efectivo ${pagos['Efectivo']:.2f} | Tarjeta ${pagos['Tarjeta']:.2f}")

product, veces = mi_caja.producto_estrella()
print(f"4. Producto estrella: {product} (vendido {veces} veces como top)")

dias_rojos = mi_caja.dias_criticos()
print(f"5. Fechas de días críticos: {', '.join(dias_rojos)}")