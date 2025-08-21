import pandas as pd
import re

def limpiar_csv(archivo_entrada, archivo_salida=None):
    """
    Lee un archivo CSV y elimina las filas donde las columnas 'catalan' o 'chino'
    están vacías o contienen solo números.
    
    Args:
        archivo_entrada (str): Ruta del archivo CSV de entrada
        archivo_salida (str): Ruta del archivo CSV de salida (opcional)
    
    Returns:
        pd.DataFrame: DataFrame limpio
    """
    
    # Leer el archivo CSV
    try:
        df = pd.read_csv(archivo_entrada)
        print(f"Archivo leído correctamente. Total de filas iniciales: {len(df)}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo_entrada}")
        return None
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None
    
    # Verificar que las columnas existan
    columnas_requeridas = ['catalan', 'chino']
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    
    if columnas_faltantes:
        print(f"Error: Las siguientes columnas no existen en el CSV: {columnas_faltantes}")
        print(f"Columnas disponibles: {list(df.columns)}")
        return None
    
    # Función para verificar si un valor es solo números
    def es_solo_numeros(valor):
        if pd.isna(valor):
            return True
        valor_str = str(valor).strip()
        if valor_str == '':
            return True
        # Verificar si contiene solo dígitos, espacios, puntos o comas
        return bool(re.match(r'^[\d\s.,]+$', valor_str))
    
    # Crear máscaras para identificar filas a eliminar
    mascara_catalan = df['catalan'].apply(es_solo_numeros)
    mascara_chino = df['chino'].apply(es_solo_numeros)
    
    # Combinar las máscaras (eliminar si cualquiera de las dos columnas cumple la condición)
    filas_a_eliminar = mascara_catalan | mascara_chino
    
    # Mostrar información sobre las filas que se eliminarán
    num_filas_eliminar = filas_a_eliminar.sum()
    print(f"\nFilas a eliminar: {num_filas_eliminar}")
    
    if num_filas_eliminar > 0:
        print("\nEjemplos de filas que se eliminarán:")
        ejemplos = df[filas_a_eliminar].head(5)[['catalan', 'chino']]
        print(ejemplos)
    
    # Filtrar el DataFrame para mantener solo las filas válidas
    df_limpio = df[~filas_a_eliminar].copy()
    
    print(f"\nTotal de filas después de la limpieza: {len(df_limpio)}")
    print(f"Filas eliminadas: {len(df) - len(df_limpio)}")
    
    # Guardar el resultado si se especifica archivo de salida
    if archivo_salida:
        try:
            df_limpio.to_csv(archivo_salida, index=False)
            print(f"\nArchivo guardado exitosamente en: {archivo_salida}")
        except Exception as e:
            print(f"Error al guardar el archivo: {e}")
    
    return df_limpio


# Uso del script
if __name__ == "__main__":
    # Configurar las rutas de los archivos
    archivo_entrada = "dataset_origen.csv"  # Cambia esto por la ruta de tu archivo
    archivo_salida = "dataset_limpio.csv"  # Archivo donde se guardará el resultado
    
    # Ejecutar la limpieza
    df_resultado = limpiar_csv(archivo_entrada, archivo_salida)
    
    # Opcional: mostrar las primeras filas del resultado
    if df_resultado is not None:
        print("\nPrimeras 5 filas del archivo limpio:")
        print(df_resultado[['catalan', 'chino']].head())