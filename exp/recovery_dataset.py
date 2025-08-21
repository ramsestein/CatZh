import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path
import glob

def recover_clean_samples(base_name="muestras_traducciones_10000", results_dir="p1_results"):
    """
    Recupera todas las muestras limpias desde los archivos de backup CSV
    y genera un nuevo archivo Excel consolidado.
    """
    
    print("🔍 INICIANDO RECUPERACIÓN DE MUESTRAS LIMPIAS")
    print(f"📂 Buscando archivos en: {results_dir}/")
    print("="*60)
    
    # Buscar todos los archivos de backup CSV
    pattern = os.path.join(results_dir, f"{base_name}_backup_*.csv")
    backup_files = glob.glob(pattern)
    
    if not backup_files:
        print(f"❌ No se encontraron archivos de backup con patrón: {pattern}")
        return None
    
    print(f"📁 Archivos de backup encontrados: {len(backup_files)}")
    
    # Ordenar por timestamp para procesar en orden cronológico
    backup_files.sort()
    
    # Leer y combinar todos los backups
    all_samples = []
    files_processed = 0
    total_rows = 0
    
    for i, backup_file in enumerate(backup_files, 1):
        try:
            # Extraer timestamp del nombre del archivo
            filename = os.path.basename(backup_file)
            timestamp_part = filename.split('_backup_')[1].replace('.csv', '')
            
            # Leer el CSV
            df = pd.read_csv(backup_file, encoding='utf-8')
            rows_in_file = len(df)
            total_rows += rows_in_file
            
            # Añadir información de origen
            df['archivo_origen'] = filename
            df['timestamp_origen'] = timestamp_part
            
            all_samples.append(df)
            files_processed += 1
            
            print(f"  ✅ [{i}/{len(backup_files)}] {filename} - {rows_in_file} registros")
            
        except Exception as e:
            print(f"  ❌ Error leyendo {backup_file}: {e}")
            continue
    
    if not all_samples:
        print("❌ No se pudieron leer archivos de backup")
        return None
    
    print(f"\n📊 Consolidando datos...")
    
    # Combinar todos los DataFrames
    combined_df = pd.concat(all_samples, ignore_index=True)
    print(f"  - Total registros combinados: {len(combined_df)}")
    
    # Eliminar duplicados basándose en 'linea' (mantener el más reciente)
    # Ordenar por timestamp para mantener el más reciente
    combined_df = combined_df.sort_values('timestamp_origen')
    clean_df = combined_df.drop_duplicates(subset=['linea'], keep='last')
    print(f"  - Registros únicos después de eliminar duplicados: {len(clean_df)}")
    
    # Preparar DataFrame final (solo columnas esenciales)
    final_df = clean_df[['linea', 'catalan', 'chino']].copy()
    final_df = final_df.sort_values('linea').reset_index(drop=True)
    
    # Generar nuevo archivo Excel
    output_excel = f"{base_name}_muestras_limpias_RECUPERADO.xlsx"
    
    print(f"\n📝 Generando nuevo archivo Excel...")
    
    try:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            # Hoja principal con muestras limpias
            final_df.to_excel(writer, sheet_name='Muestras_Limpias', index=False)
            
            # Hoja de información de recuperación
            recovery_info = pd.DataFrame({
                'Parametro': [
                    'Fecha_Recuperacion',
                    'Archivos_Backup_Procesados',
                    'Total_Registros_Encontrados',
                    'Total_Registros_Unicos',
                    'Registros_Duplicados_Eliminados',
                    'Primer_Archivo_Backup',
                    'Ultimo_Archivo_Backup'
                ],
                'Valor': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    files_processed,
                    total_rows,
                    len(final_df),
                    total_rows - len(final_df),
                    os.path.basename(backup_files[0]) if backup_files else 'N/A',
                    os.path.basename(backup_files[-1]) if backup_files else 'N/A'
                ]
            })
            recovery_info.to_excel(writer, sheet_name='Info_Recuperacion', index=False)
            
            # Hoja con estadísticas por archivo de origen
            origin_stats = combined_df.groupby('archivo_origen').agg({
                'linea': 'count',
                'timestamp_origen': 'first'
            }).rename(columns={'linea': 'registros'}).reset_index()
            origin_stats = origin_stats.sort_values('timestamp_origen')
            origin_stats.to_excel(writer, sheet_name='Estadisticas_Por_Archivo', index=False)
            
            # Hoja con análisis de calidad (opcional)
            print("  📊 Analizando calidad de datos recuperados...")
            quality_analysis = pd.DataFrame({
                'Metrica': [
                    'Total_Registros_Recuperados',
                    'Registros_Con_Catalan_Vacio',
                    'Registros_Con_Chino_Vacio',
                    'Longitud_Promedio_Catalan',
                    'Longitud_Promedio_Chino',
                    'Registros_Con_Texto_Muy_Corto',
                    'Registros_Con_Texto_Muy_Largo'
                ],
                'Valor': [
                    len(final_df),
                    final_df['catalan'].isna().sum() + (final_df['catalan'].str.len() == 0).sum(),
                    final_df['chino'].isna().sum() + (final_df['chino'].str.len() == 0).sum(),
                    final_df['catalan'].str.len().mean(),
                    final_df['chino'].str.len().mean(),
                    ((final_df['catalan'].str.len() < 5) | (final_df['chino'].str.len() < 2)).sum(),
                    ((final_df['catalan'].str.len() > 500) | (final_df['chino'].str.len() > 300)).sum()
                ]
            })
            quality_analysis.to_excel(writer, sheet_name='Analisis_Calidad', index=False)
        
        print(f"\n✅ RECUPERACIÓN COMPLETADA EXITOSAMENTE")
        print(f"📊 Archivo Excel generado: {output_excel}")
        print(f"📈 Total muestras recuperadas: {len(final_df)}")
        
        # Intentar leer estadísticas consolidadas para más información
        stats_file = os.path.join(results_dir, f"{base_name}_estadisticas_consolidadas.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            print(f"\n📊 ESTADÍSTICAS HISTÓRICAS:")
            print(f"  - Total ejecuciones detectadas: {stats.get('total_ejecuciones', 'N/A')}")
            print(f"  - Primera ejecución: {stats.get('primera_ejecucion', 'N/A')}")
            print(f"  - Última ejecución: {stats.get('ultima_ejecucion', 'N/A')}")
            print(f"  - Total registros procesados históricamente: {stats.get('total_registros_procesados', 'N/A')}")
        
        return output_excel
        
    except Exception as e:
        print(f"❌ Error generando Excel: {e}")
        
        # Fallback a CSV si falla Excel
        output_csv = f"{base_name}_muestras_limpias_RECUPERADO.csv"
        final_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"💾 Guardado como CSV alternativo: {output_csv}")
        return output_csv


def analyze_recovery_potential(base_name="muestras_traducciones_100", results_dir="p1_results"):
    """
    Analiza el potencial de recuperación antes de ejecutar el proceso completo.
    """
    print("🔍 ANÁLISIS DE POTENCIAL DE RECUPERACIÓN")
    print("="*60)
    
    if not os.path.exists(results_dir):
        print(f"❌ No existe la carpeta: {results_dir}")
        return
    
    # Buscar diferentes tipos de archivos
    patterns = {
        'backups': f"{base_name}_backup_*.csv",
        'metricas': f"{base_name}_metricas_*.txt",
        'correcciones': f"{base_name}_log_correcciones_*.csv",
        'resumenes': f"{base_name}_resumen_*.json",
        'analisis': f"{base_name}_analisis_eliminacion_*.xlsx"
    }
    
    total_files = 0
    for file_type, pattern in patterns.items():
        files = glob.glob(os.path.join(results_dir, pattern))
        count = len(files)
        total_files += count
        print(f"  📁 {file_type}: {count} archivos")
    
    print(f"\n📊 Total archivos de respaldo: {total_files}")
    
    # Estimar tamaño de datos
    backup_pattern = os.path.join(results_dir, f"{base_name}_backup_*.csv")
    backup_files = glob.glob(backup_pattern)
    
    if backup_files:
        total_size = sum(os.path.getsize(f) for f in backup_files)
        print(f"💾 Tamaño total de backups: {total_size / (1024*1024):.1f} MB")
        
        # Muestra algunos archivos como ejemplo
        print(f"\n📋 Primeros 5 archivos de backup:")
        for f in sorted(backup_files)[:5]:
            size = os.path.getsize(f) / 1024
            print(f"  - {os.path.basename(f)} ({size:.1f} KB)")
            
        if len(backup_files) > 5:
            print(f"  ... y {len(backup_files) - 5} archivos más")


def verify_recovered_file(excel_file):
    """
    Verifica la integridad del archivo recuperado.
    """
    print(f"\n🔍 VERIFICANDO ARCHIVO RECUPERADO: {excel_file}")
    print("="*60)
    
    try:
        # Leer todas las hojas
        xl_file = pd.ExcelFile(excel_file)
        print(f"📋 Hojas encontradas: {', '.join(xl_file.sheet_names)}")
        
        # Verificar hoja principal
        df = pd.read_excel(excel_file, sheet_name='Muestras_Limpias')
        print(f"\n📊 Estadísticas de muestras recuperadas:")
        print(f"  - Total registros: {len(df)}")
        print(f"  - Columnas: {', '.join(df.columns)}")
        print(f"  - Rango de líneas: {df['linea'].min()} - {df['linea'].max()}")
        print(f"  - Registros sin datos en catalán: {df['catalan'].isna().sum()}")
        print(f"  - Registros sin datos en chino: {df['chino'].isna().sum()}")
        
        # Mostrar algunas muestras
        print(f"\n📝 Primeras 3 muestras:")
        for i, row in df.head(3).iterrows():
            print(f"  [{row['linea']}] CAT: {row['catalan'][:50]}...")
            print(f"         CHI: {row['chino'][:50]}...")
            print()
            
        print("✅ Archivo verificado correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error verificando archivo: {e}")
        return False


# Función principal de recuperación
def main():
    import sys
    
    # Valores por defecto
    base_name = "muestra_traducciones_10000"
    results_dir = "p1_results"
    
    # Permitir personalización via argumentos
    if len(sys.argv) > 1:
        base_name = sys.argv[1]
    if len(sys.argv) > 2:
        results_dir = sys.argv[2]
    
    print("🚀 SISTEMA DE RECUPERACIÓN DE MUESTRAS LIMPIAS")
    print(f"📁 Base name: {base_name}")
    print(f"📂 Directorio: {results_dir}")
    print()
    
    # Primero analizar potencial
    analyze_recovery_potential(base_name, results_dir)
    
    # Preguntar si continuar
    response = input("\n¿Deseas proceder con la recuperación? (s/n): ")
    if response.lower() != 's':
        print("❌ Recuperación cancelada")
        return
    
    # Ejecutar recuperación
    recovered_file = recover_clean_samples(base_name, results_dir)
    
    if recovered_file:
        # Verificar el archivo recuperado
        verify_recovered_file(recovered_file)
        
        print(f"\n🎉 PROCESO COMPLETADO")
        print(f"📊 Archivo recuperado: {recovered_file}")
        print(f"💡 Recomendación: Revisa el archivo y renómbralo si todo está correcto")


if __name__ == "__main__":
    main()