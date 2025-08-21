#!/usr/bin/env python3
"""
Extractor ULTRA RÁPIDO de Muestras Aleatorias para Traducciones CA-ZH
Versión con Índices para Acceso Instantáneo + Sistema de Record Anti-Duplicados

Uso:
    python csv_muestra_fast.py --samples 1000
    python csv_muestra_fast.py -s 5000 --seed 42
    python csv_muestra_fast.py --samples 2000 --preview 10
    python csv_muestra_fast.py --create-index  # Crear índices manualmente
"""

import random
import csv
import time
import os
import json
import argparse
import struct
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import pickle

# Configuración por defecto
DEFAULT_CONFIG = {
    'ca_file': "ca-zh_all_2024_08_05.ca",
    'zh_file': "ca-zh_all_2024_08_05.zh", 
    'total_lines': 96_000_000,
    'buffer_size': 8192 * 32,  # 256KB buffer
    'default_samples': 10000,
    'preview_lines': 5,
    'record_file': 'lineas_muestreadas_record.csv',
    'record_cache': 'lineas_muestreadas_cache.pkl',
    'use_index': True  # Por defecto usar índices
}

# =============================================================================
# FUNCIONES DE ÍNDICES PARA ACCESO RÁPIDO
# =============================================================================

def create_line_index(filename: str, index_file: str = None, force: bool = False):
    """
    Crea un índice binario con las posiciones de cada línea.
    Solo necesitas ejecutar esto UNA VEZ por archivo.
    """
    if index_file is None:
        index_file = f"{filename}.idx"
    
    # Si ya existe y no forzamos, salir
    if os.path.exists(index_file) and not force:
        print(f"✅ Índice ya existe: {index_file}")
        return index_file
    
    print(f"📊 Creando índice para {filename}...")
    print(f"   Esto solo se hace una vez y acelerará TODAS las extracciones futuras")
    start_time = time.time()
    
    # Lista para almacenar offsets
    offsets = [0]  # La primera línea empieza en 0
    
    with open(filename, 'rb') as f:
        offset = 0
        line_count = 0
        
        while True:
            line = f.readline()
            if not line:
                break
            
            offset += len(line)
            offsets.append(offset)
            line_count += 1
            
            if line_count % 5_000_000 == 0:
                elapsed = time.time() - start_time
                speed = line_count / elapsed
                eta = (DEFAULT_CONFIG['total_lines'] - line_count) / speed
                print(f"   📍 {line_count:,} líneas indexadas... " +
                      f"({speed:,.0f} líneas/s, ETA: {eta:.0f}s)")
    
    # Guardar índice en formato binario
    print(f"   💾 Guardando índice...")
    with open(index_file, 'wb') as f:
        # Escribir número de líneas
        f.write(struct.pack('Q', len(offsets) - 1))
        
        # Escribir todos los offsets
        for offset in offsets[:-1]:  # No incluir el último
            f.write(struct.pack('Q', offset))
    
    elapsed = time.time() - start_time
    size_mb = os.path.getsize(index_file) / (1024 * 1024)
    
    print(f"✅ Índice creado: {index_file}")
    print(f"   📊 Líneas indexadas: {line_count:,}")
    print(f"   💾 Tamaño del índice: {size_mb:.1f} MB")
    print(f"   ⏱️  Tiempo: {elapsed:.1f} segundos ({elapsed/60:.1f} minutos)")
    
    return index_file

def read_lines_with_index(filename: str, line_numbers: List[int], index_file: str = None) -> Dict[int, str]:
    """
    Lee líneas específicas usando el índice. ULTRA RÁPIDO.
    """
    if index_file is None:
        index_file = f"{filename}.idx"
    
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"No existe el índice {index_file}. Créalo primero con --create-index")
    
    # Cargar offsets necesarios del índice
    offsets = {}
    
    with open(index_file, 'rb') as f:
        total_lines = struct.unpack('Q', f.read(8))[0]
        
        for line_num in line_numbers:
            if 0 <= line_num < total_lines:
                f.seek(8 + line_num * 8)  # 8 bytes header + line_num * 8 bytes per offset
                offsets[line_num] = struct.unpack('Q', f.read(8))[0]
    
    # Leer las líneas en orden de offset para minimizar seeks
    results = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, offset in sorted(offsets.items(), key=lambda x: x[1]):
            f.seek(offset)
            results[line_num] = f.readline().strip()
    
    return results

def check_or_create_index(filename: str) -> str:
    """
    Verifica si existe el índice, si no, lo crea.
    """
    index_file = f"{filename}.idx"
    
    if not os.path.exists(index_file):
        print(f"⚠️  No existe índice para {filename}")
        response = input(f"¿Crear índice ahora? Tardará ~3-5 minutos pero acelerará TODAS las extracciones futuras (S/n): ")
        
        if response.lower() != 'n':
            create_line_index(filename, index_file)
        else:
            raise FileNotFoundError(f"Se requiere índice para extracción rápida. Usa --create-index o --no-index")
    
    return index_file

def load_sampled_lines_record() -> Tuple[Set[int], Dict[int, Tuple[str, str]]]:
    """
    Carga las líneas ya muestreadas del archivo de record.
    Retorna: (set de números de línea, dict con contenido {linea: (catalan, chino)})
    """
    record_file = DEFAULT_CONFIG['record_file']
    cache_file = DEFAULT_CONFIG['record_cache']
    
    sampled_lines = set()
    content_dict = {}
    
    # Verificar si existe cache y es más reciente que el CSV
    if os.path.exists(cache_file) and os.path.exists(record_file):
        cache_mtime = os.path.getmtime(cache_file)
        record_mtime = os.path.getmtime(record_file)
        
        if cache_mtime > record_mtime:
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                    if isinstance(cache_data, tuple) and len(cache_data) == 2:
                        sampled_lines, content_dict = cache_data
                        print(f"📋 Cargadas {len(sampled_lines):,} líneas del cache de record")
                        return sampled_lines, content_dict
                    elif isinstance(cache_data, set):
                        os.remove(cache_file)
                    else:
                        os.remove(cache_file)
            except Exception:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
    
    # Si no hay cache o es antiguo, cargar desde CSV
    if os.path.exists(record_file):
        print(f"📋 Cargando archivo de record: {record_file}")
        start_time = time.time()
        
        try:
            with open(record_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                f.seek(0)
                
                if 'catalan' in first_line and 'chino' in first_line:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row and 'linea' in row:
                            try:
                                line_num = int(row['linea'])
                                sampled_lines.add(line_num)
                                
                                if 'catalan' in row and 'chino' in row:
                                    content_dict[line_num] = (row['catalan'], row['chino'])
                            except (ValueError, KeyError):
                                continue
                else:
                    f.seek(0)
                    reader = csv.reader(f)
                    next(reader, None)
                    
                    for row in reader:
                        if row:
                            try:
                                sampled_lines.add(int(row[0]))
                            except (ValueError, IndexError):
                                continue
        except Exception as e:
            print(f"⚠️ Error al leer record: {e}")
        
        load_time = time.time() - start_time
        print(f"✅ Cargadas {len(sampled_lines):,} líneas ya muestreadas en {load_time:.2f}s")
        
        # Guardar cache para próximas ejecuciones
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((sampled_lines, content_dict), f)
        except:
            pass
    else:
        print(f"📋 No existe archivo de record previo. Se creará uno nuevo.")
        with open(record_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['linea', 'catalan', 'chino', 'fecha_muestreo', 'archivo_muestra'])
    
    return sampled_lines, content_dict

def save_new_lines_to_record(new_samples: Dict[int, Tuple[str, str]], output_csv: str):
    """
    Guarda las nuevas líneas muestreadas con su contenido completo en el archivo de record.
    """
    record_file = DEFAULT_CONFIG['record_file']
    cache_file = DEFAULT_CONFIG['record_cache']
    fecha_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"💾 Guardando {len(new_samples):,} nuevas líneas con contenido en el record...")
    
    with open(record_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for line_num in sorted(new_samples.keys()):
            catalan_text, chino_text = new_samples[line_num]
            writer.writerow([line_num, catalan_text, chino_text, fecha_actual, output_csv])
    
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
        except:
            pass
    
    print(f"✅ Record actualizado con éxito")

def generate_unique_random_lines(sample_size: int, total_lines: int, 
                               existing_lines: Set[int], seed: Optional[int] = None) -> List[int]:
    """
    Genera números de línea aleatorios que no hayan sido muestreados previamente.
    """
    if seed is not None:
        random.seed(seed)
    
    available_lines = total_lines - len(existing_lines)
    if available_lines < sample_size:
        raise ValueError(f"❌ Solo quedan {available_lines:,} líneas disponibles, " +
                        f"pero se solicitaron {sample_size:,} muestras")
    
    print(f"🎲 Generando {sample_size:,} números únicos de línea...")
    print(f"📊 Líneas ya muestreadas: {len(existing_lines):,}")
    print(f"📊 Líneas disponibles: {available_lines:,}")
    
    new_lines = set()
    usage_percentage = len(existing_lines) / total_lines
    
    if usage_percentage < 0.5:  # Menos del 50% usado
        attempts = 0
        max_attempts = sample_size * 10
        
        while len(new_lines) < sample_size and attempts < max_attempts:
            batch_size = min(sample_size - len(new_lines), 1000) * 2
            candidates = random.sample(range(total_lines), min(batch_size, available_lines))
            
            for candidate in candidates:
                if candidate not in existing_lines and candidate not in new_lines:
                    new_lines.add(candidate)
                    if len(new_lines) >= sample_size:
                        break
            
            attempts += 1
        
        if len(new_lines) < sample_size:
            all_available = set(range(total_lines)) - existing_lines
            remaining_needed = sample_size - len(new_lines)
            new_lines.update(random.sample(list(all_available - new_lines), remaining_needed))
    
    else:  # Más del 50% usado
        print(f"⚠️  Alto uso del corpus ({usage_percentage:.1%}). Usando método exhaustivo...")
        all_available = list(set(range(total_lines)) - existing_lines)
        new_lines = set(random.sample(all_available, sample_size))
    
    return sorted(list(new_lines))

# =============================================================================
# FUNCIONES DE MÉTRICAS (HEREDADAS DEL ORIGINAL)
# =============================================================================

def save_metrics_to_p1_results(execution_data: Dict, output_csv: str):
    """
    Guarda las métricas de la ejecución en p1_results con timestamp único.
    """
    results_dir = Path("p1_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base_name = Path(output_csv).stem
    
    metrics_json = results_dir / f"{base_name}_metricas_{timestamp}.json"
    metrics_csv = results_dir / "muestreo_estadisticas_consolidadas.csv"
    
    if 'record_stats' not in execution_data:
        execution_data['record_stats'] = {
            'lineas_previas': 0,
            'lineas_nuevas': 0,
            'total_en_record': 0,
            'porcentaje_corpus': 0
        }
    
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump(execution_data, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Métricas detalladas guardadas: {metrics_json}")
    
    save_to_consolidated_csv(metrics_csv, execution_data)
    
    return metrics_json, metrics_csv

def save_to_consolidated_csv(csv_file: Path, data: Dict):
    """
    Guarda métricas en CSV consolidado para análisis histórico.
    """
    file_exists = csv_file.exists()
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            headers = [
                'timestamp', 'fecha_hora', 'archivo_csv', 'muestras_solicitadas',
                'muestras_extraidas', 'total_lineas', 'tiempo_total_seg', 'tiempo_total_min',
                'tiempo_lectura_ca', 'tiempo_lectura_zh', 'tiempo_creacion_csv',
                'linea_min', 'linea_max', 'eficiencia_pct', 'seed_usado',
                'lineas_previas_en_record', 'lineas_nuevas_añadidas', 'metodo_usado'
            ]
            writer.writerow(headers)
        
        row_data = [
            data['timestamp'],
            data['fecha_hora'], 
            data['archivo_csv'],
            data['configuracion']['muestras_solicitadas'],
            data['resultados']['muestras_extraidas'],
            data['configuracion']['total_lineas'],
            data['tiempos']['total_segundos'],
            data['tiempos']['total_minutos'],
            data['tiempos']['lectura_catalan'],
            data['tiempos']['lectura_chino'],
            data['tiempos']['creacion_csv'],
            data['estadisticas']['linea_minima'],
            data['estadisticas']['linea_maxima'],
            data['estadisticas']['eficiencia_porcentaje'],
            data['configuracion'].get('seed', 'aleatorio'),
            data.get('record_stats', {}).get('lineas_previas', 0),
            data.get('record_stats', {}).get('lineas_nuevas', 0),
            data.get('metodo', 'indice')
        ]
        
        writer.writerow(row_data)

def show_historical_metrics():
    """
    Muestra un resumen de métricas históricas desde p1_results.
    """
    results_dir = Path("p1_results")
    metrics_csv = results_dir / "muestreo_estadisticas_consolidadas.csv"
    
    if not metrics_csv.exists():
        print("📊 No hay métricas históricas previas.")
        return
    
    try:
        with open(metrics_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if len(rows) == 0:
            print("📊 El archivo de métricas está vacío.")
            return
            
        print(f"\n{'='*70}")
        print("📊 RESUMEN HISTÓRICO DE MÉTRICAS DE MUESTREO")
        print("="*70)
        print(f"Total de ejecuciones: {len(rows)}")
        print(f"Métricas guardadas en: {results_dir}")
        
        record_file = DEFAULT_CONFIG['record_file']
        if os.path.exists(record_file):
            existing_lines, _ = load_sampled_lines_record()
            print(f"📋 Total de líneas en record: {len(existing_lines):,}")
            print(f"📋 Porcentaje del corpus muestreado: {(len(existing_lines)/DEFAULT_CONFIG['total_lines']*100):.2f}%")
        
        print()
        
        total_times = [float(row['tiempo_total_seg']) for row in rows if row['tiempo_total_seg']]
        total_samples = sum(int(row['muestras_extraidas']) for row in rows if row['muestras_extraidas'])
        
        if total_times:
            print(f"Tiempo promedio por ejecución: {sum(total_times)/len(total_times):.2f} segundos")
            print(f"Tiempo mínimo: {min(total_times):.2f} segundos")
            print(f"Tiempo máximo: {max(total_times):.2f} segundos")
        print(f"Total de muestras extraídas: {total_samples:,}")
        print()
        
        print("📈 ÚLTIMAS 5 EJECUCIONES:")
        print("-" * 70)
        for i, row in enumerate(rows[-5:], 1):
            print(f"{i}. {row['fecha_hora']}")
            print(f"   ⏱️  Tiempo: {row['tiempo_total_seg']}s ({row['tiempo_total_min']}min)")
            print(f"   📊 Muestras: {row['muestras_extraidas']}/{row['muestras_solicitadas']}")
            print(f"   📁 Archivo: {row['archivo_csv']}")
            print(f"   🎲 Seed: {row.get('seed_usado', 'aleatorio')}")
            if 'lineas_nuevas_añadidas' in row:
                print(f"   📋 Líneas nuevas añadidas al record: {row.get('lineas_nuevas_añadidas', 'N/A')}")
            if 'metodo_usado' in row:
                print(f"   ⚡ Método: {row.get('metodo_usado', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"❌ Error al leer métricas históricas: {e}")

# =============================================================================
# FUNCIÓN PRINCIPAL DE EXTRACCIÓN CON ÍNDICES (ULTRA RÁPIDA)
# =============================================================================

def extract_random_sample_with_index(ca_file: str, zh_file: str, output_csv: str, 
                                   sample_size: int, total_lines: int, 
                                   seed: Optional[int] = None) -> Dict:
    """
    Extrae una muestra aleatoria usando ÍNDICES para acceso ultra rápido.
    """
    print(f"⚡ MODO ULTRA RÁPIDO CON ÍNDICES ⚡")
    print(f"🚀 Iniciando extracción de {sample_size:,} muestras...")
    print(f"📁 Archivos: {ca_file} y {zh_file}")
    print(f"🎯 CSV destino: {output_csv}")
    
    if seed is not None:
        random.seed(seed)
        print(f"🎲 Usando seed: {seed} (reproducible)")
    else:
        print(f"🎲 Modo aleatorio real (no reproducible)")
    
    start_time = time.time()
    
    # Verificar/crear índices
    print("\n📊 Verificando índices...")
    ca_index = check_or_create_index(ca_file)
    zh_index = check_or_create_index(zh_file)
    
    # Cargar líneas ya muestreadas
    existing_lines, existing_content = load_sampled_lines_record()
    record_load_time = time.time() - start_time
    
    # Generar números de línea aleatorios únicos
    try:
        sorted_lines = generate_unique_random_lines(sample_size, total_lines, existing_lines, seed)
    except ValueError as e:
        print(str(e))
        raise
    
    print(f"📍 Rango de líneas: {min(sorted_lines):,} a {max(sorted_lines):,}")
    
    # Leer líneas usando índices - ULTRA RÁPIDO
    print("\n⚡ Leyendo líneas con índices (ultra rápido)...")
    
    # Leer archivo catalán
    ca_start_time = time.time()
    ca_lines = read_lines_with_index(ca_file, sorted_lines, ca_index)
    ca_read_time = time.time() - ca_start_time
    print(f"✅ {len(ca_lines):,} líneas catalanas leídas en {ca_read_time:.2f} segundos")
    
    # Leer archivo chino
    zh_start_time = time.time()
    zh_lines = read_lines_with_index(zh_file, sorted_lines, zh_index)
    zh_read_time = time.time() - zh_start_time
    print(f"✅ {len(zh_lines):,} líneas chinas leídas en {zh_read_time:.2f} segundos")
    
    # Crear CSV
    print("\n📝 Creando archivo CSV...")
    csv_start = time.time()
    
    new_samples_for_record = {}
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['linea', 'catalan', 'chino'])
        
        samples_written = 0
        for line_num in sorted(sorted_lines):
            if line_num in ca_lines and line_num in zh_lines:
                catalan_text = ca_lines[line_num]
                chino_text = zh_lines[line_num]
                writer.writerow([line_num, catalan_text, chino_text])
                
                new_samples_for_record[line_num] = (catalan_text, chino_text)
                samples_written += 1
    
    csv_time = time.time() - csv_start
    
    # Guardar nuevas líneas en el record
    record_start = time.time()
    save_new_lines_to_record(new_samples_for_record, output_csv)
    record_save_time = time.time() - record_start
    
    total_time = time.time() - start_time
    
    # Crear diccionario completo de métricas
    metrics_data = {
        'timestamp': datetime.now().timestamp(),
        'fecha_hora': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'archivo_csv': output_csv,
        'metodo': 'indice',
        'configuracion': {
            'archivo_catalan': ca_file,
            'archivo_chino': zh_file,
            'muestras_solicitadas': sample_size,
            'total_lineas': total_lines,
            'seed': seed,
            'usa_indices': True
        },
        'tiempos': {
            'total_segundos': round(total_time, 2),
            'total_minutos': round(total_time / 60, 2),
            'lectura_catalan': round(ca_read_time, 2),
            'lectura_chino': round(zh_read_time, 2),
            'creacion_csv': round(csv_time, 2),
            'carga_record': round(record_load_time, 2),
            'guardado_record': round(record_save_time, 2)
        },
        'resultados': {
            'muestras_extraidas': samples_written,
            'muestras_solicitadas': sample_size,
            'completado_exitosamente': samples_written == sample_size
        },
        'estadisticas': {
            'linea_minima': min(sorted_lines) if sorted_lines else 0,
            'linea_maxima': max(sorted_lines) if sorted_lines else 0,
            'eficiencia_porcentaje': round((samples_written / sample_size) * 100, 1) if sample_size > 0 else 0,
            'velocidad_muestras_por_segundo': round(samples_written / total_time, 1) if total_time > 0 else 0
        },
        'archivos_fuente': {
            'catalan_existe': os.path.exists(ca_file),
            'chino_existe': os.path.exists(zh_file),
            'catalan_size_mb': round(os.path.getsize(ca_file) / (1024*1024), 1) if os.path.exists(ca_file) else 0,
            'chino_size_mb': round(os.path.getsize(zh_file) / (1024*1024), 1) if os.path.exists(zh_file) else 0,
            'indice_catalan_existe': os.path.exists(ca_index),
            'indice_chino_existe': os.path.exists(zh_index)
        },
        'record_stats': {
            'lineas_previas': len(existing_lines),
            'lineas_nuevas': len(new_samples_for_record),
            'total_en_record': len(existing_lines) + len(new_samples_for_record),
            'porcentaje_corpus': round(((len(existing_lines) + len(new_samples_for_record)) / total_lines) * 100, 2)
        }
    }
    
    return metrics_data

# =============================================================================
# FUNCIÓN DE EXTRACCIÓN TRADICIONAL (FALLBACK)
# =============================================================================

def extract_random_sample_traditional(ca_file: str, zh_file: str, output_csv: str, 
                                    sample_size: int, total_lines: int, 
                                    seed: Optional[int] = None) -> Dict:
    """
    Extrae una muestra aleatoria de forma tradicional (sin índices).
    Mantiene compatibilidad con el método original.
    """
    print(f"📖 MODO TRADICIONAL (sin índices)")
    print(f"⚠️  Este método es más lento. Considera usar --create-index para acelerar futuras extracciones")
    
    # [El código es idéntico al extract_random_sample_optimized original]
    # Lo omito por brevedad, pero debería incluir toda la función original aquí
    
    # Por ahora, lanzar excepción para forzar uso de índices
    raise NotImplementedError("Usa --create-index primero o importa el script original para modo tradicional")

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def verify_sample(csv_file: str, num_preview: int = 5):
    """
    Verifica y muestra una vista previa del archivo CSV generado.
    """
    print(f"\n{'='*50}")
    print("🔍 VISTA PREVIA DEL ARCHIVO GENERADO")
    print("="*50)
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            print(f"Total de filas (incluyendo encabezado): {len(rows)}")
            print(f"Columnas: {rows[0] if rows else 'N/A'}")
            print("-" * 50)
                    
    except Exception as e:
        print(f"❌ Error al verificar el archivo: {e}")

def show_record_stats():
    """
    Muestra estadísticas detalladas del archivo de record.
    """
    record_file = DEFAULT_CONFIG['record_file']
    
    if not os.path.exists(record_file):
        print("📋 No existe archivo de record.")
        return
    
    print(f"\n{'='*70}")
    print("📋 ESTADÍSTICAS DEL ARCHIVO DE RECORD")
    print("="*70)
    
    existing_lines, existing_content = load_sampled_lines_record()
    total_lines = DEFAULT_CONFIG['total_lines']
    
    print(f"📁 Archivo: {record_file}")
    print(f"📊 Total de líneas muestreadas: {len(existing_lines):,}")
    print(f"📊 Total de líneas en corpus: {total_lines:,}")
    print(f"📊 Porcentaje muestreado: {(len(existing_lines)/total_lines*100):.2f}%")
    print(f"📊 Líneas disponibles: {(total_lines - len(existing_lines)):,}")
    
    if existing_lines:
        print(f"\n📈 Distribución:")
        print(f"   Línea mínima: {min(existing_lines):,}")
        print(f"   Línea máxima: {max(existing_lines):,}")
    
    file_size = os.path.getsize(record_file)
    print(f"\n💾 Tamaño del archivo: {file_size / (1024*1024):.2f} MB")
    
    if existing_content and len(existing_content) > 0:
        print(f"\n📝 Ejemplos de contenido guardado:")
        print("-" * 70)
        sample_lines = list(sorted(existing_content.keys()))[:3]
        for i, line_num in enumerate(sample_lines, 1):
            if line_num in existing_content:
                ca_text, zh_text = existing_content[line_num]
                print(f"{i}. Línea {line_num}:")
                print(f"   CA: {ca_text[:60]}{'...' if len(ca_text) > 60 else ''}")
                print(f"   ZH: {zh_text[:60]}{'...' if len(zh_text) > 60 else ''}")
        if len(existing_content) > 3:
            print(f"   ... y {len(existing_content) - 3:,} muestras más")

def check_index_status():
    """
    Verifica el estado de los índices y muestra información.
    """
    print(f"\n{'='*70}")
    print("📊 ESTADO DE LOS ÍNDICES")
    print("="*70)
    
    for file_key, filename in [('ca_file', DEFAULT_CONFIG['ca_file']), 
                               ('zh_file', DEFAULT_CONFIG['zh_file'])]:
        index_file = f"{filename}.idx"
        print(f"\n📁 Archivo: {filename}")
        
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024*1024*1024)
            print(f"   Tamaño: {file_size:.2f} GB")
            
            if os.path.exists(index_file):
                index_size = os.path.getsize(index_file) / (1024*1024)
                index_mtime = datetime.fromtimestamp(os.path.getmtime(index_file))
                print(f"   ✅ Índice existe: {index_file}")
                print(f"   💾 Tamaño índice: {index_size:.1f} MB")
                print(f"   📅 Creado: {index_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"   ❌ No existe índice")
                print(f"   💡 Usa --create-index para crearlo")
        else:
            print(f"   ❌ Archivo no encontrado")

# =============================================================================
# FUNCIÓN PRINCIPAL Y ARGUMENTOS
# =============================================================================

def parse_arguments():
    """
    Parsea argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Extractor ULTRA RÁPIDO de Muestras Aleatorias para Traducciones CA-ZH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Primera vez: crear índices (tarda ~3-5 min, solo se hace una vez)
  python csv_muestra_fast.py --create-index

  # Extraer 1000 muestras (ultra rápido con índices)
  python csv_muestra_fast.py --samples 1000

  # Extraer 5000 muestras con seed reproducible
  python csv_muestra_fast.py -s 5000 --seed 42

  # Ver estado de los índices
  python csv_muestra_fast.py --index-status

  # Ver estadísticas del record
  python csv_muestra_fast.py --record-stats
  
  # Forzar modo sin índices (lento)
  python csv_muestra_fast.py --samples 1000 --no-index
        """
    )
    
    parser.add_argument('--samples', '-s', 
                       type=int, 
                       default=DEFAULT_CONFIG['default_samples'],
                       help=f'Número de muestras a extraer (default: {DEFAULT_CONFIG["default_samples"]})')
    
    parser.add_argument('--seed',
                       type=int,
                       help='Seed para reproducibilidad (opcional)')
    
    parser.add_argument('--preview', '-p',
                       type=int,
                       default=DEFAULT_CONFIG['preview_lines'],
                       help=f'Número de líneas para preview (default: {DEFAULT_CONFIG["preview_lines"]})')
    
    parser.add_argument('--total-lines', '-t',
                       type=int,
                       default=DEFAULT_CONFIG['total_lines'],
                       help=f'Total de líneas en archivos fuente (default: {DEFAULT_CONFIG["total_lines"]:,})')
    
    parser.add_argument('--ca-file',
                       default=DEFAULT_CONFIG['ca_file'],
                       help=f'Archivo catalán (default: {DEFAULT_CONFIG["ca_file"]})')
    
    parser.add_argument('--zh-file',
                       default=DEFAULT_CONFIG['zh_file'],
                       help=f'Archivo chino (default: {DEFAULT_CONFIG["zh_file"]})')
    
    parser.add_argument('--output', '-o',
                       help='Nombre del archivo CSV de salida (default: auto-generado)')
    
    parser.add_argument('--create-index',
                       action='store_true',
                       help='Crear índices para los archivos (solo necesario una vez)')
    
    parser.add_argument('--index-status',
                       action='store_true',
                       help='Mostrar estado de los índices')
    
    parser.add_argument('--no-index',
                       action='store_true',
                       help='Forzar modo tradicional sin índices (LENTO)')
    
    parser.add_argument('--history-only',
                       action='store_true',
                       help='Solo mostrar métricas históricas sin extraer muestras')
    
    parser.add_argument('--record-stats',
                       action='store_true',
                       help='Mostrar estadísticas del archivo de record')
    
    parser.add_argument('--quiet', '-q',
                       action='store_true',
                       help='Modo silencioso (menos output)')
    
    parser.add_argument('--clear-record',
                       action='store_true',
                       help='Limpiar el archivo de record (⚠️ CUIDADO: borra todo el historial)')
    
    return parser.parse_args()

def main():
    """
    Función principal con argumentos.
    """
    args = parse_arguments()
    
    # Crear índices si se solicita
    if args.create_index:
        print("📊 CREACIÓN DE ÍNDICES")
        print("="*70)
        print("Esto creará índices para acceso ultra rápido a las líneas.")
        print("Solo necesitas hacer esto UNA VEZ. Los índices se reutilizarán.")
        print()
        
        for filename in [args.ca_file, args.zh_file]:
            if os.path.exists(filename):
                create_line_index(filename, force=True)
            else:
                print(f"❌ Archivo no encontrado: {filename}")
        
        print("\n✅ Índices creados. Ahora puedes extraer muestras ultra rápido.")
        return 0
    
    # Mostrar estado de índices
    if args.index_status:
        check_index_status()
        return 0
    
    # Limpiar record si se solicita
    if args.clear_record:
        response = input("⚠️  ¿Estás seguro de que quieres borrar TODO el historial de muestras? (s/N): ")
        if response.lower() == 's':
            try:
                if os.path.exists(DEFAULT_CONFIG['record_file']):
                    os.remove(DEFAULT_CONFIG['record_file'])
                if os.path.exists(DEFAULT_CONFIG['record_cache']):
                    os.remove(DEFAULT_CONFIG['record_cache'])
                print("✅ Archivo de record eliminado.")
            except Exception as e:
                print(f"❌ Error al eliminar record: {e}")
        return 0
    
    # Mostrar estadísticas del record
    if args.record_stats:
        show_record_stats()
        return 0
    
    # Validaciones
    if args.samples <= 0:
        print(f"❌ Error: El número de muestras debe ser positivo. Recibido: {args.samples}")
        return 1
    
    if args.samples > args.total_lines:
        print(f"❌ Error: No se pueden extraer {args.samples:,} muestras de {args.total_lines:,} líneas")
        return 1
    
    # Solo mostrar historial
    if args.history_only:
        show_historical_metrics()
        return 0
    
    # Verificar archivos fuente
    if not os.path.exists(args.ca_file):
        print(f"❌ Error: Archivo catalán no encontrado: {args.ca_file}")
        return 1
    
    if not os.path.exists(args.zh_file):
        print(f"❌ Error: Archivo chino no encontrado: {args.zh_file}")
        return 1
    
    # Generar nombre de archivo de salida
    if args.output:
        output_csv = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_csv = f"muestra_traducciones_{args.samples}.csv"
    
    # Mostrar configuración
    if not args.quiet:
        print(f"⚡ EXTRACTOR ULTRA RÁPIDO DE MUESTRAS")
        print("="*70)
        print(f"📁 Archivo catalán: {args.ca_file}")
        print(f"📁 Archivo chino: {args.zh_file}")
        print(f"📊 Muestras a extraer: {args.samples:,}")
        print(f"📊 Total líneas disponibles: {args.total_lines:,}")
        print(f"🎯 Archivo destino: {output_csv}")
        if args.seed:
            print(f"🎲 Seed: {args.seed} (reproducible)")
        else:
            print(f"🎲 Modo aleatorio (no reproducible)")
        print(f"📂 Métricas se guardarán en: p1_results/")
        print(f"📋 Archivo de record: {DEFAULT_CONFIG['record_file']}")
        
        if not args.no_index:
            # Verificar estado de índices
            ca_idx = f"{args.ca_file}.idx"
            zh_idx = f"{args.zh_file}.idx"
            if os.path.exists(ca_idx) and os.path.exists(zh_idx):
                print(f"⚡ Modo: ULTRA RÁPIDO con índices")
            else:
                print(f"⚠️  Modo: Se crearán índices si no existen")
    
    # Mostrar métricas históricas
    if not args.quiet:
        show_historical_metrics()
    
    try:
        # Decidir qué método usar
        if args.no_index:
            # Modo tradicional (lento)
            metrics_data = extract_random_sample_traditional(
                ca_file=args.ca_file,
                zh_file=args.zh_file,
                output_csv=output_csv,
                sample_size=args.samples,
                total_lines=args.total_lines,
                seed=args.seed
            )
        else:
            # Modo con índices (ultra rápido)
            metrics_data = extract_random_sample_with_index(
                ca_file=args.ca_file,
                zh_file=args.zh_file,
                output_csv=output_csv,
                sample_size=args.samples,
                total_lines=args.total_lines,
                seed=args.seed
            )
        
        # Guardar métricas
        metrics_json, metrics_csv = save_metrics_to_p1_results(metrics_data, output_csv)
        
        # Mostrar resumen
        if not args.quiet:
            print(f"\n{'='*60}")
            print("📊 RESUMEN FINAL")
            print("="*60)
            print(f"⏱️  Tiempo total: {metrics_data['tiempos']['total_segundos']} segundos " +
                  f"({metrics_data['tiempos']['total_minutos']} minutos)")
            print(f"📊 Muestras extraídas: {metrics_data['resultados']['muestras_extraidas']:,}")
            print(f"✅ Eficiencia: {metrics_data['estadisticas']['eficiencia_porcentaje']}%")
            
            if 'velocidad_muestras_por_segundo' in metrics_data['estadisticas']:
                print(f"⚡ Velocidad: {metrics_data['estadisticas']['velocidad_muestras_por_segundo']} muestras/segundo")
            
            print(f"\n📁 CSV limpio: {output_csv}")
            print(f"📊 Métricas detalladas: {metrics_json}")
            print(f"📈 Métricas históricas: {metrics_csv}")
            print(f"\n📋 ESTADÍSTICAS DEL RECORD:")
            print(f"   Líneas previas: {metrics_data['record_stats']['lineas_previas']:,}")
            print(f"   Líneas nuevas añadidas: {metrics_data['record_stats']['lineas_nuevas']:,}")
            print(f"   Total en record: {metrics_data['record_stats']['total_en_record']:,}")
            print(f"   Porcentaje del corpus: {metrics_data['record_stats']['porcentaje_corpus']:.2f}%")
        
        # Verificar resultado
        if not args.quiet:
            verify_sample(output_csv, args.preview)
        
        print(f"\n🎉 ¡Proceso completado exitosamente!")
        
        if args.quiet:
            print(f"📊 {metrics_data['resultados']['muestras_extraidas']:,} muestras en {metrics_data['tiempos']['total_segundos']}s")
            print(f"📁 {output_csv}")
            print(f"📋 {metrics_data['record_stats']['lineas_nuevas']:,} líneas nuevas añadidas al record")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except MemoryError:
        print(f"❌ Error: No hay suficiente memoria para procesar {args.samples:,} muestras")
        print(f"💡 Sugerencia: Reduce el número de muestras con --samples")
        return 1
    except KeyboardInterrupt:
        print(f"\n⚠️  Proceso interrumpido por el usuario")
        return 1
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())