import numpy as np
import pandas as pd
import time
import re
import csv
import hashlib
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openpyxl
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pickle
import gzip
import torch
import sys
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')
EMBEDDINGS_AVAILABLE = True
EXCEL_AVAILABLE = True
TQDM_AVAILABLE = True

def phase_timer(phase_name: str):
    """Decorador para medir tiempo de fases"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            start = time.time()
            self.log(f"🚀 Iniciando {phase_name}")
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start
            self.metrics_log[f'tiempo_{phase_name.lower().replace(" ", "_").replace(":", "")}'] = elapsed
            self.log(f"✅ {phase_name} completado en {elapsed:.2f}s")
            return result
        return wrapper
    return decorator

class ChineseCorrectionPipeline:
    def __init__(self, csv_file: str, elimination_percentage: float = 5.0, interactive_mode: bool = True, improvement_threshold: float = 0.5):
        self.csv_file = csv_file
        self.elimination_percentage = elimination_percentage  # Solo usado si interactive_mode=False
        self.interactive_mode = interactive_mode  # Activar modo interactivo 
        self.improvement_threshold = improvement_threshold  # Threshold para detectar meseta
        self.df = None
        self.models = {}
        self.embeddings_cache = defaultdict(dict)  # {model_name: {text_hash: embedding}}
        self.correction_log = []
        self.duplicates_log = []
        self.metrics_log = {}
        self.start_time = time.time()
        self.embeddings_cache = defaultdict(dict)
        self.cache_lock = threading.Lock()
        self.cache_dir = Path("p1_embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.max_cache_size_gb = 2.0  # Límite de 2GB
        self.load_cache_index()
        
        # Configuración expandida de modelos
        self.models_to_load = [
            # Modelos generales multilingües (robustos)
            'paraphrase-multilingual-mpnet-base-v2',
            #'LaBSE',  
            #'distiluse-base-multilingual-cased',
            'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens',
            
            # Modelos específicos para traducción/similaridad semántica
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'sentence-transformers/distiluse-base-multilingual-cased-v2',
            'sentence-transformers/all-MiniLM-L6-v2',  # Muy rápido, buena calidad
            
            # Modelos más recientes y especializados
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/multi-qa-mpnet-base-dot-v1',  # Bueno para Q&A cross-lingual
        ]
        
        # Patrones regex compilados
        self.catalan_bad_chars = re.compile(r'[^\w\sàáèéíïóòúüçñ.,;:!?¿¡()"\'-]', re.IGNORECASE)
        self.chinese_bad_chars = re.compile(r'[^\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2f00-\u2fdf\u31c0-\u31ef\u2ff0-\u2fff\u3000-\u303f\uff00-\uffef0-9，。！？；：、""''（）【】《》〈〉·—…\\s]')
        self.multiple_spaces = re.compile(r'\s+')
        self.bad_quotes = re.compile(r'[""''`]')
        
        # HSK caracteres básicos (simplificado para demo)
        self.hsk_chars = set('的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严')
        # Configuración GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.log(f"🎮 GPU detectada: {self.gpu_name} ({self.gpu_memory:.1f} GB)")
            
            # Batch size mayor para GPU
            self.gpu_batch_size = 128  # Ajusta según tu GPU
        else:
            self.log("⚠️  No se detectó GPU, usando CPU")
            self.gpu_batch_size = 50

    def clean_gpu_memory(self):
        """Limpia memoria GPU si es necesario"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            # Opcional: forzar recolección de basura
            import gc
            gc.collect()
            
            # Mostrar uso de memoria
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.log(f"  🎮 GPU memoria: {allocated:.1f}GB usado, {reserved:.1f}GB reservado")
    
    def log(self, message: str, level: str = "INFO"):
        """Sistema de logging con timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    @phase_timer("Fase 0: Inicialización")
    def initialize_models(self):
        """Carga múltiples modelos de sentence transformers EN GPU"""
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers requerido")
            
        self.log(f"Cargando modelos de embeddings en {self.device.upper()}...")
        successful_models = []
        
        for model_name in tqdm(self.models_to_load, desc="  📥 Cargando modelos", ncols=100):
            try:
                # Cargar modelo con configuración GPU
                model = SentenceTransformer(model_name, device=self.device)
                
                # Optimizaciones para GPU
                if self.gpu_available:
                    model.eval()  # Modo evaluación
                    # Opcional: usar half precision para más velocidad
                    # model = model.half()  # Descomenta si tu GPU soporta FP16
                
                short_name = model_name.split('/')[-1].replace('sentence-transformers/', '')
                self.models[short_name] = model
                successful_models.append(short_name)
                
            except Exception as e:
                self.log(f"  ❌ Error cargando {model_name}: {e}", "ERROR")
                # Si falla GPU, intentar CPU
                if self.gpu_available and 'out of memory' in str(e).lower():
                    self.log(f"  ⚠️  Memoria GPU insuficiente, intentando CPU para {model_name}")
                    try:
                        model = SentenceTransformer(model_name, device='cpu')
                        self.models[short_name] = model
                        successful_models.append(short_name + " (CPU)")
                    except:
                        pass
                        
        if not successful_models:
            raise Exception("No se pudo cargar ningún modelo de embeddings")
            
        self.log(f"📊 Modelos cargados exitosamente: {len(successful_models)}")
        return successful_models
    
    @phase_timer("Fase 1: Limpieza Catalana")  
    def clean_catalan(self):
        """Limpieza y normalización del texto catalán"""
        corrections = 0
        eliminations = 0
        
        initial_rows = len(self.df)
        
        # Detectar caracteres problemáticos
        bad_chars_mask = self.df['catalan'].str.contains(self.catalan_bad_chars, na=False)
        bad_rows = self.df[bad_chars_mask]
        
        if len(bad_rows) > 0:
            self.log(f"  🗑️  Eliminando {len(bad_rows)} registros por caracteres problemáticos en catalán")
            for idx, row in bad_rows.iterrows():
                bad_chars = self.catalan_bad_chars.findall(row['catalan'])
                self.correction_log.append({
                    'linea': row['linea'],
                    'tipo': 'eliminacion_catalan',
                    'razon': f'caracteres_problematicos: {set(bad_chars)}',
                    'texto_original': row['catalan'][:100] + '...' if len(row['catalan']) > 100 else row['catalan']
                })
            self.df = self.df[~bad_chars_mask].reset_index(drop=True)
            eliminations = len(bad_rows)
        
        # Normalización
        original_catalan = self.df['catalan'].copy()
        
        # Corregir comillas problemáticas
        self.df['catalan'] = self.df['catalan'].str.replace(self.bad_quotes, '"', regex=True)
        
        # Normalizar espacios múltiples
        self.df['catalan'] = self.df['catalan'].str.replace(self.multiple_spaces, ' ', regex=True)
        
        # Limpiar espacios al inicio/final
        self.df['catalan'] = self.df['catalan'].str.strip()
        
        # Contar correcciones
        corrections = (original_catalan != self.df['catalan']).sum()
        
        if corrections > 0:
            self.log(f"  🔧 {corrections} textos catalanes corregidos")
            
        self.log(f"  📊 Catalán: {eliminations} eliminados, {corrections} corregidos, {len(self.df)} restantes")
        return eliminations, corrections
    
    def load_cache_index(self):
        """Carga el índice de caché del disco"""
        if self.cache_index_file.exists():
            with open(self.cache_index_file, 'r') as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {}
        
    def save_cache_index(self):
        """Guarda el índice de caché al disco"""
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f)
            
    def get_cache_path(self, model_name: str, text_hash: str) -> Path:
        """Genera la ruta del archivo de caché"""
        model_dir = self.cache_dir / model_name.replace('/', '_')
        model_dir.mkdir(exist_ok=True)
        return model_dir / f"{text_hash}.pkl.gz"
        
    def load_embedding_from_disk(self, model_name: str, text_hash: str) -> np.ndarray:
        """Carga un embedding del disco si existe"""
        cache_path = self.get_cache_path(model_name, text_hash)
        if cache_path.exists():
            try:
                with gzip.open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
        
    def save_embedding_to_disk(self, model_name: str, text_hash: str, embedding: np.ndarray):
        """Guarda un embedding al disco"""
        cache_path = self.get_cache_path(model_name, text_hash)
        with gzip.open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
        
        # Actualizar índice
        if model_name not in self.cache_index:
            self.cache_index[model_name] = {}
        self.cache_index[model_name][text_hash] = {
            'path': str(cache_path),
            'timestamp': datetime.now().isoformat()
        }
        
    def clean_old_cache(self):
        """Limpia caché antiguo si excede el límite de tamaño"""
        try:
            # Obtener lista de archivos válidos
            cache_files = []
            total_size = 0
            
            for f in self.cache_dir.rglob('*.pkl.gz'):
                try:
                    if f.exists():  # Verificar que el archivo existe
                        size = f.stat().st_size
                        mtime = f.stat().st_mtime
                        cache_files.append((f, size, mtime))
                        total_size += size
                except (OSError, FileNotFoundError):
                    # Si no se puede acceder al archivo, saltarlo
                    continue
            
            self.log(f"  🗂️  Caché actual: {total_size / (1024**3):.2f} GB ({len(cache_files)} archivos)")
            
            # Solo limpiar si excede el límite
            if total_size > self.max_cache_size_gb * 1024**3:
                self.log(f"  🧹 Limpiando caché (límite: {self.max_cache_size_gb} GB)")
                
                # Ordenar por tiempo de modificación (más antiguos primero)
                cache_files.sort(key=lambda x: x[2])  # Ordenar por mtime
                
                target_size = self.max_cache_size_gb * 1024**3 * 0.8  # Dejar 20% libre
                files_removed = 0
                size_removed = 0
                
                while total_size > target_size and cache_files:
                    file_path, file_size, _ = cache_files.pop(0)
                    
                    try:
                        if file_path.exists():  # Verificar que aún existe
                            file_path.unlink()  # Eliminar archivo
                            total_size -= file_size  # Usar el tamaño que ya obtuvimos
                            size_removed += file_size
                            files_removed += 1
                            
                            # Actualizar índice de caché si existe
                            self._remove_from_cache_index(file_path)
                            
                    except (OSError, FileNotFoundError) as e:
                        # Si no se puede eliminar, continuar con el siguiente
                        self.log(f"  ⚠️  No se pudo eliminar {file_path}: {e}")
                        continue
                
                self.log(f"  ✅ Caché limpiado: {files_removed} archivos, {size_removed / (1024**3):.2f} GB liberados")
                
        except Exception as e:
            self.log(f"  ⚠️  Error limpiando caché: {e}. Continuando...", "WARNING")

    def _remove_from_cache_index(self, file_path):
        """Remueve una entrada del índice de caché"""
        try:
            # Extraer información del path
            parts = file_path.parts
            if len(parts) >= 2:
                model_name = parts[-2]  # Directorio del modelo
                file_hash = parts[-1].replace('.pkl.gz', '')  # Hash sin extensión
                
                if model_name in self.cache_index and file_hash in self.cache_index[model_name]:
                    del self.cache_index[model_name][file_hash]
                    
                    # Si el modelo no tiene más entradas, eliminar el modelo del índice
                    if not self.cache_index[model_name]:
                        del self.cache_index[model_name]
                        
        except Exception:
            # Si hay error actualizando el índice, no es crítico
            pass
    
    @phase_timer("Fase 2: Limpieza China")
    def clean_chinese(self):
        """Limpieza y normalización del texto chino"""
        corrections = 0
        eliminations = 0
        
        # Detectar caracteres no-chinos (excepto números)
        bad_chars_mask = self.df['chino'].str.contains(self.chinese_bad_chars, na=False)
        bad_rows = self.df[bad_chars_mask]
        
        if len(bad_rows) > 0:
            self.log(f"  🗑️  Eliminando {len(bad_rows)} registros por caracteres no-chinos")
            for idx, row in bad_rows.iterrows():
                bad_chars = self.chinese_bad_chars.findall(row['chino'])
                self.correction_log.append({
                    'linea': row['linea'],
                    'tipo': 'eliminacion_chino',
                    'razon': f'caracteres_no_chinos: {set(bad_chars)}',
                    'texto_original': row['chino'][:100] + '...' if len(row['chino']) > 100 else row['chino']
                })
            self.df = self.df[~bad_chars_mask].reset_index(drop=True)
            eliminations = len(bad_rows)
        
        # Normalización
        original_chinese = self.df['chino'].copy()
        
        # Corregir comillas problemáticas  
        self.df['chino'] = self.df['chino'].str.replace(self.bad_quotes, '"', regex=True)
        
        # Normalizar espacios (el chino no debería tener muchos espacios)
        self.df['chino'] = self.df['chino'].str.replace(self.multiple_spaces, '', regex=True)
        
        # Normalizar puntuación china
        punct_replacements = {
            ',': '，',
            '.': '。', 
            '!': '！',
            '?': '？',
            ';': '；',
            ':': '：'
        }
        for en_punct, ch_punct in punct_replacements.items():
            self.df['chino'] = self.df['chino'].str.replace(en_punct, ch_punct)
        
        # Limpiar espacios al inicio/final
        self.df['chino'] = self.df['chino'].str.strip()
        
        # Contar correcciones
        corrections = (original_chinese != self.df['chino']).sum()
        
        if corrections > 0:
            self.log(f"  🔧 {corrections} textos chinos corregidos")
            
        self.log(f"  📊 Chino: {eliminations} eliminados, {corrections} corregidos, {len(self.df)} restantes")
        return eliminations, corrections
    
    @phase_timer("Fase 3: Detección de Duplicados")
    def detect_duplicates(self):
        """Detecta duplicados en múltiples niveles"""
        duplicates_found = {}
        
        # Nivel 1: Duplicados exactos en chino
        chinese_groups = self.df.groupby('chino')['linea'].apply(list)
        exact_duplicates = {text: lines for text, lines in chinese_groups.items() if len(lines) > 1}
        
        if exact_duplicates:
            duplicates_found['exactos'] = exact_duplicates
            total_duplicate_lines = sum(len(lines) for lines in exact_duplicates.values())
            self.log(f"  🔍 Duplicados exactos: {len(exact_duplicates)} grupos, {total_duplicate_lines} líneas afectadas")
        
        # Nivel 2: Duplicados casi-exactos (después de normalización básica)
        def normalize_for_comparison(text):
            if pd.isna(text):
                return ""
            # Quitar espacios y puntuación para comparación
            normalized = re.sub(r'[，。！？；：、\s]', '', str(text))
            return normalized.lower()
        
        self.df['chino_normalized'] = self.df['chino'].apply(normalize_for_comparison)
        normalized_groups = self.df.groupby('chino_normalized')['linea'].apply(list)
        near_duplicates = {norm_text: lines for norm_text, lines in normalized_groups.items() 
                          if len(lines) > 1 and norm_text != ""}
        
        if near_duplicates:
            duplicates_found['casi_exactos'] = near_duplicates
            self.log(f"  🔍 Duplicados casi-exactos: {len(near_duplicates)} grupos adicionales")
        
        # Limpiar columna temporal
        self.df.drop('chino_normalized', axis=1, inplace=True)
        
        # Guardar log de duplicados
        self.duplicates_log = duplicates_found
        
        return duplicates_found
    
    def get_text_hash(self, text: str) -> str:
        """Genera hash para caché de embeddings"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embeddings_batch(self, texts: List[str], model_name: str, batch_size: int = None) -> np.ndarray:
        """Obtiene embeddings con caché en memoria Y disco"""
        # Usar batch size de GPU si no se especifica
        if batch_size is None:
            batch_size = self.gpu_batch_size if self.gpu_available else 100

        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        # Verificar caché (memoria y disco)
        with self.cache_lock:
            for i, text in enumerate(texts):
                text_hash = self.get_text_hash(text)
                
                # Primero verificar memoria
                if text_hash in self.embeddings_cache[model_name]:
                    embeddings.append(self.embeddings_cache[model_name][text_hash])
                else:
                    # Luego verificar disco
                    #disk_embedding = self.load_embedding_from_disk(model_name, text_hash)
                    #if disk_embedding is not None:
                    #    self.embeddings_cache[model_name][text_hash] = disk_embedding
                    #    embeddings.append(disk_embedding)
                    #else:
                    embeddings.append(None)
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
        
        # Codificar textos no cacheados
        if texts_to_encode:
            model = self.models[model_name]
            
            # Para GPU, podemos procesar más textos a la vez
            num_batches = (len(texts_to_encode) + batch_size - 1) // batch_size
            
            # Solo mostrar barra si hay muchos batches
            if num_batches > 3:
                batch_iterator = tqdm(range(0, len(texts_to_encode), batch_size), 
                                    desc=f"    🔄 GPU encoding {model_name[:20]}", 
                                    total=num_batches,
                                    leave=False,
                                    ncols=100)
            else:
                batch_iterator = range(0, len(texts_to_encode), batch_size)
            
            for i in batch_iterator:
                batch_texts = texts_to_encode[i:i + batch_size]
                batch_indices = indices_to_encode[i:i + batch_size]
                
                try:
                    # Configuración específica para GPU
                    encode_kwargs = {
                        'convert_to_numpy': True,
                        'show_progress_bar': False,
                        'batch_size': batch_size,
                        'normalize_embeddings': True  # Normalizar para cosine similarity
                    }
                    
                    # Si hay GPU, podemos usar convert_to_tensor primero
                    if self.gpu_available:
                        encode_kwargs['convert_to_tensor'] = True
                        encode_kwargs['convert_to_numpy'] = False
                        
                    batch_embeddings = model.encode(batch_texts, **encode_kwargs)
                    
                    # Convertir a numpy si viene de GPU
                    if self.gpu_available:
                        batch_embeddings = batch_embeddings.cpu().numpy()
                    
                    # Almacenar en caché
                    with self.cache_lock:
                        for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                            text_hash = self.get_text_hash(text)
                            self.embeddings_cache[model_name][text_hash] = embedding
                            #self.save_embedding_to_disk(model_name, text_hash, embedding)
                            embeddings[batch_indices[j]] = embedding
                            
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and self.gpu_available:
                        self.log(f"⚠️  GPU sin memoria, reduciendo batch size", "WARNING")
                        # Limpiar memoria GPU
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Reintentar con batch más pequeño
                        smaller_batch_size = batch_size // 2
                        if smaller_batch_size > 0:
                            return self.get_embeddings_batch(texts, model_name, smaller_batch_size)
                    else:
                        raise e
                        
        return np.array(embeddings)
    
    @phase_timer("Fase 3.2: Análisis Multi-Embeddings")
    def calculate_semantic_scores(self):
        """Calcula scores semánticos usando múltiples modelos EN PARALELO"""
        catalan_texts = self.df['catalan'].tolist()
        chinese_texts = self.df['chino'].tolist()
        
        model_similarities = {}
        
        self.log(f"  🧠 Procesando {len(self.models)} modelos de embeddings EN PARALELO...")
        
        def process_single_model(model_name):
            """Procesa un modelo individual con gestión agresiva de memoria"""
            try:
                # Determinar si el modelo está en GPU
                model = self.models[model_name]
                is_gpu_model = next(model.parameters()).is_cuda
                
                # Limpiar memoria GPU antes de procesar si es modelo GPU
                if is_gpu_model and self.gpu_available:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Procesar en lotes más pequeños si es GPU
                batch_size = 32 if is_gpu_model else 100
                
                # Obtener embeddings con batch size específico
                catalan_embeddings = self.get_embeddings_batch(catalan_texts, model_name, batch_size)
                chinese_embeddings = self.get_embeddings_batch(chinese_texts, model_name, batch_size)
                
                # Calcular similitudes
                similarities = []
                for i in range(len(catalan_embeddings)):
                    cat_emb = catalan_embeddings[i].reshape(1, -1)
                    chi_emb = chinese_embeddings[i].reshape(1, -1)
                    similarity = cosine_similarity(cat_emb, chi_emb)[0][0]
                    similarities.append(similarity)
                
                # Limpiar memoria después de procesar
                if is_gpu_model and self.gpu_available:
                    del catalan_embeddings
                    del chinese_embeddings
                    torch.cuda.empty_cache()
                
                return model_name, similarities
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    self.log(f"  ⚠️  Sin memoria para {model_name}, saltando", "WARNING")
                    return model_name, None
                else:
                    raise e
            except Exception as e:
                self.log(f"  ❌ Error en modelo {model_name}: {e}", "ERROR")
                return model_name, None
        
        # Ejecutar modelos en paralelo
        with ThreadPoolExecutor(max_workers=min(len(self.models), os.cpu_count())) as executor:
            future_to_model = {executor.submit(process_single_model, model_name): model_name 
                              for model_name in self.models.keys()}
            
            # Recolectar resultados conforme se completan
            for future in tqdm(as_completed(future_to_model), total=len(self.models), 
                              desc="  📊 Modelos completados", ncols=100):
                model_name, similarities = future.result()
                if similarities is not None:
                    model_similarities[model_name] = similarities
                    self.log(f"  ✅ {model_name}: similitud promedio = {np.mean(similarities):.3f}")
        
        # Calcular promedio entre modelos (resto igual)
        all_similarities = np.array(list(model_similarities.values()))
        average_similarities = np.mean(all_similarities, axis=0)
        std_similarities = np.std(all_similarities, axis=0)
        
        self.df['semantic_score'] = average_similarities * 25
        self.df['semantic_std'] = std_similarities
        
        self.log(f"  📊 Similitud promedio final: {np.mean(average_similarities):.3f}")
        self.log(f"  📊 Desviación estándar promedio entre modelos: {np.mean(std_similarities):.3f}")
        
        return model_similarities, average_similarities
    
    @phase_timer("Fase 3.3: Métricas de Calidad Básicas")
    def calculate_basic_metrics(self):
        """Calcula métricas básicas de calidad VECTORIZADO"""
        
        self.log(f"  📏 Calculando métricas básicas para {len(self.df)} registros...")
        
        # Métrica A: Coherencia de longitud VECTORIZADA
        catalan_lengths = self.df['catalan'].str.len()
        chinese_lengths = self.df['chino'].str.len()
        length_ratios = chinese_lengths / catalan_lengths.replace(0, 1)
        
        # Usar np.select en lugar de pd.cut
        conditions = [
            (length_ratios >= 0.5) & (length_ratios <= 0.8),
            ((length_ratios >= 0.3) & (length_ratios < 0.5)) | ((length_ratios > 0.8) & (length_ratios <= 1.0)),
            ((length_ratios >= 0.2) & (length_ratios < 0.3)) | ((length_ratios > 1.0) & (length_ratios <= 1.3)),
            (length_ratios < 0.2) | (length_ratios > 1.3)
        ]
        choices = [15, 11, 6, 0]
        self.df['length_score'] = np.select(conditions, choices, default=0)
        
        # Métrica B: Diversidad de caracteres VECTORIZADA
        unique_chars = self.df['chino'].apply(lambda x: len(set(str(x))) if pd.notna(x) else 0)
        total_chars = self.df['chino'].str.len().fillna(0)
        diversity_ratio = unique_chars / total_chars.replace(0, 1)
        
        conditions = [
            diversity_ratio > 0.7,
            diversity_ratio > 0.5,
            diversity_ratio > 0.3,
            diversity_ratio >= 0
        ]
        choices = [15, 11, 7, 0]
        self.df['diversity_score'] = np.select(conditions, choices, default=0)
        
        # Métrica C: Ratio puntuación VECTORIZADA
        punct_pattern = r'[，。！？；：、""''（）【】《》〈〉·—…]'
        punct_counts = self.df['chino'].str.count(punct_pattern).fillna(0)
        punct_ratio = punct_counts / self.df['chino'].str.len().replace(0, 1)
        
        conditions = [
            (punct_ratio >= 0.05) & (punct_ratio <= 0.15),
            ((punct_ratio >= 0.02) & (punct_ratio < 0.05)) | ((punct_ratio > 0.15) & (punct_ratio <= 0.25)),
            True
        ]
        choices = [10, 7, 0]
        self.df['punctuation_score'] = np.select(conditions, choices, default=0)
        
        # Métrica D: HSK VECTORIZADA (más compleja, pero factible)
        # Crear una función vectorizada para contar caracteres HSK
        hsk_chars_set = self.hsk_chars
        
        def count_hsk_chars_vectorized(text):
            if pd.isna(text) or len(text) == 0:
                return 0, 0
            chinese_chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
            if not chinese_chars:
                return 0, 0
            hsk_count = sum(1 for c in chinese_chars if c in hsk_chars_set)
            return hsk_count, len(chinese_chars)
        
        # Aplicar y desempaquetar
        hsk_results = self.df['chino'].apply(count_hsk_chars_vectorized)
        hsk_counts = np.array([r[0] for r in hsk_results])
        chinese_counts = np.array([r[1] for r in hsk_results])
        
        hsk_ratio = np.divide(hsk_counts, chinese_counts, out=np.zeros_like(hsk_counts, dtype=float), where=chinese_counts!=0)
        
        conditions = [
            hsk_ratio >= 0.7,
            hsk_ratio >= 0.5,
            hsk_ratio >= 0.3,
            hsk_ratio >= 0
        ]
        choices = [20, 15, 8, 0]
        self.df['hsk_score'] = np.select(conditions, choices, default=0)
        
        # Métrica E: Estructura VECTORIZADA
        text_lengths = self.df['chino'].str.len().fillna(0)
        consecutive_repeats = self.df['chino'].str.contains(r'(.)\1{3,}', na=False).astype(int)
        
        conditions = [
            consecutive_repeats > 0,
            (text_lengths >= 5) & (text_lengths <= 200),
            ((text_lengths >= 2) & (text_lengths < 5)) | ((text_lengths > 200) & (text_lengths <= 300)),
            True
        ]
        choices = [5, 15, 10, 5]
        self.df['structure_score'] = np.select(conditions, choices, default=5)
        
        # Logs igual que antes
        self.log("  📊 Métricas básicas calculadas:")
        self.log(f"    - Longitud promedio: {self.df['length_score'].mean():.1f}/15")
        self.log(f"    - Diversidad promedio: {self.df['diversity_score'].mean():.1f}/15")
        self.log(f"    - Puntuación promedio: {self.df['punctuation_score'].mean():.1f}/10")
        self.log(f"    - HSK promedio: {self.df['hsk_score'].mean():.1f}/20")
        self.log(f"    - Estructura promedio: {self.df['structure_score'].mean():.1f}/15")
    
    @phase_timer("Fase 3.4: Métricas Avanzadas de Traducción")
    def calculate_translation_quality_metrics(self):
        """Métricas avanzadas específicas para calidad de traducción"""
        
        # Métrica G: Consistencia de números/fechas (10 puntos extra)
        def check_number_consistency(cat_text, chi_text):
            if pd.isna(cat_text) or pd.isna(chi_text):
                return 5  # Neutral si faltan datos
                
            # Extraer números de ambos textos
            cat_numbers = set(re.findall(r'\d+', str(cat_text)))
            chi_numbers = set(re.findall(r'\d+', str(chi_text)))
            
            # Si no hay números, dar puntos neutrales
            if not cat_numbers and not chi_numbers:
                return 8
            
            # Si hay números, verificar consistencia
            if cat_numbers == chi_numbers:
                return 10  # Perfecta consistencia
            elif len(cat_numbers & chi_numbers) > 0:
                return 6   # Alguna consistencia
            else:
                return 0   # Sin consistencia (posible error)
        
        # Métrica H: Detección de traducciones automáticas pobres (10 puntos extra)
        def detect_machine_translation_artifacts(chi_text):
            if pd.isna(chi_text):
                return 5
                
            text = str(chi_text)
            penalty = 0
            
            # Patrones típicos de traducciones automáticas pobres
            if re.search(r'[a-zA-Z]{3,}', text):  # Palabras en inglés no traducidas
                penalty += 3
            if re.search(r'\s{2,}', text):  # Espacios múltiples anormales
                penalty += 2
            if re.search(r'(.)\1{4,}', text):  # Repeticiones excesivas
                penalty += 3
            if len(text) > 0 and text.count('的') / len(text) > 0.1:  # Uso excesivo de 的
                penalty += 2
                
            return max(0, 10 - penalty)
        
        # Métrica I: Análisis de entidades nombradas (10 puntos extra)
        def analyze_named_entities_consistency(cat_text, chi_text):
            if pd.isna(cat_text) or pd.isna(chi_text):
                return 5
                
            # Detectar nombres propios (palabras en mayúscula en catalán)
            cat_proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', str(cat_text))
            
            # Si no hay nombres propios, score neutral
            if not cat_proper_nouns:
                return 8
            
            # Verificar si aparecen transliterados o traducidos en chino
            chi_text_str = str(chi_text)
            found_entities = 0
            for noun in cat_proper_nouns:
                # Buscar el nombre tal cual (para nombres internacionales)
                if noun in chi_text_str:
                    found_entities += 1
            
            if len(cat_proper_nouns) == 0:
                return 8
            
            consistency_ratio = found_entities / len(cat_proper_nouns)
            if consistency_ratio >= 0.8:
                return 10
            elif consistency_ratio >= 0.5:
                return 7
            elif consistency_ratio >= 0.2:
                return 4
            else:
                return 2
        
        # Aplicar métricas avanzadas
        self.log("  🔬 Calculando métricas avanzadas de traducción...")
        
        consistency_scores = []
        machine_scores = []
        entity_scores = []
        
        # Barra de progreso para el procesamiento
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), 
                          desc="  🔍 Análisis avanzado", ncols=100):
            consistency_scores.append(check_number_consistency(row['catalan'], row['chino']))
            machine_scores.append(detect_machine_translation_artifacts(row['chino']))
            entity_scores.append(analyze_named_entities_consistency(row['catalan'], row['chino']))
        
        self.df['number_consistency_score'] = consistency_scores
        self.df['machine_quality_score'] = machine_scores
        self.df['entity_consistency_score'] = entity_scores
        
        self.log(f"    - Consistencia números promedio: {np.mean(consistency_scores):.1f}/10")
        self.log(f"    - Calidad no-automática promedio: {np.mean(machine_scores):.1f}/10")
        self.log(f"    - Consistencia entidades promedio: {np.mean(entity_scores):.1f}/10")
        
        # Métrica H: Detección de traducciones automáticas pobres (10 puntos extra)
        def detect_machine_translation_artifacts(chi_text):
            if pd.isna(chi_text):
                return 5
                
            text = str(chi_text)
            penalty = 0
            
            # Patrones típicos de traducciones automáticas pobres
            if re.search(r'[a-zA-Z]{3,}', text):  # Palabras en inglés no traducidas
                penalty += 3
            if re.search(r'\s{2,}', text):  # Espacios múltiples anormales
                penalty += 2
            if re.search(r'(.)\1{4,}', text):  # Repeticiones excesivas
                penalty += 3
            if len(text) > 0 and text.count('的') / len(text) > 0.1:  # Uso excesivo de 的
                penalty += 2
                
            return max(0, 10 - penalty)
        
        # Métrica I: Análisis de entidades nombradas (10 puntos extra)
        def analyze_named_entities_consistency(cat_text, chi_text):
            if pd.isna(cat_text) or pd.isna(chi_text):
                return 5
                
            # Detectar nombres propios (palabras en mayúscula en catalán)
            cat_proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', str(cat_text))
            
            # Si no hay nombres propios, score neutral
            if not cat_proper_nouns:
                return 8
            
            # Verificar si aparecen transliterados o traducidos en chino
            chi_text_str = str(chi_text)
            found_entities = 0
            for noun in cat_proper_nouns:
                # Buscar el nombre tal cual (para nombres internacionales)
                if noun in chi_text_str:
                    found_entities += 1
            
            if len(cat_proper_nouns) == 0:
                return 8
            
            consistency_ratio = found_entities / len(cat_proper_nouns)
            if consistency_ratio >= 0.8:
                return 10
            elif consistency_ratio >= 0.5:
                return 7
            elif consistency_ratio >= 0.2:
                return 4
            else:
                return 2
        
        # Aplicar métricas avanzadas
        self.log("  🔬 Calculando métricas avanzadas de traducción...")
        
        consistency_scores = []
        machine_scores = []
        entity_scores = []
        
        # Barra de progreso para el procesamiento
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), 
                          desc="  🔍 Análisis avanzado", ncols=100):
            consistency_scores.append(check_number_consistency(row['catalan'], row['chino']))
            machine_scores.append(detect_machine_translation_artifacts(row['chino']))
            entity_scores.append(analyze_named_entities_consistency(row['catalan'], row['chino']))
        
        self.df['number_consistency_score'] = consistency_scores
        self.df['machine_quality_score'] = machine_scores
        self.df['entity_consistency_score'] = entity_scores
        
        self.log(f"    - Consistencia números promedio: {np.mean(consistency_scores):.1f}/10")
        self.log(f"    - Calidad no-automática promedio: {np.mean(machine_scores):.1f}/10")
        self.log(f"    - Consistencia entidades promedio: {np.mean(entity_scores):.1f}/10")

    
    @phase_timer("Fase 4: Análisis Completo de Eliminación")
    def find_optimal_elimination_percentage(self, min_percentage: float = 0.0, max_percentage: float = 10.0, 
                                          step: float = 0.5, improvement_threshold: float = 0.5):
        """
        Calcula TODOS los porcentajes de eliminación y exporta análisis completo a Excel.
        """
        
        self.log(f"🔍 Calculando TODOS los porcentajes de eliminación...")
        self.log(f"   Rango: {min_percentage}% - {max_percentage}% (pasos de {step}%)")
        self.log(f"   Exportando análisis completo a Excel...")
        
        # Generar lista de porcentajes a probar
        percentages_to_test = []
        current_pct = min_percentage
        while current_pct <= max_percentage:
            percentages_to_test.append(current_pct)
            current_pct += step
        
        analysis_results = []
        
        # Barra de progreso para análisis de porcentajes
        for pct in tqdm(percentages_to_test, desc="  📈 Analizando percentiles", ncols=100):
            # Calcular threshold para este porcentaje
            threshold = np.percentile(self.df['total_score'], pct)
            
            # Separar peores y mejores
            worst_records = self.df[self.df['total_score'] <= threshold]
            remaining_records = self.df[self.df['total_score'] > threshold]
            
            # Calcular métricas
            if len(remaining_records) == 0:
                continue  # Skip si eliminaría todo
                
            mean_worst = worst_records['total_score'].mean() if len(worst_records) > 0 else 0
            mean_remaining = remaining_records['total_score'].mean()
            mean_all = self.df['total_score'].mean()
            
            improvement = mean_remaining - mean_all
            records_eliminated = len(worst_records)
            records_remaining = len(remaining_records)
            
            # Calcular métricas adicionales para análisis
            improvement_per_record = improvement / records_eliminated if records_eliminated > 0 else 0
            efficiency_ratio = improvement / records_eliminated * 100 if records_eliminated > 0 else 0
            retention_percentage = (records_remaining / len(self.df)) * 100
            
            analysis_results.append({
                'Porcentaje_Eliminacion': pct,
                'Registros_Eliminados': records_eliminated,
                'Registros_Restantes': records_remaining,
                'Porcentaje_Retencion': retention_percentage,
                'Threshold_Score': threshold,
                'Score_Promedio_Original': mean_all,
                'Score_Promedio_Eliminados': mean_worst,
                'Score_Promedio_Final': mean_remaining,
                'Mejora_Absoluta': improvement,
                'Mejora_Por_Registro': improvement_per_record,
                'Ratio_Eficiencia': efficiency_ratio
            })
        
        # Calcular mejoras marginales
        for i in range(1, len(analysis_results)):
            prev_improvement = analysis_results[i-1]['Mejora_Absoluta']
            curr_improvement = analysis_results[i]['Mejora_Absoluta']
            marginal_improvement = curr_improvement - prev_improvement
            analysis_results[i]['Mejora_Marginal'] = marginal_improvement
            
            # Calcular ratio beneficio/costo
            prev_eliminated = analysis_results[i-1]['Registros_Eliminados']
            curr_eliminated = analysis_results[i]['Registros_Eliminados']
            additional_eliminated = curr_eliminated - prev_eliminated
            analysis_results[i]['Beneficio_Por_Eliminacion_Adicional'] = marginal_improvement / additional_eliminated if additional_eliminated > 0 else 0
        
        # Primer elemento no tiene mejora marginal
        if analysis_results:
            analysis_results[0]['Mejora_Marginal'] = analysis_results[0]['Mejora_Absoluta']
            analysis_results[0]['Beneficio_Por_Eliminacion_Adicional'] = analysis_results[0]['Mejora_Por_Registro']
        
        # Convertir a DataFrame para análisis
        df_analysis = pd.DataFrame(analysis_results)
        
        # Exportar a Excel en carpeta p1_results
        results_dir = "p1_results"
        os.makedirs(results_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(self.csv_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Microsegundos para unicidad
        excel_file = os.path.join(results_dir, f"{base_name}_analisis_eliminacion_{timestamp}.xlsx")
        
        try:
            if EXCEL_AVAILABLE:
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    # Hoja principal con análisis completo
                    df_analysis.to_excel(writer, sheet_name='Analisis_Completo', index=False)
                    
                    # Hoja con métricas por scoring
                    if hasattr(self, 'df') and not self.df.empty:
                        score_stats = pd.DataFrame({
                            'Metrica': ['semantic_score', 'length_score', 'diversity_score', 'punctuation_score', 
                                       'hsk_score', 'structure_score', 'number_consistency_score', 
                                       'machine_quality_score', 'entity_consistency_score', 'total_score'],
                            'Promedio': [self.df[col].mean() for col in ['semantic_score', 'length_score', 'diversity_score', 
                                        'punctuation_score', 'hsk_score', 'structure_score', 'number_consistency_score',
                                        'machine_quality_score', 'entity_consistency_score', 'total_score']],
                            'Desviacion': [self.df[col].std() for col in ['semantic_score', 'length_score', 'diversity_score', 
                                          'punctuation_score', 'hsk_score', 'structure_score', 'number_consistency_score',
                                          'machine_quality_score', 'entity_consistency_score', 'total_score']],
                            'Minimo': [self.df[col].min() for col in ['semantic_score', 'length_score', 'diversity_score', 
                                      'punctuation_score', 'hsk_score', 'structure_score', 'number_consistency_score',
                                      'machine_quality_score', 'entity_consistency_score', 'total_score']],
                            'Maximo': [self.df[col].max() for col in ['semantic_score', 'length_score', 'diversity_score', 
                                      'punctuation_score', 'hsk_score', 'structure_score', 'number_consistency_score',
                                      'machine_quality_score', 'entity_consistency_score', 'total_score']]
                        })
                        score_stats.to_excel(writer, sheet_name='Estadisticas_Scores', index=False)
                    
                    # Hoja con top/bottom registros para análisis manual
                    if hasattr(self, 'df') and not self.df.empty:
                        top_bottom = pd.concat([
                            self.df.nlargest(50, 'total_score')[['linea', 'catalan', 'chino', 'total_score', 'semantic_score']],
                            self.df.nsmallest(50, 'total_score')[['linea', 'catalan', 'chino', 'total_score', 'semantic_score']]
                        ])
                        top_bottom.to_excel(writer, sheet_name='Top_Bottom_50', index=False)
                
                self.log(f"📊 Análisis completo exportado a Excel: {excel_file}")
            else:
                # Fallback a CSV si no hay openpyxl
                raise ImportError("openpyxl no disponible")
                
        except Exception as e:
            self.log(f"⚠️  Error creando Excel (usando CSV): {e}")
            csv_file = os.path.join(results_dir, f"{base_name}_analisis_eliminacion_{timestamp}.csv")
            df_analysis.to_csv(csv_file, index=False)
            excel_file = csv_file
            self.log(f"📊 Análisis exportado a CSV: {csv_file}")
            self.log(f"💡 Para Excel instala: pip install openpyxl")
        
        # CALCULAR suggested_index ANTES de usarlo
        suggested_index = 0
        max_efficiency = 0
        
        for i, result in enumerate(analysis_results):
            # Buscar el punto con mejor ratio eficiencia que mantenga buena retención
            if (result['Ratio_Eficiencia'] > max_efficiency and 
                result['Porcentaje_Retencion'] >= 85 and  # Mantener al menos 85%
                result['Mejora_Absoluta'] > 0.5):  # Mejora mínima significativa
                max_efficiency = result['Ratio_Eficiencia']
                suggested_index = i
        
        suggested_percentage = analysis_results[suggested_index]['Porcentaje_Eliminacion'] if analysis_results else 5.0
        
        # Mostrar tabla completa en consola con formato mejorado
        self.log(f"\n📋 ANÁLISIS COMPLETO DE ELIMINACIÓN:")
        self.log("="*130)
        self.log(f"{'%':>4} | {'Elim':>5} | {'Rest':>5} | {'Ret%':>5} | {'Score_Orig':>10} | {'Score_Elim':>10} | {'Score_Final':>11} | {'Mejora':>8} | {'Mej.Marg':>9} | {'Efic':>6} | {'Recomendación':<12}")
        self.log("-"*130)
        
        for i, result in enumerate(analysis_results):
            mejora_marginal = result.get('Mejora_Marginal', 0)
            
            # Añadir recomendaciones visuales
            recommendation = ""
            if i == suggested_index:
                recommendation = "🤖 SUGERIDO"
            elif mejora_marginal > 0.15:
                recommendation = "📈 ALTA MEJORA"
            elif mejora_marginal > 0.05:
                recommendation = "✅ BUENA"
            elif mejora_marginal < 0.02:
                recommendation = "⚪ MESETA"
            
            self.log(f"{result['Porcentaje_Eliminacion']:4.1f} | "
                    f"{result['Registros_Eliminados']:5d} | "
                    f"{result['Registros_Restantes']:5d} | "
                    f"{result['Porcentaje_Retencion']:5.1f} | "
                    f"{result['Score_Promedio_Original']:10.2f} | "
                    f"{result['Score_Promedio_Eliminados']:10.2f} | "
                    f"{result['Score_Promedio_Final']:11.2f} | "
                    f"{result['Mejora_Absoluta']:8.2f} | "
                    f"{mejora_marginal:9.2f} | "
                    f"{result['Ratio_Eficiencia']:6.2f} | "
                    f"{recommendation:<12}")
        
        self.log("="*130)
        self.log("📊 LEYENDA:")
        self.log("  - Elim/Rest: Registros eliminados/restantes")
        self.log("  - Ret%: Porcentaje de retención de datos")
        self.log("  - Mejora: Incremento en score promedio")
        self.log("  - Mej.Marg: Mejora adicional vs punto anterior")
        self.log("  - Efic: Ratio eficiencia (mejora por registro eliminado)")
        self.log("  - 🤖 SUGERIDO: Recomendación automática del algoritmo")
        self.log("  - 📈 ALTA MEJORA: Mejora marginal > 0.15 puntos")
        self.log("  - ⚪ MESETA: Mejora marginal < 0.02 (punto de saturación)")
        
        # Añadir información final sobre la sugerencia
        self.log(f"\n🤖 SUGERENCIA AUTOMÁTICA: {suggested_percentage}% (ratio eficiencia: {max_efficiency:.2f})")
        self.log(f"📊 Archivo Excel generado: {excel_file}")
        self.log(f"👤 DECISIÓN MANUAL: Revisa la tabla y el Excel para elegir el punto óptimo")
        
        return df_analysis, suggested_percentage, excel_file
    
    def apply_automatic_suggestion(self, suggested_percentage: float):
        """Aplica automáticamente la sugerencia del algoritmo sin interacción del usuario"""
        
        self.log(f"\n🤖 Aplicando sugerencia automática: {suggested_percentage}%")
        self.log(f"📊 (Calculada mediante análisis de eficiencia y retención)")
        
        # Usar la misma lógica que apply_manual_elimination pero sin input del usuario
        chosen_percentage = suggested_percentage
        
        # Calcular threshold para el porcentaje sugerido
        threshold = np.percentile(self.df['total_score'], chosen_percentage)
        
        initial_count = len(self.df)
        
        # Aplicar filtrado con threshold sugerido
        worst_records = self.df[self.df['total_score'] <= threshold]
        self.df_filtered = self.df[self.df['total_score'] > threshold].copy()
        
        # Estadísticas finales
        mean_all = self.df['total_score'].mean()
        mean_worst = worst_records['total_score'].mean() if len(worst_records) > 0 else 0
        mean_remaining = self.df_filtered['total_score'].mean()
        
        # Log de eliminados con más detalle
        for _, row in worst_records.iterrows():
            self.correction_log.append({
                'linea': row['linea'],
                'tipo': 'eliminacion_score_automatico',
                'razon': f'score_total: {row["total_score"]:.2f} (threshold_automatico: {threshold:.2f})',
                'porcentaje_automatico': chosen_percentage,
                'detalles_base': f'sem:{row["semantic_score"]:.1f} len:{row["length_score"]:.1f} div:{row["diversity_score"]:.1f} hsk:{row["hsk_score"]:.1f}',
                'detalles_avanzados': f'num:{row["number_consistency_score"]:.1f} maq:{row["machine_quality_score"]:.1f} ent:{row["entity_consistency_score"]:.1f}',
                'texto_catalan': row['catalan'][:50] + '...' if len(row['catalan']) > 50 else row['catalan'],
                'texto_chino': row['chino'][:50] + '...' if len(row['chino']) > 50 else row['chino']
            })
        
        self.log(f"  📊 Eliminación automática aplicada:")
        self.log(f"    - Porcentaje sugerido: {chosen_percentage}%")
        self.log(f"    - Score promedio original: {mean_all:.2f}")
        self.log(f"    - Score promedio eliminados: {mean_worst:.2f}")
        self.log(f"    - Score promedio final: {mean_remaining:.2f}")
        self.log(f"    - Mejora lograda: +{mean_remaining - mean_all:.2f} puntos")
        self.log(f"  🗑️  Eliminados: {len(worst_records)} registros ({len(worst_records)/initial_count*100:.1f}%)")
        self.log(f"  ✅ Restantes: {len(self.df_filtered)} registros")
        
        # Guardar métricas
        self.metrics_log.update({
            'porcentaje_eliminacion_automatico': chosen_percentage,
            'threshold_automatico': threshold,
            'registros_originales': initial_count,
            'registros_eliminados': len(worst_records),
            'registros_finales': len(self.df_filtered),
            'score_promedio_original': mean_all,
            'score_promedio_eliminados': mean_worst,
            'score_promedio_restantes': mean_remaining,
            'mejora_relativa': mean_remaining - mean_all,
            'score_maximo_teorico': 130,
            'optimizacion_automatica': True
        })
        
        return len(worst_records), mean_worst, mean_remaining

    def apply_optimal_elimination(self, analysis_df: pd.DataFrame, suggested_percentage: float, excel_file: str):
        """Permite elección interactiva en tiempo real del porcentaje de eliminación"""
        
        self.log(f"\n🤖 Sugerencia automática: {suggested_percentage}%")
        self.log(f"📊 Análisis completo disponible en: {excel_file}")
        
        # Solicitar decisión interactiva del usuario
        print(f"\n" + "="*80)
        print("🎯 DECISIÓN INTERACTIVA: ¿Qué porcentaje de eliminación quieres usar?")
        print("="*80)
        print(f"📊 Opciones disponibles: 0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%, 3.5%, 4.0%, ... hasta 10.0%")
        print(f"🤖 Sugerencia automática: {suggested_percentage}%")
        print(f"📁 Consulta el archivo Excel para más detalles: {excel_file}")
        
        while True:
            try:
                user_input = input("\n👤 Introduce el porcentaje que quieres usar (ej: 2.5) o 'auto' para usar la sugerencia: ").strip()
                
                if user_input.lower() == 'auto':
                    chosen_percentage = suggested_percentage
                    print(f"✅ Usando sugerencia automática: {chosen_percentage}%")
                    break
                    
                chosen_percentage = float(user_input)
                
                # Validar que el porcentaje esté en el rango analizado
                if not 0.5 <= chosen_percentage <= 10.0:
                    print(f"❌ El porcentaje debe estar entre 0.5 y 10.0. Recibido: {chosen_percentage}")
                    continue
                    
                # Verificar que el porcentaje esté en los valores analizados (múltiplos de 0.5)
                if chosen_percentage % 0.5 != 0:
                    print(f"⚠️  El porcentaje debe ser múltiplo de 0.5 (ej: 1.5, 2.0, 2.5)")
                    # Ofrecer el más cercano
                    closest = round(chosen_percentage * 2) / 2
                    use_closest = input(f"¿Quieres usar {closest}% (el más cercano)? (s/n): ").strip().lower()
                    if use_closest in ['s', 'si', 'y', 'yes']:
                        chosen_percentage = closest
                        break
                    else:
                        continue
                else:
                    break
                    
            except ValueError:
                print("❌ Por favor introduce un número válido (ej: 2.5) o 'auto'")
                continue
            except KeyboardInterrupt:
                print(f"\n⚠️  Proceso interrumpido. Usando sugerencia automática: {suggested_percentage}%")
                chosen_percentage = suggested_percentage
                break
        
        print(f"\n🚀 Aplicando eliminación elegida: {chosen_percentage}%")
        return self.apply_manual_elimination(chosen_percentage)

    def calculate_final_scores_and_filter_fixed(self):
        """Versión legacy con porcentaje fijo (raramente usada)"""
        
        # Calcular score total con métricas expandidas
        self.df['base_score'] = (
            self.df['length_score'] * 0.15 +           # 15 puntos máx
            self.df['diversity_score'] * 0.15 +        # 15 puntos máx
            self.df['punctuation_score'] * 0.10 +      # 10 puntos máx
            self.df['hsk_score'] * 0.20 +              # 20 puntos máx
            self.df['structure_score'] * 0.15 +        # 15 puntos máx
            self.df['semantic_score'] * 0.25           # 25 puntos máx (de embeddings)
        )
        
        # Añadir métricas avanzadas (30 puntos adicionales)
        self.df['advanced_score'] = (
            self.df['number_consistency_score'] +      # 10 puntos máx
            self.df['machine_quality_score'] +         # 10 puntos máx  
            self.df['entity_consistency_score']        # 10 puntos máx
        )
        
        # Score total combinado
        self.df['total_score'] = self.df['base_score'] + self.df['advanced_score']
        
        # Aplicar penalizaciones adicionales
        self.apply_advanced_penalties()
        
        initial_count = len(self.df)
        
        # Calcular threshold basado en el porcentaje configurado
        threshold = np.percentile(self.df['total_score'], self.elimination_percentage)
        worst_percent = self.df[self.df['total_score'] <= threshold]
        
        # Estadísticas antes del filtrado
        mean_all = self.df['total_score'].mean()
        mean_worst = worst_percent['total_score'].mean()
        
        # Filtrar el porcentaje peor
        self.df_filtered = self.df[self.df['total_score'] > threshold].copy()
        mean_remaining = self.df_filtered['total_score'].mean()
        
        # Log de eliminados con más detalle
        for _, row in worst_percent.iterrows():
            self.correction_log.append({
                'linea': row['linea'],
                'tipo': 'eliminacion_score_fijo_legacy',
                'razon': f'score_total: {row["total_score"]:.2f} (threshold_fijo: {threshold:.2f})',
                'detalles_base': f'sem:{row["semantic_score"]:.1f} len:{row["length_score"]:.1f} div:{row["diversity_score"]:.1f} hsk:{row["hsk_score"]:.1f}',
                'detalles_avanzados': f'num:{row["number_consistency_score"]:.1f} maq:{row["machine_quality_score"]:.1f} ent:{row["entity_consistency_score"]:.1f}',
                'texto_catalan': row['catalan'][:50] + '...' if len(row['catalan']) > 50 else row['catalan'],
                'texto_chino': row['chino'][:50] + '...' if len(row['chino']) > 50 else row['chino']
            })
        
        self.log(f"  📊 Scores calculados (máximo 130 puntos):")
        self.log(f"    - Score base promedio: {self.df['base_score'].mean():.2f}/100")
        self.log(f"    - Score avanzado promedio: {self.df['advanced_score'].mean():.2f}/30")
        self.log(f"    - Score total promedio original: {mean_all:.2f}/130")
        self.log(f"    - Score promedio {self.elimination_percentage}% peor: {mean_worst:.2f}")
        self.log(f"    - Score promedio {100-self.elimination_percentage}% restante: {mean_remaining:.2f}")
        self.log(f"    - Mejora relativa: +{mean_remaining - mean_worst:.2f} puntos")
        self.log(f"  🗑️  Eliminados: {len(worst_percent)} registros ({len(worst_percent)/initial_count*100:.1f}%)")
        self.log(f"  ✅ Restantes: {len(self.df_filtered)} registros")
        
        # Guardar métricas
        self.metrics_log.update({
            'porcentaje_eliminacion': self.elimination_percentage,
            'registros_originales': initial_count,
            'registros_eliminados': len(worst_percent),
            'registros_finales': len(self.df_filtered),
            'score_promedio_original': mean_all,
            'score_promedio_eliminados': mean_worst,
            'score_promedio_restantes': mean_remaining,
            'mejora_relativa': mean_remaining - mean_worst,
            'threshold_eliminacion': threshold,
            'score_maximo_teorico': 130,
            'modo_legacy_fijo': True
        })
        
        return len(worst_percent), mean_worst, mean_remaining
    
    @phase_timer("Fase 4: Scoring y Optimización")
    def calculate_final_scores_and_filter(self, use_interactive: bool = True):
        """Calcula scores finales y aplica eliminación (interactiva, automática o fija)"""
        
        # Primero calcular todos los scores
        self.df['base_score'] = (
            self.df['length_score'] * 0.15 +           # 15 puntos máx
            self.df['diversity_score'] * 0.15 +        # 15 puntos máx
            self.df['punctuation_score'] * 0.10 +      # 10 puntos máx
            self.df['hsk_score'] * 0.15 +              # 20 puntos máx
            self.df['structure_score'] * 0.15 +        # 15 puntos máx
            self.df['semantic_score'] * 0.30           # 25 puntos máx (de embeddings)
        )
        
        # Añadir métricas avanzadas (30 puntos adicionales)
        self.df['advanced_score'] = (
            self.df['number_consistency_score'] +      # 10 puntos máx
            self.df['machine_quality_score'] +         # 10 puntos máx  
            self.df['entity_consistency_score']        # 10 puntos máx
        )
        
        # Score total combinado
        self.df['total_score'] = self.df['base_score'] + self.df['advanced_score']
        
        # Aplicar penalizaciones adicionales
        self.apply_advanced_penalties()
        
        # Decidir estrategia de eliminación
        if use_interactive:
            self.log("🎯 Iniciando análisis completo para decisión interactiva...")
            analysis_df, suggested_percentage, excel_file = self.find_optimal_elimination_percentage(
                improvement_threshold=self.improvement_threshold
            )
            # Guardar referencia al archivo Excel para los reportes
            self.excel_analysis_file = excel_file
            return self.apply_optimal_elimination(analysis_df, suggested_percentage, excel_file)
        else:
            self.log("🤖 Iniciando análisis completo para sugerencia automática...")
            analysis_df, suggested_percentage, excel_file = self.find_optimal_elimination_percentage(
                improvement_threshold=self.improvement_threshold
            )
            # Guardar referencia al archivo Excel para los reportes
            self.excel_analysis_file = excel_file
            return self.apply_automatic_suggestion(suggested_percentage)
    
    def apply_advanced_penalties(self):
        """Aplica penalizaciones avanzadas VECTORIZADO"""
        
        # Inicializar todas las penalizaciones a 0
        penalties = np.zeros(len(self.df))
        
        # Penalización por baja similitud semántica
        low_semantic_mask = self.df['semantic_score'] < 5
        penalties[low_semantic_mask] += 20
        
        # Penalización por longitudes extremas
        zero_length_mask = self.df['length_score'] == 0
        penalties[zero_length_mask] += 10
        
        # Penalización por falta de diversidad
        low_diversity_mask = self.df['diversity_score'] <= 3
        penalties[low_diversity_mask] += 15
        
        # Penalización por artefactos de traducción
        low_machine_mask = self.df['machine_quality_score'] <= 3
        penalties[low_machine_mask] += 10
        
        # Penalización acumulativa por múltiples problemas
        low_scores_count = (
            (self.df['length_score'] < 8).astype(int) +
            (self.df['diversity_score'] < 8).astype(int) +
            (self.df['punctuation_score'] < 5).astype(int) +
            (self.df['hsk_score'] < 10).astype(int) +
            (self.df['structure_score'] < 8).astype(int) +
            (self.df['semantic_score'] < 15).astype(int)
        )
        
        penalties[low_scores_count >= 4] += 25
        penalties[(low_scores_count >= 3) & (low_scores_count < 4)] += 15
        penalties[(low_scores_count >= 2) & (low_scores_count < 3)] += 8
        
        # Aplicar penalizaciones
        self.df['penalties'] = penalties
        self.df['total_score'] = self.df['total_score'] - penalties
        
        # Estadísticas
        penalized_count = (penalties > 0).sum()
        self.log(f"  ⚠️  Penalizaciones aplicadas:")
        self.log(f"    - Registros penalizados: {penalized_count}/{len(penalties)}")
        self.log(f"    - Penalización promedio: {penalties.mean():.1f} puntos")
        self.log(f"    - Penalización máxima: {penalties.max():.1f} puntos")

    @phase_timer("Fase 5: Generación de Reportes")
    def generate_reports(self):
        """Genera reportes detallados y archivos de salida con acumulación en Excel"""
        
        # Crear carpeta de resultados
        results_dir = "p1_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Crear nombre base para archivos con timestamp único
        base_name = os.path.splitext(os.path.basename(self.csv_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Incluir microsegundos para unicidad
        
        # 1. MUESTRAS LIMPIAS ACUMULATIVAS EN EXCEL (directorio principal)
        clean_samples_file = f"{base_name}_muestras_limpias.xlsx"
        self._save_accumulated_clean_samples(clean_samples_file)
        
        # 2. Reportes de métricas en carpeta p1_results (siempre únicos)
        output_files = {}
        
        # CSV de respaldo de esta ejecución
        backup_csv = os.path.join(results_dir, f"{base_name}_backup_{timestamp}.csv")
        self.df_filtered[['linea', 'catalan', 'chino']].to_csv(backup_csv, index=False, encoding='utf-8')
        output_files['backup_csv'] = backup_csv
        self.log(f"  📁 Backup CSV guardado: {backup_csv}")
        
        # Log de correcciones
        corrections_csv = os.path.join(results_dir, f"{base_name}_log_correcciones_{timestamp}.csv")
        if self.correction_log:
            corrections_df = pd.DataFrame(self.correction_log)
            corrections_df.to_csv(corrections_csv, index=False, encoding='utf-8')
            output_files['log_correcciones'] = corrections_csv
            self.log(f"  📁 Log de correcciones guardado: {corrections_csv}")
        
        # Reporte de duplicados
        duplicates_file = os.path.join(results_dir, f"{base_name}_duplicados_{timestamp}.txt")
        with open(duplicates_file, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE DUPLICADOS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Archivo fuente: {self.csv_file}\n\n")
            
            if self.duplicates_log:
                for dup_type, groups in self.duplicates_log.items():
                    f.write(f"\n{dup_type.upper()}:\n")
                    f.write("-" * 30 + "\n")
                    for i, (text, lines) in enumerate(groups.items(), 1):
                        f.write(f"{i}. Líneas {lines}: {text[:100]}{'...' if len(text) > 100 else ''}\n")
            else:
                f.write("No se encontraron duplicados.\n")
        
        output_files['reporte_duplicados'] = duplicates_file
        self.log(f"  📁 Reporte de duplicados guardado: {duplicates_file}")
        
        # Métricas finales detalladas
        total_time = time.time() - self.start_time
        self.metrics_log['tiempo_total'] = total_time
        
        metrics_file = os.path.join(results_dir, f"{base_name}_metricas_{timestamp}.txt")
        self._generate_detailed_metrics_report(metrics_file, total_time)
        output_files['metricas'] = metrics_file
        self.log(f"  📁 Métricas detalladas guardadas: {metrics_file}")
        
        # Excel con análisis completo de scores (si existe - ya está en p1_results)
        if hasattr(self, 'excel_analysis_file'):
            output_files['analisis_excel'] = self.excel_analysis_file
            self.log(f"  📁 Análisis Excel disponible: {self.excel_analysis_file}")
        
        # Resumen de esta ejecución
        summary_file = os.path.join(results_dir, f"{base_name}_resumen_{timestamp}.json")
        self._save_execution_summary(summary_file, output_files)
        output_files['resumen_json'] = summary_file
        
        # Estadísticas consolidadas
        self._update_consolidated_stats(base_name, results_dir)
        
        return output_files

    def _save_accumulated_clean_samples(self, excel_file):
        """Guarda las muestras limpias acumulando con ejecuciones anteriores"""
        
        try:
            # Preparar datos nuevos (solo columnas esenciales)
            new_samples = self.df_filtered[['linea', 'catalan', 'chino']].copy()
            
            # Verificar si existe archivo Excel previo
            if os.path.exists(excel_file):
                try:
                    # Cargar datos existentes
                    existing_samples = pd.read_excel(excel_file, sheet_name='Muestras_Limpias')
                    self.log(f"  📊 Cargadas {len(existing_samples)} muestras existentes")
                    
                    # Combinar y eliminar duplicados (basado en línea)
                    combined_samples = pd.concat([existing_samples, new_samples], ignore_index=True)
                    combined_samples = combined_samples.drop_duplicates(subset=['linea'], keep='last')
                    
                    # Estadísticas de acumulación
                    total_samples = len(combined_samples)
                    new_samples_added = len(combined_samples) - len(existing_samples)
                    
                    self.log(f"  ✅ Muestras acumuladas: {total_samples} total ({new_samples_added} nuevas)")
                    
                except Exception as e:
                    self.log(f"  ⚠️  Error leyendo Excel existente: {e}. Creando nuevo archivo.")
                    combined_samples = new_samples
                    total_samples = len(new_samples)
                    new_samples_added = len(new_samples)
            else:
                combined_samples = new_samples
                total_samples = len(new_samples)
                new_samples_added = len(new_samples)
                self.log(f"  📊 Creando nuevo archivo Excel con {total_samples} muestras")
            
            # Guardar en Excel con múltiples hojas
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Hoja principal con muestras limpias
                combined_samples.to_excel(writer, sheet_name='Muestras_Limpias', index=False)
                
                # Hoja de información de esta ejecución
                execution_info = pd.DataFrame({
                    'Parametro': [
                        'Fecha_Ejecucion', 'Archivo_Fuente', 'Registros_Procesados_Esta_Ejecucion',
                        'Registros_Limpios_Esta_Ejecucion', 'Modo_Interactivo', 'Tipo_Eliminacion',
                        'Porcentaje_Eliminacion', 'Score_Promedio_Final', 'Modelos_Embeddings_Usados'
                    ],
                    'Valor': [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        self.csv_file,
                        self.metrics_log.get('registros_originales', 0),
                        len(self.df_filtered),
                        'SÍ' if self.interactive_mode else 'NO',
                        'INTERACTIVA' if self.metrics_log.get('optimizacion_manual', False) 
                        else 'AUTOMÁTICA' if self.metrics_log.get('optimizacion_automatica', False)
                        else 'FIJA (LEGACY)',
                        self.metrics_log.get('porcentaje_eliminacion_manual', 
                                           self.metrics_log.get('porcentaje_eliminacion_automatico',
                                                              self.metrics_log.get('porcentaje_eliminacion', 0))),
                        f"{self.metrics_log.get('score_promedio_restantes', 0):.2f}/130",
                        ', '.join(list(self.models.keys()))
                    ]
                })
                execution_info.to_excel(writer, sheet_name='Info_Ultima_Ejecucion', index=False)
                
                # Hoja de estadísticas acumulativas
                stats_info = pd.DataFrame({
                    'Estadistica': [
                        'Total_Muestras_Acumuladas', 'Nuevas_Muestras_Esta_Ejecucion',
                        'Ultima_Actualizacion', 'Numero_Ejecuciones_Detectadas'
                    ],
                    'Valor': [
                        total_samples, new_samples_added,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        self._count_executions_in_results()
                    ]
                })
                stats_info.to_excel(writer, sheet_name='Estadisticas_Acumuladas', index=False)
            
            self.log(f"  📊 Excel de muestras limpias actualizado: {excel_file}")
            self.log(f"    - Total acumulado: {total_samples} muestras")
            self.log(f"    - Nuevas esta ejecución: {new_samples_added} muestras")
            
            # Actualizar métricas para incluir información acumulativa
            self.metrics_log.update({
                'muestras_acumuladas_total': total_samples,
                'muestras_nuevas_esta_ejecucion': new_samples_added,
                'archivo_excel_muestras': excel_file
            })
            
        except ImportError:
            self.log(f"  ❌ openpyxl no disponible. No se puede crear Excel de muestras acumuladas.")
            # Fallback a CSV acumulativo
            csv_file = excel_file.replace('.xlsx', '.csv')
            self._save_accumulated_clean_samples_csv(csv_file)
        except Exception as e:
            self.log(f"  ❌ Error guardando muestras acumuladas: {e}")

    def _save_accumulated_clean_samples_csv(self, csv_file):
        """Fallback: guardar muestras limpias en CSV acumulativo"""
        new_samples = self.df_filtered[['linea', 'catalan', 'chino']].copy()
        
        if os.path.exists(csv_file):
            existing_samples = pd.read_csv(csv_file)
            combined_samples = pd.concat([existing_samples, new_samples], ignore_index=True)
            combined_samples = combined_samples.drop_duplicates(subset=['linea'], keep='last')
            new_samples_added = len(combined_samples) - len(existing_samples)
        else:
            combined_samples = new_samples
            new_samples_added = len(new_samples)
        
        combined_samples.to_csv(csv_file, index=False, encoding='utf-8')
        self.log(f"  📊 CSV de muestras limpias actualizado: {csv_file}")
        self.log(f"    - Total: {len(combined_samples)}, Nuevas: {new_samples_added}")

    def _generate_detailed_metrics_report(self, metrics_file, total_time):
        """Genera reporte detallado de métricas"""
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write("MÉTRICAS FINALES DE CORRECCIÓN CHINA - EJECUCIÓN DETALLADA\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("INFORMACIÓN DE EJECUCIÓN:\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Archivo fuente: {self.csv_file}\n")
            f.write(f"Tiempo total ejecución: {total_time:.2f} segundos ({total_time/60:.2f} minutos)\n")
            f.write(f"Modo interactivo: {'SÍ' if self.interactive_mode else 'NO'}\n\n")
            
            f.write("CONFIGURACIÓN DE ELIMINACIÓN:\n")
            if self.metrics_log.get('optimizacion_manual', False):
                f.write(f"Tipo: ANÁLISIS COMPLETO + DECISIÓN INTERACTIVA\n")
                f.write(f"Porcentaje elegido: {self.metrics_log.get('porcentaje_eliminacion_manual', 0):.1f}%\n")
                f.write(f"Threshold manual: {self.metrics_log.get('threshold_manual', 0):.2f} puntos\n")
            elif self.metrics_log.get('optimizacion_automatica', False):
                f.write(f"Tipo: ANÁLISIS COMPLETO + SUGERENCIA AUTOMÁTICA\n")
                f.write(f"Porcentaje sugerido aplicado: {self.metrics_log.get('porcentaje_eliminacion_automatico', 0):.1f}%\n")
                f.write(f"Threshold automático: {self.metrics_log.get('threshold_automatico', 0):.2f} puntos\n")
            else:
                f.write(f"Tipo: PORCENTAJE FIJO (MODO LEGACY)\n")
                f.write(f"Porcentaje fijo: {self.elimination_percentage}%\n")
            f.write("\n")
            
            f.write("ESTADÍSTICAS DE PROCESAMIENTO:\n")
            f.write(f"Registros originales: {self.metrics_log.get('registros_originales', 0)}\n")
            f.write(f"Registros eliminados: {self.metrics_log.get('registros_eliminados', 0)}\n")
            f.write(f"Registros finales: {self.metrics_log.get('registros_finales', 0)}\n")
            f.write(f"Porcentaje retención: {(self.metrics_log.get('registros_finales', 0) / self.metrics_log.get('registros_originales', 1) * 100):.1f}%\n\n")
            
            f.write("ANÁLISIS DE CALIDAD:\n")
            f.write(f"Score máximo teórico: {self.metrics_log.get('score_maximo_teorico', 130)} puntos\n")
            f.write(f"Score promedio original: {self.metrics_log.get('score_promedio_original', 0):.2f}\n")
            f.write(f"Score promedio eliminados: {self.metrics_log.get('score_promedio_eliminados', 0):.2f}\n")
            f.write(f"Score promedio final: {self.metrics_log.get('score_promedio_restantes', 0):.2f}\n")
            f.write(f"Mejora conseguida: +{self.metrics_log.get('mejora_relativa', 0):.2f} puntos\n")
            f.write(f"Eficiencia final: {(self.metrics_log.get('score_promedio_restantes', 0) / 130 * 100):.1f}%\n\n")
            
            f.write("MÉTRICAS DETALLADAS POR COMPONENTE:\n")
            if hasattr(self, 'df_filtered') and not self.df_filtered.empty:
                f.write(f"📏 Coherencia de longitud: {self.df_filtered['length_score'].mean():.1f}/15\n")
                f.write(f"🔤 Diversidad de caracteres: {self.df_filtered['diversity_score'].mean():.1f}/15\n")
                f.write(f"📝 Ratio de puntuación: {self.df_filtered['punctuation_score'].mean():.1f}/10\n")
                f.write(f"🇨🇳 Complejidad HSK: {self.df_filtered['hsk_score'].mean():.1f}/20\n")
                f.write(f"🏗️ Estructura gramatical: {self.df_filtered['structure_score'].mean():.1f}/15\n")
                f.write(f"🧠 Coherencia semántica: {self.df_filtered['semantic_score'].mean():.1f}/25\n")
                f.write(f"🔢 Consistencia números: {self.df_filtered['number_consistency_score'].mean():.1f}/10\n")
                f.write(f"🤖 Calidad no-automática: {self.df_filtered['machine_quality_score'].mean():.1f}/10\n")
                f.write(f"👤 Consistencia entidades: {self.df_filtered['entity_consistency_score'].mean():.1f}/10\n")
                f.write(f"⚠️ Penalizaciones promedio: {self.df_filtered['penalties'].mean():.1f} puntos\n\n")
            
            f.write("INFORMACIÓN ACUMULATIVA:\n")
            f.write(f"Total muestras acumuladas: {self.metrics_log.get('muestras_acumuladas_total', 'N/A')}\n")
            f.write(f"Nuevas muestras esta ejecución: {self.metrics_log.get('muestras_nuevas_esta_ejecucion', 'N/A')}\n")
            f.write(f"Archivo Excel muestras: {self.metrics_log.get('archivo_excel_muestras', 'N/A')}\n\n")
            
            f.write("TIEMPO POR FASE:\n")
            for key, value in self.metrics_log.items():
                if key.startswith('tiempo_'):
                    fase = key.replace('tiempo_', '').replace('_', ' ').title()
                    f.write(f"{fase}: {value:.2f}s\n")
            f.write("\n")
            
            f.write("MODELOS DE EMBEDDINGS UTILIZADOS:\n")
            for i, model_name in enumerate(self.models.keys(), 1):
                f.write(f"{i}. {model_name}\n")
            f.write("\n")
            
            f.write("ESTADÍSTICAS DE EMBEDDINGS:\n")
            if hasattr(self, 'df_filtered') and not self.df_filtered.empty:
                f.write(f"Similitud semántica promedio: {self.df_filtered['semantic_score'].mean()/25:.3f}\n")
                f.write(f"Desviación estándar inter-modelo: {self.df_filtered['semantic_std'].mean():.3f}\n")
                f.write(f"Casos con alta discrepancia (std > 0.1): {sum(self.df_filtered['semantic_std'] > 0.1)}\n")

    def _save_execution_summary(self, summary_file, output_files):
        """Guarda resumen JSON de la ejecución"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'archivo_fuente': self.csv_file,
            'configuracion': {
                'modo_interactivo': self.interactive_mode,
                'porcentaje_eliminacion': self.metrics_log.get('porcentaje_eliminacion_manual', 
                                                              self.metrics_log.get('porcentaje_eliminacion_automatico',
                                                                                  self.metrics_log.get('porcentaje_eliminacion', 0))),
                'improvement_threshold': self.improvement_threshold,
                'tipo_optimizacion': 'interactiva' if self.metrics_log.get('optimizacion_manual', False) 
                                   else 'automatica' if self.metrics_log.get('optimizacion_automatica', False)
                                   else 'fija'
            },
            'estadisticas': {
                'registros_originales': self.metrics_log.get('registros_originales', 0),
                'registros_eliminados': self.metrics_log.get('registros_eliminados', 0),
                'registros_finales': self.metrics_log.get('registros_finales', 0),
                'score_promedio_original': round(self.metrics_log.get('score_promedio_original', 0), 2),
                'score_promedio_final': round(self.metrics_log.get('score_promedio_restantes', 0), 2),
                'mejora_conseguida': round(self.metrics_log.get('mejora_relativa', 0), 2),
                'eficiencia_final': round(self.metrics_log.get('score_promedio_restantes', 0) / 130 * 100, 1)
            },
            'acumulacion': {
                'muestras_acumuladas_total': self.metrics_log.get('muestras_acumuladas_total', 0),
                'muestras_nuevas_esta_ejecucion': self.metrics_log.get('muestras_nuevas_esta_ejecucion', 0)
            },
            'archivos_generados': output_files,
            'modelos_embeddings': list(self.models.keys()),
            'tiempo_total_segundos': round(self.metrics_log.get('tiempo_total', 0), 2)
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.log(f"  📁 Resumen JSON guardado: {summary_file}")

    def _count_executions_in_results(self):
        """Cuenta el número de ejecuciones detectadas en la carpeta de resultados"""
        results_dir = "p1_results"
        if not os.path.exists(results_dir):
            return 0
        
        # Contar archivos de resumen JSON
        json_files = [f for f in os.listdir(results_dir) if f.endswith('_resumen_') and f.endswith('.json')]
        return len(json_files)

    def _update_consolidated_stats(self, base_name, results_dir):
        """Actualiza estadísticas consolidadas de todas las ejecuciones"""
        stats_file = os.path.join(results_dir, f"{base_name}_estadisticas_consolidadas.json")
        
        # Cargar estadísticas existentes o crear nuevas
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                consolidated_stats = json.load(f)
        else:
            consolidated_stats = {
                'primera_ejecucion': datetime.now().isoformat(),
                'total_ejecuciones': 0,
                'total_registros_procesados': 0,
                'total_registros_eliminados': 0,
                'total_registros_generados': 0,
                'promedio_score_mejora': 0,
                'historial_ejecuciones': []
            }
        
        # Actualizar con datos de esta ejecución
        consolidated_stats['ultima_ejecucion'] = datetime.now().isoformat()
        consolidated_stats['total_ejecuciones'] += 1
        consolidated_stats['total_registros_procesados'] += self.metrics_log.get('registros_originales', 0)
        consolidated_stats['total_registros_eliminados'] += self.metrics_log.get('registros_eliminados', 0)
        consolidated_stats['total_registros_generados'] += self.metrics_log.get('registros_finales', 0)
        
        # Calcular promedio de mejora
        nueva_mejora = self.metrics_log.get('mejora_relativa', 0)
        if consolidated_stats['total_ejecuciones'] == 1:
            consolidated_stats['promedio_score_mejora'] = nueva_mejora
        else:
            consolidated_stats['promedio_score_mejora'] = (
                (consolidated_stats['promedio_score_mejora'] * (consolidated_stats['total_ejecuciones'] - 1) + nueva_mejora) 
                / consolidated_stats['total_ejecuciones']
            )
        
        # Añadir entrada de historial
        consolidated_stats['historial_ejecuciones'].append({
            'ejecucion': consolidated_stats['total_ejecuciones'],
            'timestamp': datetime.now().isoformat(),
            'archivo_fuente': self.csv_file,
            'registros_procesados': self.metrics_log.get('registros_originales', 0),
            'registros_finales': self.metrics_log.get('registros_finales', 0),
            'score_final': round(self.metrics_log.get('score_promedio_restantes', 0), 2),
            'mejora_conseguida': round(nueva_mejora, 2)
        })
        
        # Mantener solo últimas 50 ejecuciones en historial
        if len(consolidated_stats['historial_ejecuciones']) > 50:
            consolidated_stats['historial_ejecuciones'] = consolidated_stats['historial_ejecuciones'][-50:]
        
        # Guardar estadísticas actualizadas
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_stats, f, indent=2, ensure_ascii=False)
        
        self.log(f"  📊 Estadísticas consolidadas actualizadas: {stats_file}")
        self.log(f"    - Total ejecuciones: {consolidated_stats['total_ejecuciones']}")
        self.log(f"    - Promedio mejora score: {consolidated_stats['promedio_score_mejora']:.2f} puntos")
    
    def run_full_pipeline(self):
        """Ejecuta el pipeline completo de corrección"""
        
        self.log("🚀 INICIANDO PIPELINE DE CORRECCIÓN CHINA")
        self.log(f"📁 Archivo: {self.csv_file}")
        
        # Cargar CSV
        try:
            self.df = pd.read_csv(self.csv_file, encoding='utf-8')
            self.log(f"📊 Datos cargados: {len(self.df)} registros")
        except Exception as e:
            self.log(f"❌ Error cargando CSV: {e}", "ERROR")
            return None
        
        # Verificar columnas necesarias
        required_cols = ['linea', 'catalan', 'chino']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            self.log(f"❌ Columnas faltantes: {missing_cols}", "ERROR")
            return None
        
        # Ejecutar fases
        try:
            # Fase 0: Inicialización
            models_loaded = self.initialize_models()
            
            # Fase 1: Limpieza catalana
            cat_elim, cat_corr = self.clean_catalan()
            
            # Fase 2: Limpieza china
            chi_elim, chi_corr = self.clean_chinese()
            
            # Fase 3: Análisis
            duplicates = self.detect_duplicates()
            model_sims, avg_sims = self.calculate_semantic_scores()
            self.calculate_basic_metrics()
            self.calculate_translation_quality_metrics()
            
            # Fase 4: Scoring y filtrado (con modo interactivo o fijo)
            eliminated, mean_worst, mean_remaining = self.calculate_final_scores_and_filter(self.interactive_mode)
            
            # Fase 5: Reportes (NUEVA VERSION CON ACUMULACIÓN)
            output_files = self.generate_reports()
            
            # Resumen final
            total_time = time.time() - self.start_time
            self.log("\n" + "="*80)
            self.log("🎉 PIPELINE DE CORRECCIÓN CHINA COMPLETADO EXITOSAMENTE")
            self.log("="*80)
            self.log(f"⏱️  Tiempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
            self.log(f"🧠 Modelos embeddings utilizados: {len(models_loaded)}")
            self.log(f"📊 Registros procesados: {self.metrics_log.get('registros_originales', 0)}")
            
            if self.metrics_log.get('optimizacion_manual', False):
                self.log(f"📊 Análisis completo realizado con decisión interactiva")
                self.log(f"🎯 Porcentaje elegido interactivamente: {self.metrics_log.get('porcentaje_eliminacion_manual', 0):.1f}%")
            elif self.metrics_log.get('optimizacion_automatica', False):
                self.log(f"📊 Análisis completo realizado con sugerencia automática")
                self.log(f"🤖 Porcentaje aplicado automáticamente: {self.metrics_log.get('porcentaje_eliminacion_automatico', 0):.1f}%")
            else:
                self.log(f"📋 Porcentaje fijo usado (modo legacy): {self.elimination_percentage}%")
                self.log(f"🎯 Modo interactivo: DESACTIVADO")
                
            self.log(f"✅ Registros finales: {self.metrics_log.get('registros_finales', 0)} (calidad mejorada)")
            self.log(f"📈 Mejora en score: +{self.metrics_log.get('mejora_relativa', 0):.2f} puntos")
            self.log(f"🎯 Score final promedio: {self.metrics_log.get('score_promedio_restantes', 0):.2f}/130")
            
            # Información sobre acumulación
            if 'muestras_acumuladas_total' in self.metrics_log:
                self.log(f"🗄️  Muestras acumuladas totales: {self.metrics_log['muestras_acumuladas_total']}")
                self.log(f"🆕 Nuevas muestras esta ejecución: {self.metrics_log['muestras_nuevas_esta_ejecucion']}")
                self.log(f"📊 Excel acumulativo: {self.metrics_log.get('archivo_excel_muestras', 'N/A')}")
            
            self.log(f"📁 Archivos generados: {len(output_files)}")
            self.log(f"📂 Carpeta de métricas: p1_results/")

            #self.save_cache_index()
            #self.clean_old_cache()
            
            return {
                'success': True,
                'output_files': output_files,
                'metrics': self.metrics_log,
                'models_used': list(self.models.keys()),
                'total_time': total_time
            }
            
        except Exception as e:
            self.log(f"❌ Error en pipeline: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

def main():
    """Función principal con configuración avanzada"""
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Pipeline de Corrección China con Multi-Embeddings y Decisión Interactiva')
    parser.add_argument('csv_file', nargs='?', help='Archivo CSV a procesar')
    parser.add_argument('--percentage', '-p', type=float, default=5.0, 
                       help='Porcentaje de peores registros a eliminar (SOLO usado en modo legacy fijo)')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Usar sugerencia automática en lugar de modo interactivo')
    parser.add_argument('--models', '-m', nargs='+', 
                       help='Modelos específicos a usar (opcional)')
    parser.add_argument('--improvement-threshold', type=float, default=0.5,
                       help='Threshold de mejora marginal para sugerencia automática (default: 0.5)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Limpiar caché de embeddings antes de ejecutar')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Forzar uso de CPU incluso si hay GPU disponible')
    parser.add_argument('--gpu-batch-size', type=int, default=256,
                       help='Tamaño de batch para GPU (default: 256)')
    parser.add_argument('--fp16', action='store_true',
                       help='Usar precisión FP16 en GPU (más rápido, menos preciso)')
    
    args = parser.parse_args()
    
    # Limpiar caché si se solicita
    if args.clear_cache:
        import shutil
        if os.path.exists('p1_embeddings_cache'):
            shutil.rmtree('p1_embeddings_cache')
            print("🗑️  Caché de embeddings eliminado")
    
    # Si no se proporciona archivo, buscar un CSV en el directorio actual
    if not args.csv_file:
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'muestra_traducciones' in f]
        if csv_files:
            csv_file = csv_files[0]
            print(f"📁 Usando archivo encontrado: {csv_file}")
        else:
            print("Uso: python p1_embeddings.py <archivo_csv> [opciones]")
            print("Ejemplos:")
            print("  python p1_embeddings.py archivo.csv                    # Modo INTERACTIVO (recomendado)")
            print("  python p1_embeddings.py archivo.csv --no-interactive   # Modo AUTOMÁTICO (usa sugerencia)")
            print("O coloca un archivo CSV con 'muestra_traducciones' en el nombre en este directorio")
            return
    else:
        csv_file = args.csv_file
    
    if not os.path.exists(csv_file):
        print(f"❌ Archivo no encontrado: {csv_file}")
        return
    
    # Configuración del modo
    interactive_mode = not args.no_interactive
    
    # Información inicial
    print(f"🚀 Iniciando pipeline de corrección china MEJORADO")
    print(f"📁 Archivo: {csv_file}")
    print(f"📂 Métricas se guardarán en: p1_results/")
    print(f"📊 Muestras limpias se acumularán en Excel")
    
    if interactive_mode:
        print(f"🎯 Modo: INTERACTIVO")
        print(f"   1️⃣  Se calculará análisis completo (0.5% - 10.0%)")
        print(f"   2️⃣  Se mostrará tabla con todas las opciones")
        print(f"   3️⃣  Podrás elegir el porcentaje en tiempo real")
        print(f"   4️⃣  Se aplicará inmediatamente tu elección")
    else:
        print(f"🤖 Modo: SUGERENCIA AUTOMÁTICA")
        print(f"   1️⃣  Se calculará análisis completo (0.5% - 10.0%)")
        print(f"   2️⃣  Se aplicará automáticamente la sugerencia del algoritmo")
        print(f"   3️⃣  No requiere intervención manual")
    
    print(f"🧠 Usando hasta 9 modelos de embeddings para máxima robustez")
    
    # CREAR PIPELINE AQUÍ (ANTES de intentar modificarlo)
    pipeline = ChineseCorrectionPipeline(csv_file, args.percentage, interactive_mode, args.improvement_threshold)
    
    # AHORA SÍ podemos modificar configuraciones del pipeline
    if args.no_gpu:
        pipeline.device = 'cpu'
        pipeline.gpu_available = False
        print("🖥️  Forzando uso de CPU")
    
    if args.gpu_batch_size:
        pipeline.gpu_batch_size = args.gpu_batch_size
        print(f"📦 Batch size GPU configurado: {args.gpu_batch_size}")
    
    # Configurar modelos específicos si se proporcionan
    if args.models:
        pipeline.models_to_load = args.models
        print(f"🎯 Modelos personalizados: {args.models}")
    
    # Ejecutar pipeline
    result = pipeline.run_full_pipeline()
    
    if result and result.get('success'):
        print(f"\n🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
        print(f"⏱️  Tiempo total: {result['total_time']:.2f} segundos ({result['total_time']/60:.2f} minutos)")
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        metrics = result['metrics']
        print(f"  📝 Registros procesados: {metrics.get('registros_originales', 0)}")
        
        if metrics.get('optimizacion_manual', False):
            print(f"  🎯 Porcentaje elegido interactivamente: {metrics.get('porcentaje_eliminacion_manual', 0):.1f}%")
            print(f"  🗑️  Registros eliminados (elegido): {metrics.get('registros_eliminados', 0)}")
        elif metrics.get('optimizacion_automatica', False):
            print(f"  🤖 Porcentaje sugerido automáticamente: {metrics.get('porcentaje_eliminacion_automatico', 0):.1f}%")
            print(f"  🗑️  Registros eliminados (automático): {metrics.get('registros_eliminados', 0)}")
        else:
            print(f"  📋 Porcentaje usado (fijo): {metrics.get('porcentaje_eliminacion', 0):.1f}%")
            print(f"  🗑️  Registros eliminados: {metrics.get('registros_eliminados', 0)}")
        
        print(f"  ✅ Registros finales: {metrics.get('registros_finales', 0)}")
        print(f"  📈 Score original: {metrics.get('score_promedio_original', 0):.2f}/130")
        print(f"  🚀 Score final: {metrics.get('score_promedio_restantes', 0):.2f}/130")
        print(f"  ⬆️  Mejora conseguida: +{metrics.get('mejora_relativa', 0):.2f} puntos")
        
        efficiency = (metrics.get('score_promedio_restantes', 0) / 130) * 100
        print(f"  🎯 Eficiencia final: {efficiency:.1f}%")
        
        # Información de acumulación
        if 'muestras_acumuladas_total' in metrics:
            print(f"\n📊 ACUMULACIÓN DE MUESTRAS:")
            print(f"  🗄️  Total acumulado: {metrics['muestras_acumuladas_total']} muestras")
            print(f"  🆕 Nuevas esta ejecución: {metrics['muestras_nuevas_esta_ejecucion']} muestras")
            print(f"  📊 Archivo Excel: {metrics.get('archivo_excel_muestras', 'N/A')}")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        for file_type, file_path in result['output_files'].items():
            if file_type == 'analisis_excel':
                print(f"  - 📊 ANÁLISIS EXCEL: {file_path}")
                print(f"    ↳ Tabla completa de eliminación 0.5% - 10.0%")
                print(f"    ↳ Estadísticas de scores por métrica")
                print(f"    ↳ Top/Bottom 50 registros para análisis manual")
            else:
                print(f"  - {file_type}: {file_path}")
            
        print(f"\n🧠 MODELOS EMBEDDINGS UTILIZADOS: {len(result.get('models_used', []))}")
        for model in result.get('models_used', []):
            print(f"  - {model}")
            
    else:
        print(f"❌ Error en el proceso: {result.get('error', 'Error desconocido')}")

if __name__ == "__main__":
    main()