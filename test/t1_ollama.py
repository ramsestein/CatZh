#!/usr/bin/env python3
"""
eval_direct_fixed.py - Evaluación paralela con limpieza correcta de tags <think>
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import re
from typing import Dict, List
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ===============================
# CONFIGURACIÓN
# ===============================

class Config:
    """Configuración con paralelización"""
    def __init__(self):
        # Archivos
        self.dataset_path = "test-llm-esemble3.jsonl"
        self.results_dir = Path("results_direct")
        self.results_dir.mkdir(exist_ok=True)
        
        # Modelos Ollama
        self.llm_models = [
            "qwen3:0.6b",
            "yi:9b", 
            "deepseek-r1:1.5b"
            #"llama3.2:3b"
        ]
        
        # Parámetros de evaluación
        self.iterations_per_question = 5
        self.test_mode = False
        self.num_test = 10
        
        # Paralelización
        self.max_parallel_calls = 10  # Workers simultáneos
        self.timeout_seconds = 60
        
        # API Ollama
        self.ollama_url = "http://localhost:11434"


# ===============================
# GESTOR DE DATOS
# ===============================

class DataManager:
    """Gestiona el dataset"""
    
    def __init__(self, config: Config):
        self.config = config
        self.questions = []
        self.blocks = {
            "normal_telegraphic": (1, 102),
            "formal_technical": (103, 204),
            "slang_youth": (205, 306),
            "grammatical_errors": (307, 408),
            "verbose_poetic": (409, 510),
            "minimalist_extreme": (511, 612)
        }
    
    def load_dataset(self) -> List[Dict]:
        """Carga las preguntas del archivo JSONL"""
        print(f"📂 Cargando dataset desde {self.config.dataset_path}...")
        
        if not Path(self.config.dataset_path).exists():
            print(f"❌ Error: No se encuentra {self.config.dataset_path}")
            return []
        
        questions = []
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    data['id'] = idx
                    data['block'] = self._get_block(idx)
                    questions.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error en línea {idx}: {e}")
        
        self.questions = questions
        
        if self.config.test_mode:
            self.questions = self.questions[:self.num_test]
            print(f"🧪 Modo TEST: Solo {len(self.questions)} preguntas")
        
        print(f"✅ {len(self.questions)} preguntas cargadas")
        self._print_summary()
        
        return self.questions
    
    def _get_block(self, idx: int) -> str:
        """Determina el bloque de la pregunta"""
        for block_name, (start, end) in self.blocks.items():
            if start <= idx <= end:
                return block_name
        return "unknown"
    
    def _print_summary(self):
        """Imprime resumen del dataset"""
        print("\n📊 Distribución por bloques:")
        block_counts = {}
        for q in self.questions:
            block = q['block']
            block_counts[block] = block_counts.get(block, 0) + 1
        
        for block, count in sorted(block_counts.items()):
            print(f"   {block}: {count} preguntas")

# ===============================
# EVALUADOR PARALELO MEJORADO
# ===============================

class ParallelOllamaEvaluator:
    """Evaluador con procesamiento paralelo y limpieza correcta"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = []
        self.results_lock = Lock()
        self.completed_count = 0
        self.failed_count = 0
        self.progress_lock = Lock()
        
        # Estadísticas de tags encontrados
        self.tags_found = {
            'think': 0,
            'thinking': 0,
            'no_tags': 0
        }
    
    def test_connection(self) -> bool:
        """Verifica conexión con Ollama"""
        print("\n🔌 Verificando Ollama...")
        
        try:
            response = requests.get(f"{self.config.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = [m['name'] for m in models]
                print(f"✅ Ollama OK. Modelos: {available}")
                
                missing = []
                for model in self.config.llm_models:
                    if not any(model in m for m in available):
                        missing.append(model)
                
                if missing:
                    print(f"⚠️ Faltan: {missing}")
                    return False
                
                return True
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def clean_response(self, response: str, model: str) -> str:
        """
        Limpia la respuesta según el formato de cada modelo
        VERSIÓN CORREGIDA para <think> tags
        """
        original_length = len(response)
        cleaned = response
        
        # 1. Limpiar tags <think>...</think> (qwen3 y deepseek)
        if '<think>' in cleaned:
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            self.tags_found['think'] += 1
        
        # 2. Por si acaso, también limpiar <thinking>...</thinking>
        elif '<thinking>' in cleaned:
            cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            self.tags_found['thinking'] += 1
        
        # 3. Limpiar variaciones con mayúsculas
        cleaned = re.sub(r'<THINK>.*?</THINK>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<Think>.*?</Think>', '', cleaned, flags=re.DOTALL)
        
        # 4. Limpiar espacios y saltos de línea extras
        cleaned = cleaned.strip()
        cleaned = re.sub(r'\n+', ' ', cleaned)  # Reemplazar múltiples saltos por espacio
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Reemplazar múltiples espacios por uno
        
        # 5. Si aún es muy largo, buscar la respuesta china
        if len(cleaned) > 20:  # Debería ser solo 是 o 不是
            # Buscar patrones de respuesta al final
            patterns = [
                r'(是|不是|否|si|sí|no)$',  # Al final
                r'^(是|不是|否|si|sí|no)',  # Al principio
                r'答案[：:]\s*(是|不是|否|si|sí|no)',  # Después de "答案:"
                r'回答[：:]\s*(是|不是|否|si|sí|no)',  # Después de "回答:"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, cleaned)
                if match:
                    cleaned = match.group(1) if match.group(1) else match.group(0)
                    break
        
        # Estadística
        if original_length == len(cleaned):
            self.tags_found['no_tags'] += 1
        
        return cleaned.strip()
    
    def evaluate_single_task(self, task: Dict) -> Dict:
        """Evalúa una tarea individual"""
        model = task['model']
        prompt = task['prompt']
        expected = task['expected']
        question_id = task['question_id']
        iteration = task['iteration']
        block = task['block']
        
        session_id = f"q{question_id}_m{model.replace(':', '')}_i{iteration}"
        
        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "seed": hash(session_id) % 2**32,
                        "num_ctx": 2048
                    }
                },
                timeout=self.config.timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '')
                
                # LIMPIAR RESPUESTA CON LA FUNCIÓN CORREGIDA
                cleaned_response = self.clean_response(raw_response, model)
                
                # Calcular métricas
                metrics = self._calculate_metrics(cleaned_response, expected, raw_response)
                
                return {
                    'success': True,
                    'model': model,
                    'question_id': question_id,
                    'iteration': iteration,
                    'block': block,
                    'raw_response': raw_response[:500],  # Guardar para debug
                    'cleaned_response': cleaned_response,
                    'expected': expected,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    **metrics
                }
            else:
                return {
                    'success': False,
                    'model': model,
                    'question_id': question_id,
                    'iteration': iteration,
                    'error': f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'model': model,
                'question_id': question_id,
                'iteration': iteration,
                'error': str(e)
            }
    
    def _calculate_metrics(self, cleaned: str, expected: str, raw: str) -> Dict:
        """Calcula métricas de la respuesta"""
        # Métrica 1: Coincidencia exacta
        exact_match = cleaned == expected
        
        # Métrica 2: Contiene la respuesta
        contains_yes = any(yes in cleaned for yes in ['是', 'si', 'sí'])
        contains_no = any(neg in cleaned for neg in ['不是', '否', '不', 'no'])
        
        # Métrica 3: Respuesta correcta según esperado
        if expected == '是':
            contains_correct = contains_yes and not contains_no
        else:  # expected es alguna forma de "no"
            contains_correct = contains_no or (not contains_yes)
        
        # Métrica 4: Detectar si había tags
        had_think_tags = '<think>' in raw.lower() or '<thinking>' in raw.lower()
        
        return {
            'exact_match': exact_match,
            'contains_correct': contains_correct,
            'contains_yes': contains_yes,
            'contains_no': contains_no,
            'response_length': len(cleaned),
            'raw_length': len(raw),
            'had_think_tags': had_think_tags
        }
    
    def run_parallel_evaluation(self, questions: List[Dict]):
        """Ejecuta evaluación en paralelo"""
        print("\n" + "="*60)
        print(f"⚡ EVALUACIÓN PARALELA - {self.config.max_parallel_calls} workers")
        print("="*60)
        
        # Crear todas las tareas
        tasks = []
        for question in questions:
            for model in self.config.llm_models:
                for iteration in range(self.config.iterations_per_question):
                    task = {
                        'model': model,
                        'prompt': question['input'],
                        'expected': question.get('output', ''),
                        'question_id': question['id'],
                        'iteration': iteration,
                        'block': question['block']
                    }
                    tasks.append(task)
        
        total_tasks = len(tasks)
        print(f"📊 Total evaluaciones: {total_tasks}")
        
        # Procesar en paralelo
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_calls) as executor:
            futures = {executor.submit(self.evaluate_single_task, task): task 
                      for task in tasks}
            
            with tqdm(total=total_tasks, desc="Evaluando") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        
                        with self.results_lock:
                            self.results.append(result)
                        
                        with self.progress_lock:
                            if result.get('success'):
                                self.completed_count += 1
                            else:
                                self.failed_count += 1
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'OK': self.completed_count,
                            'Err': self.failed_count
                        })
                        
                    except Exception as e:
                        print(f"\n⚠️ Error: {e}")
                        self.failed_count += 1
                        pbar.update(1)
        
        print(f"\n✅ Completadas: {self.completed_count}")
        print(f"❌ Fallidas: {self.failed_count}")

# ===============================
# GENERADOR DE REPORTES
# ===============================

class ReportGenerator:
    """Genera reportes mejorados"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate_report(self, results: List[Dict]):
        """Genera reporte completo"""
        print("\n" + "="*60)
        print("GENERANDO REPORTE FINAL")
        print("="*60)
        
        # Filtrar resultados exitosos
        df = pd.DataFrame([r for r in results if r.get('success')])
        
        if df.empty:
            print("❌ No hay resultados exitosos")
            return
        
        # RESUMEN GENERAL
        print(f"\n📊 RESUMEN GENERAL")
        print("-" * 40)
        print(f"Total evaluaciones: {len(df)}")
        print(f"Preguntas únicas: {df['question_id'].nunique()}")
        print(f"Con tags <think>: {df['had_think_tags'].sum()} ({df['had_think_tags'].mean():.1%})")
        
        # POR MODELO
        print(f"\n📊 RESULTADOS POR MODELO")
        print("-" * 40)
        
        model_summary = []
        for model in self.config.llm_models:
            model_df = df[df['model'] == model]
            if not model_df.empty:
                exact = model_df['exact_match'].mean()
                correct = model_df['contains_correct'].mean()
                with_tags = model_df['had_think_tags'].mean()
                
                print(f"\n{model}:")
                print(f"  Coincidencia exacta: {exact:.2%}")
                print(f"  Contiene correcta: {correct:.2%}")
                print(f"  Usó tags <think>: {with_tags:.1%}")
                
                model_summary.append({
                    'model': model,
                    'exact_match': exact,
                    'contains_correct': correct,
                    'used_think_tags': with_tags
                })
        
        # COMPARACIÓN POR BLOQUES
        print(f"\n📊 EXACTITUD POR BLOQUE Y MODELO")
        print("-" * 40)
        
        pivot = df.pivot_table(
            values='exact_match',
            index='block',
            columns='model',
            aggfunc='mean'
        )
        
        for block in sorted(df['block'].unique()):
            print(f"\n{block}:")
            for model in self.config.llm_models:
                if model in pivot.columns:
                    value = pivot.loc[block, model]
                    print(f"  {model}: {value:.2%}")
        
        # GUARDAR ARCHIVOS
        output_csv = self.config.results_dir / "results_esemble3.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n💾 CSV guardado: {output_csv}")
        
        # Resumen JSON
        summary = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_evaluations': len(df),
                'questions': df['question_id'].nunique(),
                'iterations_per_question': self.config.iterations_per_question
            },
            'global_metrics': {
                'exact_match': df['exact_match'].mean(),
                'contains_correct': df['contains_correct'].mean(),
                'had_think_tags': df['had_think_tags'].mean()
            },
            'by_model': model_summary,
            'by_block': pivot.to_dict()
        }
        
        output_json = self.config.results_dir / "summary_results_esemble3.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"📊 JSON guardado: {output_json}")

# ===============================
# FUNCIÓN PRINCIPAL
# ===============================

def main():
    """Función principal"""
    print("="*60)
    print("⚡ EVALUACIÓN DATASET")
    print("="*60)
    
    config = Config()
    
    # Configuración
    print("\n⚙️ CONFIGURACIÓN")
    workers = input(f"Workers paralelos [{config.max_parallel_calls}]: ").strip()
    if workers.isdigit():
        config.max_parallel_calls = int(workers)
    
    test = input("¿Modo TEST (10 preguntas)? [s/N]: ").lower() == 's'
    config.test_mode = test
    
    # Cargar datos
    data_manager = DataManager(config)
    questions = data_manager.load_dataset()
    
    if not questions:
        return
    
    # Evaluador
    evaluator = ParallelOllamaEvaluator(config)
    
    if not evaluator.test_connection():
        return
    
    # Confirmar
    total = len(questions) * len(config.llm_models) * config.iterations_per_question
    print(f"\n📋 Resumen:")
    print(f"   Preguntas: {len(questions)}")
    print(f"   Evaluaciones totales: {total}")
    print(f"   Workers: {config.max_parallel_calls}")
    
    if input("\n¿Iniciar? [S/n]: ").lower() == 'n':
        return
    
    # Ejecutar
    start = time.time()
    evaluator.run_parallel_evaluation(questions)
    
    # Reporte
    report = ReportGenerator(config)
    report.generate_report(evaluator.results)
    
    elapsed = time.time() - start
    print(f"\n⏱️ Tiempo: {elapsed/60:.1f} minutos")
    print(f"⚡ Velocidad: {len(evaluator.results)/elapsed:.1f} eval/seg")

if __name__ == "__main__":
    main()