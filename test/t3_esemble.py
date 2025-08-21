#!/usr/bin/env python3
"""
Script para crear datasets con ranking semántico de traducciones.

Compara las traducciones de 6 modelos usando embeddings semánticos,
rankea por similitud, y crea datasets con los mejores 2 y 3.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import warnings
from collections import defaultdict, Counter
import torch

# Suprimir warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class MultiModelSemanticRanker:
    """Rankeador con votación de múltiples modelos de embedding"""
    
    def __init__(self, model_paths: List[str] = None, device: str = "auto"):
        if model_paths is None:
            model_paths = [
                "sentence-transformers/LaBSE",
                "BAAI/bge-large-zh-v1.5", 
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            ]
        
        self.model_paths = model_paths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.models = []
        self._load_models()
    
    def _load_models(self):
        from sentence_transformers import SentenceTransformer
        
        print(f"Cargando {len(self.model_paths)} modelos de embedding en {self.device}:")
        for i, model_path in enumerate(self.model_paths):
            print(f"  {i+1}. {model_path}")
            model = SentenceTransformer(model_path, device=self.device)
            # Optimizar para GPU
            if self.device.type == "cuda":
                model.half()  # Usar FP16 para ahorrar memoria y acelerar
            self.models.append(model)
        print(f"✅ Todos los modelos cargados en {self.device}")
    
    def rank_by_voting_consensus(self, texts: List[str], labels: List[str]) -> List[Tuple[str, str, float, Dict]]:
        if len(texts) <= 1:
            return [(texts[0], labels[0], 1.0, {})] if texts else []
        
        # Calcular rankings con cada modelo
        all_rankings = []
        model_scores = {}
        
        for i, model in enumerate(self.models):
            model_name = f"model_{i+1}"
            
            # Calcular embeddings y similitudes con batch size optimizado
            batch_size = 64 if self.device.type == "cuda" else 32
            embeddings = model.encode(
                texts, 
                batch_size=batch_size,
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            sim_matrix = np.dot(embeddings, embeddings.T)
            
            # Calcular scores de consenso
            scores = []
            for j in range(len(texts)):
                other_similarities = [sim_matrix[j][k] for k in range(len(texts)) if j != k]
                avg_similarity = float(np.mean(other_similarities)) if other_similarities else 0.0
                scores.append(avg_similarity)
            
            # Crear ranking para este modelo
            ranked = list(zip(texts, labels, scores))
            ranked.sort(key=lambda x: x[2], reverse=True)
            all_rankings.append(ranked)
            model_scores[model_name] = {labels[j]: scores[j] for j in range(len(labels))}
        
        # Votación por ranking promedio
        final_scores = {}
        voting_details = {}
        
        for label in labels:
            # Obtener posición en cada ranking (1-based)
            positions = []
            individual_scores = []
            
            for ranking in all_rankings:
                for pos, (text, lbl, score) in enumerate(ranking):
                    if lbl == label:
                        positions.append(pos + 1)  # 1-based position
                        individual_scores.append(score)
                        break
            
            # Score final = promedio de scores individuales
            final_score = float(np.mean(individual_scores))
            avg_position = float(np.mean(positions))
            
            final_scores[label] = final_score
            voting_details[label] = {
                "positions": positions,
                "individual_scores": individual_scores,
                "avg_position": avg_position
            }
        
        # Crear ranking final
        final_ranking = []
        for i, label in enumerate(labels):
            text = texts[i]
            score = final_scores[label]
            details = voting_details[label]
            final_ranking.append((text, label, score, details))
        
        # Ordenar por score final
        final_ranking.sort(key=lambda x: x[2], reverse=True)
        
        return final_ranking

def read_jsonl(file_path: Path) -> List[dict]:
    """Lee un archivo JSONL."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decodificando línea: {line[:50]}... Error: {e}")
    return data


def write_jsonl(data: List[dict], file_path: Path):
    """Escribe una lista de diccionarios a un archivo JSONL."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_translated_datasets(base_path: Path, input_field: str = "input") -> Dict[str, List[dict]]:
    """
    Carga todos los datasets traducidos.
    
    Args:
        base_path: Directorio base donde están los datasets
        input_field: Campo que contiene las traducciones
        
    Returns:
        Diccionario {modelo: datos}
    """
    # Patrones de archivos esperados
    patterns = {
        "aina": "*-aina.jsonl", 
        "google": "*-gt.jsonl",
        "nllb": "*-nllb200.jsonl",
        "mbart": "*-mbart.jsonl",
        "opus": "*-opus.jsonl",
        "m2m100": "*-m2m100.jsonl"
    }
    
    datasets = {}
    
    for model_name, pattern in patterns.items():
        files = list(base_path.glob(pattern))
        if files:
            # Tomar el primer archivo que coincida
            file_path = files[0]
            print(f"Cargando {model_name}: {file_path.name}")
            data = read_jsonl(file_path)
            if data:
                datasets[model_name] = data
                print(f"  ✅ {len(data)} elementos cargados")
            else:
                print(f"  ⚠️  Archivo vacío")
        else:
            print(f"  ❌ No encontrado patrón: {pattern}")
    
    return datasets


def align_datasets(datasets: Dict[str, List[dict]], input_field: str = "input") -> List[Dict[str, Any]]:
    """
    Alinea los datasets por índice para comparar las mismas preguntas.
    
    Returns:
        Lista de diccionarios con todas las traducciones de cada pregunta
    """
    if not datasets:
        return []
    
    # Obtener el tamaño mínimo (en caso de que algunos datasets sean más cortos)
    min_size = min(len(data) for data in datasets.values())
    print(f"Alineando datasets: {min_size} elementos comunes")
    
    aligned_data = []
    
    for i in range(min_size):
        item = {"index": i, "translations": {}, "original": None}
        
        for model_name, data in datasets.items():
            translation = data[i].get(input_field, "")
            item["translations"][model_name] = translation
            
            # Guardar datos originales del primer dataset (para referencia)
            if item["original"] is None:
                original_item = data[i].copy()
                original_item.pop(input_field, None)  # Remover la traducción
                item["original"] = original_item
        
        aligned_data.append(item)
    
    return aligned_data


def create_ranked_datasets(aligned_data: List[Dict[str, Any]], 
                          ranker: MultiModelSemanticRanker,
                          top_k_list: List[int] = [2, 3]) -> Tuple[Dict[int, List[dict]], Dict[str, Any]]:    
    """
    Crea datasets rankeados y estadísticas.
    
    Args:
        aligned_data: Datos alineados de todos los modelos
        ranker: Instancia del rankeador semántico
        top_k_list: Lista de valores k para crear datasets (ej: [2, 3])
        
    Returns:
        Tupla (datasets_rankeados, estadísticas)
    """
    print(f"🧮 Calculando rankings semánticos para {len(aligned_data)} elementos...")
    
    # Estadísticas
    model_contributions = defaultdict(int)  # Cuántas veces cada modelo está en top-k
    model_rankings = defaultdict(list)      # Posiciones de ranking de cada modelo
    similarity_scores = []                  # Todos los scores de similitud
    model_points = defaultdict(int)  # Sistema de puntos acumulativo
    
    # Datasets de salida
    ranked_datasets = {k: [] for k in top_k_list}
    
    for item in tqdm(aligned_data, desc="Procesando rankings"):
        translations = item["translations"]
        
        # Extraer textos y labels
        texts = list(translations.values())
        labels = list(translations.keys())
        
        # Rankear por votación de múltiples modelos
        ranked = ranker.rank_by_voting_consensus(texts, labels)        
        
        # Guardar estadísticas y asignar puntos
        num_models = len(ranked)
        for i, (text, model, score, voting_details) in enumerate(ranked):
            model_rankings[model].append(i + 1)  # Posición (1-based)
            similarity_scores.append(float(score))
            # Sistema de puntos: 1º lugar = 6pts, 2º = 5pts, ..., 6º = 1pt
            points = num_models - i
            model_points[model] += points
        
        # Crear elementos para cada top-k
        for k in top_k_list:
            top_k_ranked = ranked[:k]
            
            # Contar contribuciones
            for _, model, _, _ in top_k_ranked:
                model_contributions[f"{model}_top{k}"] += 1
            
            # Crear prompt duplicado/triplicado
            top_texts = [text for text, _, _, _ in top_k_ranked]
            
            if k == 2:
                prompt_template = "我给你传递重复的提示，请按照它们传达的指令操作，这两个是相同的：{}，{}"
            elif k == 3:
                prompt_template = "我给你传递三重提示，请按照它们传达的指令操作，这三个是相同的：{}，{}，{}"
            else:
                # Para k > 3, formato general
                texts_str = "，".join(top_texts)
                prompt_template = f"我给你传递重复{k}次的提示，请按照它们传达的指令操作，这些都是相同的：{texts_str}"
            
            # Formatear prompt
            if k <= 3:
                formatted_prompt = prompt_template.format(*top_texts)
            else:
                formatted_prompt = prompt_template
            
            # Crear elemento del dataset limpio
            new_item = {
                "input": formatted_prompt,
                "output": item["original"].get("output", "") if item["original"] else ""
            }
            
            ranked_datasets[k].append(new_item)
    
    # Calcular estadísticas finales
    stats = {
        "total_questions": len(aligned_data),
        "models_analyzed": list(set(model_rankings.keys())),
        "model_total_points": dict(model_points),
        "model_avg_points": {
            model: float(total_points / len(aligned_data)) for model, total_points in model_points.items()
        },
        "model_avg_ranking": {
            model: float(np.mean(rankings)) for model, rankings in model_rankings.items()
        },
        "model_contributions_by_topk": dict(model_contributions),
        "ranking_distribution": {
            model: {f"pos_{i+1}": rankings.count(i+1) for i in range(6)}
            for model, rankings in model_rankings.items()
        },
        "overall_similarity_stats": {
            "mean": float(np.mean(similarity_scores)),
            "std": float(np.std(similarity_scores)),
            "min": float(np.min(similarity_scores)),
            "max": float(np.max(similarity_scores))
        }
    }
    
    return ranked_datasets, stats


def print_statistics(stats: Dict[str, Any]):
    """Imprime estadísticas de manera legible."""
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DE RANKING SEMÁNTICO")
    print("="*60)
    
    print(f"\n📈 Resumen general:")
    print(f"  Total de preguntas analizadas: {stats['total_questions']}")
    print(f"  Modelos analizados: {', '.join(stats['models_analyzed'])}")
    
    print(f"\n🏆 Ranking promedio por modelo (menor = mejor):")
    for model, avg_rank in sorted(stats['model_avg_ranking'].items(), key=lambda x: x[1]):
        print(f"  {model:12} → {avg_rank:.2f}")
    
    print(f"\n🎯 Puntos totales por modelo (mayor = mejor):")
    for model, points in sorted(stats['model_total_points'].items(), key=lambda x: x[1], reverse=True):
        avg_points = stats['model_avg_points'][model]
        print(f"  {model:12} → {points:4d} pts (avg: {avg_points:.2f})")
    
    print(f"\n📊 Distribución de posiciones:")
    for model in sorted(stats['ranking_distribution'].keys()):
        dist = stats['ranking_distribution'][model]
        dist_str = " ".join([f"{pos}:{count}" for pos, count in dist.items() if count > 0])
        print(f"  {model:12} → {dist_str}")
    
    print(f"\n📊 Contribuciones por TOP-K:")
    contributions = stats['model_contributions_by_topk']
    
    # Agrupar por top-k
    top_k_groups = defaultdict(dict)
    for key, count in contributions.items():
        if '_top' in key:
            model, top_k = key.rsplit('_top', 1)
            top_k_groups[top_k][model] = count
    
    for top_k, models in sorted(top_k_groups.items()):
        print(f"\n  TOP-{top_k}:")
        total = sum(models.values())
        for model, count in sorted(models.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"    {model:12} → {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n📊 Estadísticas de similitud:")
    sim_stats = stats['overall_similarity_stats']
    print(f"  Media: {sim_stats['mean']:.4f}")
    print(f"  Desv. estándar: {sim_stats['std']:.4f}")
    print(f"  Rango: [{sim_stats['min']:.4f}, {sim_stats['max']:.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Crea datasets con ranking semántico de traducciones"
    )
    
    # Configuración de entrada
    parser.add_argument("--data_dir", type=str, default=".",
                       help="Directorio donde están los datasets traducidos")
    parser.add_argument("--input_field", type=str, default="input",
                       help="Campo que contiene las traducciones")
    
    # Configuración del modelo de embeddings
    parser.add_argument("--embedding_models", type=str, nargs=3, 
                   default=["sentence-transformers/LaBSE", "BAAI/bge-large-zh-v1.5", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"],
                   help="3 modelos de embeddings para votación")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Dispositivo para embeddings")
    
    # Configuración de salida
    parser.add_argument("--output_dir", type=str, default="./ranked_datasets",
                       help="Directorio de salida")
    parser.add_argument("--top_k", type=int, nargs="+", default=[2, 3],
                       help="Valores de k para crear datasets (ej: --top_k 2 3)")
    
    # Parámetros adicionales
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size para cálculo de embeddings")
    
    args = parser.parse_args()
    
    # Configurar paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🚀 INICIANDO RANKING SEMÁNTICO DE TRADUCCIONES")
    print("="*60)
    # Mostrar info de GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU detectada: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("💻 Usando CPU")
    print(f"📁 Directorio de datos: {data_dir}")
    print(f"📁 Directorio de salida: {output_dir}")
    print(f"🤖 Modelos de embeddings: {args.embedding_models}")
    print(f"📊 TOP-K valores: {args.top_k}")
    print("="*60)
    
    try:
        # 1. Cargar modelo de embeddings
        print("\n1️⃣ Cargando modelos de embeddings...")
        ranker = MultiModelSemanticRanker(
            model_paths=args.embedding_models,
            device=args.device
        )
        
        # 2. Cargar todos los datasets traducidos
        print("\n2️⃣ Cargando datasets traducidos...")
        datasets = load_translated_datasets(data_dir, args.input_field)
        
        if not datasets:
            raise ValueError("No se encontraron datasets traducidos")
        
        print(f"✅ Cargados {len(datasets)} datasets: {list(datasets.keys())}")
        
        # 3. Alinear datasets
        print("\n3️⃣ Alineando datasets...")
        aligned_data = align_datasets(datasets, args.input_field)
        
        if not aligned_data:
            raise ValueError("No se pudieron alinear los datasets")
        
        # 4. Crear rankings y datasets
        print("\n4️⃣ Creando rankings semánticos...")
        ranked_datasets, stats = create_ranked_datasets(
            aligned_data, ranker, args.top_k
        )
        
        # 5. Guardar datasets rankeados
        print("\n5️⃣ Guardando datasets rankeados...")
        for k, dataset in ranked_datasets.items():
            output_path = output_dir / f"top{k}_ranked_dataset.jsonl"
            write_jsonl(dataset, output_path)
            print(f"  ✅ TOP-{k}: {len(dataset)} elementos → {output_path}")
        
        # 6. Guardar estadísticas
        stats_path = output_dir / "ranking_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Estadísticas → {stats_path}")
        
        # 7. Mostrar estadísticas
        print_statistics(stats)
        
        print(f"\n✅ ¡Proceso completado exitosamente!")
        print(f"📁 Resultados guardados en: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()