# 🌐 Benchmark de Traducción Catalán-Chino mediante Evaluación LLM

## 📊 Resumen del Proyecto

Sistema innovador de evaluación de calidad de traducción automática catalán→chino que utiliza la capacidad de LLMs para responder preguntas Sí/No como métrica de calidad. Este benchmark evalúa 6 modelos de traducción mediante 3 LLMs evaluadores sobre un dataset de 612 preguntas en 6 estilos lingüísticos diferentes.

### 🎯 Características Principales

- **612 preguntas Sí/No** en catalán con respuestas esperadas en chino (是/不是)
- **6 modelos de traducción** evaluados: NLLB-200, AINA, Google Translate, M2M100, mBART, OpusMT
- **3 LLMs evaluadores**: Qwen3:0.6b, Yi:9b, DeepSeek-R1:1.5b
- **6 estilos lingüísticos**: Desde formal técnico hasta minimalista extremo
- **Sistema de ensemble** con ranking semántico usando 3 modelos de embeddings
- **9,040+ evaluaciones** por modelo (5 iteraciones × 612 preguntas × 3 LLMs)

## 🔬 Metodología

### Pipeline de Evaluación de 3 Fases

**FASE 1: TRADUCCIÓN**
- Dataset Original: 612 preguntas en catalán
- 6 Modelos de Traducción: NLLB-200, AINA, Google Translate, M2M100, mBART, OpusMT
- Resultado: 6 datasets traducidos al chino

**FASE 2: EVALUACIÓN CON LLMs**
- 3 LLMs evalúan cada dataset traducido
- 5 iteraciones por pregunta para robustez estadística
- Métricas: exact_match, contains_correct, had_think_tags
- Total: 9,040 evaluaciones por modelo

**FASE 3: ENSEMBLE SEMÁNTICO**
- Ranking usando 3 modelos de embeddings: LaBSE, BGE-large-zh, Paraphrase-multilingual-mpnet
- Votación por consenso para determinar mejores traducciones
- Generación de datasets mejorados (top-2 y top-3)

### Métricas de Evaluación

- **exact_match**: Coincidencia exacta con respuesta esperada (是/不是) [0-100%]
- **contains_correct**: Respuesta contiene el elemento correcto [0-100%]
- **had_think_tags**: Modelo usó tags de razonamiento interno [0-100%]
- **semantic_similarity**: Similitud semántica entre traducciones [0-1]

## 📈 Resultados Principales

### Ranking Global de Modelos de Traducción

| Ranking | Modelo | Exact Match | Contains Correct | Score Ensemble |
|---------|--------|-------------|------------------|----------------|
| 1º | NLLB-200 | 46.98% | 68.97% | 4.40 pts |
| 2º | AINA | 46.98% | 68.97% | 4.58 pts |
| 3º | Google Translate | 41.97% | 63.28% | 3.82 pts |
| 4º | M2M100 | 16.60% | 53.88% | 4.10 pts |
| 5º | mBART | 19.55% | 54.28% | 2.97 pts |
| 6º | OpusMT | 0.00% | 97.32% | 1.14 pts |

### Rendimiento por Estilo Lingüístico (Exact Match %)

| Modelo | Normal Telegráfico | Formal Técnico | Slang Juvenil | Con Errores | Verboso Poético | Minimalista |
|--------|-------------------|----------------|---------------|-------------|-----------------|-------------|
| NLLB | 52.4% | 60.4% | 64.6% | 51.8% | 46.5% | 42.1% |
| AINA | 52.4% | 60.8% | 53.3% | 47.6% | 56.1% | 23.8% |
| Google | 71.9% | 70.6% | 60.0% | 59.5% | 55.6% | 30.0% |
| M2M100 | 6.0% | 16.9% | 5.1% | 3.6% | 16.1% | 18.4% |
| mBART | 11.8% | 21.0% | 14.3% | 15.4% | 21.5% | 12.3% |
| OpusMT | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## 📂 Dataset de Evaluación

El dataset consta de 612 preguntas organizadas en 6 bloques estilísticos:

| Bloque | Rango | Estilo | Características |
|--------|-------|--------|-----------------|
| 1 | 1-102 | Normal telegráfico | Formato estándar, conciso |
| 2 | 103-204 | Formal técnico | Lenguaje culto, ligeramente verborreico |
| 3 | 205-306 | Slang juvenil | Jerga callejera, informal |
| 4 | 307-408 | Con errores | Errores gramaticales no agresivos |
| 5 | 409-510 | Verboso poético | Arcaico, muy elaborado |
| 6 | 511-612 | Minimalista extremo | Mínima expresión posible |

## 🚀 Uso del Sistema

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/catalan-chinese-benchmark.git
cd catalan-chinese-benchmark

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar Ollama (para evaluación con LLMs locales)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:0.6b
ollama pull yi:9b
ollama pull deepseek-r1:1.5b

###Ejecutar Evaluación Completa

```
bash
# FASE 1: Traducir dataset con todos los modelos
python t2_nllb200.py datasets/test-catalan.jsonl --output_file test-llm-nllb200.jsonl
python t2_aina.py datasets/test-catalan.jsonl --output_file test-llm-aina.jsonl
python t2_google.py datasets/test-catalan.jsonl --output_file test-llm-gt.jsonl
python t2_m2m100.py datasets/test-catalan.jsonl --output_file test-llm-m2m100.jsonl
python t2_mbart.py datasets/test-catalan.jsonl --output_file test-llm-mbart.jsonl
python t2_opus.py datasets/test-catalan.jsonl --output_file test-llm-opus.jsonl

# FASE 2: Evaluar con Ollama
python t1_ollama.py
# Configuración interactiva:
# - Workers paralelos: 10 (ajustar según CPU)
# - Modo TEST: No (para evaluación completa)

# FASE 3: Crear ensemble semántico
python t3_ensemble.py --data_dir . --output_dir ./ranked_datasets --top_k 2 3
```
###Evaluación Rápida (Modo Test)

```
bash
# Solo 10 preguntas para prueba rápida
python t1_ollama.py --test_mode --num_test 10
```

###📊 Estructura del Proyecto

catalan-chinese-benchmark/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python
├── requirements_tpu.txt              # Dependencias para TPU/Colab
│
├── datasets/
│   ├── test-catalan.jsonl           # Dataset original (612 preguntas)
│   └── explain_dataset.txt          # Descripción de bloques
│
├── evaluation/
│   ├── t1_ollama.py                 # Evaluación con LLMs locales
│   ├── t2_nllb200.py                # Traducción con NLLB-200
│   ├── t2_aina.py                   # Traducción con AINA
│   ├── t2_google.py                 # Traducción con Google
│   ├── t2_m2m100.py                 # Traducción con M2M100
│   ├── t2_mbart.py                  # Traducción con mBART
│   ├── t2_opus.py                   # Traducción con OpusMT
│   └── t3_ensemble.py               # Ensemble semántico
│
├── results/
│   ├── summary_results_*.json       # Resultados por modelo
│   ├── ranking_statistics.json      # Estadísticas de ensemble
│   └── results_direct/              # Resultados detallados
│
├── training/                         # Sistema de entrenamiento (opcional)
│   ├── train_all.py                 # Entrenamiento multi-modelo
│   ├── universal_trainer.py         # Lógica de entrenamiento
│   ├── model_configs.py             # Configuraciones de modelos
│   └── checkpoint_manager.py        # Gestión de checkpoints
│
└── ranked_datasets/                  # Datasets generados por ensemble
    ├── top2_ranked_dataset.jsonl    # Mejores 2 traducciones
    └── top3_ranked_dataset.jsonl    # Mejores 3 traducciones


###📉 Análisis de Resultados
##Hallazgos Clave

- OpusMT: Paradoja interesante - 97.3% contains_correct pero 0% exact_match, sugiere que entiende el contenido pero no respeta el formato de respuesta
- NLLB y AINA: Mejor balance entre comprensión y formato, lideran en el ranking del ensemble
- Google Translate: Rendimiento sólido y consistente en todos los estilos lingüísticos

Impacto del estilo: El bloque minimalista extremo es el más desafiante para todos los modelos
Think tags: Los modelos que usan razonamiento interno (Qwen3, DeepSeek) muestran mayor precisión

##Contribuciones del Ensemble (Top-2)

AINA: 340 contribuciones (27.8%)
NLLB: 321 contribuciones (26.2%)
M2M100: 257 contribuciones (21.0%)
Google: 195 contribuciones (15.9%)
mBART: 111 contribuciones (9.1%)

###🔧 Configuración Avanzada
Ajustar Parámetros de Evaluación
```
`python
# En t1_ollama.py
config.iterations_per_question = 10  # Más iteraciones para mayor robustez
config.max_parallel_calls = 20       # Más workers para evaluación rápida
config.timeout_seconds = 120         # Timeout mayor para modelos lentos
```

##Agregar Nuevo Modelo de Traducción

Crear script t2_nuevo_modelo.py basado en plantilla existente
Implementar clase NuevoModeloTranslator
Agregar a pipeline de evaluación

##Agregar Nuevo LLM Evaluador
```
python
# En t1_ollama.py, agregar a config.llm_models
self.llm_models = [
    "qwen3:0.6b",
    "yi:9b",
    "deepseek-r1:1.5b",
    "nuevo-modelo:version"  # Nuevo LLM
]
```

###📝 Reproducibilidad

Para reproducir exactamente los resultados:

Versiones de software: Ver requirements.txt para versiones exactas
Semillas aleatorias: Todas fijadas a 42
Datasets: Disponibles en carpeta datasets/
Configuración: Documentada en archivos de configuración

###🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

Fork el repositorio
Crea una rama para tu feature (git checkout -b feature/NuevaCaracteristica)
Commit tus cambios (git commit -m 'Agregar nueva característica')
Push a la rama (git push origin feature/NuevaCaracteristica)
Abre un Pull Request

###📄 Licencia
Este proyecto está bajo licencia MIT. Ver archivo LICENSE para más detalles.

###📧 Contacto
Para preguntas o colaboraciones, contactar a: [tu-email@ejemplo.com]

###🙏 Agradecimientos

Modelos de traducción de código abierto
Comunidad Ollama por infraestructura de LLMs locales
Sentence Transformers por modelos de embeddings

