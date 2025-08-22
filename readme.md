# 🌐 Introducción de modelos de traducción para mejora de rendimiento de LLM

# Tareas pte
- Traductor en tandem aina + qwen3 -> Flores200
- Corroborar lo de semantica correlación con modelo grande

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
```

### Ejecutar Evaluación Completa

```
bash
# FASE 1: Traducir dataset con todos los modelos
python t2_nllb200.py datasets/test-catalan.jsonl --output_file test-llm-nllb200.jsonl
python t2_aina.py datasets/test-catalan.jsonl --output_file test-llm-aina.jsonl
python t2_google.py datasets/test-catalan.jsonl --output_file test-llm-gt.jsonl
python t2_m2m100.py datasets/test-catalan.jsonl --output_file test-llm-m2m100.jsonl
python t2_mbart.py datasets/test-catalan.jsonl --output_file test-llm-mbart.jsonl
python t2_opus.py datasets/test-catalan.jsonl --output_file test-llm-opus.jsonl
```

## FASE 2: Evaluar con Ollama
```
bash
python t1_ollama.py
# Configuración interactiva:
# - Workers paralelos: 10 (ajustar según CPU)
# - Modo TEST: No (para evaluación completa)
```

## FASE 3: Crear ensemble semántico
```
bash
python t3_ensemble.py --data_dir . --output_dir ./ranked_datasets --top_k 2 3
```
### Evaluación Rápida (Modo Test)

```
bash
# Solo 10 preguntas para prueba rápida
python t1_ollama.py --test_mode --num_test 10
```

### 📊 Estructura del Proyecto

```
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
```

### 🔧Configuración Avanzada
Ajustar Parámetros de Evaluación
```
`python
# En t1_ollama.py
config.iterations_per_question = 10  # Más iteraciones para mayor robustez
config.max_parallel_calls = 20       # Más workers para evaluación rápida
config.timeout_seconds = 120         # Timeout mayor para modelos lentos
```

### Agregar Nuevo Modelo de Traducción

Crear script t2_nuevo_modelo.py basado en plantilla existente
Implementar clase NuevoModeloTranslator
Agregar a pipeline de evaluación

### Agregar Nuevo LLM Evaluador
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
## 📝Reproducibilidad

Para reproducir exactamente los resultados:

Versiones de software: Ver requirements.txt para versiones exactas
Semillas aleatorias: Todas fijadas a 42
Datasets: Disponibles en carpeta datasets/
Configuración: Documentada en archivos de configuración

## 📈 Resultados Principales

### 🔬 Experimento de Control: Llama3.2:3b

### Contexto del Experimento
Llama3.2:3b se utilizó como modelo de control para evaluar el comportamiento de un LLM que **no conoce ni catalán ni chino** cuando se le presentan directamente preguntas Sí/No en catalán y se espera respuesta en formato chino (是/不是).

#### Métricas Principales de Control

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Exact Match** | 0.0% | No puede generar formato chino |
| **Contains Correct** | 97.3% | Comprende perfectamente el contenido |
| **Idioma de Respuesta** | 94.6% Español, 2.1% Catalán | Responde en idiomas conocidos |
| **Respuestas Vacías** | 84.6% | Alta tasa de no-respuesta al formato esperado |
| **Consistencia** | 85.5% | Muy consistente en sus respuestas |

#### Análisis por Bloques Estilísticos

| Bloque | Contains Correct | Respuesta Media | Consistencia |
|--------|-----------------|-----------------|--------------|
| **Normal Telegráfico** | 100% | 3.3 chars | 98.8% |
| **Formal Técnico** | 100% | 3.0 chars | 98.5% |
| **Slang Juvenil** | 100% | 3.6 chars | 97.8% |
| **Errores Gramaticales** | 99.4% | 32.2 chars | 87.0% |
| **Verboso Poético** | 100% | 3.0 chars | 99.0% |
| **Minimalista Extremo** | 84.5% | 333.5 chars | 31.9% |

#### Patrones de Respuesta Observados

#### Distribución de Respuestas
- **"Sí" en español**: 973 casos (31.8%)
- **"No" en español**: 1,631 casos (53.3%)
- **Respuestas elaboradas**: 16.6% (con explicación)
- **Sin respuesta clara**: 80.8%

#### Top Palabras en Respuestas
1. "No." - 1,631 ocurrencias
2. "Sí." - 973 ocurrencias
3. Palabras en español: "de", "la", "que", "el", "es"
4. Sin términos en chino en las respuestas

#### Hallazgos Clave del Control

1. **Comprensión Perfecta**: 97.3% de comprensión del contenido demuestra que el modelo entiende las preguntas en catalán a pesar de no estar entrenado específicamente en este idioma

2. **Incapacidad de Formato**: 0% exact match confirma que sin conocimiento del chino, no puede generar el formato de respuesta esperado (是/不是)

3. **Respuesta en Idioma Conocido**: El modelo defaultea a español (94.6%) como idioma de respuesta, su idioma más cercano al catalán

4. **Alta Consistencia**: 85.5% de consistencia indica que el modelo es muy predecible en su comportamiento, especialmente en bloques simples

5. **Degradación con Complejidad**: El bloque minimalista extremo muestra una caída dramática en consistencia (31.9%) y comprensión (84.5%)

#### Comparación con Modelos de Traducción

| Aspecto | Llama3.2:3b (Control) | Mejor Traductor (AINA) |
|---------|----------------------|------------------------|
| **Comprensión Semántica** | 97.3% | 69.0% |
| **Formato Correcto** | 0% | 47.0% |
| **Consistencia** | 85.5% | 74.4% |
| **Respuestas Válidas** | 15.4% (intentos) | 98.8% (intentos) |

#### Implicaciones del Control

1. **Validación del Pipeline**: Confirma que la traducción es esencial para obtener respuestas en formato chino

2. **Benchmark de Comprensión**: Establece un techo de 97.3% para la comprensión semántica pura de las preguntas

3. **Importancia del Conocimiento Lingüístico**: Demuestra que sin conocimiento específico del chino, incluso con comprensión perfecta, es imposible generar el formato de respuesta correcto

4. **Robustez Cross-lingüística**: Sugiere que los LLMs pueden comprender idiomas relacionados (catalán) a través de idiomas conocidos (español)

5. **Limitaciones del Minimalismo**: El estilo minimalista extremo es desafiante incluso para la comprensión, no solo para la traducción

### Ranking Global de Modelos de Traducción

| Ranking | Modelo | Exact Match | Contains Correct |
|---------|--------|-------------|------------------|
| 1º | **AINA** | 46.98% | 68.97% |
| 2º | **Google Translate** | 41.97% | 63.28% |
| 3º | **Ensemble (Top-3)** | 25.84% | 62.35% |
| 4º | **mBART** | 19.55% | 54.28% |
| 5º | **M2M100** | 16.60% | 53.88% |
| 6º | **CAT (Directo)¹** | 2.70% | 89.85% |

¹ *CAT: Evaluación directa en catalán sin traducción - Alta comprensión semántica (89.85%) pero falla en formato de respuesta chino*  
² *Llama3.2:3b: Modelo base de trabajo del proyecto*

#### Rendimiento por Estilo Lingüístico (Exact Match % - Promedio entre evaluadores)

| Modelo | Formal Técnico | Errores Gramaticales | Minimalista Extremo | Normal Telegráfico | Slang Juvenil | Verboso Poético |
|--------|----------------|---------------------|-------------------|------------------|---------------|-----------------|
| **AINA** | 56.5% | 45.1% | 28.1% | 53.0% | 50.9% | 41.1% |
| **Google** | 50.1% | 46.1% | 18.3% | 37.6% | 43.6% | 34.4% |
| **Ensemble** | 31.0% | 21.2% | 16.1% | 31.6% | 24.8% | 30.4% |
| **mBART** | 27.7% | 20.7% | 13.7% | 18.4% | 20.2% | 20.2% |
| **M2M100** | 23.8% | 7.3% | 19.7% | 15.5% | 12.2% | 21.2% |
| **CAT** | 3.3% | 0.3% | 1.1% | 4.6% | 2.3% | 4.2% |

#### Métricas Detalladas por Modelo Evaluador

#### Modelo: AINA
| LLM Evaluador | Exact Match | Contains Correct |
|---------------|-------------|------------------|
| Qwen3:0.6b | 55.64% | 70.86% |
| Yi:9b | 36.10% | 72.60% |
| DeepSeek-R1:1.5b | 49.06% | 63.47% |

#### Modelo: Google Translate  
| LLM Evaluador | Exact Match | Contains Correct |
|---------------|-------------|------------------|
| Qwen3:0.6b | 57.98% | 65.05% |
| Yi:9b | 26.32% | 66.71% |
| DeepSeek-R1:1.5b | 41.57% | 58.03% |

#### Modelo: CAT (Evaluación Directa en Catalán)
| LLM Evaluador | Exact Match | Contains Correct |
|---------------|-------------|------------------|
| Qwen3:0.6b | 7.85% | 88.25% |
| Yi:9b | 0.00% | 95.01% |
| DeepSeek-R1:1.5b | 0.03% | 86.59% |

### 📊 Análisis Detallado de Resultados

#### 🔍 Análisis de Patrones de Respuesta

| Modelo | Sesgo de Respuesta | Dirección | Respuestas Mixtas | Respuestas Vacías |
|--------|-------------------|-----------|-------------------|-------------------|
| **AINA** | 24.7% | Hacia "Sí" | 10.2% | 1.2% |
| **Google Translate** | 17.3% | Hacia "Sí" | 11.6% | 1.1% |
| **M2M100** | 42.7% | Hacia "Sí" | 49.9% | 2.1% |
| **mBART** | 34.5% | Hacia "Sí" | 36.9% | 1.6% |
| **CAT (Directo)** | 15.1% | Hacia "No" | 24.7% | 46.4% |
| **Ensemble Top-3** | 41.0% | Hacia "Sí" | 27.0% | 0.7% |

#### 📏 Análisis de Longitud y Verbosidad

| Modelo | Longitud Media | Mediana | Longitud Óptima | Correlación Longitud-Precisión |
|--------|---------------|---------|-----------------|--------------------------------|
| **AINA** | 19.3 chars | 1 | 1 (73.1% precisión) | -0.271 |
| **Google Translate** | 26.7 chars | 1 | 1 (69.1% precisión) | -0.275 |
| **M2M100** | 141.8 chars | 102 | 1 (85.6% precisión) | -0.408 |
| **mBART** | 116.4 chars | 38 | 1 (78.6% precisión) | -0.341 |
| **CAT (Directo)** | 185.2 chars | 98 | 2 (46.7% precisión) | -0.133 |
| **Ensemble Top-3** | 57.3 chars | 3 | 1 (72.0% precisión) | -0.268 |

#### 🎯 Análisis de Consistencia

| Modelo | Score de Consistencia | Preguntas Determinísticas | Preguntas Caóticas | Alta Confianza |
|--------|----------------------|--------------------------|-------------------|----------------|
| **AINA** | 74.4% ± 18.5% | 26 | 2 | 595 |
| **Google Translate** | 71.8% ± 22.2% | 5 | 16 | 575 |
| **M2M100** | 22.6% ± 25.1% | 3 | 220 | 276 |
| **mBART** | 38.5% ± 29.2% | 4 | 123 | 394 |
| **CAT (Directo)** | 33.5% ± 21.0% | 0 | 73 | 385 |
| **Ensemble Top-3** | 53.5% ± 25.7% | 1 | 17 | 487 |

### 📈 Análisis Temporal y Fatiga

#### Degradación de Rendimiento (Q1 → Q4)

| Modelo | Q1 Accuracy | Q4 Accuracy | Tendencia | Degradación |
|--------|------------|------------|-----------|-------------|
| **AINA** | 54.0% | 37.8% | ↓ Degradación | -16.2% |
| **Google Translate** | 49.7% | 31.0% | ↓ Degradación | -18.7% |
| **M2M100** | 18.5% | 21.4% | ↑ Mejora | +2.9% |
| **mBART** | 20.6% | 16.3% | ↓ Degradación | -4.3% |
| **CAT (Directo)** | 4.0% | 2.6% | ↓ Degradación | -1.4% |
| **Ensemble Top-3** | 29.8% | 21.4% | ↓ Degradación | -8.4% |

### 💪 Métricas de Robustez

#### Sensibilidad al Estilo y Adaptabilidad

| Modelo | Sensibilidad (σ) | Rango de Variación | Resiliencia a Errores | Tendencia de Complejidad |
|--------|-----------------|-------------------|----------------------|-------------------------|
| **AINA** | 0.074 | 21.8% | -7.9% drop | ↓ -0.037 (degrada) |
| **Google Translate** | 0.080 | 22.7% | -4.9% drop | ↓ -0.045 (degrada) |
| **M2M100** | 0.057 | 16.5% | -8.3% drop | ↑ +0.005 (mejora) |
| **mBART** | 0.039 | 13.0% | +0.5% mejora | ↓ -0.011 (degrada) |
| **CAT (Directo)** | 0.016 | 4.3% | -4.3% drop | ↓ -0.003 (degrada) |
| **Ensemble Top-3** | 0.057 | 15.4% | -10.1% drop | ↓ -0.022 (degrada) |

### 🔬 Casos Especiales

#### Análisis de Falsos Positivos y Casos Extremos

| Modelo | Falsos Positivos | Respuestas Extremas (>P95) | Preguntas Perfectas | Preguntas Imposibles |
|--------|-----------------|---------------------------|-------------------|---------------------|
| **AINA** | 22.0% (1,988) | 1,498 casos (max: 947) | 25 (4.1%) | 37 (6.0%) |
| **Google Translate** | 21.3% (1,939) | 1,742 casos (max: 1,113) | 5 (0.8%) | 38 (6.2%) |
| **M2M100** | 37.3% (3,370) | 381 casos (max: 2,384) | 3 (0.5%) | 308 (50.3%) |
| **mBART** | 34.7% (3,125) | 708 casos (max: 1,639) | 4 (0.7%) | 249 (40.7%) |
| **CAT (Directo)** | 89.0% (7,888) | 179 casos (max: 1,821) | 0 (0%) | 506 (82.7%) |
| **Ensemble Top-3** | 36.5% (3,223) | 1,743 casos (max: 1,907) | 1 (0.2%) | 118 (19.3%) |

### 📉 Rendimiento por Complejidad Estilística

#### Degradación Promedio por Bloque (ordenado por dificultad)

| Bloque | AINA | Google | M2M100 | mBART | CAT | Ensemble |
|--------|------|--------|--------|-------|-----|----------|
| **Normal Telegráfico** | 53.0% | 51.0% | 15.5% | 18.0% | 4.6% | 31.3% |
| **Formal Técnico** | 56.5% | 48.4% | 23.8% | 26.6% | 3.3% | 31.1% |
| **Errores Gramaticales** | 45.1% | 46.0% | 7.2% | 18.6% | 0.3% | 21.2% |
| **Slang Juvenil** | 50.9% | 43.6% | 12.2% | 20.2% | 2.3% | 24.7% |
| **Verboso Poético** | 41.3% | 34.3% | 21.3% | 20.3% | 4.4% | 30.1% |
| **Minimalista Extremo** | 34.7% | 28.3% | 19.7% | 13.6% | 1.2% | 15.9% |

### 🎯 Métricas de Calidad Agregadas

| Modelo | F1-Score Promedio | Precisión Ponderada | Tasa Comprensión Real | Índice de Formato |
|--------|------------------|--------------------|--------------------|-------------------|
| **AINA** | 46.9% | 49.2% | 22.0% | 47.6% éxito cuando intenta |
| **Google Translate** | 41.9% | 44.5% | 21.3% | 42.4% éxito cuando intenta |
| **M2M100** | 16.6% | 16.7% | 37.3% | 17.0% éxito cuando intenta |
| **mBART** | 19.5% | 20.1% | 34.7% | 19.9% éxito cuando intenta |
| **CAT (Directo)** | 2.7% | 3.0% | 87.1% | 5.0% éxito cuando intenta |
| **Ensemble Top-3** | 25.7% | 27.2% | 36.5% | 26.0% éxito cuando intenta |

### 📊 Conclusiones del Análisis Detallado

1. **Modelo más consistente**: Llama3.2:3b (85.5%) pero con 0% de precisión
2. **Modelo más preciso**: AINA (46.9% F1-Score) con buena consistencia (74.4%)
3. **Mejor balance precisión-consistencia**: AINA y Google Translate
4. **Mayor degradación temporal**: Google Translate (-18.7%) y AINA (-16.2%)
5. **Más sensible al estilo**: Google Translate (σ=0.080) y AINA (σ=0.074)
6. **Bloque más desafiante**: Minimalista extremo (promedio 15.5% entre todos los modelos)
7. **Bloque más fácil**: Formal técnico (promedio 28.3% entre todos los modelos)

### Hallazgos Clave

1. **AINA lidera en precisión global** con el mejor balance entre exact match (46.98%) y comprensión semántica (68.97%), estableciéndose como el modelo de traducción más efectivo para el par catalán→chino

2. **Experimento de control (CAT)**: La evaluación directa en catalán demuestra que los LLMs comprenden excelentemente el contenido (89.85% contains_correct) pero no pueden generar respuestas en el formato chino esperado (是/不是), validando la necesidad de traducción

3. **Modelo Llama3.2:3b**: Aunque muestra comprensión casi perfecta del contenido (97.32%), su incapacidad para generar el formato de respuesta correcto lo hace inadecuado como traductor directo, justificando para él también el pipeline de traducción+evaluación

4. **Impacto del estilo lingüístico**: 
   - **Formal técnico**: Mejores resultados generales (AINA: 56.5%, Google: 50.1%)
   - **Minimalista extremo**: El más desafiante para todos los modelos (AINA: 28.1%, Google: 18.3%)
   - **Errores gramaticales**: Impacto moderado en la precisión, sugiriendo robustez de los modelos

5. **Efecto del razonamiento interno**: 
   - Modelos con think tags (Qwen3, DeepSeek-R1): Mayor precisión en exact match
   - Yi:9b sin think tags: Mayor contains_correct pero menor precisión exacta
   - Correlación positiva entre uso de razonamiento interno y precisión de formato

6. **Ensemble semántico**: Con 25.84% de exact match, el ensemble no supera a los mejores modelos individuales pero ofrece valor para aplicaciones que requieren diversidad y consenso entre traducciones

### Distribución de Evaluaciones

- **Total de evaluaciones**: 54,172 (promedio 9,029 por modelo)
- **Preguntas evaluadas**: 612 en 6 estilos lingüísticos distintos
- **Iteraciones por pregunta**: 5 para robustez estadística
- **LLMs evaluadores**: 3 modelos (Qwen3:0.6b, Yi:9b, DeepSeek-R1:1.5b)
- **Modelos de traducción evaluados**: 6 + 2 controles (CAT directo, Llama3.2:3b base)

### Contribuciones del Ensemble (Top-2)

AINA: 340 contribuciones (27.8%)
NLLB: 321 contribuciones (26.2%)
M2M100: 257 contribuciones (21.0%)
Google: 195 contribuciones (15.9%)
mBART: 111 contribuciones (9.1%)

## 🤝Contribuciones

Las contribuciones son bienvenidas. Por favor:

Fork el repositorio
Crea una rama para tu feature (git checkout -b feature/NuevaCaracteristica)
Commit tus cambios (git commit -m 'Agregar nueva característica')
Push a la rama (git push origin feature/NuevaCaracteristica)
Abre un Pull Request

## 📄Licencia
Este proyecto está bajo licencia MIT. Ver archivo LICENSE para más detalles.

## 🙏Agradecimientos

Modelos de traducción de código abierto
Comunidad Ollama por infraestructura de LLMs locales
Sentence Transformers por modelos de embeddings

