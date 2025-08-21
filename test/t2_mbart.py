#!/usr/bin/env python3
"""
Script para traducir los inputs de un dataset JSONL del catalán al chino
usando el modelo local mBART-large-50.

Genera un nuevo JSONL con los inputs traducidos.
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import warnings

# Importar configuraciones de modelos
from train.model_configs_tpu import MODEL_CONFIGS

# Suprimir warnings de transformers
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class MBARTTranslator:
    """Traductor usando el modelo mBART-large-50"""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Inicializa el traductor mBART-large-50.
        
        Args:
            model_path: Path al modelo (por defecto usa la configuración)
            device: Dispositivo a usar ("auto", "cuda", "cpu")
        """
        self.config = MODEL_CONFIGS["mbart_large"]
        if model_path:
            self.config.model_id = model_path
            
        # Configurar dispositivo
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Usando dispositivo: {self.device}")
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo y tokenizer mBART."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            print(f"Cargando modelo mBART-large-50 desde: {self.config.model_id}")
            
            # Cargar tokenizer sin configurar idioma automáticamente
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                src_lang=None,  # No configurar idioma fuente automáticamente
                tgt_lang=None   # No configurar idioma destino automáticamente
            )
            
            # Cargar modelo
            if self.device.type == "cuda":
                # Usar half precision en GPU para ahorrar memoria (mBART es grande)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                # CPU
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_id,
                    torch_dtype=torch.float32
                )
                self.model = self.model.to(self.device)
            
            # Configurar idiomas para mBART
            # Catalán: "ca_XX" -> Chino: "zh_CN"
            self.src_lang = self.config.special_tokens["src_lang"]  # "ca_XX"
            self.tgt_lang = self.config.special_tokens["tgt_lang"]  # "zh_CN"
            
            # Configurar el tokenizer para el idioma fuente
            # Para modelos fine-tuned, configurar idiomas manualmente
            #try:
            #    self.tokenizer.src_lang = self.src_lang
            #except (KeyError, ValueError):
            #    print(f"⚠️  {self.src_lang} no está en el vocabulario base, usando configuración manual")
            #    # No configurar src_lang automáticamente
            
            print(f"✅ Modelo mBART-large-50 cargado exitosamente")
            print(f"📋 Configuración: {self.src_lang} → {self.tgt_lang}")
            
        except Exception as e:
            print(f"❌ Error cargando el modelo: {e}")
            raise
    
    def translate_batch(self, texts: List[str], batch_size: int = None, 
                       max_length: int = 128) -> List[str]:
        """
        Traduce una lista de textos.
        
        Args:
            texts: Lista de textos en catalán
            batch_size: Tamaño del batch (por defecto usa la configuración)
            max_length: Longitud máxima de la secuencia
            
        Returns:
            Lista de textos traducidos al chino
        """
        if batch_size is None:
            batch_size = self.config.batch_size
            
        # Ajustar batch_size según la GPU disponible (mBART es más grande)
        if self.device.type == "cuda":
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                if gpu_memory < 8:  # Menos de 8GB
                    batch_size = min(batch_size, 8)
                elif gpu_memory < 16:  # Menos de 16GB
                    batch_size = min(batch_size, 16)
                else:  # 16GB+
                    batch_size = min(batch_size, 32)
            except:
                batch_size = min(batch_size, 8)  # Conservative default
        else:
            batch_size = min(batch_size, 4)  # CPU es más lento para modelos grandes
        
        print(f"Usando batch_size: {batch_size}")
        
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), 
                     desc="Traduciendo con mBART"):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Tokenizar con idioma fuente configurado
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # Generar traducción
                with torch.no_grad():
                    # Para mBART, necesitamos obtener el token ID del idioma destino
                    # Intentar obtener el ID del token destino
                    try:
                        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
                        if forced_bos_token_id == self.tokenizer.unk_token_id:
                            # Si devuelve UNK, probar sin forzar idioma
                            forced_bos_token_id = None
                    except:
                        forced_bos_token_id = None
                        print(f"⚠️  No se pudo configurar idioma destino {self.tgt_lang}")

                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=max_length,
                        num_beams=4,
                        do_sample=False,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decodificar
                batch_translations = self.tokenizer.batch_decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                
                results.extend(batch_translations)
                
            except Exception as e:
                print(f"\n⚠️  Error en batch {i//batch_size + 1}: {e}")
                # En caso de error, mantener textos originales
                results.extend(batch_texts)
                
                # Liberar memoria si hay error
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        
        return results
    
    def translate_single(self, text: str, max_length: int = 128) -> str:
        """Traduce un solo texto."""
        return self.translate_batch([text], batch_size=1, max_length=max_length)[0]


def read_jsonl(file_path: Path) -> List[dict]:
    """Lee un archivo JSONL y devuelve una lista de diccionarios."""
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


def translate_dataset_inputs(input_file: Path, output_file: Path, 
                           translator: MBARTTranslator,
                           input_field: str = "inputs",
                           batch_size: int = None,
                           max_length: int = 128):
    """
    Traduce los inputs de un dataset JSONL usando el modelo mBART.
    
    Args:
        input_file: Archivo JSONL de entrada
        output_file: Archivo JSONL de salida
        translator: Instancia del traductor mBART
        input_field: Campo que contiene el texto a traducir
        batch_size: Tamaño del batch
        max_length: Longitud máxima de secuencia
    """
    
    # Leer el dataset original
    print(f"📖 Leyendo dataset: {input_file}")
    data = read_jsonl(input_file)
    
    if not data:
        raise ValueError(f"No se pudieron leer datos de {input_file}")
    
    # Verificar que el campo existe
    if input_field not in data[0]:
        available_fields = list(data[0].keys())
        raise ValueError(f"Campo '{input_field}' no encontrado. Campos disponibles: {available_fields}")
    
    # Extraer textos a traducir
    texts_to_translate = []
    for item in data:
        text = item.get(input_field, "")
        if isinstance(text, str):
            texts_to_translate.append(text)
        else:
            texts_to_translate.append(str(text))
    
    print(f"🔤 Traduciendo {len(texts_to_translate)} textos con modelo mBART...")
    
    # Traducir los textos
    translated_texts = translator.translate_batch(
        texts_to_translate,
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Crear el nuevo dataset con textos traducidos
    translated_data = []
    for i, item in enumerate(data):
        new_item = item.copy()
        new_item[input_field] = translated_texts[i]
        translated_data.append(new_item)
    
    # Guardar el resultado
    write_jsonl(translated_data, output_file)
    print(f"💾 Dataset traducido guardado en: {output_file}")
    
    return translated_data


def main():
    parser = argparse.ArgumentParser(
        description="Traduce los inputs de un dataset JSONL del catalán al chino usando el modelo mBART-large-50"
    )
    
    # Archivos de entrada y salida
    parser.add_argument("input_file", type=str,
                       help="Archivo JSONL de entrada")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Archivo JSONL de salida (por defecto: test-llm-nllb200.jsonl)")
    
    # Campo a traducir
    parser.add_argument("--input_field", type=str, default="input",
                       help="Campo que contiene el texto a traducir (por defecto: 'input')")
    
    # Configuración del modelo
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path personalizado al modelo mBART (opcional)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Dispositivo a usar")
    
    # Configuración de traducción
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Tamaño del batch (por defecto: usa configuración del modelo)")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Longitud máxima de secuencia")
    
    args = parser.parse_args()
    
    # Configurar paths
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_path}")
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        stem = input_path.stem
        output_path = input_path.parent / "test-llm-mbart.jsonl"
    
    # Mostrar configuración
    print("="*60)
    print("CONFIGURACIÓN DE TRADUCCIÓN - MODELO mBART-LARGE-50")
    print("="*60)
    print(f"📁 Archivo de entrada: {input_path}")
    print(f"📁 Archivo de salida: {output_path}")
    print(f"🏷️  Campo a traducir: {args.input_field}")
    print(f"🤖 Modelo: mBART-large-50")
    print(f"🌐 Dirección: ca_XX → zh_CN")
    print(f"⚙️  Dispositivo: {args.device}")
    print(f"📦 Batch size: {args.batch_size or 'automático'}")
    print(f"📏 Max length: {args.max_length}")
    print("="*60)
    
    # Verificar si el archivo de salida ya existe
    if output_path.exists():
        response = input(f"El archivo {output_path} ya existe. ¿Sobrescribir? (y/N): ")
        if response.lower() not in ['y', 'yes', 'sí', 's']:
            print("Operación cancelada.")
            return
    
    try:
        # Crear traductor
        print("\n🚀 Inicializando modelo mBART-large-50...")
        translator = MBARTTranslator(
            model_path=args.model_path,
            device=args.device
        )
        
        # Traducir dataset
        translated_data = translate_dataset_inputs(
            input_file=input_path,
            output_file=output_path,
            translator=translator,
            input_field=args.input_field,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        print(f"\n✅ ¡Traducción completada exitosamente!")
        print(f"📄 {len(translated_data)} elementos traducidos")
        print(f"💾 Guardado en: {output_path}")
        
        # Mostrar ejemplo del resultado
        if translated_data:
            print(f"\n📋 Ejemplo de traducción:")
            # Buscar el elemento original para comparar
            original_data = read_jsonl(input_path)
            if original_data:
                original_text = original_data[0].get(args.input_field, "N/A")
                translated_text = translated_data[0].get(args.input_field, "N/A")
                print(f"   Original (ca_XX): {str(original_text)[:100]}...")
                print(f"   Traducido (zh_CN): {str(translated_text)[:100]}...")
        
    except Exception as e:
        print(f"\n❌ Error durante la traducción: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Limpiar memoria GPU si se usó
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()