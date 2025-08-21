#!/usr/bin/env python3
"""
Script para traducir los inputs de un dataset JSONL del catalán al chino
usando Google Translate.

Genera un nuevo JSONL con los inputs traducidos.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional
from enum import Enum
from tqdm import tqdm

# ===== Configuración de Google Translate =====
class TranslationAPI(Enum):
    DEEP_TRANSLATOR = "deep_translator"
    GOOGLETRANS = "googletrans"
    GOOGLE_CLOUD = "google_cloud"


class GoogleTranslator:
    def __init__(self, api_type: TranslationAPI = TranslationAPI.DEEP_TRANSLATOR, 
                 credentials_path: Optional[str] = None):
        self.api_type = api_type
        self.translator = None
        self.client = None
        
        if api_type == TranslationAPI.DEEP_TRANSLATOR:
            try:
                from deep_translator import GoogleTranslator as DeepGoogleTranslator
                self.translator_class = DeepGoogleTranslator
            except ImportError:
                raise ImportError(
                    "deep-translator no está instalado. Instalar con:\n"
                    "pip install deep-translator"
                )
        
        elif api_type == TranslationAPI.GOOGLETRANS:
            try:
                try:
                    from googletrans import Translator
                    self.translator = Translator()
                except ModuleNotFoundError as e:
                    if "cgi" in str(e):
                        raise ImportError(
                            "Error: El módulo 'cgi' no está disponible (Python 3.13+).\n"
                            "Soluciones:\n"
                            "1. Usar --api deep_translator (recomendado)\n"
                            "2. Actualizar httpx: pip install httpx --upgrade\n"
                            "3. Usar Python < 3.13"
                        )
                    raise
            except ImportError:
                raise ImportError(
                    "googletrans no está instalado o tiene problemas de compatibilidad.\n"
                    "Usar --api deep_translator (recomendado)"
                )
        
        elif api_type == TranslationAPI.GOOGLE_CLOUD:
            try:
                from google.cloud import translate_v2 as translate
                if credentials_path:
                    import os
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                self.client = translate.Client()
            except ImportError:
                raise ImportError(
                    "google-cloud-translate no está instalado. Instalar con:\n"
                    "pip install google-cloud-translate"
                )
    
    def translate_batch(self, texts: List[str], src_lang: str = "ca", 
                       tgt_lang: str = "zh-CN", batch_size: int = 10,
                       delay_seconds: float = 0.1) -> List[str]:
        """Traduce una lista de textos con manejo de errores."""
        results = []
        
        if self.api_type == TranslationAPI.DEEP_TRANSLATOR:
            tgt_lang_deep = "zh-CN" if tgt_lang.lower() in ["zh-cn", "zh", "chinese"] else tgt_lang
            src_lang_deep = "ca" if src_lang.lower() in ["ca", "catalan"] else src_lang
        
        for i in tqdm(range(0, len(texts), batch_size), 
                     desc="Traduciendo con Google Translate"):
            batch = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch:
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        if self.api_type == TranslationAPI.DEEP_TRANSLATOR:
                            translator = self.translator_class(
                                source=src_lang_deep,
                                target=tgt_lang_deep
                            )
                            translated = translator.translate(text)
                        
                        elif self.api_type == TranslationAPI.GOOGLETRANS:
                            result = self.translator.translate(
                                text, 
                                src=src_lang, 
                                dest=tgt_lang.lower().replace("-", "_")
                            )
                            translated = result.text
                        
                        elif self.api_type == TranslationAPI.GOOGLE_CLOUD:
                            result = self.client.translate(
                                text,
                                source_language=src_lang,
                                target_language=tgt_lang
                            )
                            translated = result['translatedText']
                        
                        batch_results.append(translated)
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"\nError traduciendo después de {max_retries} intentos: {str(e)[:100]}")
                            print(f"Texto problemático: {text[:50]}...")
                            # En caso de error, mantener el texto original
                            batch_results.append(text)
                        else:
                            time.sleep(1 * retry_count)  # Backoff exponencial
            
            results.extend(batch_results)
            
            # Pausa entre batches para evitar rate limiting
            if i + batch_size < len(texts):
                time.sleep(delay_seconds)
        
        return results


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
                           translator: GoogleTranslator,
                           input_field: str = "inputs",
                           batch_size: int = 10,
                           delay_seconds: float = 0.1):
    """
    Traduce los inputs de un dataset JSONL.
    
    Args:
        input_file: Archivo JSONL de entrada
        output_file: Archivo JSONL de salida
        translator: Instancia del traductor de Google
        input_field: Campo que contiene el texto a traducir (por defecto "inputs")
        batch_size: Tamaño del batch para la traducción
        delay_seconds: Pausa entre batches
    """
    
    # Leer el dataset original
    print(f"Leyendo dataset: {input_file}")
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
            # Si no es string, convertir a string
            texts_to_translate.append(str(text))
    
    print(f"Traduciendo {len(texts_to_translate)} textos...")
    
    # Traducir los textos
    translated_texts = translator.translate_batch(
        texts_to_translate,
        src_lang="ca",
        tgt_lang="zh-CN",
        batch_size=batch_size,
        delay_seconds=delay_seconds
    )
    
    # Crear el nuevo dataset con textos traducidos
    translated_data = []
    for i, item in enumerate(data):
        new_item = item.copy()  # Copiar todos los campos originales
        new_item[input_field] = translated_texts[i]  # Reemplazar el campo traducido
        translated_data.append(new_item)
    
    # Guardar el resultado
    write_jsonl(translated_data, output_file)
    print(f"Dataset traducido guardado en: {output_file}")
    
    return translated_data


def main():
    parser = argparse.ArgumentParser(
        description="Traduce los inputs de un dataset JSONL del catalán al chino usando Google Translate"
    )
    
    # Archivos de entrada y salida
    parser.add_argument("input_file", type=str,
                       help="Archivo JSONL de entrada")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Archivo JSONL de salida (por defecto: test-llm-gt.jsonl)")
    
    # Campo a traducir
    parser.add_argument("--input_field", type=str, default="input",
                       help="Campo que contiene el texto a traducir (por defecto: 'inputs')")
    
    # Configuración de API
    parser.add_argument("--api", type=str, default="deep_translator",
                       choices=["deep_translator", "googletrans", "google_cloud"],
                       help="API a usar (recomendado: deep_translator)")
    parser.add_argument("--credentials", type=str, default=None,
                       help="Path al archivo de credenciales JSON de Google Cloud")
    
    # Configuración de traducción
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Tamaño del batch para llamadas a la API")
    parser.add_argument("--delay", type=float, default=0.1,
                       help="Pausa en segundos entre batches")
    
    # Idiomas
    parser.add_argument("--src_lang", type=str, default="ca",
                       help="Código de idioma fuente (por defecto: 'ca')")
    parser.add_argument("--tgt_lang", type=str, default="zh-CN",
                       help="Código de idioma destino (por defecto: 'zh-CN')")
    
    args = parser.parse_args()
    
    # Configurar paths
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_path}")
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # Generar nombre automático
        stem = input_path.stem
        output_path = input_path.parent / "test-llm-gt.jsonl"
    
    # Configurar API
    api_type = TranslationAPI[args.api.upper().replace("-", "_")]
    
    # Mostrar configuración
    print("="*60)
    print("CONFIGURACIÓN DE TRADUCCIÓN")
    print("="*60)
    print(f"Archivo de entrada: {input_path}")
    print(f"Archivo de salida: {output_path}")
    print(f"Campo a traducir: {args.input_field}")
    print(f"API: {args.api}")
    print(f"Dirección: {args.src_lang} → {args.tgt_lang}")
    print(f"Batch size: {args.batch_size}")
    print(f"Delay: {args.delay}s")
    print("="*60)
    
    # Verificar si el archivo de salida ya existe
    if output_path.exists():
        response = input(f"El archivo {output_path} ya existe. ¿Sobrescribir? (y/N): ")
        if response.lower() not in ['y', 'yes', 'sí', 's']:
            print("Operación cancelada.")
            return
    
    try:
        # Crear traductor
        translator = GoogleTranslator(
            api_type=api_type,
            credentials_path=args.credentials
        )
        
        # Traducir dataset
        translated_data = translate_dataset_inputs(
            input_file=input_path,
            output_file=output_path,
            translator=translator,
            input_field=args.input_field,
            batch_size=args.batch_size,
            delay_seconds=args.delay
        )
        
        print(f"\n✅ ¡Traducción completada exitosamente!")
        print(f"📄 {len(translated_data)} elementos traducidos")
        print(f"💾 Guardado en: {output_path}")
        
        # Mostrar ejemplo del resultado
        if translated_data:
            print(f"\n📋 Ejemplo de traducción:")
            example = translated_data[0]
            original = example.get(args.input_field + "_original", "N/A")
            translated = example.get(args.input_field, "N/A")
            print(f"   Original (ca): {str(original)[:100]}...")
            print(f"   Traducido (zh): {str(translated)[:100]}...")
        
    except Exception as e:
        print(f"\n❌ Error durante la traducción: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()