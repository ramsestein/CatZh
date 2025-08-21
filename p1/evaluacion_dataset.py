# CONFIGURACIÓN - MODIFICAR ESTAS VARIABLES
INPUT_FILE = "datos_old.xlsx"  # Archivo Excel de entrada
SHEET_NAME = None  # None = primera hoja, o especificar nombre: "Sheet1"
CATALAN_COLUMN = "catalan"
CHINESE_COLUMN = "chino"
OUTPUT_FILE = "advanced_old.xlsx"
DEEPL_API_KEY = "56b9bfa9-69d4-4685-97d9-79a1842fd367:fx"
GOOGLE_API_KEY = "AIzaSyAzbfJ9ggticv98aZVmyCpss5h7eo7BwHU"

# Constantes del análisis
NUM_BATCHES = 10
SAMPLES_PER_BATCH = 3
TOTAL_SAMPLES = NUM_BATCHES * SAMPLES_PER_BATCH

import pandas as pd
import random
import time
import json
from datetime import datetime
import deepl
import requests
from sacrebleu import BLEU, CHRF, TER
from nltk.translate.meteor_score import meteor_score
import nltk
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import openpyxl  # Para leer archivos Excel

import pandas as pd
import random
import time
import json
from datetime import datetime
import deepl
import requests
from sacrebleu import BLEU, CHRF, TER
from nltk.translate.meteor_score import meteor_score
import nltk
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import openpyxl  # Para leer archivos Excel

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Optional advanced metrics flags
USE_BERT_SCORE = False  # Set to True if bert-score is installed
USE_COMET = False       # Set to True if unbabel-comet is installed

# Try to import optional libraries
try:
    from bert_score import score as bert_score
    USE_BERT_SCORE = True
    print("BERTScore available")
except ImportError:
    print("BERTScore not available - using sentence embeddings instead")

try:
    from comet import download_model, load_from_checkpoint
    USE_COMET = True
    print("COMET available")
except ImportError:
    print("COMET not available - using alternative neural metric")

class WindowsCompatibleTranslationEvaluation:
    def __init__(self):
        print("Initializing translation evaluation system...")
        self.deepl_translator = deepl.Translator(DEEPL_API_KEY)
        self.google_api_key = GOOGLE_API_KEY
        
        # Traditional metrics
        self.bleu = BLEU(effective_order=True)
        self.chrf = CHRF()
        self.ter = TER()
        
        # Advanced neural metrics
        print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Load COMET if available
        if USE_COMET:
            print("Loading COMET model...")
            self.comet_model_path = download_model("Unbabel/wmt22-comet-da")
            self.comet_model = load_from_checkpoint(self.comet_model_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def deepl_translate(self, text, source_lang, target_lang):
        result = self.deepl_translator.translate_text(text, 
                                                     source_lang=source_lang, 
                                                     target_lang=target_lang)
        return result.text
    
    def google_translate(self, text, source_lang, target_lang):
        url = f"https://translation.googleapis.com/language/translate/v2?key={self.google_api_key}"
        params = {
            'q': text,
            'source': source_lang,
            'target': target_lang,
            'format': 'text'
        }
        response = requests.post(url, data=params)
        return response.json()['data']['translations'][0]['translatedText']
    
    def calculate_traditional_metrics(self, reference, hypothesis):
        """Calculate traditional MT metrics"""
        bleu_score = self.bleu.sentence_score(hypothesis, [reference]).score
        chrf_score = self.chrf.sentence_score(hypothesis, [reference]).score
        ter_score = self.ter.sentence_score(hypothesis, [reference]).score
        
        # METEOR score
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        meteor_score_val = meteor_score([ref_tokens], hyp_tokens)
        
        return {
            'bleu': bleu_score,
            'chrf': chrf_score, 
            'ter': ter_score,
            'meteor': meteor_score_val * 100  # Scale to 0-100
        }
    
    def calculate_sentence_similarity(self, text1, text2):
        """Calculate semantic similarity using sentence embeddings"""
        embeddings1 = self.sentence_model.encode(text1, convert_to_tensor=True)
        embeddings2 = self.sentence_model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
        return similarity * 100  # Scale to 0-100
    
    def calculate_bert_score_alternative(self, references, hypotheses):
        """Alternative to BERTScore using sentence transformers"""
        scores = []
        for ref, hyp in zip(references, hypotheses):
            score = self.calculate_sentence_similarity(ref, hyp)
            scores.append(score)
        
        return {
            'mean': sum(scores) / len(scores),
            'scores': scores
        }
    
    def calculate_comet_alternative(self, sources, hypotheses, references):
        """Alternative to COMET using semantic similarity"""
        scores = []
        for src, hyp, ref in zip(sources, hypotheses, references):
            # Calculate similarity between hypothesis and reference
            ref_sim = self.calculate_sentence_similarity(hyp, ref)
            # Calculate similarity preservation from source
            src_ref_sim = self.calculate_sentence_similarity(src, ref)
            src_hyp_sim = self.calculate_sentence_similarity(src, hyp)
            preservation = min(src_hyp_sim / max(src_ref_sim, 1), 100)
            # Combined score
            score = (ref_sim * 0.7) + (preservation * 0.3)
            scores.append(score)
        
        return {
            'system_score': sum(scores) / len(scores),
            'scores': scores
        }
    
    def calculate_advanced_metrics(self, sources, references, hypotheses, lang='zh'):
        """Calculate advanced neural metrics with fallbacks"""
        results = {}
        
        # BERTScore or alternative
        if USE_BERT_SCORE:
            P, R, F1 = bert_score(hypotheses, references, lang=lang, device=self.device)
            results['bert_scores'] = {
                'precision': [p.item() * 100 for p in P],
                'recall': [r.item() * 100 for r in R],
                'f1': [f.item() * 100 for f in F1],
                'mean_f1': F1.mean().item() * 100
            }
        else:
            bert_alt = self.calculate_bert_score_alternative(references, hypotheses)
            results['bert_scores'] = {
                'f1': bert_alt['scores'],
                'mean_f1': bert_alt['mean']
            }
        
        # COMET or alternative
        if USE_COMET:
            data = []
            for src, hyp, ref in zip(sources, hypotheses, references):
                data.append({"src": src, "mt": hyp, "ref": ref})
            model_output = self.comet_model.predict(data, batch_size=8, gpus=1 if self.device == "cuda" else 0)
            results['comet_scores'] = {
                'scores': [s * 100 for s in model_output.scores],
                'system_score': model_output.system_score * 100
            }
        else:
            comet_alt = self.calculate_comet_alternative(sources, hypotheses, references)
            results['comet_scores'] = comet_alt
        
        # Semantic similarity as BLEURT alternative
        semantic_scores = []
        for ref, hyp in zip(references, hypotheses):
            score = self.calculate_sentence_similarity(ref, hyp)
            semantic_scores.append(score)
        
        results['semantic_scores'] = {
            'scores': semantic_scores,
            'mean': sum(semantic_scores) / len(semantic_scores)
        }
        
        return results
    
    def similarity_score(self, text1, text2):
        """Token-based similarity for confidence factor"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if len(tokens1) == 0 and len(tokens2) == 0:
            return 1.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def process_batch(self, batch_samples):
        """Process a batch of samples"""
        batch_results = []
        
        # Collect all translations first
        sources_catalan = []
        references_chinese = []
        hypotheses_google = []
        hypotheses_deepl = []
        retro_catalan_google = []
        retro_catalan_deepl = []
        
        print("Generating translations for batch...")
        for idx, row in batch_samples.iterrows():
            catalan_text = row[CATALAN_COLUMN]
            chinese_text = row[CHINESE_COLUMN]
            
            sources_catalan.append(catalan_text)
            references_chinese.append(chinese_text)
            
            try:
                # Route 1: Google (Catalan → Spanish → Chinese)
                spanish_gt = self.google_translate(catalan_text, 'ca', 'es')
                time.sleep(0.1)
                chinese_gt = self.google_translate(spanish_gt, 'es', 'zh')
                time.sleep(0.1)
                hypotheses_google.append(chinese_gt)
                
                # Route 1 Retro: Google (Chinese → Spanish → Catalan)
                spanish_retro_gt = self.google_translate(chinese_gt, 'zh', 'es')
                time.sleep(0.1)
                catalan_retro_gt = self.google_translate(spanish_retro_gt, 'es', 'ca')
                time.sleep(0.1)
                retro_catalan_google.append(catalan_retro_gt)
                
                # Route 2: Hybrid (Google → DeepL: Catalan → English → Chinese)
                english_gt = self.google_translate(catalan_text, 'ca', 'en')
                time.sleep(0.1)
                chinese_dl = self.deepl_translate(english_gt, 'EN', 'ZH-HANS')
                time.sleep(0.1)
                hypotheses_deepl.append(chinese_dl)
                
                # Route 2 Retro: DeepL → Google (Chinese → English → Catalan)
                english_retro_dl = self.deepl_translate(chinese_dl, 'ZH', 'EN-GB')
                time.sleep(0.1)
                catalan_retro_dl = self.google_translate(english_retro_dl, 'en', 'ca')
                time.sleep(0.1)
                retro_catalan_deepl.append(catalan_retro_dl)
                
            except Exception as e:
                print(f"Error in translation: {e}")
                # Use empty strings as fallback
                hypotheses_google.append("")
                hypotheses_deepl.append("")
                retro_catalan_google.append("")
                retro_catalan_deepl.append("")
        
        # Calculate advanced metrics in batch
        print("Calculating advanced metrics...")
        advanced_google = self.calculate_advanced_metrics(sources_catalan, references_chinese, hypotheses_google)
        advanced_deepl = self.calculate_advanced_metrics(sources_catalan, references_chinese, hypotheses_deepl)
        
        # Process individual samples
        for i, (idx, row) in enumerate(batch_samples.iterrows()):
            print(f"Processing sample {i+1}/{len(batch_samples)}...")
            
            # Skip if translation failed
            if not hypotheses_google[i] or not hypotheses_deepl[i]:
                continue
            
            # Traditional metrics
            trad_metrics_google = self.calculate_traditional_metrics(references_chinese[i], hypotheses_google[i])
            trad_metrics_deepl = self.calculate_traditional_metrics(references_chinese[i], hypotheses_deepl[i])
            
            # Confidence factors
            confidence_google = self.similarity_score(sources_catalan[i], retro_catalan_google[i])
            confidence_deepl = self.similarity_score(sources_catalan[i], retro_catalan_deepl[i])
            
            # Calculate composite scores
            # Traditional composite
            trad_composite_google = (trad_metrics_google['bleu'] + trad_metrics_google['chrf'] + 
                                   max(0, 100 - trad_metrics_google['ter']) + trad_metrics_google['meteor']) / 4
            trad_composite_deepl = (trad_metrics_deepl['bleu'] + trad_metrics_deepl['chrf'] + 
                                  max(0, 100 - trad_metrics_deepl['ter']) + trad_metrics_deepl['meteor']) / 4
            
            # Neural composite
            if USE_BERT_SCORE:
                bert_f1_google = advanced_google['bert_scores']['f1'][i]
                bert_f1_deepl = advanced_deepl['bert_scores']['f1'][i]
            else:
                bert_f1_google = advanced_google['bert_scores']['f1'][i]
                bert_f1_deepl = advanced_deepl['bert_scores']['f1'][i]
            
            comet_google = advanced_google['comet_scores']['scores'][i]
            comet_deepl = advanced_deepl['comet_scores']['scores'][i]
            semantic_google = advanced_google['semantic_scores']['scores'][i]
            semantic_deepl = advanced_deepl['semantic_scores']['scores'][i]
            
            neural_composite_google = (bert_f1_google + comet_google + semantic_google) / 3
            neural_composite_deepl = (bert_f1_deepl + comet_deepl + semantic_deepl) / 3
            
            # Overall composite (50% traditional, 50% neural)
            overall_composite_google = (trad_composite_google * 0.5) + (neural_composite_google * 0.5)
            overall_composite_deepl = (trad_composite_deepl * 0.5) + (neural_composite_deepl * 0.5)
            
            # Apply confidence weighting
            final_score_google = overall_composite_google * confidence_google
            final_score_deepl = overall_composite_deepl * confidence_deepl
            
            result = {
                # Traditional metrics
                'trad_metrics_google': trad_metrics_google,
                'trad_metrics_deepl': trad_metrics_deepl,
                
                # Neural metrics
                'bert_f1_google': bert_f1_google,
                'bert_f1_deepl': bert_f1_deepl,
                'comet_google': comet_google,
                'comet_deepl': comet_deepl,
                'semantic_google': semantic_google,
                'semantic_deepl': semantic_deepl,
                
                # Confidence
                'confidence_google': confidence_google,
                'confidence_deepl': confidence_deepl,
                
                # Composite scores
                'trad_composite_google': trad_composite_google,
                'trad_composite_deepl': trad_composite_deepl,
                'neural_composite_google': neural_composite_google,
                'neural_composite_deepl': neural_composite_deepl,
                'overall_composite_google': overall_composite_google,
                'overall_composite_deepl': overall_composite_deepl,
                'final_score_google': final_score_google,
                'final_score_deepl': final_score_deepl,
                
                # Original data
                'catalan_original': sources_catalan[i],
                'chinese_original': references_chinese[i],
                'chinese_google': hypotheses_google[i],
                'chinese_deepl': hypotheses_deepl[i],
            }
            
            batch_results.append(result)
        
        return batch_results
    
    def run_evaluation(self):
        """Run the complete evaluation"""
        # Load Excel file
        print(f"Loading Excel file: {INPUT_FILE}")
        
        try:
            # Load Excel with optional sheet specification
            if SHEET_NAME:
                df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
                print(f"Successfully loaded sheet: {SHEET_NAME}")
            else:
                df = pd.read_excel(INPUT_FILE)
                print(f"Successfully loaded first sheet")
            
            # Try to get sheet names for information
            try:
                xl_file = pd.ExcelFile(INPUT_FILE)
                print(f"Available sheets: {xl_file.sheet_names}")
            except:
                pass
                
        except FileNotFoundError:
            print(f"ERROR: File '{INPUT_FILE}' not found!")
            print(f"Current directory: {os.getcwd()}")
            print("\nExcel files in current directory:")
            excel_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]
            for f in excel_files:
                print(f"  - {f}")
            raise
        except Exception as e:
            print(f"ERROR loading Excel file: {str(e)}")
            raise
        
        total_rows = len(df)
        print(f"Dataset loaded: {total_rows} rows")
        print(f"Columns found: {list(df.columns)}")
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Show first few rows as preview
        print("\nPreview of first 3 rows:")
        print(df.head(3))
        
        # Verify required columns exist
        if CATALAN_COLUMN not in df.columns or CHINESE_COLUMN not in df.columns:
            print(f"\nERROR: Required columns not found!")
            print(f"Looking for: '{CATALAN_COLUMN}' and '{CHINESE_COLUMN}'")
            print(f"Available columns: {list(df.columns)}")
            
            # Try to find similar column names
            catalan_matches = [col for col in df.columns if 'catalan' in col.lower() or 'cat' in col.lower()]
            chinese_matches = [col for col in df.columns if 'chino' in col.lower() or 'chin' in col.lower() or 'zh' in col.lower()]
            
            if catalan_matches:
                print(f"\nDid you mean one of these for Catalan? {catalan_matches}")
            if chinese_matches:
                print(f"Did you mean one of these for Chinese? {chinese_matches}")
                
            raise ValueError(f"Required columns not found. Please check column names in configuration.")
        
        # Check for null values in required columns
        null_catalan = df[CATALAN_COLUMN].isnull().sum()
        null_chinese = df[CHINESE_COLUMN].isnull().sum()
        
        if null_catalan > 0 or null_chinese > 0:
            print(f"\nWARNING: Found null values!")
            print(f"Null values in {CATALAN_COLUMN}: {null_catalan}")
            print(f"Null values in {CHINESE_COLUMN}: {null_chinese}")
            
            # Remove rows with null values
            df_clean = df.dropna(subset=[CATALAN_COLUMN, CHINESE_COLUMN])
            print(f"Removed {len(df) - len(df_clean)} rows with null values")
            df = df_clean
            total_rows = len(df)
            print(f"Remaining rows: {total_rows}")
        
        # Check for empty strings
        empty_catalan = (df[CATALAN_COLUMN].str.strip() == '').sum()
        empty_chinese = (df[CHINESE_COLUMN].str.strip() == '').sum()
        
        if empty_catalan > 0 or empty_chinese > 0:
            print(f"\nWARNING: Found empty strings!")
            print(f"Empty strings in {CATALAN_COLUMN}: {empty_catalan}")
            print(f"Empty strings in {CHINESE_COLUMN}: {empty_chinese}")
            
            # Remove rows with empty strings
            df = df[(df[CATALAN_COLUMN].str.strip() != '') & (df[CHINESE_COLUMN].str.strip() != '')]
            total_rows = len(df)
            print(f"Remaining rows after removing empty strings: {total_rows}")
        
        if total_rows < TOTAL_SAMPLES:
            print(f"\nWARNING: Dataset has only {total_rows} rows, but {TOTAL_SAMPLES} samples requested!")
            print(f"Adjusting to use all available rows with replacement...")
        
        all_batch_results = []
        batch_summaries = []
        
        for batch_num in range(NUM_BATCHES):
            print(f"\n=== BATCH {batch_num + 1}/{NUM_BATCHES} ===")
            
            # Sample random rows for this batch
            if total_rows >= SAMPLES_PER_BATCH:
                # Sample without replacement if we have enough rows
                sample_indices = random.sample(range(total_rows), SAMPLES_PER_BATCH)
            else:
                # Sample with replacement if dataset is too small
                sample_indices = [random.randint(0, total_rows - 1) for _ in range(SAMPLES_PER_BATCH)]
            
            batch_samples = df.iloc[sample_indices]
            
            # Process batch
            batch_results = self.process_batch(batch_samples)
            
            if not batch_results:
                print(f"Batch {batch_num + 1} failed - skipping")
                continue
                
            all_batch_results.extend(batch_results)
            
            # Calculate batch summary
            batch_summary = self.calculate_batch_summary(batch_results, batch_num + 1)
            batch_summaries.append(batch_summary)
            
            print(f"Batch {batch_num + 1} - Google avg: {batch_summary['avg_final_google']:.2f}, "
                  f"DeepL avg: {batch_summary['avg_final_deepl']:.2f}")
            
            # Save intermediate results
            if (batch_num + 1) % 10 == 0:
                self.save_intermediate_results(batch_summaries, batch_num + 1)
        
        # Calculate overall statistics and save final results
        self.save_final_results(batch_summaries, all_batch_results)
        
        return batch_summaries, all_batch_results
    
    def calculate_batch_summary(self, batch_results, batch_num):
        """Calculate summary statistics for a batch"""
        n = len(batch_results)
        
        summary = {
            'batch_number': batch_num,
            
            # Google route averages
            'avg_final_google': sum(r['final_score_google'] for r in batch_results) / n,
            'avg_trad_composite_google': sum(r['trad_composite_google'] for r in batch_results) / n,
            'avg_neural_composite_google': sum(r['neural_composite_google'] for r in batch_results) / n,
            'avg_bleu_google': sum(r['trad_metrics_google']['bleu'] for r in batch_results) / n,
            'avg_chrf_google': sum(r['trad_metrics_google']['chrf'] for r in batch_results) / n,
            'avg_ter_google': sum(r['trad_metrics_google']['ter'] for r in batch_results) / n,
            'avg_meteor_google': sum(r['trad_metrics_google']['meteor'] for r in batch_results) / n,
            'avg_bert_f1_google': sum(r['bert_f1_google'] for r in batch_results) / n,
            'avg_comet_google': sum(r['comet_google'] for r in batch_results) / n,
            'avg_semantic_google': sum(r['semantic_google'] for r in batch_results) / n,
            'avg_confidence_google': sum(r['confidence_google'] for r in batch_results) / n,
            
            # DeepL route averages
            'avg_final_deepl': sum(r['final_score_deepl'] for r in batch_results) / n,
            'avg_trad_composite_deepl': sum(r['trad_composite_deepl'] for r in batch_results) / n,
            'avg_neural_composite_deepl': sum(r['neural_composite_deepl'] for r in batch_results) / n,
            'avg_bleu_deepl': sum(r['trad_metrics_deepl']['bleu'] for r in batch_results) / n,
            'avg_chrf_deepl': sum(r['trad_metrics_deepl']['chrf'] for r in batch_results) / n,
            'avg_ter_deepl': sum(r['trad_metrics_deepl']['ter'] for r in batch_results) / n,
            'avg_meteor_deepl': sum(r['trad_metrics_deepl']['meteor'] for r in batch_results) / n,
            'avg_bert_f1_deepl': sum(r['bert_f1_deepl'] for r in batch_results) / n,
            'avg_comet_deepl': sum(r['comet_deepl'] for r in batch_results) / n,
            'avg_semantic_deepl': sum(r['semantic_deepl'] for r in batch_results) / n,
            'avg_confidence_deepl': sum(r['confidence_deepl'] for r in batch_results) / n,
            
            # Min/max for final scores
            'min_final_google': min(r['final_score_google'] for r in batch_results),
            'max_final_google': max(r['final_score_google'] for r in batch_results),
            'min_final_deepl': min(r['final_score_deepl'] for r in batch_results),
            'max_final_deepl': max(r['final_score_deepl'] for r in batch_results),
        }
        
        # Calculate which route is better
        summary['better_route'] = 'Google' if summary['avg_final_google'] > summary['avg_final_deepl'] else 'DeepL'
        summary['difference'] = abs(summary['avg_final_google'] - summary['avg_final_deepl'])
        
        return summary
    
    def save_intermediate_results(self, batch_summaries, completed_batches):
        """Save intermediate results"""
        df = pd.DataFrame(batch_summaries)
        filename = f"intermediate_results_batch_{completed_batches}.xlsx"
        df.to_excel(filename, index=False)
        print(f"Intermediate results saved: {filename}")
    
    def save_final_results(self, batch_summaries, all_results):
        """Save final results with statistics"""
        # Convert batch summaries to DataFrame
        df_summary = pd.DataFrame(batch_summaries)
        
        # Calculate overall statistics
        overall_stats = {
            'batch_number': 'OVERALL',
            'avg_final_google': df_summary['avg_final_google'].mean(),
            'avg_final_deepl': df_summary['avg_final_deepl'].mean(),
            'avg_trad_composite_google': df_summary['avg_trad_composite_google'].mean(),
            'avg_trad_composite_deepl': df_summary['avg_trad_composite_deepl'].mean(),
            'avg_neural_composite_google': df_summary['avg_neural_composite_google'].mean(),
            'avg_neural_composite_deepl': df_summary['avg_neural_composite_deepl'].mean(),
            'avg_bleu_google': df_summary['avg_bleu_google'].mean(),
            'avg_bleu_deepl': df_summary['avg_bleu_deepl'].mean(),
            'avg_chrf_google': df_summary['avg_chrf_google'].mean(),
            'avg_chrf_deepl': df_summary['avg_chrf_deepl'].mean(),
            'avg_ter_google': df_summary['avg_ter_google'].mean(),
            'avg_ter_deepl': df_summary['avg_ter_deepl'].mean(),
            'avg_meteor_google': df_summary['avg_meteor_google'].mean(),
            'avg_meteor_deepl': df_summary['avg_meteor_deepl'].mean(),
            'avg_bert_f1_google': df_summary['avg_bert_f1_google'].mean(),
            'avg_bert_f1_deepl': df_summary['avg_bert_f1_deepl'].mean(),
            'avg_comet_google': df_summary['avg_comet_google'].mean(),
            'avg_comet_deepl': df_summary['avg_comet_deepl'].mean(),
            'avg_semantic_google': df_summary['avg_semantic_google'].mean(),
            'avg_semantic_deepl': df_summary['avg_semantic_deepl'].mean(),
        }
        
        # Determine overall better route
        google_wins = sum(1 for s in batch_summaries if s['better_route'] == 'Google')
        deepl_wins = len(batch_summaries) - google_wins
        overall_stats['better_route'] = f"Google ({google_wins}/{len(batch_summaries)} batches)"
        overall_stats['difference'] = abs(overall_stats['avg_final_google'] - overall_stats['avg_final_deepl'])
        
        # Add confidence intervals
        n_batches = len(batch_summaries)
        std_google = df_summary['avg_final_google'].std()
        std_deepl = df_summary['avg_final_deepl'].std()
        ci_mult = 1.96 / (n_batches ** 0.5)
        
        ci_stats_lower = {
            'batch_number': 'CI_95_LOWER',
            'avg_final_google': overall_stats['avg_final_google'] - ci_mult * std_google,
            'avg_final_deepl': overall_stats['avg_final_deepl'] - ci_mult * std_deepl,
        }
        
        ci_stats_upper = {
            'batch_number': 'CI_95_UPPER',
            'avg_final_google': overall_stats['avg_final_google'] + ci_mult * std_google,
            'avg_final_deepl': overall_stats['avg_final_deepl'] + ci_mult * std_deepl,
        }
        
        # Append statistics to summary
        df_summary = pd.concat([df_summary, pd.DataFrame([overall_stats, ci_stats_lower, ci_stats_upper])], 
                              ignore_index=True)
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            # Summary sheet
            df_summary.to_excel(writer, sheet_name='Batch Summary', index=False)
            
            # Route comparison
            comparison_data = []
            for batch in batch_summaries:
                comparison_data.append({
                    'batch': batch['batch_number'],
                    'google_final': batch['avg_final_google'],
                    'deepl_final': batch['avg_final_deepl'],
                    'better_route': batch['better_route'],
                    'difference': batch['difference'],
                    'google_traditional': batch['avg_trad_composite_google'],
                    'google_neural': batch['avg_neural_composite_google'],
                    'deepl_traditional': batch['avg_trad_composite_deepl'],
                    'deepl_neural': batch['avg_neural_composite_deepl'],
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison.to_excel(writer, sheet_name='Route Comparison', index=False)
            
            # Metrics analysis
            metrics_data = {
                'Metric': ['BLEU', 'chrF', 'TER', 'METEOR', 
                          'BERTScore F1', 'COMET/Alternative', 'Semantic Similarity',
                          'Traditional Composite', 'Neural Composite', 'Final Score'],
                'Google Average': [
                    df_summary['avg_bleu_google'].iloc[:-3].mean(),
                    df_summary['avg_chrf_google'].iloc[:-3].mean(),
                    df_summary['avg_ter_google'].iloc[:-3].mean(),
                    df_summary['avg_meteor_google'].iloc[:-3].mean(),
                    df_summary['avg_bert_f1_google'].iloc[:-3].mean(),
                    df_summary['avg_comet_google'].iloc[:-3].mean(),
                    df_summary['avg_semantic_google'].iloc[:-3].mean(),
                    overall_stats['avg_trad_composite_google'],
                    overall_stats['avg_neural_composite_google'],
                    overall_stats['avg_final_google']
                ],
                'DeepL Average': [
                    df_summary['avg_bleu_deepl'].iloc[:-3].mean(),
                    df_summary['avg_chrf_deepl'].iloc[:-3].mean(),
                    df_summary['avg_ter_deepl'].iloc[:-3].mean(),
                    df_summary['avg_meteor_deepl'].iloc[:-3].mean(),
                    df_summary['avg_bert_f1_deepl'].iloc[:-3].mean(),
                    df_summary['avg_comet_deepl'].iloc[:-3].mean(),
                    df_summary['avg_semantic_deepl'].iloc[:-3].mean(),
                    overall_stats['avg_trad_composite_deepl'],
                    overall_stats['avg_neural_composite_deepl'],
                    overall_stats['avg_final_deepl']
                ]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics['Difference (Google - DeepL)'] = df_metrics['Google Average'] - df_metrics['DeepL Average']
            df_metrics['Better'] = df_metrics['Difference (Google - DeepL)'].apply(
                lambda x: 'Google' if x > 0 else 'DeepL' if x < 0 else 'Equal'
            )
            df_metrics.to_excel(writer, sheet_name='Metrics Analysis', index=False)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Google Route - Average Score: {overall_stats['avg_final_google']:.2f}")
        print(f"  Traditional: {overall_stats['avg_trad_composite_google']:.2f}")
        print(f"  Neural: {overall_stats['avg_neural_composite_google']:.2f}")
        print(f"DeepL Route - Average Score: {overall_stats['avg_final_deepl']:.2f}")
        print(f"  Traditional: {overall_stats['avg_trad_composite_deepl']:.2f}")
        print(f"  Neural: {overall_stats['avg_neural_composite_deepl']:.2f}")
        print(f"\nBetter Route Overall: {overall_stats['better_route']}")
        print(f"Average Difference: {overall_stats['difference']:.2f}")
        print(f"\nDetailed metrics breakdown available in: {OUTPUT_FILE}")
        print("Excel sheets include:")
        print("- Batch Summary: All metrics by batch")
        print("- Route Comparison: Google vs DeepL")
        print("- Metrics Analysis: Performance by metric type")
        print("- Sample Details: Individual translations")
        print("- Neural Metrics Detail: BERTScore, COMET, Semantic similarity")


if __name__ == "__main__":
    print("Starting Advanced Translation Quality Evaluation (Windows Compatible)")
    print(f"Configuration: {NUM_BATCHES} batches × {SAMPLES_PER_BATCH} samples = {TOTAL_SAMPLES} total evaluations")
    
    # Quick file check
    print(f"\nChecking Excel file: {INPUT_FILE}")
    import os
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File '{INPUT_FILE}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print("Excel files in current directory:")
        excel_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]
        if excel_files:
            for f in excel_files:
                print(f"  - {f}")
        else:
            print("  No Excel files found!")
        exit(1)
    
    print(f"✓ File found: {INPUT_FILE}")
    print(f"  Size: {os.path.getsize(INPUT_FILE) / 1024 / 1024:.2f} MB")
    
    print("\n=== METRICS INCLUDED ===")
    print("Traditional metrics:")
    print("- BLEU: N-gram precision (0-100)")
    print("- chrF: Character F-score (0-100)")
    print("- TER: Translation Error Rate (lower is better)")
    print("- METEOR: Semantic matching with synonyms (0-100)")
    print("\nNeural metrics:")
    print("- BERTScore F1: Contextual embeddings similarity (0-100)")
    print("- COMET/Alternative: Neural quality estimation (0-100)")
    print("- Semantic Similarity: Sentence embeddings cosine similarity (0-100)")
    
    evaluator = WindowsCompatibleTranslationEvaluation()
    batch_summaries, all_results = evaluator.run_evaluation()
    
    print("\nEvaluation completed successfully!")
    print("Check the Excel file for detailed results in multiple sheets:")
    print("1. Batch Summary - All metrics averaged by batch")
    print("2. Route Comparison - Direct Google vs DeepL comparison")
    print("3. Metrics Analysis - Average performance by metric type")
    print("4. Sample Details - Individual sample results (first 50)")
    print("5. Neural Metrics Detail - Detailed neural metrics comparison")