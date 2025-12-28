
import os
import re
import json
import time
import random
import unicodedata
import pathlib
import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, AutoModelForSeq2SeqLM

# 8. Set environment variables for MLOps compliance
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Ensure Windows compatibility for sub-processes
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Global Configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/bart-large-cnn"
DATA_DIR = "./local-test-data"
OUTPUT_DIR = "./summarization_results"
MODEL_SAVE_DIR = "./agent_workspace/trained_models"

# Ensure directories exist
pathlib.Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

class ReviewDataset(Dataset):
    """Custom Dataset for Customer Reviews."""
    def __init__(self, dataframe: pd.DataFrame, tokenizer: BartTokenizer, max_length: int = 1024):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = str(item['cleaned_text'])
        review_id = item['review_id']
        
        # 2. Tokenization & Truncation (1,024 tokens)
        inputs = self.tokenizer(
            text, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "review_id": review_id,
            "original_text": text
        }

def clean_text(text: str) -> str:
    """2. Noise Reduction: Strip HTML, normalize Unicode, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Normalize Unicode characters (remove accents, etc.)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Collapse multiple whitespaces/newlines into single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_text_column(df: pd.DataFrame) -> str:
    """1. Structured Parsing: Target column discovery logic."""
    cols = df.columns.tolist()
    # Search for specific names
    for target in ["review", "text", "content"]:
        for col in cols:
            if target in col.lower():
                return col
    
    # Fallback: Select object column with highest average string length
    obj_cols = df.select_dtypes(include=[object]).columns
    if len(obj_cols) > 0:
        lengths = df[obj_cols].apply(lambda x: x.str.len().mean())
        return lengths.idxmax()
    
    # Ultimate fallback
    return cols[0]

def ingest_and_preprocess() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """1 & 2. Data Ingestion, Cleaning, and Splitting."""
    all_data = []
    path = pathlib.Path(DATA_DIR)
    
    # 1. Recursive Scanning
    files = list(path.rglob("*.csv")) + list(path.rglob("*.json")) + list(path.rglob("*.txt"))
    
    if not files:
        print("Warning: No files found in local-test-data. Generating dummy data for validation.")
        dummy_text = (
            "The product arrived on time and works perfectly. I am very happy with the quality "
            "and the customer service was excellent. I would definitely recommend this to anyone "
            "looking for a reliable solution. The packaging was secure and the instructions were clear. "
            "I have been using it for a week now and haven't encountered any issues. Great value for money."
        )
        dummy_df = pd.DataFrame({"review": [dummy_text] * 20})
        dummy_df.to_csv(path / "dummy_reviews.csv", index=False)
        files = [path / "dummy_reviews.csv"]

    for file_path in files:
        try:
            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix == ".json":
                df = pd.read_json(file_path)
            elif file_path.suffix == ".txt":
                # 1. Unstructured Parsing
                with open(file_path, 'r', encoding='utf-8') as f:
                    df = pd.DataFrame({"text": [f.read()]})
            
            if df.empty:
                continue
                
            text_col = find_text_column(df)
            df = df[[text_col]].rename(columns={text_col: "original_text"})
            df['source_file'] = file_path.name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Error parsing {file_path}: {e}. Skipping.")

    if not all_data:
        raise ValueError("No valid data could be loaded.")

    # 1. Consolidation
    consolidated_df = pd.concat(all_data, ignore_index=True)
    consolidated_df['review_id'] = [f"rev_{i:04d}" for i in range(len(consolidated_df))]
    
    # 2. Cleaning
    consolidated_df['cleaned_text'] = consolidated_df['original_text'].apply(clean_text)
    
    # 2. Length Filtering (> 10 words)
    consolidated_df = consolidated_df[consolidated_df['cleaned_text'].apply(lambda x: len(x.split()) >= 10)]
    
    if consolidated_df.empty:
        raise ValueError("No records passed the length filtering (>10 words).")

    # 2. Data Split (70% Training, 20% Validation, 10% Testing)
    # Note: Primary task is zero-shot, but split is required for consistency.
    train_df = consolidated_df.sample(frac=0.7, random_state=SEED)
    temp_df = consolidated_df.drop(train_df.index)
    
    # Handle small datasets gracefully
    if len(temp_df) >= 3:
        val_df = temp_df.sample(frac=0.66, random_state=SEED)
        test_df = temp_df.drop(val_df.index)
    else:
        val_df = temp_df
        test_df = temp_df
    
    print(f"Data split completed. Total: {len(consolidated_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def run_inference(model, test_loader, tokenizer):
    """4 & 5. Zero-shot inference and Metrics calculation."""
    model.eval()
    results = []
    latencies = []
    
    # Reset peak memory stats for logging
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    print("Starting zero-shot inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            start_time = time.time()
            
            # 4. Decoding Hyperparameters (HPO)
            summary_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                max_length=130,
                min_length=30,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True
            )
            
            end_time = time.time()
            batch_latency = (end_time - start_time) * 1000 # ms
            latencies.append(batch_latency / len(input_ids))
            
            # Decode summaries
            summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            for i in range(len(summaries)):
                # 5. Metrics Calculation: Compression Ratio
                orig_tokens = tokenizer.encode(batch['original_text'][i], truncation=True, max_length=1024)
                summ_tokens = tokenizer.encode(summaries[i])
                
                orig_len = len(orig_tokens)
                summ_len = len(summ_tokens)
                comp_ratio = summ_len / orig_len if orig_len > 0 else 0
                
                results.append({
                    "review_id": batch['review_id'][i],
                    "original_text": batch['original_text'][i],
                    "generated_summary": summaries[i],
                    "compression_ratio": round(comp_ratio, 4)
                })

    # 5. Performance Logging
    avg_latency = np.mean(latencies) if latencies else 0
    peak_vram = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2) if torch.cuda.is_available() else 0
    
    performance_scores = {
        "avg_inference_latency_ms_per_review": round(float(avg_latency), 2),
        "avg_compression_ratio": round(float(np.mean([r['compression_ratio'] for r in results])), 4) if results else 0
    }
    
    complexity_scores = {
        "peak_vram_usage_mb": round(float(peak_vram), 2),
        "model_parameters": sum(p.numel() for p in model.parameters())
    }

    # 5. Final Export
    output_file = os.path.join(OUTPUT_DIR, "summary_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results exported to {output_file}")
        
    return performance_scores, complexity_scores

def main():
    # Step 1 & 2: Data Ingestion and Preprocessing
    train_df, val_df, test_df = ingest_and_preprocess()
    
    # Step 3: Model Configuration and Optimization
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # 3. Precision: FP16 (Half-Precision) for GPU
    if torch.cuda.is_available():
        model = model.half()
        print("Using FP16 Half-Precision for inference.")
    
    model.to(DEVICE)
    
    # Step 4: Inference Batching (Batch size 8, num_workers 0)
    test_dataset = ReviewDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Step 5: Evaluation and Inference
    performance, complexity = run_inference(model, test_loader, tokenizer)
    
    # Step 6: Prepare for Deployment (Save model and tokenizer)
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    deploy_status = f"Model and tokenizer saved to {MODEL_SAVE_DIR}"
    print(deploy_status)
    
    # Save metrics to JSON for the automation pipeline
    metrics = {
        "performance": performance,
        "complexity": complexity,
        "deployment_status": deploy_status,
        "test_samples_processed": len(test_df)
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics

if __name__ == "__main__":
    try:
        # Ensure the script is compatible with Windows sub-process execution
        final_metrics = main()
        print("\n--- Execution Summary ---")
        print(json.dumps(final_metrics, indent=4))
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        # Create a basic error metrics file so the pipeline knows it failed
        error_metrics = {"status": "failed", "error": str(e)}
        try:
            with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
                json.dump(error_metrics, f, indent=4)
        except:
            pass
