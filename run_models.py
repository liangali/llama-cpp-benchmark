#!/usr/bin/env python3
"""
llama.cpp GGUF Model Benchmark Script
Runs multiple GGUF models and collects performance metrics
"""

import subprocess
import re
import sys
import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

MODEL_DIR = r"C:\data\models\gguf"
LLAMA_CLI = r"C:\data\code\llama_cpp_benchmark_code\llama-cpp-benchmark\build-vulkan\bin\Release\llama-completion.exe"

INPUT_PROMPT = """Explain the concept of machine learning in detail. Include topics such as supervised learning, unsupervised learning, reinforcement learning, neural networks, deep learning, backpropagation, gradient descent, overfitting, underfitting, bias-variance tradeoff, regularization, optimization algorithms, batch normalization, dropout, transfer learning, fine-tuning, hyperparameter tuning, model evaluation metrics like accuracy, precision, recall, F1 score, confusion matrix, ROC curve, AUC, loss functions, activation functions, convolutional neural networks, recurrent neural networks, long short-term memory, transformers, attention mechanism, self-attention, multi-head attention, positional encoding, embedding layers, pooling layers, fully connected layers, skip connections, residual networks, generative adversarial networks, variational autoencoders, autoencoders, dimensionality reduction, PCA, t-SNE, clustering, k-means, hierarchical clustering, DBSCAN, anomaly detection, recommendation systems, collaborative filtering, content-based filtering, natural language processing, word embeddings, word2vec, GloVe, BERT, GPT, text classification, sentiment analysis, named entity recognition, machine translation, question answering, text summarization, chatbot"""

MAX_TOKENS = 600
CTX_SIZE = 4096

MODELS = [
    "Qwen3.5-0.8B-Q4_1.gguf",
    "Qwen3.5-2B-Q4_1.gguf",
    "Qwen3.5-4B-Q4_1.gguf",
    "Qwen3.5-9B-Q4_1.gguf",
    "Qwen3.5-35B-A3B-Q4_1.gguf",
    "Qwen3.5-35B-A3B-Q4_K_M.gguf",
]


@dataclass
class ModelResult:
    name: str
    model_size_gb: float
    params_b: float
    load_time_ms: float
    prefill_time_ms: float
    prefill_tokens: int
    prefill_tps: float
    decode_time_ms: float
    decode_tokens: int
    decode_tps: float
    total_time_ms: float
    total_tokens: int
    vram_used_mb: float
    vram_free_mb: float
    success: bool
    error: Optional[str] = None


def find_llama_cli() -> Optional[str]:
    """Find available llama CLI executable"""
    search_paths = [
        r"C:\data\code\llama_cpp_benchmark_code\llama-cpp-benchmark\build-vulkan\bin\Release\llama-completion.exe",
        r"C:\data\code\llama_cpp_benchmark_code\llama-cpp-benchmark\build-vulkan\bin\Release\llama-simple.exe",
        r"C:\data\code\llama_cpp_benchmark_code\llama-cpp-benchmark\build-vulkan\bin\Release\llama-cli.exe",
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None


def parse_model_info(output: str) -> tuple:
    """Parse model size and parameters from output"""
    model_size_gb = 0.0
    params_b = 0.0
    
    # Parse file size - try multiple patterns
    size_match = re.search(r'file\s+size\s*=\s*([\d.]+)\s*GiB', output, re.IGNORECASE)
    if not size_match:
        size_match = re.search(r'file\s+size\s*=\s*([\d.]+)\s*MiB', output, re.IGNORECASE)
    if size_match:
        size_val = float(size_match.group(1))
        if size_val < 100:  # Likely in GiB
            model_size_gb = size_val
        else:  # Convert MiB to GiB
            model_size_gb = size_val / 1024
    
    # Parse model params - try multiple patterns
    params_match = re.search(r'model\s+params\s*=\s*([\d.]+)\s*B', output, re.IGNORECASE)
    if params_match:
        params_b = float(params_match.group(1))
    
    return model_size_gb, params_b


def parse_performance(output: str) -> dict:
    """Parse performance metrics from output"""
    result = {
        'load_time_ms': 0.0,
        'prefill_time_ms': 0.0,
        'prefill_tokens': 0,
        'prefill_tps': 0.0,
        'decode_time_ms': 0.0,
        'decode_tokens': 0,
        'decode_tps': 0.0,
        'total_time_ms': 0.0,
        'total_tokens': 0,
        'vram_total_mb': 0.0,
        'vram_used_mb': 0.0,
        'vram_free_mb': 0.0,
    }
    
    # Check for errors first
    if 'error:' in output.lower():
        return result
    
    # Parse load time
    load_match = re.search(r'load\s+time\s*=\s*([\d.]+)\s*ms', output, re.IGNORECASE)
    if load_match:
        result['load_time_ms'] = float(load_match.group(1))
    
    # Parse prompt eval (prefill) - handle "inf" case
    prefill_match = re.search(
        r'prompt\s+eval\s+time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+tokens\s*\(\s*([\d.]+)\s*ms\s+per\s+token[,\s]*\s*([\d.]+|inf)\s+tokens\s+per\s+second',
        output, re.IGNORECASE
    )
    if prefill_match:
        result['prefill_time_ms'] = float(prefill_match.group(1))
        result['prefill_tokens'] = int(prefill_match.group(2))
        tps_str = prefill_match.group(4)
        result['prefill_tps'] = float(tps_str) if tps_str != 'inf' else 0.0
    
    # Parse eval (decode)
    decode_match = re.search(
        r'(?:^|\s)eval\s+time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+runs?\s*\(\s*([\d.]+)\s*ms\s+per\s+token[,\s]*\s*([\d.]+)\s+tokens\s+per\s+second',
        output, re.IGNORECASE | re.MULTILINE
    )
    if decode_match:
        result['decode_time_ms'] = float(decode_match.group(1))
        result['decode_tokens'] = int(decode_match.group(2))
        result['decode_tps'] = float(decode_match.group(4))
    
    # Parse total time
    total_match = re.search(r'total\s+time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+tokens', output, re.IGNORECASE)
    if total_match:
        result['total_time_ms'] = float(total_match.group(1))
        result['total_tokens'] = int(total_match.group(2))
    
    # Parse VRAM - Format: Vulkan0 (...) | total = used + (compute = model + ops + other) + unaccounted |
    # Example: | 32624 = 10447 + (21283 = 20642 + 142 + 498) + 893 |
    vram_match = re.search(
        r'Vulkan0.*?\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*\(.*?\)\s*\+\s*(\d+)\s*\|',
        output, re.IGNORECASE
    )
    if vram_match:
        result['vram_total_mb'] = float(vram_match.group(1))
        result['vram_used_mb'] = float(vram_match.group(2))
        result['vram_free_mb'] = float(vram_match.group(3))
    
    return result


def run_model(model_path: str, llama_cli: str) -> ModelResult:
    """Run a single model and collect performance metrics"""
    print(f"\n{'='*60}")
    print(f"  Running: {Path(model_path).name}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        return ModelResult(
            name=Path(model_path).name,
            model_size_gb=0, params_b=0,
            load_time_ms=0, prefill_time_ms=0, prefill_tokens=0, prefill_tps=0,
            decode_time_ms=0, decode_tokens=0, decode_tps=0,
            total_time_ms=0, total_tokens=0,
            vram_used_mb=0, vram_free_mb=0,
            success=False,
            error=f"Model not found: {model_path}"
        )
    
    cmd = [
        llama_cli,
        "-m", model_path,
        "-p", INPUT_PROMPT,
        "-n", str(MAX_TOKENS),
        "-c", str(CTX_SIZE),
        "--no-mmap",
        "-ngl", "99",
        "-fa", "on",
        "--perf",
        "-no-cnv"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            encoding='utf-8',
            errors='replace'
        )
        output = result.stdout + result.stderr
        
        model_size_gb, params_b = parse_model_info(output)
        perf = parse_performance(output)
        
        return ModelResult(
            name=Path(model_path).name,
            model_size_gb=model_size_gb,
            params_b=params_b,
            load_time_ms=perf['load_time_ms'],
            prefill_time_ms=perf['prefill_time_ms'],
            prefill_tokens=perf['prefill_tokens'],
            prefill_tps=perf['prefill_tps'],
            decode_time_ms=perf['decode_time_ms'],
            decode_tokens=perf['decode_tokens'],
            decode_tps=perf['decode_tps'],
            total_time_ms=perf['total_time_ms'],
            total_tokens=perf['total_tokens'],
            vram_used_mb=perf['vram_used_mb'],
            vram_free_mb=perf['vram_free_mb'],
            success=True
        )
        
    except subprocess.TimeoutExpired:
        return ModelResult(
            name=Path(model_path).name,
            model_size_gb=0, params_b=0,
            load_time_ms=0, prefill_time_ms=0, prefill_tokens=0, prefill_tps=0,
            decode_time_ms=0, decode_tokens=0, decode_tps=0,
            total_time_ms=0, total_tokens=0,
            vram_used_mb=0, vram_free_mb=0,
            success=False,
            error="Timeout"
        )
    except Exception as e:
        return ModelResult(
            name=Path(model_path).name,
            model_size_gb=0, params_b=0,
            load_time_ms=0, prefill_time_ms=0, prefill_tokens=0, prefill_tps=0,
            decode_time_ms=0, decode_tokens=0, decode_tps=0,
            total_time_ms=0, total_tokens=0,
            vram_used_mb=0, vram_free_mb=0,
            success=False,
            error=str(e)
        )


def print_results_table(results: List[ModelResult]):
    """Print results in a nice table format"""
    # Header
    print("\n")
    print("+" + "="*100 + "+")
    print("|" + " "*30 + "LLAMA.CPP VULKAN BENCHMARK RESULTS" + " "*33 + "|")
    print("+" + "="*100 + "+")
    
    # Column headers
    header = (
        f"| {'Model':<28} "
        f"{'Size':>6} "
        f"{'Params':>7} "
        f"{'Load':>7} "
        f"{'Prefill':>10} "
        f"{'Prefill':>10} "
        f"{'Decode':>10} "
        f"{'Decode':>10} "
        f"|"
    )
    print(header)
    
    header2 = (
        f"| {'':<28} "
        f"{'(GB)':>6} "
        f"{'(B)':>7} "
        f"{'(ms)':>7} "
        f"{'(ms)':>10} "
        f"{'(t/s)':>10} "
        f"{'(ms)':>10} "
        f"{'(t/s)':>10} "
        f"|"
    )
    print(header2)
    print("+" + "="*100 + "+")
    
    # Data rows
    for r in results:
        if r.success:
            row = (
                f"| {r.name:<28} "
                f"{r.model_size_gb:>6.2f} "
                f"{r.params_b:>7.1f} "
                f"{r.load_time_ms:>7.0f} "
                f"{r.prefill_time_ms:>10.1f} "
                f"{r.prefill_tps:>10.1f} "
                f"{r.decode_time_ms:>10.0f} "
                f"{r.decode_tps:>10.1f} "
                f"|"
            )
        else:
            row = (
                f"| {r.name:<28} "
                f"{'FAILED':>48} "
                f"{r.error if r.error else 'Unknown error':<20} "
                f"|"
            )
        print(row)
    
    print("+" + "="*100 + "+")
    
    # Detailed breakdown table
    print("\n\n")
    print("+" + "="*110 + "+")
    print("|" + " "*40 + "DETAILED PERFORMANCE BREAKDOWN" + " "*41 + "|")
    print("+" + "="*110 + "+")
    
    # Column headers
    header = (
        f"| {'Model':<26} "
        f"{'Total':>10} "
        f"{'Total':>10} "
        f"{'VRAM':>12} "
        f"{'VRAM':>10} "
        f"{'Prefill':>12} "
        f"{'Decode':>12} "
        f"{'Load':>10} "
        f"|"
    )
    print(header)
    
    header2 = (
        f"| {'':<26} "
        f"{'(ms)':>10} "
        f"{'Tokens':>10} "
        f"{'Used(MB)':>12} "
        f"{'Free(MB)':>10} "
        f"{'Tokens':>12} "
        f"{'Tokens':>12} "
        f"{'(ms)':>10} "
        f"|"
    )
    print(header2)
    print("+" + "="*110 + "+")
    
    for r in results:
        if r.success:
            row = (
                f"| {r.name:<26} "
                f"{r.total_time_ms:>10.0f} "
                f"{r.total_tokens:>10} "
                f"{r.vram_used_mb:>12.0f} "
                f"{r.vram_free_mb:>10.0f} "
                f"{r.prefill_tokens:>12} "
                f"{r.decode_tokens:>12} "
                f"{r.load_time_ms:>10.0f} "
                f"|"
            )
        else:
            row = f"| {r.name:<26} {'FAILED':>83} |"
        print(row)
    
    print("+" + "="*110 + "+")


def main():
    print("="*60)
    print("  llama.cpp GGUF Model Benchmark")
    print("="*60)
    print(f"\n[INFO] Model directory: {MODEL_DIR}")
    print(f"[INFO] Max tokens: {MAX_TOKENS}")
    print(f"[INFO] Context size: {CTX_SIZE}")
    
    # Find llama CLI
    llama_cli = find_llama_cli()
    if not llama_cli:
        print(f"\n[ERROR] Could not find llama CLI executable!")
        print(f"[ERROR] Searched in:")
        print(f"   - {MODEL_DIR}")
        sys.exit(1)
    
    print(f"[INFO] Using: {llama_cli}")
    
    results: List[ModelResult] = []
    
    for model in MODELS:
        model_path = os.path.join(MODEL_DIR, model)
        result = run_model(model_path, llama_cli)
        results.append(result)
    
    print_results_table(results)


if __name__ == "__main__":
    main()
