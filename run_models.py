#!/usr/bin/env python3
"""
llama.cpp GGUF Model Benchmark Script
Runs multiple GGUF models and collects performance metrics
"""

import subprocess
import re
import sys
import os
import argparse
import glob
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime
from datetime import datetime


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


def find_llama_cli(script_dir: str) -> Optional[str]:
    """Find available llama CLI executable based on script location"""
    build_dir = os.path.join(script_dir, "build-vulkan", "bin", "Release")
    
    search_files = [
        "llama-completion.exe",
        "llama-cli.exe", 
        "llama-simple.exe",
        "llama-simple-chat.exe",
    ]
    
    for fname in search_files:
        fpath = os.path.join(build_dir, fname)
        if os.path.exists(fpath):
            return fpath
    
    return None


def find_gguf_models(model_folder: str, model_filter: Optional[List[str]] = None) -> List[str]:
    """Find all GGUF model files in the specified folder"""
    patterns = [
        os.path.join(model_folder, "*.gguf"),
        os.path.join(model_folder, "*.GGUF"),
    ]
    
    models = []
    for pattern in patterns:
        models.extend(glob.glob(pattern))
    
    # Sort by file name first
    models.sort(key=lambda x: os.path.basename(x).lower())
    
    # If no filter, return all found models (deduplicated)
    if not model_filter:
        seen = set()
        unique_models = []
        for m in models:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)
        return unique_models
    
    # Filter models - use exact match first
    filtered = []
    filter_lower = [f.lower() for f in model_filter]
    
    for m in models:
        basename = os.path.basename(m)
        basename_lower = basename.lower()
        
        # Check for exact match
        if basename in model_filter:
            filtered.append(m)
            continue
        
        # Check for case-insensitive exact match
        if basename_lower in filter_lower:
            filtered.append(m)
            continue
    
    # Remove any remaining duplicates
    seen = set()
    unique_filtered = []
    for m in filtered:
        if m not in seen:
            seen.add(m)
            unique_filtered.append(m)
    
    return unique_filtered


def get_input_prompt() -> str:
    """Generate input prompt with approximately 1024 tokens"""
    prompt = """Explain the concept of machine learning in detail. Include topics such as supervised learning, unsupervised learning, reinforcement learning, neural networks, deep learning, backpropagation, gradient descent, overfitting, underfitting, bias-variance tradeoff, regularization, optimization algorithms, batch normalization, dropout, transfer learning, fine-tuning, hyperparameter tuning, model evaluation metrics like accuracy, precision, recall, F1 score, confusion matrix, ROC curve, AUC, loss functions, activation functions, convolutional neural networks, recurrent neural networks, long short-term memory, transformers, attention mechanism, self-attention, multi-head attention, positional encoding, embedding layers, pooling layers, fully connected layers, skip connections, residual networks, generative adversarial networks, variational autoencoders, autoencoders, dimensionality reduction, PCA, t-SNE, clustering, k-means, hierarchical clustering, DBSCAN, anomaly detection, recommendation systems, collaborative filtering, content-based filtering, natural language processing, word embeddings, word2vec, GloVe, BERT, GPT, text classification, sentiment analysis, named entity recognition, machine translation, question answering, text summarization, chatbot, reinforcement learning from human feedback, RLHF, proximal policy optimization, actor-critic methods, Q-learning, deep Q-network, Monte Carlo tree search, AlphaGo, AlphaZero, robotics, computer vision, object detection, YOLO, SSD, R-CNN, semantic segmentation, U-Net, image classification, image generation, stable diffusion, DALL-E, Midjourney, style transfer, super resolution, image denoising, edge detection, feature extraction, HOG, SIFT, SURF, ORB, camera calibration, stereo vision, depth estimation, 3D reconstruction, SLAM, LIDAR, point clouds, graph neural networks, graph embedding, knowledge graphs, federated learning, distributed training, model compression, quantization, pruning, knowledge distillation, neural architecture search, AutoML, meta-learning, few-shot learning, zero-shot learning, prompt engineering, in-context learning, chain-of-thought prompting, tree-of-thought, retrieval-augmented generation, RAG, vector databases, embedding similarity search, cosine similarity, euclidean distance, manhattan distance, jaccard similarity, edit distance, levenshtein distance, perplexity, BLEU score, ROUGE score, METEOR, CIDEr, SPICE, beam search, greedy search, nucleus sampling, top-k sampling, temperature scaling, repetition penalty, length penalty, presence penalty, frequency penalty, context length, max tokens, temperature, top-p, top-k, stop sequences, streaming, batch inference, latency, throughput, FLOPs, MACs, GPU memory, VRAM, model size, parameter count, inference speed, tokens per second, time to first token, memory bandwidth, compute capability, CUDA cores, tensor cores, ray tracing, mesh shaders, workflow automation, MLOps, ML pipeline, CI/CD for ML, experiment tracking, MLflow, Weights & Biases, TensorBoard, model versioning, data versioning, DVC, ML metadata, lineage tracking, model registry, feature store, feature engineering, feature selection, feature importance, correlation analysis, mutual information, chi-square test, ANOVA, statistical tests, hypothesis testing, p-value, confidence interval, Bayesian inference, maximum likelihood estimation, expectation-maximization, Gaussian mixture models, hidden Markov models, viterbi algorithm, forward-backward algorithm, particle filters, Kalman filters, extended Kalman filters, unscented Kalman filters, graph cut, belief propagation, loopy belief propagation, mean field inference, variational inference, Monte Carlo methods, importance sampling, rejection sampling, Gibbs sampling, Metropolis-Hastings, Hamiltonian Monte Carlo, No-U-Turn Sampler, automatic differentiation, symbolic differentiation, numerical differentiation, computational graph, forward pass, backward pass, automatic mixed precision, FP16, BF16, INT8, INT4, model fusion, model averaging, ensemble methods, bagging, boosting, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, Random Forest, Decision Trees, CART, ID3, C4.5, entropy, information gain, Gini impurity, pruning strategies, minimum description length, cost-complexity pruning, early stopping, cross-validation, k-fold cross-validation, stratified k-fold, leave-one-out, bootstrap, jackknife, out-of-bag estimation, bias, variance, irreducible error, optimum complexity, learning curve, validation curve, ROC convex hull, precision-recall curve, average precision, mean average precision, mAP, IoU, non-maximum suppression, anchor boxes, region proposals, sliding window, image pyramids, feature pyramids, multi-scale detection, data augmentation, random cropping, flipping, rotation, scaling, color jittering, cutout, mixup, cutmix, AutoAugment, RandAugment, test time augmentation, synthetic data generation, data synthesis, domain randomization, simulation, physics engines, Bullet, MuJoCo, OpenAI Gym, DeepMind Control Suite"""
    return prompt


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
    vram_match = re.search(
        r'Vulkan0.*?\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*\(.*?\)\s*\+\s*(\d+)\s*\|',
        output, re.IGNORECASE
    )
    if vram_match:
        result['vram_total_mb'] = float(vram_match.group(1))
        result['vram_used_mb'] = float(vram_match.group(2))
        result['vram_free_mb'] = float(vram_match.group(3))
    
    return result


def run_model(model_path: str, llama_cli: str, input_prompt: str, max_tokens: int, ctx_size: int, log_file: str) -> ModelResult:
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
        "-p", input_prompt,
        "-n", str(max_tokens),
        "-c", str(ctx_size),
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
            timeout=600,
            encoding='utf-8',
            errors='replace'
        )
        output = result.stdout + result.stderr
        
        # Find where system logs start (after "ggml_vulkan:")
        system_log_marker = "ggml_vulkan:"
        marker_pos = output.find(system_log_marker)
        
        if marker_pos > 0:
            generated_text = output[:marker_pos].strip()
            system_logs = output[marker_pos:]
        else:
            generated_text = output.strip()
            system_logs = ""
        
        # Save log to file with proper formatting
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("INPUT PROMPT:\n")
            f.write("="*80 + "\n")
            f.write(input_prompt + "\n")
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("MODEL OUTPUT (prompt + generated text):\n")
            f.write("="*80 + "\n")
            f.write(generated_text + "\n")
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("SYSTEM LOGS:\n")
            f.write("="*80 + "\n")
            f.write(system_logs)
        
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


def print_results_markdown(results: List[ModelResult], output_tokens: int) -> str:
    """Generate markdown formatted results"""
    md = []
    
    # Title
    md.append("# llama.cpp Vulkan Benchmark Results\n")
    md.append(f"**Output Tokens:** {output_tokens} | **Context Size:** 4096\n")
    md.append("---\n")
    
    # Summary Table
    md.append("## Performance Summary\n")
    md.append("| Model | Size (GB) | Params (B) | Load (ms) | Prefill (ms) | Prefill (t/s) | Decode (ms) | Decode (t/s) |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    for r in results:
        if r.success:
            md.append(f"| {r.name} | {r.model_size_gb:.2f} | {r.params_b:.1f} | {r.load_time_ms:.0f} | {r.prefill_time_ms:.1f} | {r.prefill_tps:.1f} | {r.decode_time_ms:.0f} | {r.decode_tps:.1f} |")
        else:
            md.append(f"| {r.name} | FAILED | - | - | - | - | - | - |")
    
    md.append("\n")
    
    # Detailed Table
    md.append("## Detailed Performance Breakdown\n")
    md.append("| Model | Total (ms) | Total Tokens | VRAM Used (MB) | VRAM Free (MB) | Prefill Tokens | Decode Tokens | Load (ms) |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    for r in results:
        if r.success:
            md.append(f"| {r.name} | {r.total_time_ms:.0f} | {r.total_tokens} | {r.vram_used_mb:.0f} | {r.vram_free_mb:.0f} | {r.prefill_tokens} | {r.decode_tokens} | {r.load_time_ms:.0f} |")
        else:
            md.append(f"| {r.name} | FAILED | - | - | - | - | - | - |")
    
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="llama.cpp GGUF Model Benchmark")
    parser.add_argument("--model-folder", type=str, required=True, help="Folder containing GGUF model files")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Exact model file names to run (e.g., Qwen3.5-0.8B-Q4_1.gguf)")
    parser.add_argument("--output-tokens", type=int, default=600, help="Number of tokens to generate (default: 600)")
    parser.add_argument("--context-size", type=int, default=4096, help="Context size (default: 4096)")
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup logs directory with timestamp subfolder
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamped folder for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(logs_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Find llama CLI
    llama_cli = find_llama_cli(script_dir)
    if not llama_cli:
        print(f"[ERROR] Could not find llama CLI executable!")
        print(f"[ERROR] Please run llama-cpp-setup.bat first to build llama.cpp")
        sys.exit(1)
    
    # Find models (with optional filter)
    models = find_gguf_models(args.model_folder, args.models)
    if not models:
        print(f"[ERROR] No GGUF files found in: {args.model_folder}")
        if args.models:
            print(f"[ERROR] After filtering with: {args.models}")
            print(f"[ERROR] Available models in folder:")
            all_models = find_gguf_models(args.model_folder)
            for m in all_models:
                print(f"  - {os.path.basename(m)}")
        sys.exit(1)
    
    # Input prompt (~1024 tokens)
    input_prompt = get_input_prompt()
    
    # Print header
    print("="*60)
    print("  llama.cpp GGUF Model Benchmark")
    print("="*60)
    print(f"\n[INFO] Model folder: {args.model_folder}")
    print(f"[INFO] Found {len(models)} model(s)")
    print(f"[INFO] Output tokens: {args.output_tokens}")
    print(f"[INFO] Context size: {args.context_size}")
    print(f"[INFO] Using: {llama_cli}")
    print(f"[INFO] Logs directory: {run_dir}")
    
    results: List[ModelResult] = []
    
    for model_path in models:
        # Generate log file name with timestamp
        model_name = Path(model_path).stem
        log_file = os.path.join(run_dir, f"{model_name}.log")
        
        result = run_model(model_path, llama_cli, input_prompt, args.output_tokens, args.context_size, log_file)
        results.append(result)
        
        print(f"[INFO] Log saved: {log_file}")
    
    # Print markdown results
    md_output = print_results_markdown(results, args.output_tokens)
    print("\n" + md_output)
    
    # Save markdown to run directory
    md_file = os.path.join(run_dir, f"benchmark_results.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_output)
    print(f"\n[INFO] Results saved to: {md_file}")


if __name__ == "__main__":
    main()
