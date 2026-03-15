@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "MODEL_DIR=C:\data\models\gguf"
set "BUILD_DIR=C:\data\code\llama_cpp_benchmark_code\llama-cpp-benchmark\build-vulkan\bin\Release"

set "LLAMA_CLI="
for %%D in ("%BUILD_DIR%\llama-completion.exe" "%BUILD_DIR%\llama-simple.exe" "%BUILD_DIR%\llama-simple-chat.exe") do (
    if exist "%%~D" (
        if not defined LLAMA_CLI set "LLAMA_CLI=%%~D"
    )
)

if not defined LLAMA_CLI (
    echo [ERROR] No suitable llama executable found in:
    echo [ERROR]   %BUILD_DIR%
    exit /b 1
)

set "INPUT_PROMPT=Explain the concept of machine learning in detail. Include topics such as supervised learning, unsupervised learning, reinforcement learning, neural networks, deep learning, backpropagation, gradient descent, overfitting, underfitting, bias-variance tradeoff, regularization, optimization algorithms, batch normalization, dropout, transfer learning, fine-tuning, hyperparameter tuning, model evaluation metrics like accuracy, precision, recall, F1 score, confusion matrix, ROC curve, AUC, loss functions, activation functions, convolutional neural networks, recurrent neural networks, long short-term memory, transformers, attention mechanism, self-attention, multi-head attention, positional encoding, embedding layers, pooling layers, fully connected layers, skip connections, residual networks, generative adversarial networks, variational autoencoders, autoencoders, dimensionality reduction, PCA, t-SNE, clustering, k-means, hierarchical clustering, DBSCAN, anomaly detection, recommendation systems, collaborative filtering, content-based filtering, natural language processing, word embeddings, word2vec, GloVe, BERT, GPT, text classification, sentiment analysis, named entity recognition, machine translation, question answering, text summarization, chatbot, reinforcement learning from human feedback, RLHF, proximal policy optimization, actor-critic methods, Q-learning, deep Q-network, Monte Carlo tree search, AlphaGo, AlphaZero, robotics, computer vision, object detection, YOLO, SSD, R-CNN, semantic segmentation, U-Net, image classification, image generation, stable diffusion, DALL-E, Midjourney, style transfer, super resolution, image denoising, edge detection, feature extraction, HOG, SIFT, SURF, ORB, camera calibration, stereo vision, depth estimation, 3D reconstruction, SLAM, LIDAR, point clouds, graph neural networks, graph embedding, knowledge graphs, federated learning, distributed training, model compression, quantization, pruning, knowledge distillation, neural architecture search, AutoML, meta-learning, few-shot learning, zero-shot learning, prompt engineering, in-context learning, chain-of-thought prompting, tree-of-thought, retrieval-augmented generation, RAG, vector databases, embedding similarity search, cosine similarity, euclidean distance, manhattan distance, jaccard similarity, edit distance, levenshtein distance, perplexity, BLEU score, ROUGE score, METEOR, CIDEr, SPICE, beam search, greedy search, nucleus sampling, top-k sampling, temperature scaling, repetition penalty, length penalty, presence penalty, frequency penalty, context length, max tokens, temperature, top-p, top-k, stop sequences, streaming, batch inference, latency, throughput, FLOPs, MACs, GPU memory, VRAM, model size, parameter count, inference speed, tokens per second, time to first token, memory bandwidth, compute capability, CUDA cores, tensor cores, ray tracing, mesh shaders, workflow automation, MLOps, ML pipeline, CI/CD for ML, experiment tracking, MLflow, Weights & Biases, TensorBoard, model versioning, data versioning, DVC, ML metadata, lineage tracking, model registry, feature store, feature engineering, feature selection, feature importance, correlation analysis, mutual information, chi-square test, ANOVA, statistical tests, hypothesis testing, p-value, confidence interval, Bayesian inference, maximum likelihood estimation, expectation-maximization, Gaussian mixture models, hidden Markov models, viterbi algorithm, forward-backward algorithm, particle filters, Kalman filters, extended Kalman filters, unscented Kalman filters, graph cut, belief propagation, loopy belief propagation, mean field inference, variational inference, Monte Carlo methods, importance sampling, rejection sampling, Gibbs sampling, Metropolis-Hastings, Hamiltonian Monte Carlo, No-U-Turn Sampler, automatic differentiation, symbolic differentiation, numerical differentiation, computational graph, forward pass, backward pass, automatic mixed precision, FP16, BF16, INT8, INT4, model fusion, model averaging, ensemble methods, bagging, boosting, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, Random Forest, Decision Trees, CART, ID3, C4.5, entropy, information gain, Gini impurity, pruning strategies, minimum description length, cost-complexity pruning, early stopping, cross-validation, k-fold cross-validation, stratified k-fold, leave-one-out, bootstrap, jackknife, out-of-bag estimation, bias, variance, irreducible error, optimum complexity, learning curve, validation curve, ROC convex hull, precision-recall curve, average precision, mean average precision, mAP, IoU, non-maximum suppression, anchor boxes, region proposals, sliding window, image pyramids, feature pyramids, multi-scale detection, data augmentation, random cropping, flipping, rotation, scaling, color jittering, cutout, mixup, cutmix, AutoAugment, RandAugment, test time augmentation, synthetic data generation, data synthesis, domain randomization, simulation, physics engines, Bullet, MuJoCo, OpenAI Gym, DeepMind Control Suite"

set "MAX_TOKENS=600"

echo ========================================
echo   llama.cpp GGUF Model Benchmark
echo ========================================
echo.
echo [INFO] Using llama-cli: %LLAMA_CLI%
echo [INFO] Model directory: %MODEL_DIR%
echo [INFO] Input prompt length: ~1000 tokens
echo [INFO] Generation length: %MAX_TOKENS% tokens
echo.

if not exist "%LLAMA_CLI%" (
    echo [ERROR] llama executable not found at:
    echo [ERROR]   %LLAMA_CLI%
    exit /b 1
)

set "MODELS[0]=Qwen3.5-0.8B-Q4_1.gguf"
set "MODELS[1]=Qwen3.5-2B-Q4_1.gguf"
set "MODELS[2]=Qwen3.5-4B-Q4_1.gguf"
set "MODELS[3]=Qwen3.5-9B-Q4_1.gguf"
set "MODELS[4]=Qwen3.5-35B-A3B-Q4_1.gguf"
set "MODELS[5]=Qwen3.5-35B-A3B-Q4_K_M.gguf"
set "MODEL_COUNT=6"

for /L %%i in (0,1,5) do (
    call :run_model !MODELS[%%i]!
)

echo.
echo ========================================
echo   Benchmark Complete
echo ========================================
exit /b 0

:run_model
set "MODEL_PATH=%MODEL_DIR%\%~1"

if not exist "!MODEL_PATH!" (
    echo [WARNING] Model not found: !MODEL_PATH!
    echo.
    exit /b 0
)

echo.
echo ========================================
echo   Running: %~1
echo ========================================
echo.
echo [INFO] Model: %~1
echo [INFO] Path: !MODEL_PATH!
echo.
echo [OUTPUT]
echo ----------------------------------------------------------------------

"!LLAMA_CLI!" -m "!MODEL_PATH!" -p "%INPUT_PROMPT%" -n 600 -c 4096 --no-mmap -ngl 99 -fa on --perf -no-cnv 2>&1

echo.
echo ----------------------------------------------------------------------
echo.

exit /b 0
