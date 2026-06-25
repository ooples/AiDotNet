---
title: "ONNXNER<T>"
description: "ONNX-NER: Generic ONNX Runtime-based NER model for high-performance inference with any exported NER model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

ONNX-NER: Generic ONNX Runtime-based NER model for high-performance inference with any
exported NER model.

## For Beginners

ONNX-NER is not a specific NER architecture, but rather a way to run
any NER model that has been exported to the ONNX format. ONNX is like a "universal translator"
for neural network models - it lets you train in any framework and run inference efficiently.
Use ONNX-NER when you have a pre-trained ONNX model file and want the fastest possible
inference, or when you want to deploy NER models without depending on PyTorch/TensorFlow.

## How It Works

ONNX-NER provides a model-agnostic wrapper for running any NER model exported to ONNX format
through ONNX Runtime. This enables high-performance inference regardless of the original
training framework (PyTorch, TensorFlow, JAX, etc.).

**ONNX Runtime Optimizations:**

- **Graph optimizations:** Operator fusion (MatMul+Add, Conv+BN), constant folding,

and redundant computation elimination

- **Hardware acceleration:** CUDA, TensorRT, DirectML, OpenVINO, CoreML execution providers
- **Quantization:** INT8/FP16 inference for reduced memory and faster execution
- **Memory optimization:** Memory-efficient attention patterns, memory arena pre-allocation
- **Threading:** Intra-op and inter-op parallelism configuration

**Supported Model Sources:**

- HuggingFace Transformers (exported via optimum or torch.onnx.export)
- PyTorch models (exported via torch.onnx.export)
- TensorFlow/Keras models (exported via tf2onnx)
- Custom models exported to ONNX format

**Performance (typical speedups over PyTorch):**

- CPU: 2-4x faster with graph optimizations
- GPU (CUDA): 1.5-3x faster with TensorRT
- INT8 quantized: 3-5x faster than FP32 with ~0.5% F1 loss

**Common ONNX NER Models:**

- dslim/bert-base-NER (ONNX): ~92.4% F1, ~5ms/sentence on GPU
- elastic/distilbert-base-cased-finetuned-conll03-english: ~91.1% F1, ~2ms/sentence on GPU
- Jean-Baptiste/camembert-ner (French NER): ~90.5% F1

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ONNXNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates an ONNX-NER model in ONNX inference mode. |
| `ONNXNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an ONNX-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

