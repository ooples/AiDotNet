---
title: "LoRA"
description: "All 39 public types in the AiDotNet.lora namespace, organized by kind."
section: "API Reference"
---

**39** public types in this namespace, organized by kind.

## Models & Types (34)

| Type | Summary |
|:-----|:--------|
| [`AdaLoRAAdapter<T>`](/docs/reference/wiki/lora/adaloraadapter/) | Adaptive Low-Rank Adaptation (AdaLoRA) adapter that dynamically allocates parameter budgets among weight matrices. |
| [`ChainLoRAAdapter<T>`](/docs/reference/wiki/lora/chainloraadapter/) | Chain-of-LoRA adapter that implements sequential composition of multiple LoRA adapters. |
| [`DVoRAAdapter<T>`](/docs/reference/wiki/lora/dvoraadapter/) | DVoRA (DoRA + VeRA) adapter - combines DoRA's magnitude-direction decomposition with VeRA's extreme parameter efficiency. |
| [`DeltaLoRAAdapter<T>`](/docs/reference/wiki/lora/deltaloraadapter/) | Delta-LoRA adapter that focuses on parameter-efficient delta updates with momentum. |
| [`DenseLoRAAdapter<T>`](/docs/reference/wiki/lora/denseloraadapter/) | LoRA adapter specifically for Dense and FullyConnected layers with 1D input/output shapes. |
| [`DoRAAdapter<T>`](/docs/reference/wiki/lora/doraadapter/) | DoRA (Weight-Decomposed Low-Rank Adaptation) adapter for parameter-efficient fine-tuning with improved stability. |
| [`DyLoRAAdapter<T>`](/docs/reference/wiki/lora/dyloraadapter/) | DyLoRA (Dynamic LoRA) adapter that trains with multiple ranks simultaneously. |
| [`FloraAdapter<T>`](/docs/reference/wiki/lora/floraadapter/) | Implements Flora (Low-Rank Adapters Are Secretly Gradient Compressors) adapter for memory-efficient fine-tuning. |
| [`GLoRAAdapter<T>`](/docs/reference/wiki/lora/gloraadapter/) | Generalized LoRA (GLoRA) implementation that adapts both weights AND activations. |
| [`GraphConvolutionalLoRAAdapter<T>`](/docs/reference/wiki/lora/graphconvolutionalloraadapter/) | LoRA adapter for Graph Convolutional layers, enabling parameter-efficient fine-tuning of GNN models. |
| [`HRAAdapter<T>`](/docs/reference/wiki/lora/hraadapter/) | HRA (Hybrid Rank Adaptation) adapter that combines low-rank and full-rank updates for optimal parameter efficiency. |
| [`LoHaAdapter<T>`](/docs/reference/wiki/lora/lohaadapter/) | LoHa (Low-Rank Hadamard Product Adaptation) adapter for parameter-efficient fine-tuning. |
| [`LoKrAdapter<T>`](/docs/reference/wiki/lora/lokradapter/) | LoKr (Low-Rank Kronecker Product Adaptation) adapter for parameter-efficient fine-tuning. |
| [`LoRADropAdapter<T>`](/docs/reference/wiki/lora/loradropadapter/) | LoRA-drop implementation: LoRA with dropout regularization. |
| [`LoRAFAAdapter<T>`](/docs/reference/wiki/lora/lorafaadapter/) | LoRA-FA (LoRA with Frozen A matrix) adapter for parameter-efficient fine-tuning. |
| [`LoRAPlusAdapter<T>`](/docs/reference/wiki/lora/loraplusadapter/) | LoRA+ adapter that uses optimized learning rates for faster convergence and better performance. |
| [`LoRAXSAdapter<T>`](/docs/reference/wiki/lora/loraxsadapter/) | LoRA-XS (Extremely Small) adapter for ultra-parameter-efficient fine-tuning using SVD with trainable scaling matrix. |
| [`LoRETTAAdapter<T>`](/docs/reference/wiki/lora/lorettaadapter/) | LoRETTA (Low-Rank Economic Tensor-Train Adaptation) adapter for parameter-efficient fine-tuning. |
| [`LoftQAdapter<T>`](/docs/reference/wiki/lora/loftqadapter/) | LoftQ (LoRA-Fine-Tuning-Quantized) adapter that combines quantization and LoRA with improved initialization. |
| [`LongLoRAAdapter<T>`](/docs/reference/wiki/lora/longloraadapter/) | LongLoRA adapter that efficiently extends LoRA to handle longer context lengths using shifted sparse attention. |
| [`MoRAAdapter<T>`](/docs/reference/wiki/lora/moraadapter/) | Implements MoRA (High-Rank Updating for Parameter-Efficient Fine-Tuning) adapter. |
| [`MultiLoRAAdapter<T>`](/docs/reference/wiki/lora/multiloraadapter/) | Multi-task LoRA adapter that manages multiple task-specific LoRA layers for complex multi-task learning scenarios. |
| [`NOLAAdapter<T>`](/docs/reference/wiki/lora/nolaadapter/) | Implements NOLA (Compressing LoRA using Linear Combination of Random Basis) adapter for extreme parameter efficiency. |
| [`PiSSAAdapter<T>`](/docs/reference/wiki/lora/pissaadapter/) | Principal Singular Values and Singular Vectors Adaptation (PiSSA) adapter for parameter-efficient fine-tuning. |
| [`QALoRAAdapter<T>`](/docs/reference/wiki/lora/qaloraadapter/) | Quantization-Aware LoRA (QA-LoRA) adapter that combines parameter-efficient fine-tuning with group-wise quantization awareness. |
| [`QLoRAAdapter<T>`](/docs/reference/wiki/lora/qloraadapter/) | QLoRA (Quantized LoRA) adapter for parameter-efficient fine-tuning with 4-bit quantized base weights. |
| [`ReLoRAAdapter<T>`](/docs/reference/wiki/lora/reloraadapter/) | Restart LoRA (ReLoRA) adapter that periodically merges and restarts LoRA training for continual learning. |
| [`RoSAAdapter<T>`](/docs/reference/wiki/lora/rosaadapter/) | RoSA (Robust Adaptation) adapter for parameter-efficient fine-tuning with improved robustness to distribution shifts. |
| [`SLoRAAdapter<T>`](/docs/reference/wiki/lora/sloraadapter/) | S-LoRA adapter for scalable serving of thousands of concurrent LoRA adapters. |
| [`StandardLoRAAdapter<T>`](/docs/reference/wiki/lora/standardloraadapter/) | Standard LoRA implementation (original LoRA algorithm). |
| [`TiedLoRAAdapter<T>`](/docs/reference/wiki/lora/tiedloraadapter/) | Tied-LoRA adapter - LoRA with weight tying for extreme parameter efficiency across deep networks. |
| [`VBLoRAAdapter<T>`](/docs/reference/wiki/lora/vbloraadapter/) | Vector Bank LoRA (VB-LoRA) adapter that uses shared parameter banks for efficient multi-client deployment. |
| [`VeRAAdapter<T>`](/docs/reference/wiki/lora/veraadapter/) | VeRA (Vector-based Random Matrix Adaptation) adapter - an extreme parameter-efficient variant of LoRA. |
| [`XLoRAAdapter<T>`](/docs/reference/wiki/lora/xloraadapter/) | X-LoRA (Mixture of LoRA Experts) adapter that uses multiple LoRA experts with learned routing. |

## Layers (1)

| Type | Summary |
|:-----|:--------|
| [`LoRALayer<T>`](/docs/reference/wiki/lora/loralayer/) | Implements Low-Rank Adaptation (LoRA) layer for parameter-efficient fine-tuning of neural networks. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`LoRAAdapterBase<T>`](/docs/reference/wiki/lora/loraadapterbase/) | Abstract base class for LoRA (Low-Rank Adaptation) adapters that wrap existing layers. |

## Enums (2)

| Type | Summary |
|:-----|:--------|
| [`QuantizationType<T>`](/docs/reference/wiki/lora/quantizationtype/) | Specifies the type of 4-bit quantization to use for base layer weights. |
| [`QuantizationType<T>`](/docs/reference/wiki/lora/quantizationtype-2/) | Specifies the type of 4-bit quantization to use for base layer weights. |

## Options & Configuration (1)

| Type | Summary |
|:-----|:--------|
| [`DefaultLoRAConfiguration<T>`](/docs/reference/wiki/lora/defaultloraconfiguration/) | Default LoRA configuration that applies LoRA to all layers with trainable weight matrices. |

