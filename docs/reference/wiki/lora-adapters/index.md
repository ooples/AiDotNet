---
title: "LoRA / PEFT Adapters"
description: "Every LoRA / PEFT Adapters type in AiDotNet, auto-generated with compile-checked examples."
section: "Reference"
---

Every LoRA / PEFT Adapters type in AiDotNet — each with a beginner-friendly explanation and, where the snippet compiles against the live library, a runnable example.

| Type | Summary |
|:-----|:--------|
| [`AdaLoRAAdapter`](./AdaLoRAAdapter.md) | Adaptive Low-Rank Adaptation (AdaLoRA) adapter that dynamically allocates parameter budgets among weight matrices. |
| [`ChainLoRAAdapter`](./ChainLoRAAdapter.md) | Chain-of-LoRA adapter that implements sequential composition of multiple LoRA adapters. |
| [`DeltaLoRAAdapter`](./DeltaLoRAAdapter.md) | Delta-LoRA adapter that focuses on parameter-efficient delta updates with momentum. |
| [`DenseLoRAAdapter`](./DenseLoRAAdapter.md) | LoRA adapter specifically for Dense and FullyConnected layers with 1D input/output shapes. |
| [`DoRAAdapter`](./DoRAAdapter.md) | DoRA (Weight-Decomposed Low-Rank Adaptation) adapter for parameter-efficient fine-tuning with improved stability. |
| [`DVoRAAdapter`](./DVoRAAdapter.md) | DVoRA (DoRA + VeRA) adapter - combines DoRA's magnitude-direction decomposition with VeRA's extreme parameter efficiency. |
| [`DyLoRAAdapter`](./DyLoRAAdapter.md) | DyLoRA (Dynamic LoRA) adapter that trains with multiple ranks simultaneously. |
| [`FloraAdapter`](./FloraAdapter.md) | Implements Flora (Low-Rank Adapters Are Secretly Gradient Compressors) adapter for memory-efficient fine-tuning. |
| [`GLoRAAdapter`](./GLoRAAdapter.md) | Generalized LoRA (GLoRA) implementation that adapts both weights AND activations. |
| [`GraphConvolutionalLoRAAdapter`](./GraphConvolutionalLoRAAdapter.md) | LoRA adapter for Graph Convolutional layers, enabling parameter-efficient fine-tuning of GNN models. |
| [`HRAAdapter`](./HRAAdapter.md) | HRA (Hybrid Rank Adaptation) adapter that combines low-rank and full-rank updates for optimal parameter efficiency. |
| [`LoftQAdapter`](./LoftQAdapter.md) | LoftQ (LoRA-Fine-Tuning-Quantized) adapter that combines quantization and LoRA with improved initialization. |
| [`LoHaAdapter`](./LoHaAdapter.md) | LoHa (Low-Rank Hadamard Product Adaptation) adapter for parameter-efficient fine-tuning. |
| [`LoKrAdapter`](./LoKrAdapter.md) | LoKr (Low-Rank Kronecker Product Adaptation) adapter for parameter-efficient fine-tuning. |
| [`LongLoRAAdapter`](./LongLoRAAdapter.md) | LongLoRA adapter that efficiently extends LoRA to handle longer context lengths using shifted sparse attention. |
| [`LoRAAdapterBase`](./LoRAAdapterBase.md) | Abstract base class for LoRA (Low-Rank Adaptation) adapters that wrap existing layers. |
| [`LoRADropAdapter`](./LoRADropAdapter.md) | LoRA-drop implementation: LoRA with dropout regularization. |
| [`LoRAFAAdapter`](./LoRAFAAdapter.md) | LoRA-FA (LoRA with Frozen A matrix) adapter for parameter-efficient fine-tuning. |
| [`LoRAPlusAdapter`](./LoRAPlusAdapter.md) | LoRA+ adapter that uses optimized learning rates for faster convergence and better performance. |
| [`LoRAXSAdapter`](./LoRAXSAdapter.md) | LoRA-XS (Extremely Small) adapter for ultra-parameter-efficient fine-tuning using SVD with trainable scaling matrix. |
| [`LoRETTAAdapter`](./LoRETTAAdapter.md) | LoRETTA (Low-Rank Economic Tensor-Train Adaptation) adapter for parameter-efficient fine-tuning. |
| [`MoRAAdapter`](./MoRAAdapter.md) | Implements MoRA (High-Rank Updating for Parameter-Efficient Fine-Tuning) adapter. |
| [`MultiLoRAAdapter`](./MultiLoRAAdapter.md) | Multi-task LoRA adapter that manages multiple task-specific LoRA layers for complex multi-task learning scenarios. |
| [`NOLAAdapter`](./NOLAAdapter.md) | Implements NOLA (Compressing LoRA using Linear Combination of Random Basis) adapter for extreme parameter efficiency. |
| [`PiSSAAdapter`](./PiSSAAdapter.md) | Principal Singular Values and Singular Vectors Adaptation (PiSSA) adapter for parameter-efficient fine-tuning. |
| [`QALoRAAdapter`](./QALoRAAdapter.md) | Quantization-Aware LoRA (QA-LoRA) adapter that combines parameter-efficient fine-tuning with group-wise quantization awareness. |
| [`QLoRAAdapter`](./QLoRAAdapter.md) | QLoRA (Quantized LoRA) adapter for parameter-efficient fine-tuning with 4-bit quantized base weights. |
| [`ReLoRAAdapter`](./ReLoRAAdapter.md) | Restart LoRA (ReLoRA) adapter that periodically merges and restarts LoRA training for continual learning. |
| [`RoSAAdapter`](./RoSAAdapter.md) | RoSA (Robust Adaptation) adapter for parameter-efficient fine-tuning with improved robustness to distribution shifts. |
| [`SLoRAAdapter`](./SLoRAAdapter.md) | S-LoRA adapter for scalable serving of thousands of concurrent LoRA adapters. |
| [`StandardLoRAAdapter`](./StandardLoRAAdapter.md) | Standard LoRA implementation (original LoRA algorithm). |
| [`TiedLoRAAdapter`](./TiedLoRAAdapter.md) | Tied-LoRA adapter - LoRA with weight tying for extreme parameter efficiency across deep networks. |
| [`VBLoRAAdapter`](./VBLoRAAdapter.md) | Vector Bank LoRA (VB-LoRA) adapter that uses shared parameter banks for efficient multi-client deployment. |
| [`VeRAAdapter`](./VeRAAdapter.md) | VeRA (Vector-based Random Matrix Adaptation) adapter - an extreme parameter-efficient variant of LoRA. |
| [`XLoRAAdapter`](./XLoRAAdapter.md) | X-LoRA (Mixture of LoRA Experts) adapter that uses multiple LoRA experts with learned routing. |
