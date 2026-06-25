---
title: "LoRA / PEFT Adapters"
description: "Every LoRA / PEFT Adapters type in AiDotNet, auto-generated with compile-checked examples."
section: "Reference"
---

Every LoRA / PEFT Adapters type in AiDotNet — each with a beginner-friendly explanation and, where the snippet compiles against the live library, a runnable example.

| Type | Summary |
|:-----|:--------|
| [`AdaLoRAAdapter`](/docs/reference/wiki/lora-adapters/adaloraadapter/) | Adaptive Low-Rank Adaptation (AdaLoRA) adapter that dynamically allocates parameter budgets among weight matrices. |
| [`ChainLoRAAdapter`](/docs/reference/wiki/lora-adapters/chainloraadapter/) | Chain-of-LoRA adapter that implements sequential composition of multiple LoRA adapters. |
| [`DeltaLoRAAdapter`](/docs/reference/wiki/lora-adapters/deltaloraadapter/) | Delta-LoRA adapter that focuses on parameter-efficient delta updates with momentum. |
| [`DenseLoRAAdapter`](/docs/reference/wiki/lora-adapters/denseloraadapter/) | LoRA adapter specifically for Dense and FullyConnected layers with 1D input/output shapes. |
| [`DoRAAdapter`](/docs/reference/wiki/lora-adapters/doraadapter/) | DoRA (Weight-Decomposed Low-Rank Adaptation) adapter for parameter-efficient fine-tuning with improved stability. |
| [`DVoRAAdapter`](/docs/reference/wiki/lora-adapters/dvoraadapter/) | DVoRA (DoRA + VeRA) adapter - combines DoRA's magnitude-direction decomposition with VeRA's extreme parameter efficiency. |
| [`DyLoRAAdapter`](/docs/reference/wiki/lora-adapters/dyloraadapter/) | DyLoRA (Dynamic LoRA) adapter that trains with multiple ranks simultaneously. |
| [`FloraAdapter`](/docs/reference/wiki/lora-adapters/floraadapter/) | Implements Flora (Low-Rank Adapters Are Secretly Gradient Compressors) adapter for memory-efficient fine-tuning. |
| [`GLoRAAdapter`](/docs/reference/wiki/lora-adapters/gloraadapter/) | Generalized LoRA (GLoRA) implementation that adapts both weights AND activations. |
| [`GraphConvolutionalLoRAAdapter`](/docs/reference/wiki/lora-adapters/graphconvolutionalloraadapter/) | LoRA adapter for Graph Convolutional layers, enabling parameter-efficient fine-tuning of GNN models. |
| [`HRAAdapter`](/docs/reference/wiki/lora-adapters/hraadapter/) | HRA (Hybrid Rank Adaptation) adapter that combines low-rank and full-rank updates for optimal parameter efficiency. |
| [`LoftQAdapter`](/docs/reference/wiki/lora-adapters/loftqadapter/) | LoftQ (LoRA-Fine-Tuning-Quantized) adapter that combines quantization and LoRA with improved initialization. |
| [`LoHaAdapter`](/docs/reference/wiki/lora-adapters/lohaadapter/) | LoHa (Low-Rank Hadamard Product Adaptation) adapter for parameter-efficient fine-tuning. |
| [`LoKrAdapter`](/docs/reference/wiki/lora-adapters/lokradapter/) | LoKr (Low-Rank Kronecker Product Adaptation) adapter for parameter-efficient fine-tuning. |
| [`LongLoRAAdapter`](/docs/reference/wiki/lora-adapters/longloraadapter/) | LongLoRA adapter that efficiently extends LoRA to handle longer context lengths using shifted sparse attention. |
| [`LoRAAdapterBase`](/docs/reference/wiki/lora-adapters/loraadapterbase/) | Abstract base class for LoRA (Low-Rank Adaptation) adapters that wrap existing layers. |
| [`LoRADropAdapter`](/docs/reference/wiki/lora-adapters/loradropadapter/) | LoRA-drop implementation: LoRA with dropout regularization. |
| [`LoRAFAAdapter`](/docs/reference/wiki/lora-adapters/lorafaadapter/) | LoRA-FA (LoRA with Frozen A matrix) adapter for parameter-efficient fine-tuning. |
| [`LoRAPlusAdapter`](/docs/reference/wiki/lora-adapters/loraplusadapter/) | LoRA+ adapter that uses optimized learning rates for faster convergence and better performance. |
| [`LoRAXSAdapter`](/docs/reference/wiki/lora-adapters/loraxsadapter/) | LoRA-XS (Extremely Small) adapter for ultra-parameter-efficient fine-tuning using SVD with trainable scaling matrix. |
| [`LoRETTAAdapter`](/docs/reference/wiki/lora-adapters/lorettaadapter/) | LoRETTA (Low-Rank Economic Tensor-Train Adaptation) adapter for parameter-efficient fine-tuning. |
| [`MoRAAdapter`](/docs/reference/wiki/lora-adapters/moraadapter/) | Implements MoRA (High-Rank Updating for Parameter-Efficient Fine-Tuning) adapter. |
| [`MultiLoRAAdapter`](/docs/reference/wiki/lora-adapters/multiloraadapter/) | Multi-task LoRA adapter that manages multiple task-specific LoRA layers for complex multi-task learning scenarios. |
| [`NOLAAdapter`](/docs/reference/wiki/lora-adapters/nolaadapter/) | Implements NOLA (Compressing LoRA using Linear Combination of Random Basis) adapter for extreme parameter efficiency. |
| [`PiSSAAdapter`](/docs/reference/wiki/lora-adapters/pissaadapter/) | Principal Singular Values and Singular Vectors Adaptation (PiSSA) adapter for parameter-efficient fine-tuning. |
| [`QALoRAAdapter`](/docs/reference/wiki/lora-adapters/qaloraadapter/) | Quantization-Aware LoRA (QA-LoRA) adapter that combines parameter-efficient fine-tuning with group-wise quantization awareness. |
| [`QLoRAAdapter`](/docs/reference/wiki/lora-adapters/qloraadapter/) | QLoRA (Quantized LoRA) adapter for parameter-efficient fine-tuning with 4-bit quantized base weights. |
| [`ReLoRAAdapter`](/docs/reference/wiki/lora-adapters/reloraadapter/) | Restart LoRA (ReLoRA) adapter that periodically merges and restarts LoRA training for continual learning. |
| [`RoSAAdapter`](/docs/reference/wiki/lora-adapters/rosaadapter/) | RoSA (Robust Adaptation) adapter for parameter-efficient fine-tuning with improved robustness to distribution shifts. |
| [`SLoRAAdapter`](/docs/reference/wiki/lora-adapters/sloraadapter/) | S-LoRA adapter for scalable serving of thousands of concurrent LoRA adapters. |
| [`StandardLoRAAdapter`](/docs/reference/wiki/lora-adapters/standardloraadapter/) | Standard LoRA implementation (original LoRA algorithm). |
| [`TiedLoRAAdapter`](/docs/reference/wiki/lora-adapters/tiedloraadapter/) | Tied-LoRA adapter - LoRA with weight tying for extreme parameter efficiency across deep networks. |
| [`VBLoRAAdapter`](/docs/reference/wiki/lora-adapters/vbloraadapter/) | Vector Bank LoRA (VB-LoRA) adapter that uses shared parameter banks for efficient multi-client deployment. |
| [`VeRAAdapter`](/docs/reference/wiki/lora-adapters/veraadapter/) | VeRA (Vector-based Random Matrix Adaptation) adapter - an extreme parameter-efficient variant of LoRA. |
| [`XLoRAAdapter`](/docs/reference/wiki/lora-adapters/xloraadapter/) | X-LoRA (Mixture of LoRA Experts) adapter that uses multiple LoRA experts with learned routing. |
