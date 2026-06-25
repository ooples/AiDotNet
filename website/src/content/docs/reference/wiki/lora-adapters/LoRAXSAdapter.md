---
title: "LoRAXSAdapter"
description: "LoRA-XS (Extremely Small) adapter for ultra-parameter-efficient fine-tuning using SVD with trainable scaling matrix."
section: "Reference"
---

_LoRA / PEFT Adapters_

LoRA-XS (Extremely Small) adapter for ultra-parameter-efficient fine-tuning using SVD with trainable scaling matrix.

## For Beginners

Think of LoRA-XS as "ultra-compressed LoRA". Imagine you have a large language model with huge weight matrices (e.g., 4096×4096): Standard LoRA (rank 8): - Creates two matrices: A (4096×8) and B (8×4096) - Total parameters: 4096*8 + 8*4096 = 65,536 parameters - Both matrices are trainable LoRA-XS (rank 8): - Decomposes pretrained weights with SVD into U, Σ, V - Keeps top 8 singular vectors (U_8, Σ_8, V_8) FROZEN - Trains only R matrix: 8×8 = 64 parameters - Achieves similar or better performance with 1000x fewer parameters! It's like having two fixed "coordinate systems" from the pretrained model, and you only train a small "rotation matrix" between them. The fixed coordinate systems capture the pretrained knowledge, while the rotation matrix adapts to your task. Example workflow: 1. Load pretrained model weights W 2. Compute SVD: W = U Σ V^T 3. Extract top-r components: U_r, Σ_r, V_r 4. Create LoRA-XS adapter with these frozen bases 5. Train only the tiny R matrix (64 params for rank 8) 6. Deploy with merged weights: W' = W + U_r Σ_r R V_r^T

## How It Works

LoRA-XS achieves extreme parameter efficiency by leveraging SVD of pretrained weights to create frozen orthonormal bases (U and V matrices), with only a small r×r trainable matrix R positioned between them. This architecture reduces parameter count to r² instead of 2nr (standard LoRA), achieving 100x+ reduction while matching or exceeding full fine-tuning performance. 

**Architecture Comparison:** - Standard LoRA: W' = W + BA, where A ∈ ℝ^(d×r), B ∈ ℝ^(r×d) (2dr parameters) - LoRA-XS: W' = W + U_r Σ_r R V_r^T, where only R ∈ ℝ^(r×r) is trainable (r² parameters) - U_r and V_r are frozen orthonormal bases from SVD of pretrained W - Σ_r is the frozen diagonal matrix of top-r singular values 

**Key Innovation:** Instead of training both A and B matrices (standard LoRA), LoRA-XS: 1. Computes SVD of pretrained weights: W = U Σ V^T 2. Freezes U_r (top-r left singular vectors) and V_r^T (top-r right singular vectors) 3. Freezes Σ_r (top-r singular values as diagonal matrix) 4. Trains only R (r×r mixing matrix) that interpolates between frozen bases 5. Parameter count is independent of hidden dimensions: only r² trainable parameters 

**Performance Metrics (from paper):** RoBERTa-large on GLUE (6 tasks): - LoRA-XS (rank 16): 88.03% avg accuracy, 24.6K parameters - Standard LoRA (rank 16): Similar accuracy, 100x more parameters - Full fine-tuning: 88.0% avg accuracy, ~125M parameters per task LLaMA2-7B on Commonsense Reasoning: - LoRA-XS: 80.5% avg accuracy, 3.67M parameters - Standard LoRA: 77.6% avg accuracy, 56M parameters (15x more) Mistral-7B on GSM8K (Math Reasoning): - LoRA-XS: 70.35% accuracy, 3.67M parameters - Standard LoRA: 67.70% accuracy, 168M parameters (46x more) GPT-3 Personalization (1M models): - LoRA-XS: 96GB total storage - Standard LoRA: 144TB total storage (1500x reduction) 

**Mathematical Formulation:** Forward pass computes: output = (W + U_r Σ_r R V_r^T) * input = W * input + (U_r Σ_r) * (R * (V_r^T * input)) Where: - W is frozen pretrained weights - U_r ∈ ℝ^(d_out × r): frozen left singular vectors (orthonormal columns) - Σ_r ∈ ℝ^(r × r): frozen diagonal matrix of singular values - R ∈ ℝ^(r × r): trainable mixing matrix (only trainable component!) - V_r^T ∈ ℝ^(r × d_in): frozen right singular vectors (orthonormal rows) 

**Why This Works:** The SVD provides an optimal orthonormal basis for representing weight updates. By freezing these bases and training only the mixing matrix R, LoRA-XS achieves: - Drastically fewer parameters (r² vs 2dr) - Better generalization (constrained to pretrained subspace) - Faster convergence (optimal basis from initialization) - No inference overhead (can be merged back into W) - Scalable personalization (parameter count independent of model size) 

**References:** - Paper: "LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters" - arXiv: 2405.17604 (May 2024) - GitHub: MohammadrezaBanaei/LoRA-XS - Key Innovation: Parameter count O(r²) instead of O(dr), enabling extreme efficiency

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new LoRAXSAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured LoRAXSAdapter (rank {config.Rank}).");
```

