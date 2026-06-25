---
title: "MoRAAdapter"
description: "Implements MoRA (High-Rank Updating for Parameter-Efficient Fine-Tuning) adapter."
section: "Reference"
---

_LoRA / PEFT Adapters_

Implements MoRA (High-Rank Updating for Parameter-Efficient Fine-Tuning) adapter.

## For Beginners

MoRA is like an upgraded version of LoRA that can learn
more complex changes to a model while using the same amount of memory.

Think of it like this:

- LoRA is like having 2 small notebooks to write changes (matrices A and B)
- MoRA is like having 1 square notebook plus a compression/decompression scheme

The key insight: By compressing the input, applying changes in compressed space,
and then decompressing, MoRA can make higher-rank updates that capture more
complex patterns. This is especially useful when you're teaching the model
entirely new facts or concepts, not just adapting its existing knowledge.

Example: If you're fine-tuning a model to learn medical terminology, MoRA
will be better at memorizing the new terms, while LoRA might be better at
learning to reason about medical cases using existing knowledge.

## How It Works

**Paper Reference:** "MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning"
by Ting Jiang, Shaohan Huang, et al. (arXiv:2405.12130, May 2024)

MoRA addresses a fundamental limitation of LoRA: the low-rank constraint restricts the model's
ability to learn and memorize new knowledge. While LoRA uses two rectangular matrices (A and B)
to create low-rank updates, MoRA uses a single square matrix M combined with non-parameter-sharing
operators to achieve high-rank updates while maintaining the same parameter count.

**Key Innovations:**

1. **High-Rank Updates**: Unlike LoRA's rank-r updates (r << d), MoRA achieves rank-r̂

updates where r̂ can equal the full dimension d, enabling the model to learn richer representations.

2. **Square Matrix M**: Instead of LoRA's A (d×r) and B (r×d) matrices, MoRA uses a single

square matrix M (r×r) where r = sqrt(d×d / 2). For the same parameter count as LoRA,
MoRA achieves much higher effective rank.

3. **Non-Parameter-Sharing Operators**: MoRA uses rotation, permutation, or other linear

transformations that don't add trainable parameters but enable dimension compression
and decompression around the square matrix M.

4. **Input Compression / Output Decompression**: The architecture is:
- Compress: Input (d) to Compressed (r) via rotation/permutation
- Transform: Compressed (r) to Transformed (r) via trainable matrix M
- Decompress: Transformed (r) to Output (d) via inverse rotation/permutation

**Architecture Comparison:**

LoRA: W = W₀ + BA where A ∈ ℝ^(d×r), B ∈ ℝ^(r×d)

- Parameters: 2dr
- Rank: r (low-rank constraint)
- Typical r: 8-64

MoRA: W = W₀ + R_d^(-1) M R_c where M ∈ ℝ^(r×r)

- Parameters: r²
- Rank: min(r, d) (can be full-rank)
- For same param count as LoRA: r = sqrt(2dr), so rank ≈ sqrt(2dr)
- Example: LoRA with r=8, d=1024 has 16,384 params and rank 8

MoRA with same params: r=128, rank 128 (16× higher!)

**Performance (from paper):**

Compared to LoRA on various tasks:

- **Memory-Intensive Tasks**: MoRA significantly outperforms LoRA
* Continual Pretraining: ~15% better perplexity
* Instruction Tuning: ~8% better accuracy on knowledge-intensive QA
- **Reasoning Tasks**: MoRA performs comparably to LoRA
* Mathematical Reasoning: Similar performance (within 1-2%)
- **Parameter Efficiency**: Same parameter count as LoRA
- **Training Speed**: Slightly slower than LoRA due to rotation operations (≈5-10% overhead)

**When to Use MoRA vs LoRA:**

Use MoRA when:

- Task requires memorizing new facts or knowledge
- Domain adaptation with significant vocabulary changes
- Continual learning scenarios
- You need the model to "remember" rather than just "adapt"

Use LoRA when:

- Task is primarily reasoning or pattern recognition
- Minimal new knowledge acquisition needed
- Training speed is critical
- Standard parameter-efficient fine-tuning is sufficient

**Implementation Details:**

This implementation uses rotation matrices as the non-parameter-sharing operators:

- Compression R_c: Projects input from dimension d to dimension r
- Decompression R_d: Projects from dimension r back to dimension d
- These are generated using random orthogonal matrices (Gram-Schmidt orthogonalization)
- They remain fixed during training (non-trainable)

Alternative operators mentioned in the paper (not implemented here):

- RoPE-based rotations (Rotary Position Embeddings)
- Random permutations
- Structured rotations (e.g., Hadamard transforms)

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new MoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured MoRAAdapter (rank {config.Rank}).");
```

