---
title: "ChainLoRAAdapter"
description: "Chain-of-LoRA adapter that implements sequential composition of multiple LoRA adapters."
section: "Reference"
---

_LoRA / PEFT Adapters_

Chain-of-LoRA adapter that implements sequential composition of multiple LoRA adapters.

## For Beginners

Imagine you're learning a complex skill in stages: 1. First, you learn the basics (Adapter 1) 2. Then you practice and the basics become automatic (Merge) 3. Next, you learn intermediate techniques on top of the basics (Adapter 2) 4. Again, you practice until they're automatic (Merge) 5. Finally, you learn advanced skills building on everything before (Adapter 3) Chain-of-LoRA works the same way: each adapter learns something new, then it's consolidated into the model, and the next adapter can focus on the next refinement. This stepwise approach often achieves better results than trying to learn everything at once.

## How It Works

Chain-of-LoRA (COLA) is an advanced LoRA technique that enables sequential composition of multiple LoRA adaptations through an iterative optimization framework. Unlike standard LoRA which applies a single low-rank adaptation, COLA builds a chain of adaptations where each adapter is trained, merged into the model, and then a new adapter is initialized for further refinement. 

This approach bridges the performance gap between standard LoRA and full fine-tuning by employing residual learning principles. Each iteration in the chain adds incremental improvements to the model's task-specific performance without incurring additional computational costs or memory overhead during inference. 

**Key Concepts:****Sequential Adaptation:** Chain-of-LoRA applies adaptations in sequence (Task A → Task B → Task C), where each stage builds upon the previous one. This is inspired by the Frank-Wolfe optimization algorithm, which makes greedy updates along the direction of maximum improvement. **Merge and Re-initialize:** After training each LoRA adapter, the learned weights are merged back into the base layer, and a new LoRA adapter is initialized. This "tying a knot" process allows the model to consolidate learned knowledge before adding new adaptations. **Knowledge Preservation:** By freezing the base layer and only training the LoRA components, the chain preserves previously learned knowledge while allowing new task-specific adaptations. Each adapter in the chain captures a specific aspect of the task or a refinement step. **Incremental Fine-tuning Pipeline:** COLA enables continual learning scenarios where tasks are presented sequentially, and the model must adapt to new tasks while maintaining performance on previous ones. 

**Benefits of Chain-of-LoRA:** - **Better Performance:** Achieves up to 6.47% relative accuracy gain over standard LoRA - **No Extra Overhead:** After merging, inference cost is identical to the base model - **Modular Adaptation:** Each adapter can be trained, tested, and validated independently - **Catastrophic Forgetting Mitigation:** Sequential merging helps preserve prior knowledge - **Task Chaining:** Naturally supports multi-task learning and transfer learning scenarios - **Flexible Deployment:** Can deploy the full chain or selected adapters as needed 

**Research Reference:** Based on "Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning" (arXiv:2401.04151, January 2024). The paper demonstrates that sequential low-rank adaptations can significantly improve task performance compared to single-stage LoRA, especially on complex reasoning and multi-step tasks. 

**Usage Example:**

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new ChainLoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured ChainLoRAAdapter (rank {config.Rank}).");
```

