---
title: "MultiLoRAAdapter"
description: "Multi-task LoRA adapter that manages multiple task-specific LoRA layers for complex multi-task learning scenarios."
section: "Reference"
---

_LoRA / PEFT Adapters_

Multi-task LoRA adapter that manages multiple task-specific LoRA layers for complex multi-task learning scenarios.

## For Beginners

Think of MultiLoRA as having one teacher (the base layer) and multiple students (task-specific LoRA adapters), each specializing in different subjects. In regular LoRA: - You have one base layer (the teacher) - One LoRA adapter (one student learning one subject) - Output = base + lora_adaptation In MultiLoRA: - You have one base layer (the teacher) - Multiple LoRA adapters (multiple students, each specializing in different tasks) - Output = base + task_specific_lora_adaptation This is powerful for: 1. Multi-domain learning: Train on medical, legal, and technical documents simultaneously 2. Multi-lingual models: One adapter per language 3. Multi-task learning: Sentiment analysis, named entity recognition, question answering, etc. 4. Continual learning: Add new tasks without forgetting old ones Example use case: - Base: Pre-trained language model - Task 1: Sentiment analysis (rank=4) - Task 2: Named entity recognition (rank=8) - Task 3: Question answering (rank=16) You can switch between tasks at runtime, and each task only trains its specific LoRA weights!

## How It Works

MultiLoRA extends the basic LoRA concept to handle multiple tasks simultaneously within a single layer. Instead of having one LoRA adaptation, it maintains a dictionary of task-specific LoRA layers, with a routing mechanism to select the appropriate adapter for each task. 

Key features: - Multiple task-specific LoRA adapters sharing the same base layer - Dynamic task switching during inference and training - Per-task rank configuration for optimal parameter efficiency - Shared base layer weights across all tasks - Task-specific merging for deployment

