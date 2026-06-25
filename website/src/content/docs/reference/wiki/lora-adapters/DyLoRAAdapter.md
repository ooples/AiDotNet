---
title: "DyLoRAAdapter"
description: "DyLoRA (Dynamic LoRA) adapter that trains with multiple ranks simultaneously."
section: "Reference"
---

_LoRA / PEFT Adapters_

DyLoRA (Dynamic LoRA) adapter that trains with multiple ranks simultaneously.

## For Beginners

DyLoRA is like LoRA with a superpower - flexibility! Standard LoRA problem: - You choose rank=8 and train - Later realize rank=4 would work fine (save memory/speed) - Or need rank=16 for better quality - Must retrain from scratch with the new rank DyLoRA solution: - Train once with multiple ranks (e.g., [2, 4, 8, 16]) - Deploy with ANY of those ranks without retraining - Switch between ranks at runtime based on device capabilities How it works: 1. Train with MaxRank (e.g., 16) but randomly use smaller ranks during training 2. Nested dropout ensures each rank works independently 3. After training, pick deployment rank based on needs (2=fastest, 16=best quality) Use cases: - Deploy same model to mobile (rank=2) and server (rank=16) - Dynamic quality scaling based on battery level - A/B testing different rank/quality trade-offs - Training once, deploying everywhere Example: Train with ActiveRanks=[2,4,8], deploy with: - Rank=2 for mobile devices (98% parameter reduction, good quality) - Rank=4 for tablets (95% parameter reduction, better quality) - Rank=8 for desktops (90% parameter reduction, best quality)

## How It Works

DyLoRA extends the standard LoRA approach by training multiple rank configurations simultaneously using a nested dropout technique. This allows a single trained adapter to be deployed at different rank levels without retraining, providing flexibility for different hardware constraints or performance requirements. 

The key innovation is nested dropout: during training, for each forward pass, a random rank r is selected from the active ranks, and only the first r components of matrices A and B are used. This ensures that smaller ranks can function independently and don't rely on higher-rank components.

