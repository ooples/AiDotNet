---
title: "TrainingStageType"
description: "Types of training stages in a multi-stage training pipeline."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Types of training stages in a multi-stage training pipeline.

## How It Works

Training pipelines consist of multiple stages, each with a specific purpose.
Modern LLM training typically follows patterns like:

## Fields

| Field | Summary |
|:-----|:--------|
| `AdapterMerging` | Adapter merging stage. |
| `AdversarialTraining` | Adversarial training for robustness. |
| `AgenticTraining` | Agentic behavior training stage. |
| `AudioLanguageAlignment` | Audio-language alignment stage. |
| `BestOfNSampling` | Best-of-N sampling and fine-tuning. |
| `ChainOfThoughtTraining` | Chain-of-thought training stage. |
| `Checkpoint` | Checkpoint and validation stage. |
| `CodeFineTuning` | Code-specific fine-tuning stage. |
| `Constitutional` | Constitutional AI training stage. |
| `ContinuedPreTraining` | Continued pre-training on domain-specific data. |
| `ContrastivePreference` | Contrastive Preference Optimization (CPO, NCA, Safe-NCA). |
| `Cooldown` | Cooldown stage with learning rate decay. |
| `CurriculumLearning` | Curriculum learning stage with progressively harder examples. |
| `Custom` | Custom user-defined training logic. |
| `Evaluation` | Evaluation-only stage (no training). |
| `FeatureDistillation` | Feature distillation stage. |
| `GroupRelativePolicyOptimization` | Group Relative Policy Optimization (GRPO) - DeepSeek's approach. |
| `HarmlessnessTraining` | Harmlessness training stage. |
| `HelpfulnessTraining` | Helpfulness training stage. |
| `HonestyTraining` | Honesty/truthfulness training stage. |
| `InstructionTuning` | Instruction tuning - SFT specifically for following instructions. |
| `JailbreakResistance` | Jailbreak resistance training. |
| `KnowledgeDistillation` | Knowledge distillation from teacher model. |
| `LoRAAdapterTraining` | LoRA/QLoRA adapter training stage. |
| `LongContextTraining` | Long context training stage. |
| `MathReasoningFineTuning` | Math/reasoning fine-tuning stage. |
| `ModelMerging` | Model merging stage. |
| `MultiTurnConversation` | Multi-turn conversation training stage. |
| `OddsRatioPreference` | Odds Ratio Preference Optimization (ORPO). |
| `OutcomeRewardModelTraining` | Outcome reward model training (final answer rewards). |
| `PreTraining` | Pre-training on large unlabeled text corpora (next token prediction). |
| `PreferenceOptimization` | Direct Preference Optimization family (DPO, IPO, KTO, SimPO, CPO, R-DPO, etc.). |
| `PreferenceRanking` | Preference Ranking Optimization (PRO). |
| `PrefixTuning` | Prefix tuning stage. |
| `ProcessRewardModelTraining` | Process reward model training (step-level rewards). |
| `ProgressiveTraining` | Progressive training with increasing model capacity. |
| `PromptTuning` | Prompt tuning stage. |
| `ProximalPolicyOptimization` | Proximal Policy Optimization (PPO) stage. |
| `RankResponses` | Rank Responses to align Human Feedback (RRHF). |
| `RedTeamingTraining` | Red-teaming data training. |
| `ReinforcementLearning` | Reinforcement Learning from Human Feedback (RLHF). |
| `ReinforcementLearningAIFeedback` | Reinforcement Learning from AI Feedback (RLAIF). |
| `ReinforcementLearningVerifiable` | Reinforcement Learning with Verifiable Rewards (RLVR). |
| `RejectionSampling` | Rejection sampling optimization (RSO). |
| `ResponseDistillation` | Response distillation stage. |
| `RewardModelTraining` | Reward model training stage. |
| `SafetyAlignment` | Safety alignment training. |
| `SelfPlay` | Self-Play Fine-Tuning (SPIN). |
| `SelfRewarding` | Self-rewarding language models stage. |
| `SelfTraining` | Self-training / self-improvement stage. |
| `SequenceLikelihoodCalibration` | Sequence Likelihood Calibration (SLiC-HF). |
| `SupervisedFineTuning` | Supervised fine-tuning on input-output pairs. |
| `SyntheticDataTraining` | Synthetic data generation and training. |
| `ToolUseTraining` | Tool use training stage. |
| `VisionLanguagePreTraining` | Vision-language pre-training stage. |
| `VisualInstructionTuning` | Visual instruction tuning stage. |
| `Warmup` | Warmup stage with lower learning rate. |

