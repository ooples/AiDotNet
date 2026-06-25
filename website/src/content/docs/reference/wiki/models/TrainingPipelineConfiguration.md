---
title: "TrainingPipelineConfiguration<T, TInput, TOutput>"
description: "Configuration for a multi-step training pipeline with customizable stages."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration for a multi-step training pipeline with customizable stages.

## For Beginners

Think of this as a recipe with multiple cooking steps.
Just like you might marinate, then sear, then bake - training can have multiple
phases where each phase teaches the model something different.

## How It Works

A training pipeline defines a sequence of training stages that are executed in order.
Each stage can have its own training method, optimizer, learning rate, dataset, and
evaluation criteria. This enables advanced training workflows like:

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointDirectory` | Gets or sets the directory for intermediate checkpoints. |
| `DefaultBatchSize` | Gets or sets the default batch size for all stages. |
| `DefaultLearningRate` | Gets or sets the default learning rate for all stages. |
| `DefaultOptimizer` | Gets or sets the default optimizer type for all stages. |
| `Description` | Gets or sets the description of this pipeline. |
| `EnableAutoSelection` | Gets or sets whether to use automatic pipeline selection when no stages are defined. |
| `EnableExperimentTracking` | Gets or sets whether to log to WandB or similar experiment trackers. |
| `EvaluateAfterEachStage` | Gets or sets whether to run evaluation after each stage. |
| `EvaluationMetrics` | Gets or sets the evaluation metrics to track. |
| `ExperimentName` | Gets or sets the experiment name for tracking. |
| `GlobalEarlyStopping` | Gets or sets the global early stopping configuration applied across stages. |
| `GlobalEvaluationData` | Gets or sets the evaluation data to use across all stages. |
| `GlobalSeed` | Gets or sets the global random seed for reproducibility across all stages. |
| `InterStageCallbacks` | Gets or sets callback actions to execute between stages. |
| `LogDirectory` | Gets or sets the logging directory. |
| `MaxCheckpointsToKeep` | Gets or sets the maximum number of checkpoints to keep. |
| `Metadata` | Gets or sets custom metadata for the pipeline. |
| `MixedPrecisionDType` | Gets or sets the mixed precision data type. |
| `Name` | Gets or sets the name of this pipeline for identification. |
| `OnPipelineComplete` | Gets or sets callback actions to execute when the pipeline completes. |
| `OnPipelineError` | Gets or sets callback actions to execute on pipeline failure. |
| `OnPipelineStart` | Gets or sets callback actions to execute before the pipeline starts. |
| `ResumeCheckpointPath` | Gets or sets the specific checkpoint path to resume from. |
| `ResumeFromCheckpoint` | Gets or sets whether to resume from the latest checkpoint. |
| `SaveIntermediateCheckpoints` | Gets or sets whether to save checkpoints between stages. |
| `Stages` | Gets or sets the ordered list of training stages in the pipeline. |
| `Tags` | Gets or sets tags for categorizing the pipeline. |
| `UseMixedPrecision` | Gets or sets whether to use mixed precision training globally. |
| `VerboseLogging` | Gets or sets whether to enable verbose logging. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAdapterMergingStage(Action<TrainingStage<,,>>)` | Adds an adapter merging stage. |
| `AddAgenticStage(Action<TrainingStage<,,>>)` | Adds an agentic behavior training stage. |
| `AddCPOStage(Action<TrainingStage<,,>>)` | Adds a Contrastive Preference Optimization (CPO) stage. |
| `AddChainOfThoughtStage(Action<TrainingStage<,,>>)` | Adds a chain-of-thought training stage. |
| `AddCheckpointStage(Action<TrainingStage<,,>>)` | Adds a checkpoint stage. |
| `AddCodeFineTuningStage(Action<TrainingStage<,,>>)` | Adds a code fine-tuning stage. |
| `AddConstitutionalAIStage(String[],Action<TrainingStage<,,>>)` | Adds a Constitutional AI stage. |
| `AddCustomStage(String,Func<IFullModel<,,>,FineTuningData<,,>,CancellationToken,Task<IFullModel<,,>>>,Action<TrainingStage<,,>>)` | Adds a custom training stage with user-defined logic. |
| `AddDPOStage(Action<TrainingStage<,,>>)` | Adds a Direct Preference Optimization (DPO) stage. |
| `AddDistillationStage(IFullModel<,,>,Action<TrainingStage<,,>>)` | Adds a knowledge distillation stage. |
| `AddEvaluationStage(Action<TrainingStage<,,>>)` | Adds an evaluation-only stage (no training, just metrics). |
| `AddGRPOStage(Action<TrainingStage<,,>>)` | Adds a GRPO (Group Relative Policy Optimization) stage. |
| `AddHarmlessnessStage(Action<TrainingStage<,,>>)` | Adds a harmlessness training stage. |
| `AddHelpfulnessStage(Action<TrainingStage<,,>>)` | Adds a helpfulness training stage. |
| `AddIPOStage(Action<TrainingStage<,,>>)` | Adds an Identity Preference Optimization (IPO) stage. |
| `AddInstructionTuningStage(Action<TrainingStage<,,>>)` | Adds an instruction tuning stage (specialized SFT). |
| `AddKTOStage(Action<TrainingStage<,,>>)` | Adds a Kahneman-Tversky Optimization (KTO) stage. |
| `AddLoRAStage(Int32,Action<TrainingStage<,,>>)` | Adds a LoRA adapter training stage. |
| `AddMathReasoningStage(Action<TrainingStage<,,>>)` | Adds a math reasoning fine-tuning stage. |
| `AddMultiTurnConversationStage(Action<TrainingStage<,,>>)` | Adds a multi-turn conversation training stage. |
| `AddORPOStage(Action<TrainingStage<,,>>)` | Adds an Odds Ratio Preference Optimization (ORPO) stage. |
| `AddPPOStage(Action<TrainingStage<,,>>)` | Adds a PPO stage. |
| `AddPROStage(Action<TrainingStage<,,>>)` | Adds a Preference Ranking Optimization (PRO) stage. |
| `AddPreferenceStage(FineTuningMethodType,Action<TrainingStage<,,>>)` | Adds a generic preference optimization stage with configurable method. |
| `AddProcessRewardModelStage(Action<TrainingStage<,,>>)` | Adds a process reward model (PRM) training stage. |
| `AddQLoRAStage(Int32,Int32,Action<TrainingStage<,,>>)` | Adds a QLoRA (quantized LoRA) stage. |
| `AddRLAIFStage(Action<TrainingStage<,,>>)` | Adds an RLAIF (RL from AI Feedback) stage. |
| `AddRLHFStage(Action<TrainingStage<,,>>)` | Adds an RLHF (PPO-based) stage. |
| `AddRLStage(FineTuningMethodType,Action<TrainingStage<,,>>)` | Adds a generic reinforcement learning stage. |
| `AddRRHFStage(Action<TrainingStage<,,>>)` | Adds a Rank Responses to align Human Feedback (RRHF) stage. |
| `AddRejectionSamplingStage(Action<TrainingStage<,,>>)` | Adds a Rejection Sampling Optimization (RSO) stage. |
| `AddRewardModelStage(Action<TrainingStage<,,>>)` | Adds a reward model training stage. |
| `AddRobustDPOStage(Action<TrainingStage<,,>>)` | Adds a Robust DPO (R-DPO) stage. |
| `AddSFTStage(Action<TrainingStage<,,>>)` | Adds a supervised fine-tuning (SFT) stage. |
| `AddSLiCStage(Action<TrainingStage<,,>>)` | Adds a Sequence Likelihood Calibration (SLiC-HF) stage. |
| `AddSPINStage(Action<TrainingStage<,,>>)` | Adds a Self-Play Fine-Tuning (SPIN) stage. |
| `AddSafetyAlignmentStage(Action<TrainingStage<,,>>)` | Adds a safety alignment stage. |
| `AddSelfRewardingStage(Action<TrainingStage<,,>>)` | Adds a self-rewarding stage. |
| `AddSimPOStage(Action<TrainingStage<,,>>)` | Adds a Simple Preference Optimization (SimPO) stage. |
| `AddStage(TrainingStage<,,>)` | Adds a training stage to the pipeline. |
| `AddSyntheticDataStage(IFullModel<,,>,Action<TrainingStage<,,>>)` | Adds a synthetic data training stage. |
| `AddToolUseStage(Action<TrainingStage<,,>>)` | Adds a tool use training stage. |
| `AgentTraining(FineTuningData<,,>,FineTuningData<,,>,FineTuningData<,,>)` | Creates an agent/tool-use training pipeline. |
| `AnthropicClaude(FineTuningData<,,>,String[])` | Creates an Anthropic Claude-style pipeline (SFT → Constitutional AI → RLHF). |
| `Auto(FineTuningData<,,>)` | Automatically selects an appropriate pipeline based on available data. |
| `CodeModel(FineTuningData<,,>,FineTuningData<,,>)` | Creates a code model training pipeline. |
| `ConstitutionalAI(FineTuningData<,,>,String[])` | Creates a Constitutional AI pipeline (SFT → CAI critique/revision → preference). |
| `CurriculumLearning(ValueTuple<String,FineTuningData<,,>>[])` | Creates a curriculum learning pipeline with progressively harder stages. |
| `DeepSeek(FineTuningData<,,>,FineTuningData<,,>)` | Creates a DeepSeek-style pipeline (SFT → GRPO). |
| `FullRLHF(FineTuningData<,,>,FineTuningData<,,>)` | Creates a full RLHF pipeline (SFT → Reward Model → PPO). |
| `InstructGPT(FineTuningData<,,>,FineTuningData<,,>,FineTuningData<,,>)` | Creates an OpenAI InstructGPT-style pipeline (SFT → Reward Model → PPO). |
| `IterativeRefinement(Int32,FineTuningData<,,>,FineTuningData<,,>)` | Creates an iterative refinement pipeline that runs multiple DPO rounds. |
| `IterativeSPIN(Int32,FineTuningData<,,>)` | Creates an iterative SPIN pipeline. |
| `KnowledgeDistillation(IFullModel<,,>,FineTuningData<,,>)` | Creates a distillation pipeline (teacher → student). |
| `Llama2(FineTuningData<,,>,FineTuningData<,,>)` | Creates a Meta Llama 2-style pipeline (SFT → Rejection Sampling → DPO). |
| `LoRAFineTuning(Int32,FineTuningData<,,>,FineTuningData<,,>)` | Creates a memory-efficient LoRA fine-tuning pipeline. |
| `MathReasoning(FineTuningData<,,>,FineTuningData<,,>)` | Creates a math reasoning model pipeline. |
| `ORPOAlignment(FineTuningData<,,>)` | Creates a reference-free alignment pipeline using ORPO. |
| `QLoRAFineTuning(Int32,Int32,FineTuningData<,,>)` | Creates a QLoRA fine-tuning pipeline for maximum memory efficiency. |
| `SafetyFocused(FineTuningData<,,>,FineTuningData<,,>,String[])` | Creates a safety-focused training pipeline. |
| `SimPOAlignment(FineTuningData<,,>,FineTuningData<,,>)` | Creates a SimPO alignment pipeline (reference-free, simple). |
| `StandardAlignment(FineTuningData<,,>,FineTuningData<,,>)` | Creates a standard SFT → DPO pipeline (most common alignment workflow). |
| `Validate` | Validates the pipeline configuration. |
| `ValidateOrThrow` | Throws an exception if the pipeline is invalid. |

