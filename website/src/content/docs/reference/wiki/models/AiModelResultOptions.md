---
title: "AiModelResultOptions<T, TInput, TOutput>"
description: "Represents the configuration options for creating a AiModelResult."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Represents the configuration options for creating a AiModelResult.

## For Beginners

This class is like a settings container for creating a trained model.

Instead of writing code like this (hard to read):

You can write this (easy to read):

Benefits:

- You only set the options you need (everything else has sensible defaults)
- Named properties make it clear what each setting does
- Easy to see all available settings via IntelliSense
- Adding new options doesn't break existing code

## How It Works

This class consolidates all the configuration parameters needed to construct a AiModelResult
into a single, organized object. Instead of passing 20+ parameters to constructors, this options
class groups related settings together for better readability and maintainability.

The options are organized into logical categories:

- Core Model: The trained model and its optimization/normalization data
- Ethical AI: Bias detection and fairness evaluation components
- RAG: Retrieval-Augmented Generation components for document retrieval
- Graph RAG: Knowledge graph components for enhanced retrieval
- Prompt Engineering: Templates, chains, optimizers, and analysis tools
- Fine-tuning: LoRA and meta-learning configurations
- Agent & Reasoning: AI agent and advanced reasoning configurations
- Deployment: Export, caching, versioning, and telemetry settings
- Inference: JIT compilation and optimization configurations
- Tokenization: Text tokenizer and configuration

## Properties

| Property | Summary |
|:-----|:--------|
| `AllowNondeterminism` | Determinism policy. |
| `AugmentationConfig` | Gets or sets the unified augmentation configuration. |
| `AutoMLSummary` | Gets or sets an optional AutoML run summary for this trained model. |
| `BenchmarkReport` | Gets or sets the most recent benchmark report produced during model build/evaluation. |
| `BiasDetector` | Gets or sets the bias detector for identifying potential biases in model predictions. |
| `CheckpointManager` | Gets or sets the checkpoint manager for model persistence operations. |
| `CheckpointPath` | Gets or sets the checkpoint path where the model was saved during training. |
| `CrossValidationResult` | Gets or sets the results from cross-validation. |
| `DataVersionHash` | Gets or sets the data version hash for the training data. |
| `DeploymentConfiguration` | Gets or sets the deployment configuration for model export and production use. |
| `ExperimentId` | Gets or sets the experiment ID that this run belongs to. |
| `ExperimentRun` | Gets or sets the experiment run associated with this model. |
| `ExperimentRunId` | Gets or sets the experiment run ID from experiment tracking. |
| `ExperimentTracker` | Gets or sets the experiment tracker used during training. |
| `FairnessEvaluator` | Gets or sets the fairness evaluator for computing fairness metrics. |
| `FewShotExampleSelector` | Gets or sets the few-shot example selector for choosing examples to include in prompts. |
| `GraphStore` | Gets or sets the graph store backend for persistent graph storage. |
| `HybridGraphRetriever` | Gets or sets the hybrid retriever combining vector search with graph traversal. |
| `HyperparameterOptimizationResult` | Gets or sets the hyperparameter optimization result. |
| `HyperparameterTrialId` | Gets or sets the hyperparameter optimization trial ID. |
| `Hyperparameters` | Gets or sets the hyperparameters used for training. |
| `InferenceOptimizationConfig` | Gets or sets the inference optimization configuration. |
| `InterpretabilityOptions` | Gets or sets the interpretability options for model explanation methods. |
| `JitCompilationConfig` | JIT compilation configuration applied on every Predict call. |
| `JitCompiledFunction` | Gets or sets the JIT-compiled prediction function for accelerated inference. |
| `KnowledgeDistillationOptions` | Gets or sets the knowledge-distillation options configured via `KnowledgeDistillationOptions{`. |
| `KnowledgeGraph` | Gets or sets the knowledge graph for entity-relationship-based retrieval. |
| `LoRAConfiguration` | Gets or sets the LoRA configuration for parameter-efficient fine-tuning. |
| `MemoryConfig` | Gets or sets the memory management configuration for training. |
| `MetaLearner` | Gets or sets the meta-learner for few-shot adaptation capabilities. |
| `MetaTrainingResult` | Gets or sets the results from meta-learning training. |
| `Model` | Gets or sets the trained model used for making predictions. |
| `ModelRegistry` | Gets or sets the model registry for version and lifecycle management. |
| `ModelVersion` | Gets or sets the model version from the model registry. |
| `OptimizationResult` | Gets or sets the results of the optimization process that created the model. |
| `PostprocessingFitSample` | Optional sample of model-output predictions (NOT training targets) handed in alongside an unfitted `PostprocessingPipeline` so the `AiModelResult` ctor can lazy-fit the pipeline at construction time instead of throwing. |
| `PostprocessingPipeline` | Gets or sets the postprocessing pipeline configured via `PostprocessingPipeline{`. |
| `PreprocessingInfo` | Gets or sets the preprocessing pipeline information for data transformation. |
| `ProfilingReport` | Gets or sets the profiling report from training and inference operations. |
| `ProgramSynthesisModel` | Gets or sets the Program Synthesis model used for code tasks (optional). |
| `ProgramSynthesisServingClient` | Gets or sets the Program Synthesis Serving client (optional). |
| `ProgramSynthesisServingClientOptions` | Gets or sets the options used to create a default `ProgramSynthesisServingClient` when no explicit client is provided. |
| `PromptAnalyzer` | Gets or sets the prompt analyzer for computing prompt metrics and validation. |
| `PromptCompressor` | Gets or sets the prompt compressor for reducing token counts. |
| `PromptOptimizer` | Gets or sets the prompt optimizer for automatically improving prompts. |
| `PromptTemplate` | Gets or sets the prompt template for formatting model inputs. |
| `QuantizationInfo` | Gets or sets information about model quantization. |
| `QueryProcessors` | Gets or sets the query processors for preprocessing search queries. |
| `RagGenerator` | Gets or sets the generator for creating answers from retrieved context. |
| `RagReranker` | Gets or sets the reranker for improving document relevance ordering. |
| `RagRetriever` | Gets or sets the retriever for finding relevant documents during inference. |
| `ReasoningConfig` | Gets or sets the reasoning configuration for advanced reasoning capabilities. |
| `RegisteredModelName` | Gets or sets the registered model name in the model registry. |
| `SafetyFilterConfiguration` | Gets or sets the safety filter configuration used to validate inputs and filter outputs during inference. |
| `TextVectorizer` | Gets or sets the fitted text vectorizer used to turn raw text into model features. |
| `TokenizationConfig` | Gets or sets the tokenization configuration. |
| `Tokenizer` | Gets or sets the tokenizer for text encoding and decoding. |
| `TrainingMetricsHistory` | Gets or sets the training metrics history. |
| `TrainingMonitor` | Gets or sets the training monitor for accessing training diagnostics. |
| `WeightStreamingReport` | Weight-streaming activity report from the underlying `WeightRegistry` if streaming engaged during the build (issue #1222 task #186). |

