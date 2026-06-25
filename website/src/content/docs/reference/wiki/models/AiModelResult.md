---
title: "AiModelResult<T, TInput, TOutput>"
description: "Partial class containing Test-Time Augmentation (TTA) prediction methods."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Partial class containing Test-Time Augmentation (TTA) prediction methods.

## For Beginners

This class represents a complete, ready-to-use predictive model.

When working with machine learning models:

- You need to store not just the model itself, but also how to prepare data for it
- You want to keep track of how the model was created and how well it performs
- You need to be able to save the model and load it later

This class handles all of that by storing:

- The actual model that makes predictions
- Information about how the model was optimized
- How to normalize/scale input data before making predictions
- Metadata about the model (like feature names, creation date, etc.)

It also provides methods to:

- Make predictions on new data
- Save the model to a file
- Load a model from a file

This makes it easy to train a model once and then use it many times in different applications.

## How It Works

This class encapsulates a trained predictive model along with all the information needed to use it for making 
predictions on new data. It includes the model itself, the results of the optimization process that created the 
model, normalization information for preprocessing input data and postprocessing predictions, and metadata about 
the model. The class also provides methods for serializing and deserializing the model, allowing it to be saved 
to and loaded from files.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AiModelResult` | Initializes a new instance of the AiModelResult class with default values. |
| `AiModelResult(AiModelResultOptions<,,>)` | Initializes a new instance of the AiModelResult class using an options object for clean configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccelerationSnapshot` | Snapshot of the SIMD, GPU, and native-BLAS acceleration state captured when this model was built. |
| `AllowNondeterminism` | Builder's determinism policy, re-asserted on every Predict. |
| `AutoMLSummary` | Gets the AutoML summary for this model, if AutoML was used during building. |
| `BenchmarkReport` | Gets the most recent benchmark report produced for this model, if available. |
| `BiasDetector` | Gets or sets the bias detector used for ethical AI evaluation. |
| `CausalDiscoveryResult` | Gets the causal discovery result, if causal discovery was configured. |
| `CheckpointManager` | Gets or sets the checkpoint manager for model persistence operations. |
| `CheckpointPath` | Gets or sets the checkpoint path where the model was saved during training. |
| `CrossValidationResult` | Gets or sets the results from cross-validation. |
| `DataVersionHash` | Gets or sets the data version hash for the training data. |
| `DefaultLossFunction` | Gets the default loss function used by this model for gradient computation. |
| `DeploymentConfiguration` | Gets the deployment configuration for model export, caching, versioning, A/B testing, and telemetry. |
| `EnsureModel` | Gets the model, throwing if it has not been set. |
| `Evaluation` | Gets the trained model's evaluation metrics, computed once from the data the model was built on and cached. |
| `ExperimentId` | Gets or sets the experiment ID that this run belongs to. |
| `ExperimentRun` | Gets or sets the experiment run associated with this model. |
| `ExperimentRunId` | Gets or sets the experiment run ID from experiment tracking. |
| `ExperimentTracker` | Gets or sets the experiment tracker used during training. |
| `FairnessEvaluator` | Gets or sets the fairness evaluator used for ethical AI evaluation. |
| `FewShotExampleSelector` | Gets or sets the few-shot example selector for dynamic example selection. |
| `GraphStore` | Gets or sets the graph store backend for persistent graph storage. |
| `HasFewShotExampleSelector` | Checks whether a few-shot example selector is configured and available for use. |
| `HasPromptAnalyzer` | Checks whether a prompt analyzer is configured and available for use. |
| `HasPromptCompressor` | Checks whether a prompt compressor is configured and available for use. |
| `HasPromptOptimizer` | Checks whether a prompt optimizer is configured and available for use. |
| `HasPromptTemplate` | Checks whether a prompt template is configured and available for use. |
| `HasTokenizer` | Gets whether a tokenizer is configured for this model. |
| `HybridGraphRetriever` | Gets or sets the hybrid graph retriever for combined vector + graph retrieval. |
| `HyperparameterOptimizationResult` | Gets or sets the hyperparameter optimization result. |
| `HyperparameterTrialId` | Gets or sets the hyperparameter optimization trial ID. |
| `Hyperparameters` | Gets or sets the hyperparameters used for training. |
| `InterpretabilityOptions` | Gets or sets the interpretability options for model explanation methods. |
| `JitCompilationConfig` | JIT compilation config carried from the builder. |
| `JitCompiledFunction` | Gets the JIT-compiled prediction function for accelerated inference. |
| `KnowledgeDistillationOptions` | Knowledge-distillation options configured via `KnowledgeDistillationOptions{`. |
| `KnowledgeGraph` | Gets or sets the knowledge graph for graph-enhanced retrieval. |
| `KnowledgeGraphResult` | Gets the knowledge graph processing results including trained embeddings, community structure, and link prediction evaluation. |
| `LayerCategorySummary` |  |
| `LayerCount` | Gets the total number of layers in the model, if the model supports layer-level access. |
| `LoRAConfiguration` | Gets or sets the LoRA configuration for parameter-efficient fine-tuning. |
| `MetaLearner` | Gets or sets the meta-learner used for few-shot adaptation and fine-tuning. |
| `MetaTrainingResult` | Gets or sets the results from meta-training. |
| `Model` | Gets or sets the underlying model used for making predictions. |
| `ModelMetaData` | Gets or sets the metadata associated with the model. |
| `ModelRegistry` | Gets or sets the model registry for version and lifecycle management. |
| `ModelVersion` | Gets or sets the model version from the model registry. |
| `OptimizationResult` | Gets or sets the results of the optimization process that created the model. |
| `Options` | Gets the options used to create this model result. |
| `ParameterCount` | Gets the number of parameters in the underlying model, or 0 when the model has no trainable parameter vector (e.g. |
| `PostprocessingPipeline` | Postprocessing pipeline configured via `PostprocessingPipeline{`. |
| `PreprocessingInfo` | Gets or sets the preprocessing information for data transformation. |
| `ProfilingReport` | Gets the profiling report captured during training and/or inference, if profiling was enabled. |
| `PromptAnalyzer` | Gets or sets the prompt analyzer for measuring prompt metrics. |
| `PromptCompressor` | Gets or sets the prompt compressor for reducing prompt length. |
| `PromptOptimizer` | Gets or sets the prompt optimizer used for automatic prompt improvement. |
| `PromptTemplate` | Gets or sets the prompt template used for generating prompts during inference. |
| `QuantizationInfo` | Gets information about model quantization, including strategy, bit width, and compression statistics. |
| `QueryProcessors` | Gets or sets the query processors used for RAG query preprocessing during inference. |
| `RagGenerator` | Gets or sets the generator used for RAG answer generation during inference. |
| `RagReranker` | Gets or sets the reranker used for RAG document reranking during inference. |
| `RagRetriever` | Gets or sets the retriever used for RAG document retrieval during inference. |
| `ReasoningConfig` | Gets the reasoning configuration for advanced Chain-of-Thought, Tree-of-Thoughts, and Self-Consistency reasoning. |
| `RegisteredModelName` | Gets or sets the registered model name in the model registry. |
| `SafetyPipeline` | Gets the composable safety pipeline for content safety evaluation. |
| `SerializedModelData` | Gets the serialized model payload for the facade-hidden `Model`. |
| `SupportsParameterInitialization` |  |
| `TensorsOperationProfile` | Per-tensor-op performance profile captured when the builder opted in via `EnableTensorsOpProfiling()`. |
| `TextVectorizer` | The fitted text vectorizer, when the model was trained on text via `ConfigureTextVectorizer(...)`. |
| `TokenizationConfig` | Gets or sets the tokenization configuration. |
| `Tokenizer` | Gets or sets the tokenizer used for text processing. |
| `TotalEstimatedFlops` | Gets the total estimated FLOPs (floating-point operations) for a single forward pass. |
| `TotalTrainableParameters` | Gets the total number of trainable parameters across all layers. |
| `TrainingMetricsHistory` | Gets or sets the training metrics history. |
| `TrainingMonitor` | Gets or sets the training monitor for accessing training diagnostics. |
| `WeightStreamingReport` | Gets the weight-streaming activity report from the underlying `WeightRegistry` if streaming was engaged during the build (whether explicitly via `WeightStreamingConfig)` or auto-detected from parameter count). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(,,Int32,Double)` | Quickly adapts the model to a new task using a few examples (few-shot learning). |
| `AggregatePredictions(List<>,PredictionAggregationMethod,Func<,Vector<>>,Nullable<Double>)` | Aggregates multiple predictions into a single prediction based on the specified method. |
| `AiDotNet#Interfaces#IModelSerializer#LoadModel(String)` | Explicit implementation of IModelSerializer.LoadModel to avoid confusion with static LoadModel method. |
| `AnalyzePrompt(String)` | Analyzes a prompt and returns detailed metrics about its structure and characteristics. |
| `ApplyGradients(Vector<>,)` | Applies pre-computed gradients to update the model parameters. |
| `AttachGraphComponents(KnowledgeGraph<>,IGraphStore<>,HybridGraphRetriever<>)` | Attaches Graph RAG components to a AiModelResult instance. |
| `AttachPromptEngineering(IPromptTemplate,IPromptOptimizer<>,IFewShotExampleSelector<>,IPromptAnalyzer,IPromptCompressor)` | Attaches prompt engineering components to this result. |
| `AttachTokenizer(ITokenizer,TokenizationConfig)` | Attaches tokenization components to the model result. |
| `BeginInferenceSession` | Begins an inference session for stateful inference features (e.g., KV-cache). |
| `CalculateConfidence(List<Vector<>>,PredictionAggregationMethod)` | Calculates confidence based on prediction agreement. |
| `CalculateDataSetStatsInternal(,,PredictionType)` | Internal method to calculate dataset statistics. |
| `CalculateStandardDeviation(List<Vector<>>)` | Calculates the standard deviation across predictions. |
| `Clone` | Creates a shallow copy of this AiModelResult. |
| `CompressPrompt(String,CompressionOptions)` | Compresses a prompt to reduce its token count while preserving essential meaning. |
| `ComputeGeometricMean(List<Vector<>>)` | Computes the geometric mean of vectors using log-sum-exp for numerical stability. |
| `ComputeGradients(,,ILossFunction<>)` | Computes gradients of the loss function with respect to model parameters WITHOUT updating parameters. |
| `ComputeMax(List<Vector<>>)` | Computes the element-wise maximum of vectors. |
| `ComputeMean(List<Vector<>>)` | Computes the element-wise mean of vectors. |
| `ComputeMedian(List<Vector<>>)` | Computes the element-wise median of vectors. |
| `ComputeMin(List<Vector<>>)` | Computes the element-wise minimum of vectors. |
| `ComputeR2Score(Vector<>,Vector<>)` | Computes R² score for regression evaluation. |
| `ComputeVote(List<Vector<>>)` | Computes majority voting for classification predictions. |
| `ComputeWeightedMean(List<Vector<>>)` | Computes weighted mean based on confidence scores. |
| `ConvertOutputToScalar()` | Converts the model's TOutput to a scalar value. |
| `ConvertOutputToVector()` | Converts model output to a vector for aggregation. |
| `ConvertToTensorSafe(,String)` | Safely converts a potentially nullable output value to a Tensor, throwing if the value is null. |
| `ConvertVectorToInput(Vector<>)` | Converts a Vector to the model's TInput type. |
| `ConvertVectorToOutput(Vector<>)` | Converts an aggregated vector back to the output type. |
| `CreateActivationFunction` | Creates a function that returns both output and layer activations for DeepLIFT. |
| `CreateClassProbabilityFunction` | Creates a function that returns class probabilities. |
| `CreateClassificationFunction` | Creates a function that returns the predicted class index. |
| `CreateDeploymentRuntime(String,String,String)` | Creates a deployment runtime for production features like versioning, A/B testing, caching, and telemetry. |
| `CreateNetworkInfoFunction` | Creates a function that returns output, activations, and weights for LRP. |
| `CreatePredictionFunction` | Creates a prediction function that takes a Matrix and returns a Vector of predictions. |
| `CreateScalarPredictionFunction` | Creates a scalar prediction function that takes a Vector and returns a scalar value. |
| `CreateTensorPredictionFunction` | Creates a prediction function for tensor input/output (used by CNN explainers). |
| `CreateVectorPredictionFunction` | Creates a vector prediction function (Vector in, Vector out). |
| `DeepCopy` | Creates a copy of this AiModelResult with deep-copied core model components. |
| `DeepReasonAsync(String,IChatClient<>,CancellationToken)` | Performs deep, thorough reasoning on a complex problem using extensive exploration and verification. |
| `DenormalizeVarianceIfSupported(Tensor<>)` | Attempts to denormalize variance based on the target preprocessing pipeline. |
| `Deserialize(Byte[])` | Deserializes a model from a byte array. |
| `DetectHallucinations(String)` | Detects potential hallucinations in the given output text. |
| `DetectPII(String)` | Detects personally identifiable information (PII) in the given text. |
| `DetectWatermark(String)` | Detects watermarks in the given text content. |
| `Detokenize(List<Int32>,Boolean)` | Decodes token IDs back into text. |
| `DispatchModelInference()` | Single-source-of-truth inference dispatch for `Predict(`. |
| `Dispose` |  |
| `Dispose(Boolean)` | Disposes the contained model. |
| `EnforceSafetyPolicy(SafetyReport,Boolean)` | Enforces the safety policy on a report, throwing if the content is blocked. |
| `EvaluateAudioSafety(Vector<>,Int32)` | Evaluates audio content for safety using the configured safety pipeline. |
| `EvaluateBenchmarkAsync(IBenchmark<>,IChatClient<>,Nullable<Int32>,CancellationToken)` | Evaluates a reasoning benchmark using the configured facade (prompt chain or agent reasoning). |
| `EvaluateBenchmarksAsync(BenchmarkingOptions,CancellationToken)` | Runs benchmark suites against this model using the unified benchmark runner. |
| `EvaluateBenchmarksAsync(BenchmarkingOptions,IChatClient<>,CancellationToken)` | Runs benchmark suites against this model using the unified benchmark runner, with an optional chat client for reasoning / generative benchmark suites. |
| `EvaluateBenchmarksAsync(IChatClient<>,CancellationToken)` | Forwarding overload for callers that supply only a chat client (reasoning / generative suites with default options). |
| `EvaluateFairness(String)` | Evaluates text for bias and fairness issues. |
| `EvaluateFull(OptimizationInputData<,,>,Nullable<PredictionType>)` | Evaluates the model across training, validation, and test datasets. |
| `EvaluateHumanEvalPassAtKAsync(Int32,Nullable<Int32>,CancellationToken)` | Evaluates the model on HumanEval using the configured dataset path (via env var) and returns a benchmark report. |
| `EvaluateImageSafety(Tensor<>)` | Evaluates image content for safety using the configured safety pipeline. |
| `EvaluateProgramIoAsync(ProgramEvaluateIoRequest,CancellationToken)` | Evaluates a program against input/output test cases via AiDotNet.Serving. |
| `EvaluateRobustness([],[],Double)` | Evaluates the model's robustness using the default attack configuration. |
| `EvaluateRobustness([],[],IAdversarialAttack<,,>)` | Evaluates the model's robustness against a specific adversarial attack. |
| `EvaluateRobustnessWithAutoAttack([],[],Double)` | Evaluates the model's robustness using AutoAttack (ensemble of diverse attacks). |
| `EvaluateTextSafety(String)` | Evaluates text content for safety using the configured safety pipeline. |
| `EvaluateVideoSafety(IReadOnlyList<Tensor<>>,Double)` | Evaluates video content for safety using the configured safety pipeline. |
| `ExecuteCodeTask(CodeTaskRequestBase)` | Executes a structured code task using the configured program synthesis model. |
| `ExecuteCodeTaskAsync(CodeTaskRequestBase,CancellationToken)` | Executes a structured code task, optionally delegating to AiDotNet.Serving when configured and preferred. |
| `ExecuteProgramAsync(ProgramExecuteRequest,CancellationToken)` | Executes a sandboxed program via AiDotNet.Serving. |
| `ExecuteSqlAsync(SqlExecuteRequest,CancellationToken)` | Executes SQL via AiDotNet.Serving. |
| `ExplainGlobalWithSHAP(Matrix<>,Matrix<>)` | Explains multiple predictions using SHAP values and returns global feature importance. |
| `ExplainWithAttentionVisualization(Vector<>,Int32,Int32,Int32,String[])` | Visualizes attention patterns for transformer-based models. |
| `ExplainWithDeepLIFT(Vector<>,Vector<>,DeepLIFTRule)` | Explains a prediction using DeepLIFT attribution. |
| `ExplainWithGradCAM(Vector<>,Int32[],Int32[],Boolean)` | Generates a Grad-CAM heatmap for CNN model predictions. |
| `ExplainWithGradientSHAP(Vector<>,Matrix<>,Int32,Int32)` | Explains a prediction using GradientSHAP (combines Integrated Gradients with SHAP). |
| `ExplainWithIntegratedGradients(Vector<>,Vector<>,Int32)` | Explains a prediction using Integrated Gradients attribution. |
| `ExplainWithLIME(Vector<>)` | Explains a single prediction using LIME (Local Interpretable Model-agnostic Explanations). |
| `ExplainWithLRP(Vector<>,LRPRule,Double)` | Explains a prediction using Layer-wise Relevance Propagation (LRP). |
| `ExplainWithPrototypes(Vector<>,Matrix<>,Vector<>,Int32,DistanceMetric)` | Explains a prediction using similar examples from a prototype set. |
| `ExplainWithSHAP(Vector<>,Matrix<>)` | Explains a single prediction using SHAP (SHapley Additive exPlanations) values. |
| `ExportToCoreML(String)` | Exports the model to CoreML format for deployment on Apple devices (iOS, macOS). |
| `ExportToOnnx(String)` | Exports the model to ONNX format for cross-platform deployment. |
| `ExportToTFLite(String)` | Exports the model to TensorFlow Lite format for mobile and edge deployment. |
| `ExportToTensorRT(String)` | Exports the model to TensorRT format for high-performance inference on NVIDIA GPUs. |
| `FindPathInGraph(String,String)` | Finds the shortest path between two nodes in the knowledge graph. |
| `FineTune(,,Int32,,,Double)` | Performs comprehensive fine-tuning on a dataset to optimize for a specific task. |
| `ForecastWithTimeSeriesModelInternal(TimeSeriesModelBase<>,)` | Makes predictions using the model on the provided input data. |
| `FormatPrompt(Dictionary<String,String>)` | Formats a prompt using the configured prompt template. |
| `GenerateAdversarialExample(,,Double)` | Generates an adversarial example for a given input. |
| `GenerateAnswer(String,Nullable<Int32>,Nullable<Int32>,Dictionary<String,Object>)` | Generates a grounded answer using the configured RAG pipeline during inference. |
| `GetAccumulatedLocalEffects(Matrix<>,Int32)` | Computes Accumulated Local Effects (ALE) for all features in the dataset. |
| `GetActiveFeatureIndices` | Gets the indices of features that are actively used by the underlying model. |
| `GetCheckpointManager` | Gets the checkpoint manager for model persistence operations. |
| `GetContrastiveExplanation(Vector<>,Int32,Int32)` | Generates a contrastive explanation answering "Why X and not Y?" |
| `GetDataSetStats(,,Nullable<PredictionType>)` | Gets evaluation statistics for a single dataset. |
| `GetExperimentInfo` | Gets experiment tracking information as a structured object. |
| `GetExperimentRun` | Gets the experiment run associated with this model, if experiment tracking was configured. |
| `GetExperimentTracker` | Gets the experiment tracker used during training, if configured. |
| `GetFeatureALE(Matrix<>,Int32,Int32)` | Computes Accumulated Local Effects for a specific feature. |
| `GetFeatureImportance` | Gets the feature importance scores from the underlying model. |
| `GetFeatureInteractions(Matrix<>,Int32)` | Detects and quantifies feature interactions using Friedman's H-statistic. |
| `GetFederatedLearningMetadata` | Gets federated learning training metadata if this model was produced via federated learning. |
| `GetGlobalSurrogateExplanation(Matrix<>)` | Trains a global surrogate model to approximate the complex model's behavior. |
| `GetHyperparameterOptimizationResult` | Gets the hyperparameter optimization result, if optimization was used. |
| `GetHyperparameters` | Gets the hyperparameters used for training. |
| `GetModelMetadata` | Gets the metadata associated with the model. |
| `GetModelRegistry` | Gets the model registry for version and lifecycle management. |
| `GetModelRegistryInfo` | Gets model registry information as a structured object. |
| `GetNodeRelationships(String,EdgeDirection)` | Gets all edges (relationships) connected to a node in the knowledge graph. |
| `GetParameters` | Gets the parameters of the underlying model. |
| `GetPermutationFeatureImportance(Matrix<>,Vector<>,Func<Vector<>,Vector<>,>)` | Computes permutation feature importance for the model. |
| `GetSafetyConfig` | Gets the overall safety configuration report for the current pipeline. |
| `GetSafetyReport` | Gets a full safety report by evaluating all configured safety modules on the last prediction output. |
| `GetSaliencyMap(Vector<>,SaliencyMethod,Int32,Double)` | Generates a saliency map showing input gradients. |
| `GetTrainingInfrastructureMetadata` | Gets training infrastructure metadata as a dictionary. |
| `GetTrainingMetricsHistory` | Gets the training metrics history. |
| `GetTrainingMonitor` | Gets the training monitor for accessing training diagnostics. |
| `HybridRetrieve(Vector<>,Int32,Int32,Int32)` | Retrieves results using hybrid vector + graph search for enhanced context retrieval. |
| `IsCleanForDefaultVectorInputFilter(Vector<>,SafetyFilter<>)` | Returns true when the default numeric `SafetyFilter` would pass `input` through unchanged — i.e. |
| `IsFeatureUsed(Int32)` | Checks if a specific feature is used by the underlying model. |
| `IsSafeOutput(String)` | Quickly checks whether the given text output is safe according to the configured safety pipeline. |
| `LoadFromFile(String)` | Loads the model from a file. |
| `LoadState(Stream)` | Loads the prediction model result's state from a stream. |
| `OptimizePrompt(String,Func<String,>,Int32)` | Optimizes a prompt to improve its effectiveness using an evaluation function. |
| `OptimizePromptAsync(String,Func<String,Task<>>,Int32,CancellationToken)` | Optimizes a prompt asynchronously using an async evaluation function. |
| `PredictText(IEnumerable<String>)` | Predicts directly from raw text using the fitted text vectorizer captured during training. |
| `PredictWithDefense()` | Makes a prediction with adversarial preprocessing applied if configured. |
| `PredictWithTestTimeAugmentation(,Func<,>,Func<,Vector<>>,IAugmentationPolicy<,>)` | Makes a prediction using Test-Time Augmentation for improved accuracy. |
| `PredictWithTestTimeAugmentation(,IAugmentationPolicy<,>)` | Makes a prediction using Test-Time Augmentation when input/output types match augmentation types. |
| `ProcessQueryWithProcessors(String)` | Processes a query through all configured query processors in sequence. |
| `QueryKnowledgeGraph(String,Int32)` | Queries the knowledge graph to find related nodes by entity name or label. |
| `QuickReasonAsync(String,IChatClient<>,CancellationToken)` | Quickly solves a problem with minimal reasoning overhead for fast answers. |
| `ReasonAsync(String,IChatClient<>,ReasoningMode,ReasoningConfig,CancellationToken)` | Solves a problem using advanced reasoning strategies like Chain-of-Thought, Tree-of-Thoughts, or Self-Consistency. |
| `ReasonWithConsensusAsync(String,IChatClient<>,Int32,CancellationToken)` | Solves a problem multiple times using different approaches and returns the consensus answer. |
| `RetrieveDocuments(String,Nullable<Int32>,Boolean,Dictionary<String,Object>)` | Retrieves relevant documents without generating an answer during inference. |
| `RunSafetyBenchmarks` | Runs the comprehensive safety benchmark suite against the configured safety pipeline. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveModel(String)` | Saves the model to a file. |
| `SaveState(Stream)` | Saves the prediction model result's current state to a stream. |
| `SelectFewShotExamples(String,Int32)` | Selects relevant few-shot examples for a given query or context. |
| `Serialize` | Serializes the model to a byte array. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Setting active feature indices is not supported on AiModelResult. |
| `SetCausalDiscoveryResult(CausalDiscoveryResult<>)` | Sets the causal discovery result. |
| `SetParameters(Vector<>)` | Setting parameters is not supported on AiModelResult. |
| `ToDataSetStatsInternal(OptimizationResult<,,>.DatasetResult)` | Maps an optimization dataset result (already computed during the build) onto the richer `DataSetStats` surface exposed through `Evaluation`. |
| `Tokenize(String)` | Tokenizes text using the configured tokenizer. |
| `TokenizeBatch(List<String>)` | Tokenizes multiple texts in a batch. |
| `TokenizeCode(String,ProgramLanguage,EncodingOptions,CodeTokenizationPipelineOptions)` | Tokenizes code using the canonical code-tokenization pipeline (supports AST extraction when enabled). |
| `Train(,)` | Training is not supported on AiModelResult. |
| `TraverseGraph(String,Int32)` | Traverses the knowledge graph starting from a node using breadth-first search. |
| `TryCreateActivationsFunction` | Creates a function to get layer activations for DeepLIFT. |
| `TryCreateAttentionExtractor` | Creates a function that extracts attention matrices from transformer layers. |
| `TryCreateAttentionWeightsFunction` | Creates a function to get attention weights at a specific layer. |
| `TryCreateFeatureGradientFunction` | Attempts to create a function that extracts feature maps and their gradients from a CNN. |
| `TryCreateFeatureMapFunction` | Creates a function to get feature maps at a specific layer for a given class. |
| `TryCreateLayerActivationsFunction` | Creates a function to get all layer activations for LRP. |
| `TryCreateLayerWeightsFunction` | Creates a function to get layer weights for LRP. |
| `TryCreateMultipliersFunction` | Creates a function to compute DeepLIFT multipliers. |
| `TryCreateTensorGradientFunction` | Creates a function to get gradients for tensor input at specific output and layer. |
| `TryPopulateClusteringMetricsInternal(ModelEvaluationData<,,>)` | Computes internal clustering quality metrics for unsupervised clustering models, so that `Evaluation` reports cluster-appropriate scores instead of (meaningless) supervised error stats. |
| `ValidateInputSafety(String)` | Validates input content for safety before passing it to the model. |
| `ValidateOutputSafety(String)` | Validates output content for safety after model prediction. |
| `ValidatePrompt(String,ValidationOptions)` | Validates a prompt and returns any detected issues or warnings. |
| `WithParameters(Vector<>)` | Creates a new instance with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_featureSelectionState` | Applies feature selection to prediction input if the optimizer selected a subset of features. |
| `_layerCategorySummary` | Gets a summary of layer categories in this model, showing how many layers belong to each category. |

