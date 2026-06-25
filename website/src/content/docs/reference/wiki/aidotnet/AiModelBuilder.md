---
title: "AiModelBuilder<T, TInput, TOutput>"
description: "Build-pipeline orchestration partial of `AiModelBuilder`: the streaming and standard supervised build/optimize paths."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet`

Build-pipeline orchestration partial of `AiModelBuilder`:
the streaming and standard supervised build/optimize paths. Split out of the main file
(audit-2026-05 finding #12) to keep AiModelBuilder.cs reviewable; no behaviour change.

## For Beginners

Think of this class as a recipe builder for creating AI models.
You add different ingredients (like data normalization, feature selection, etc.)
and then "cook" (build) the final model. This approach makes it easy to customize
your model without having to understand all the complex details at once.

## How It Works

This class uses the builder pattern to configure various components of a machine learning model
before building and using it for predictions.

**Training Infrastructure Example:** Complete example showing experiment tracking,
checkpointing, model registry, and hyperparameter optimization working together:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AiModelBuilder(AiDotNetLicenseKey)` | Creates a new `AiModelBuilder` with default (empty) configuration. |
| `AiModelBuilder(String,AiDotNetLicenseKey)` | Creates a new `AiModelBuilder` with configuration loaded from a YAML file. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfiguredAdversarialRobustness` | Internal accessor exposing the most recently configured `AdversarialRobustnessConfiguration` so unit tests can verify `AdversarialRobustnessConfiguration{` retained the user-supplied (or default) instance. |
| `ConfiguredModel` | Gets the configured model instance for use by domain-specific partial class methods. |
| `IsOnnxExportableRequired` | True if the user opted into ONNX-export validation on this builder. |
| `SegmentationVisualization` | Gets the configured segmentation visualization settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateStreamingInputs(List<>)` | Aggregates a list of input samples into a single TInput structure. |
| `AggregateStreamingOutputs(List<>)` | Aggregates a list of output samples into a single TOutput structure. |
| `AllowNondeterminism` | Opts out of the builder's deterministic-by-default policy. |
| `ApplyAdvancedGpuExecutionOptions` | Applies advanced GPU execution options from the configuration. |
| `ApplyGradientCheckpointingFromMemoryConfig` | Pushes the configured gradient-checkpointing segment size onto the current model. |
| `ApplyMemoryConfiguration` | Applies memory management configuration to models that support it. |
| `ApplyQuantizationIfConfigured(IFullModel<,,>,QuantizationConfig,IEnumerable<>)` | Applies quantization to the model if configured. |
| `ApplyTemperatureScalingToProbabilities(Tensor<>,,Int32,Int32)` | Applies temperature scaling to probabilities and renormalizes to maintain a valid distribution. |
| `ApplyTrialHyperparameters(Object,Dictionary<String,Object>)` | Applies trial hyperparameters from HPO to the optimizer options. |
| `ApplyWeightStreamingConfig` | Applies `_weightStreamingConfig` to the constructed neural-network model. |
| `ApplyWeightStreamingConfigTo(NeuralNetworkBase<>)` | Applies the builder's `_weightStreamingConfig` to a specific neural-network instance. |
| `AttachAdversarialRobustness(AiModelResult<,,>)` | Threads any `_adversarialRobustnessConfiguration` set via `AdversarialRobustnessConfiguration{` into the constructed `AiModelResult` so the runtime adversarial-robustness API (`PredictWithDefense`, `EvaluateRobustness`) actually picks up th… |
| `AutoSelectCausalAlgorithm(Int32)` | Auto-selects a causal discovery algorithm based on data characteristics. |
| `BuildAsync` | Builds a predictive model using data from ConfigureDataLoader() or meta-learning from ConfigureMetaLearning(). |
| `BuildCompiledPredictFunction(IFullModel<,,>)` | Wraps a trained model's `Predict` in a `CompiledModelCache`-backed function matching the `JitCompiledFunction` shape expected by `AiModelResult`. |
| `BuildMetaLearningInternalAsync` | Internal method that performs meta-learning training. |
| `BuildRLInternalAsync(Int32,Boolean)` | Internal method that performs reinforcement learning training. |
| `BuildStreamingSupervisedAsync(IStreamingDataLoader<,,>)` | Performs true streaming supervised training without materializing all data in memory. |
| `BuildSupervisedInternalAsync(,,CancellationToken)` | Internal method that performs supervised training with the provided input features and output values. |
| `BuildWeightStreamingReport` | Constructs a `WeightStreamingReport` from the model's streaming state if streaming was engaged (auto-detect or explicit opt-in), otherwise returns null. |
| `ComputeConformalClassificationThreshold(Vector<>,Double)` | Computes the conformal score threshold for classification prediction sets. |
| `ComputeConformalRegressionQuantile(Vector<>,Double)` | Computes the conformal regression residual quantile for a desired confidence level. |
| `ComputeDataVersionHash(Matrix<>,Vector<>,INumericOperations<>)` | Computes a robust hash of the training data for version control and lineage tracking. |
| `ConfigureActivationFunction(IActivationFunction<>)` | Configures the activation function for neural network layers. |
| `ConfigureActiveLearning(IActiveLearningStrategy<>)` | Configures an active learning strategy for intelligently selecting training samples. |
| `ConfigureAdversarialAttack(IAdversarialAttack<,,>)` | Configures an adversarial attack method for evaluating model robustness. |
| `ConfigureAdversarialDefense(IAdversarialDefense<,,>)` | Configures an adversarial defense method for improving model robustness. |
| `ConfigureAdversarialRobustness(AdversarialRobustnessConfiguration<,,>)` | Configures adversarial robustness and AI safety features for the model. |
| `ConfigureAnomalyDetector(IAnomalyDetector<>)` | Configures an anomaly detection algorithm for identifying unusual data points. |
| `ConfigureAudioEffect(IAudioEffect<>)` | Configures an audio effect for audio signal processing pipelines. |
| `ConfigureAudioEnhancer(IAudioEnhancer<>)` | Configures an audio enhancer for improving audio quality. |
| `ConfigureAudioGenerator(IAudioGenerator<>)` | Configures an audio generator for creating audio from various inputs. |
| `ConfigureAugmentation(AugmentationConfig)` | Configures data augmentation for training and inference. |
| `ConfigureAugmentation(AugmentationConfig<,>)` | Strongly-typed overload of `AugmentationConfig)` that accepts the generic `AugmentationConfig` (introduced in review #1368 to replace the `object?`-typed custom-augmenter slot). |
| `ConfigureAutoML(AutoMLOptions<,,>)` | Configures AutoML using facade-style options (recommended for most users). |
| `ConfigureAutoML(IAutoMLModel<,,>)` | Configures an AutoML model for automatic machine learning optimization. |
| `ConfigureBenchmark(IBenchmark<>)` | Configures a benchmark for evaluating and comparing model performance systematically. |
| `ConfigureBenchmarking(BenchmarkingOptions)` | Configures benchmarking to run standardized benchmark suites and attach a structured report to the built model. |
| `ConfigureBiasDetector(IBiasDetector<>)` | Configures the bias detector component for ethical AI evaluation. |
| `ConfigureCausalDiscovery(Action<CausalDiscoveryOptions>)` | Configures causal structure discovery to learn a DAG from the training data. |
| `ConfigureCausalDiscovery(CausalDiscoveryOptions)` | Configures causal structure discovery with a pre-built options object. |
| `ConfigureCausalInference(ICausalModel<>)` | Configures a causal inference model for understanding cause-and-effect relationships. |
| `ConfigureCertifiedDefense(ICertifiedDefense<,,>)` | Configures a certified defense for providing formal robustness guarantees. |
| `ConfigureCheckpointManager(ICheckpointManager<,,>)` | Configures checkpoint management for saving and restoring training state. |
| `ConfigureClassificationMetric(IClassificationMetric<>)` | Configures a classification metric for evaluating classifier performance. |
| `ConfigureClassifier(IClassifier<>)` | Configures a classification algorithm for categorizing data into discrete classes. |
| `ConfigureClusterMetric(IClusterMetric<>)` | Configures an internal cluster metric for evaluating clustering quality without ground truth labels. |
| `ConfigureClustering(IClustering<>)` | Configures a clustering algorithm for grouping similar data points together. |
| `ConfigureContinualLearning(IContinualLearner<,,>)` | Configures a continual learning trainer that can learn new tasks without forgetting old ones. |
| `ConfigureCrossValidation(ICrossValidator<,,>)` | Configures the cross-validation strategy for model evaluation. |
| `ConfigureCurriculumLearning(CurriculumLearningOptions<,,>)` | Configures curriculum learning for training with ordered sample difficulty. |
| `ConfigureCurriculumScheduler(ICurriculumScheduler<>)` | Configures a curriculum scheduler for ordering training samples by difficulty. |
| `ConfigureDataLoader(IDataLoader<>)` | Configures the data loader for providing training data. |
| `ConfigureDataPreparation(Action<DataPreparationPipeline<>>)` | Configures data preparation operations that change row count (outlier removal, augmentation). |
| `ConfigureDataSplitter(IDataSplitter<>)` | Configures a data splitting strategy for dividing datasets into train/test/validation sets. |
| `ConfigureDataTransformer(IDataTransformer<,,>)` | Configures a data transformer for preprocessing or postprocessing data transformations. |
| `ConfigureDataVersionControl(IDataVersionControl<>)` | Configures data version control for tracking dataset changes. |
| `ConfigureDiffusionModel(IDiffusionModel<>)` | Configures a diffusion model for generative tasks (image/audio/video generation). |
| `ConfigureDistanceMetric(IDistanceMetric<>)` | Configures a distance metric for measuring similarity between data points. |
| `ConfigureDistillationStrategy(IDistillationStrategy<>)` | Configures a knowledge distillation strategy for transferring knowledge between models. |
| `ConfigureDistributedTraining(ICommunicationBackend<>,DistributedStrategy,IShardingConfiguration<>)` | Configures distributed training across multiple GPUs or machines. |
| `ConfigureDocumentModel(IDocumentModel<>)` | Configures a document model for document understanding and processing. |
| `ConfigureDocumentStore(IDocumentStore<>)` | Configures a document store for persisting and retrieving documents with vector similarity search. |
| `ConfigureDocumentTransformers(IFullModel<,,>)` | Applies GPU acceleration configuration to the global AiDotNetEngine based on user settings. |
| `ConfigureDriftDetection(IDriftDetector<>)` | Configures a drift detector for monitoring changes in data distribution over time. |
| `ConfigureEmbeddingModel(IEmbeddingModel<>)` | Configures an embedding model for learning dense vector representations. |
| `ConfigureEnvironment(IEnvironment<>)` | Configures a reinforcement learning environment for agent training. |
| `ConfigureExperimentTracker(IExperimentTracker<>)` | Configures experiment tracking for logging and organizing training runs. |
| `ConfigureExplorationStrategy(IExplorationStrategy<>)` | Configures an exploration strategy for reinforcement learning agents. |
| `ConfigureExternalClusterMetric(IExternalClusterMetric<>)` | Configures an external cluster metric for evaluating clustering quality against ground truth labels. |
| `ConfigureFairnessEvaluator(IFairnessEvaluator<>)` | Configures the fairness evaluator component for ethical AI evaluation. |
| `ConfigureFederatedLearning(FederatedLearningOptions,IAggregationStrategy<IFullModel<,,>>,IClientSelectionStrategy,IFederatedServerOptimizer<>,IFederatedHeterogeneityCorrection<>,IHomomorphicEncryptionProvider<>,IPrivateSetIntersection,ISecureComputationProtocol<>,ITeeProvider<>,IZkProofSystem,IFederatedUnlearner<>,IFederatedDriftDetector<>,IClientContributionEvaluator<>,IFairnessConstraint<>)` | Enables federated learning training using the provided options. |
| `ConfigureFinancialModel(IFinancialModel<>)` | Configures a financial model for quantitative finance and risk analysis. |
| `ConfigureFineTuning(FineTuningConfiguration<,,>)` | Configures fine-tuning for the model using preference learning, RLHF, or other alignment methods. |
| `ConfigureFitDetector(IFitDetector<,,>)` | Configures how to detect if the model is overfitting or underfitting. |
| `ConfigureFitnessCalculator(IFitnessCalculator<,,>)` | Configures how to measure the model's performance. |
| `ConfigureGaussianProcess(IGaussianProcess<>)` | Configures a Gaussian process model for probabilistic predictions with uncertainty estimates. |
| `ConfigureGpuDiagnostics(GpuDiagnosticsOptions)` | Controls whether GPU backend diagnostic output is written to `Console` or routed through a custom sink. |
| `ConfigureHyperparameterOptimizer(IHyperparameterOptimizer<,,>,HyperparameterSearchSpace,Int32)` | Configures hyperparameter optimization for automatic tuning of model settings. |
| `ConfigureInferenceOptimizations(InferenceOptimizationConfig)` | Configures inference-time optimizations for faster predictions. |
| `ConfigureInterpolation(IInterpolation<>)` | Configures a 1D interpolation method for estimating values between known data points. |
| `ConfigureInterpolation2D(I2DInterpolation<>)` | Configures a 2D interpolation method for estimating values on a surface between known data points. |
| `ConfigureInterpretability(InterpretabilityOptions)` | Configures model interpretability and explainability features. |
| `ConfigureJitCompilation(JitCompilationConfig)` | Enables JIT (Just-In-Time) compilation for the built model's forward and backward passes. |
| `ConfigureKernelFunction(IKernelFunction<>)` | Configures the kernel function for kernel-based methods (SVM, Gaussian processes, etc.). |
| `ConfigureKnowledgeDistillation(KnowledgeDistillationOptions<,,>)` | Configures knowledge distillation to train a smaller, faster student model from a larger teacher model. |
| `ConfigureKnowledgeGraph(Action<KnowledgeGraphOptions>)` | Configures advanced knowledge graph capabilities including embeddings, community detection, link prediction, temporal queries, and KG construction. |
| `ConfigureLayer(ILayer<>)` | Configures a neural network layer for building custom network architectures. |
| `ConfigureLearningRateScheduler(ILearningRateScheduler)` | Configures a learning rate scheduler that adjusts the learning rate during training. |
| `ConfigureLicenseKey(AiDotNetLicenseKey)` |  |
| `ConfigureLinkFunction(ILinkFunction<>)` | Configures a link function for generalized linear models (GLMs). |
| `ConfigureLoRA(ILoRAConfiguration<>)` | Configures LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. |
| `ConfigureLossFunction(ILossFunction<>)` | Configures the loss function used to measure prediction error during training. |
| `ConfigureMatrixDecomposition(IMatrixDecomposition<>)` | Configures a matrix decomposition method for linear algebra operations. |
| `ConfigureMemoryManagement(TrainingMemoryConfig)` | Configures memory management for training including gradient checkpointing, activation pooling, and model sharding. |
| `ConfigureMetaLearning(IMetaLearner<,,>)` | Configures a meta-learning algorithm for training models that can quickly adapt to new tasks. |
| `ConfigureModel(IFullModel<,,>)` | Configures the core algorithm to use for predictions. |
| `ConfigureModelCompressionStrategy(IModelCompressionStrategy<>)` | Configures a model compression strategy for reducing model size and inference cost. |
| `ConfigureModelExplainer(IModelExplainer<>)` | Configures a model explainer for understanding model predictions. |
| `ConfigureModelOptions(ModelOptions)` | Configures model options that control training behavior and hyperparameters. |
| `ConfigureModelRegistry(IModelRegistry<,,>)` | Configures model registry for centralized model storage and versioning. |
| `ConfigureNoiseScheduler(INoiseScheduler<>)` | Configures a noise scheduler for diffusion model training and sampling. |
| `ConfigureOnlineLearning(IOnlineLearningModel<>)` | Configures an online learning model that updates incrementally with new data. |
| `ConfigurePDESpecification(IPDESpecification<>)` | Configures a PDE specification for physics-informed neural network training. |
| `ConfigurePipelineParallelism(IPipelineSchedule<>,IPipelinePartitionStrategy<>,ActivationCheckpointConfig,Int32)` | Configures pipeline-specific options for pipeline parallel training. |
| `ConfigurePlanCaching(String)` | Enables disk-backed caching of compiled inference plans. |
| `ConfigurePointCloudModel(IPointCloudModel<>)` | Configures a point cloud model for 3D data processing. |
| `ConfigurePostprocessing(Action<PostprocessingPipeline<,,>>)` | Configures the output postprocessing pipeline for the model using a fluent builder. |
| `ConfigurePostprocessing(IDataTransformer<,,>)` | Configures the output postprocessing using a single transformer. |
| `ConfigurePostprocessing(PostprocessingPipeline<,,>)` | Configures the output postprocessing using an existing pipeline. |
| `ConfigurePreprocessing(Action<PreprocessingPipeline<,,>>)` | Configures a preprocessing pipeline using a builder action. |
| `ConfigurePreprocessing(IDataTransformer<,,>)` | Configures a single preprocessing transformer. |
| `ConfigurePreprocessing(PreprocessingPipeline<,,>)` | Configures a pre-built preprocessing pipeline. |
| `ConfigureProfiling(ProfilingConfig)` | Configures performance profiling for training and inference operations. |
| `ConfigureProgramSynthesis(ProgramSynthesisOptions)` | Configures program synthesis (code generation / repair) settings with sensible defaults. |
| `ConfigureProgramSynthesisServing(ProgramSynthesisServingClientOptions,IProgramSynthesisServingClient)` | Configures program synthesis to use `AiDotNet.Serving` for program execution and evaluation (optional). |
| `ConfigureQueryStrategy(IQueryStrategy<,,>)` | Configures a query strategy for active learning sample selection. |
| `ConfigureRLAgent(IRLAgent<>)` | Configures a reinforcement learning agent for learning through interaction with an environment. |
| `ConfigureRadialBasisFunction(IRadialBasisFunction<>)` | Configures a radial basis function for RBF networks and interpolation. |
| `ConfigureReasoning(ReasoningConfig)` | Configures advanced reasoning capabilities for the model using Chain-of-Thought, Tree-of-Thoughts, and Self-Consistency strategies. |
| `ConfigureRegression(IRegression<>)` | Configures a regression algorithm for predicting continuous numeric values. |
| `ConfigureRegressionMetric(IRegressionMetric<>)` | Configures a regression metric for evaluating regression model performance. |
| `ConfigureRegularization(IRegularization<,,>)` | Configures regularization to prevent overfitting in the model. |
| `ConfigureReinforcementLearning(RLTrainingOptions<>)` | Configures reinforcement learning options for training an RL agent. |
| `ConfigureRetrievalAugmentedGeneration(IRetriever<>,IReranker<>,IGenerator<>,IEnumerable<IQueryProcessor>,IGraphStore<>,KnowledgeGraph<>,IDocumentStore<>)` | Configures the retrieval-augmented generation (RAG) components for use during model inference. |
| `ConfigureSSLMethod(ISSLMethod<>)` | Configures a self-supervised learning method for learning representations without labeled data. |
| `ConfigureSafety(Action<SafetyConfig>)` | Configures the comprehensive safety pipeline for input validation and output filtering. |
| `ConfigureScoringRule(IScoringRule<>)` | Configures a scoring rule for evaluating probabilistic predictions. |
| `ConfigureSegmentationVisualization(SegmentationVisualizationConfig)` | Configures visualization settings for segmentation overlays. |
| `ConfigureSelfSupervisedLearning(Action<SSLConfig>)` | Configures self-supervised learning for unsupervised representation learning. |
| `ConfigureSelfSupervisedLearning(Action<SSLConfig>,Func<IFullModel<,,>,SSLConfig,CancellationToken,Task<IFullModel<,,>>>)` | Configures self-supervised learning with a typed pretraining hook (`AiDotNet`#1361). |
| `ConfigureSimilarityMetric(ISimilarityMetric<>)` | Configures a similarity metric for vector similarity search operations. |
| `ConfigureSpeechRecognizer(ISpeechRecognizer<>)` | Configures a speech recognition model for converting spoken audio to text. |
| `ConfigureStoppingCriterion(IStoppingCriterion<>)` | Configures a stopping criterion for active learning loops. |
| `ConfigureSurvivalAnalysis(ISurvivalModel<>)` | Configures a survival analysis model for time-to-event prediction. |
| `ConfigureTargetScaling(PreprocessingPipeline<,,>)` | Configures the optimization algorithm to find the best model parameters. |
| `ConfigureTextToSpeech(ITextToSpeech<>)` | Configures a text-to-speech model for converting written text to spoken audio. |
| `ConfigureTextVectorizer(ITextVectorizer<>)` | Configures a text vectorizer for converting text data into numeric feature vectors. |
| `ConfigureTimeSeriesDecomposition(ITimeSeriesDecomposition<>)` | Configures a time series decomposition method for separating time series into components. |
| `ConfigureTool(IAgentTool)` | Configures a tool for agent-based systems and function calling. |
| `ConfigureTrainingGroups(IReadOnlyList<IReadOnlyList<Int32>>)` | Configures GROUPED training for ranking-style objectives: each inner list is a set of TRAINING row indices forming one coherent query group (e.g. |
| `ConfigureTrainingMonitor(ITrainingMonitor<>)` | Configures training monitoring for real-time visibility into training progress. |
| `ConfigureTrainingPipeline(TrainingPipelineConfiguration<,,>)` | Configures a multi-stage training pipeline for advanced training workflows. |
| `ConfigureUncertaintyQuantification(UncertaintyQuantificationOptions,UncertaintyCalibrationData<,>)` | Configures uncertainty quantification (UQ) for inference-time uncertainty estimates. |
| `ConfigureVideoModel(IVideoModel<>)` | Configures a video model for video understanding and generation tasks. |
| `ConfigureWaveletFunction(IWaveletFunction<>)` | Configures a wavelet function for time-frequency analysis and signal decomposition. |
| `ConfigureWeightStreaming(WeightStreamingConfig)` |  |
| `ConfigureWindowFunction(IWindowFunction<>)` | Configures a window function for signal processing and spectral analysis. |
| `ConvertHyperparameterValue(Object,Type)` | Converts a hyperparameter value to the target property type. |
| `CreateDeepEnsembleMemberOptimizationInputData(OptimizationInputData<,,>,Int32,Int32)` | Creates per-member optimization input data for deep ensembles. |
| `CreateDefaultAugmentationConfig` | Creates a default augmentation configuration with auto-detected modality settings. |
| `CreateNasAutoMLModel(AutoMLSearchStrategy)` | Creates a NAS-based AutoML model with the specified strategy. |
| `CreateOptimizerForEnsembleMember(IFullModel<,,>,IOptimizer<,,>)` | Creates an optimizer instance for an ensemble member based on the template optimizer type and options. |
| `CreateProfilerSession` | Creates a ProfilerSession if profiling is enabled; otherwise returns null. |
| `DeriveModelType(IFullModel<,,>)` | Derives the open generic type definition from the actual model instance. |
| `DeserializeModel(Byte[])` | Converts a byte array back into a usable predictive model. |
| `EnableTensorsOpProfiling` | Enables low-level per-tensor-op profiling via Tensors' `PerformanceProfiler.Instance`. |
| `FitPostprocessingIfNeeded(IFullModel<,,>,,String)` | Fits the configured `_postprocessingPipeline` on the model's training-set predictions BEFORE attaching it to an `AiModelResultOptions`. |
| `FitTemperatureFromProbabilities(Tensor<>,Vector<Int32>,Int32,Int32)` | Fits a temperature scaling value using log-probabilities derived from predicted probabilities. |
| `GatherRows(Tensor<>,IReadOnlyList<Int32>)` | Collects all data from a streaming data loader into aggregated features and labels. |
| `GenerateTrialEncryptionKey` | Generates a deterministic encryption key for trial-mode saves. |
| `GetSampleRowIndices(Int32)` | Gets sample row indices for data hashing (first, middle, last). |
| `LoadModel(String)` | Loads a previously saved model from a file. |
| `MapHyperparameterToProperty(String)` | Maps common hyperparameter names from search spaces to property names on optimizer options. |
| `NextGaussian(Random,Double,Double)` | Generates a Gaussian random value using the Box-Muller transform. |
| `PerturbInitialParametersIfSupported(IFullModel<,,>,Int32,Int32,Double)` | Perturbs initial model parameters to avoid ensemble member collapse to identical solutions. |
| `Predict(,AiModelResult<,,>)` | Uses a trained model to make predictions on new data. |
| `ReportAccelerationStatus(Action<String>)` | Captures a snapshot of the active acceleration environment (SIMD, GPU, native BLAS) at build time, logs it through `logger` if supplied, and surfaces the structured snapshot on `PredictionModelResult.AccelerationSnapshot`. |
| `RequireOnnxExportable` | Marks the builder as requiring an ONNX-exportable result. |
| `ResolveModalityAugmenter(AugmentationConfig)` | Resolves a modality-specific built-in augmenter from `config`'s settings blocks. |
| `SaveModel(AiModelResult<,,>,String)` | Saves a trained model to a file so it can be used later without retraining. |
| `SelectBestNasStrategy(SearchSpaceBase<>,Int32,NASOptions<>)` | Auto-selects the best NAS strategy based on task characteristics. |
| `SelectKthInPlace(Vector<>,Int32)` | Selects the k-th smallest element in-place using a Quickselect partitioning strategy. |
| `SerializeModel(AiModelResult<,,>)` | Converts a trained model into a byte array for storage or transmission. |
| `SetGpuExecutionEnvironmentVariables(GpuExecutionOptions)` | Sets environment variables for GPU execution options. |
| `SetPostprocessingFitMaxRows(Nullable<Int32>)` | Caps the number of training rows that the post-train pipeline-fit step feeds into `bestSolution.Predict(...)`. |
| `Swap(Span<>,Int32,Int32)` | Swaps two elements in a span. |
| `ToPascalCase(String)` | Converts a hyperparameter name to PascalCase for property lookup. |
| `TryBootstrapTrainingData(,,Random,,)` | Attempts to bootstrap the training data (sample with replacement) for deep-ensemble diversity. |
| `TryComputeAndAttachDeepEnsembleModels(AiModelResult<,,>,IFullModel<,,>,OptimizationInputData<,,>,IOptimizer<,,>,UncertaintyQuantificationOptions)` | Computes deep ensemble members and attaches them to the result when deep ensemble uncertainty quantification is enabled. |
| `TryComputeAndAttachUncertaintyCalibrationArtifacts(AiModelResult<,,>)` | Computes calibration artifacts (not raw calibration data) and attaches them to the final result. |
| `TryComputeClassificationCalibrationArtifacts(AiModelResult<,,>,,Vector<Int32>,UncertaintyQuantificationOptions,UncertaintyCalibrationArtifacts<>)` | Computes classification calibration artifacts including ECE, conformal threshold, and optional temperature scaling. |
| `TryComputeRegressionConformalArtifacts(AiModelResult<,,>,,,UncertaintyCalibrationArtifacts<>)` | Computes conformal regression artifacts from calibration data. |
| `TrySliceFirstNSamples(,Int32)` | Returns a prefix slice of `x` containing at most `maxRows` leading samples. |
| `TrySliceFirstSampleForLoRAWarmup()` | Carves a 1-sample probe off the training input for LoRA warmup forwards. |
| `UseDirectTrainingPath(IFullModel<,,>)` | Decides whether BuildSupervisedInternalAsync should route through the model's own `Train` method instead of the outer optimizer's clone-evaluate-select loop. |
| `ValidateLicense` | Validates the configured license key, reusing a cached validator to preserve in-memory state (e.g., offline grace period tracking). |
| `ValidateOnnxExportableIfRequired(AiModelResult<,,>)` | If the builder was marked with `RequireOnnxExportable`, throws `OnnxExportUnsupportedException` when the supplied trained model contains a non-exportable layer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_allowNondeterminism` | When `true`, `BuildAsync` does NOT force deterministic math on the engine. |
| `_postprocessingFitMaxRows` | Optional cap on the number of training rows fed into the post-train pipeline-fit Predict call (review #1368 C7HAu). |
| `_weightStreamingConfig` | User-supplied weight-streaming overrides. |

