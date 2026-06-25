---
title: "IAiModelBuilder<T, TInput, TOutput>"
description: "Defines a builder pattern interface for creating and configuring predictive models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a builder pattern interface for creating and configuring predictive models.

## How It Works

This interface provides a fluent API for setting up all components of a machine learning model.

**For Beginners:** Think of this as a step-by-step recipe builder for creating AI models.
Just like building a custom sandwich where you choose the bread, fillings, and condiments,
this builder lets you choose different components for your AI model.

The builder pattern makes it easy to:

- Configure your model piece by piece
- Change only the parts you want while keeping default settings for the rest
- Create different variations of models without writing repetitive code

## Methods

| Method | Summary |
|:-----|:--------|
| `AllowNondeterminism` | Opts out of the builder's deterministic-by-default policy. |
| `BuildAsync` | Asynchronously builds a meta-trained model that can quickly adapt to new tasks. |
| `ConfigureABTesting(ABTestingConfig)` | Configures A/B testing to compare multiple model versions by splitting traffic. |
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
| `ConfigureAutoML(AutoMLOptions<,,>)` | Configures AutoML using facade-style options (recommended for most users). |
| `ConfigureAutoML(IAutoMLModel<,,>)` | Configures an AutoML model for automatic machine learning optimization. |
| `ConfigureBenchmark(IBenchmark<>)` | Configures a benchmark for evaluating and comparing model performance. |
| `ConfigureBenchmarking(BenchmarkingOptions)` | Configures benchmarking to run standardized benchmark suites and attach a structured report to the built model. |
| `ConfigureBiasDetector(IBiasDetector<>)` | Configures the bias detector component for ethical AI evaluation. |
| `ConfigureCaching(CacheConfig)` | Configures model caching to avoid reloading models from disk repeatedly. |
| `ConfigureCausalDiscovery(Action<CausalDiscoveryOptions>)` | Configures causal structure discovery to learn a DAG from the training data. |
| `ConfigureCausalDiscovery(CausalDiscoveryOptions)` | Configures causal structure discovery with a pre-built options object. |
| `ConfigureCausalInference(ICausalModel<>)` | Configures a causal inference model for understanding cause-and-effect relationships. |
| `ConfigureCertifiedDefense(ICertifiedDefense<,,>)` | Configures a certified defense for providing formal robustness guarantees. |
| `ConfigureCheckpointManager(ICheckpointManager<,,>)` | Configures checkpoint management for saving and restoring training state. |
| `ConfigureClassificationMetric(IClassificationMetric<>)` | Configures a classification metric for evaluating classifier performance. |
| `ConfigureClassifier(IClassifier<>)` | Configures a classification algorithm for categorizing data into discrete classes. |
| `ConfigureClusterMetric(IClusterMetric<>)` | Configures an internal cluster metric for evaluating clustering quality without ground truth labels. |
| `ConfigureClustering(IClustering<>)` | Configures a clustering algorithm for grouping similar data points together. |
| `ConfigureCompression(CompressionConfig)` | Configures model compression for reducing model size during serialization. |
| `ConfigureContinualLearning(IContinualLearner<,,>)` | Configures a continual learning trainer that can learn new tasks without forgetting old ones. |
| `ConfigureCrossValidation(ICrossValidator<,,>)` | Configures the cross-validation strategy for model evaluation. |
| `ConfigureCurriculumLearning(CurriculumLearningOptions<,,>)` | Configures curriculum learning for training models with progressively harder samples. |
| `ConfigureCurriculumScheduler(ICurriculumScheduler<>)` | Configures a curriculum scheduler for ordering training samples by difficulty. |
| `ConfigureDataLoader(IDataLoader<>)` | Configures the data loader for providing training data. |
| `ConfigureDataPreparation(Action<DataPreparationPipeline<>>)` | Configures the data preparation pipeline for row-changing operations. |
| `ConfigureDataSplitter(IDataSplitter<>)` | Configures a data splitting strategy for dividing datasets into train/test/validation sets. |
| `ConfigureDataTransformer(IDataTransformer<,,>)` | Configures a data transformer for preprocessing or postprocessing data transformations. |
| `ConfigureDataVersionControl(IDataVersionControl<>)` | Configures data version control for tracking dataset changes. |
| `ConfigureDiffusionModel(IDiffusionModel<>)` | Configures a diffusion model for generative tasks (image/audio/video generation). |
| `ConfigureDistanceMetric(IDistanceMetric<>)` | Configures a distance metric for measuring similarity between data points. |
| `ConfigureDistillationStrategy(IDistillationStrategy<>)` | Configures a knowledge distillation strategy for transferring knowledge between models. |
| `ConfigureDistributedTraining(ICommunicationBackend<>,DistributedStrategy,IShardingConfiguration<>)` | Configures distributed training across multiple GPUs or machines. |
| `ConfigureDocumentModel(IDocumentModel<>)` | Configures a document model for document understanding and processing. |
| `ConfigureDocumentStore(IDocumentStore<>)` | Configures a document store for persisting and retrieving documents with vector similarity search. |
| `ConfigureDriftDetection(IDriftDetector<>)` | Configures a drift detector for monitoring changes in data distribution over time. |
| `ConfigureEmbeddingModel(IEmbeddingModel<>)` | Configures an embedding model for learning dense vector representations. |
| `ConfigureEnvironment(IEnvironment<>)` | Configures a reinforcement learning environment for agent training. |
| `ConfigureExperimentTracker(IExperimentTracker<>)` | Configures experiment tracking for organizing and logging ML experiments. |
| `ConfigureExplorationStrategy(IExplorationStrategy<>)` | Configures an exploration strategy for reinforcement learning agents. |
| `ConfigureExport(ExportConfig)` | Configures export settings for deploying the model to different platforms. |
| `ConfigureExternalClusterMetric(IExternalClusterMetric<>)` | Configures an external cluster metric for evaluating clustering quality against ground truth labels. |
| `ConfigureFairnessEvaluator(IFairnessEvaluator<>)` | Configures the fairness evaluator component for ethical AI evaluation. |
| `ConfigureFederatedLearning(FederatedLearningOptions,IAggregationStrategy<IFullModel<,,>>,IClientSelectionStrategy,IFederatedServerOptimizer<>,IFederatedHeterogeneityCorrection<>,IHomomorphicEncryptionProvider<>,IPrivateSetIntersection,ISecureComputationProtocol<>,ITeeProvider<>,IZkProofSystem,IFederatedUnlearner<>,IFederatedDriftDetector<>,IClientContributionEvaluator<>,IFairnessConstraint<>)` | Enables federated learning training using the provided options. |
| `ConfigureFinancialModel(IFinancialModel<>)` | Configures a financial model for quantitative finance and risk analysis. |
| `ConfigureFineTuning(FineTuningConfiguration<,,>)` | Configures fine-tuning for the model using preference learning, RLHF, or other alignment methods. |
| `ConfigureFitDetector(IFitDetector<,,>)` | Configures the fit detector component for the model. |
| `ConfigureFitnessCalculator(IFitnessCalculator<,,>)` | Configures the fitness calculator component for the model. |
| `ConfigureGaussianProcess(IGaussianProcess<>)` | Configures a Gaussian process model for probabilistic predictions with uncertainty estimates. |
| `ConfigureGpuAcceleration(GpuAccelerationConfig)` | Enables GPU acceleration for training and inference with optional configuration. |
| `ConfigureGpuDiagnostics(GpuDiagnosticsOptions)` | Controls GPU backend diagnostic output visibility and routing. |
| `ConfigureHyperparameterOptimizer(IHyperparameterOptimizer<,,>,HyperparameterSearchSpace,Int32)` | Configures hyperparameter optimization for automatic tuning of model settings. |
| `ConfigureInferenceOptimizations(InferenceOptimizationConfig)` | Configures inference-time optimizations for faster predictions. |
| `ConfigureInterpolation(IInterpolation<>)` | Configures a 1D interpolation method for estimating values between known data points. |
| `ConfigureInterpolation2D(I2DInterpolation<>)` | Configures a 2D interpolation method for estimating values on a surface between known data points. |
| `ConfigureJitCompilation(JitCompilationConfig)` | Enables JIT (Just-In-Time) compilation for the built model's forward-pass replay. |
| `ConfigureKernelFunction(IKernelFunction<>)` | Configures the kernel function for kernel-based methods (SVM, Gaussian processes, etc.). |
| `ConfigureKnowledgeDistillation(KnowledgeDistillationOptions<,,>)` | Configures knowledge distillation for training a smaller student model from a larger teacher model. |
| `ConfigureKnowledgeGraph(Action<KnowledgeGraphOptions>)` | Configures advanced knowledge graph capabilities including embeddings, community detection, link prediction, temporal queries, and KG construction. |
| `ConfigureLayer(ILayer<>)` | Configures a neural network layer for building custom network architectures. |
| `ConfigureLearningRateScheduler(ILearningRateScheduler)` | Configures a learning rate scheduler that adjusts the learning rate during training. |
| `ConfigureLicenseKey(AiDotNetLicenseKey)` | Configures a license key for encrypted model loading and saving with optional online validation. |
| `ConfigureLinkFunction(ILinkFunction<>)` | Configures a link function for generalized linear models (GLMs). |
| `ConfigureLoRA(ILoRAConfiguration<>)` | Configures LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. |
| `ConfigureLossFunction(ILossFunction<>)` | Configures the loss function used to measure prediction error during training. |
| `ConfigureMatrixDecomposition(IMatrixDecomposition<>)` | Configures a matrix decomposition method for linear algebra operations. |
| `ConfigureMetaLearning(IMetaLearner<,,>)` | Configures a meta-learning algorithm (MAML, Reptile, SEAL) for training models that can quickly adapt to new tasks. |
| `ConfigureMixedPrecision(MixedPrecisionConfig)` | Configures mixed-precision training for faster neural network training with reduced memory usage. |
| `ConfigureModel(IFullModel<,,>)` | Configures the prediction model algorithm to use. |
| `ConfigureModelCompressionStrategy(IModelCompressionStrategy<>)` | Configures a model compression strategy for reducing model size and inference cost. |
| `ConfigureModelExplainer(IModelExplainer<>)` | Configures a model explainer for understanding model predictions. |
| `ConfigureModelOptions(ModelOptions)` | Configures model options that control training behavior and hyperparameters. |
| `ConfigureModelRegistry(IModelRegistry<,,>)` | Configures model registry for centralized model storage and versioning. |
| `ConfigureNoiseScheduler(INoiseScheduler<>)` | Configures a noise scheduler for diffusion model training and sampling. |
| `ConfigureOnlineLearning(IOnlineLearningModel<>)` | Configures an online learning model that updates incrementally with new data. |
| `ConfigureOptimizer(IOptimizer<,,>)` | Configures the optimization algorithm for the model. |
| `ConfigurePDESpecification(IPDESpecification<>)` | Configures a PDE specification for physics-informed neural network training. |
| `ConfigurePipelineParallelism(IPipelineSchedule<>,IPipelinePartitionStrategy<>,ActivationCheckpointConfig,Int32)` | Configures pipeline-specific options for pipeline parallel training. |
| `ConfigurePlanCaching(String)` | Enables disk-backed caching of compiled inference plans in the supplied directory. |
| `ConfigurePointCloudModel(IPointCloudModel<>)` | Configures a point cloud model for 3D data processing. |
| `ConfigurePostprocessing(Action<PostprocessingPipeline<,,>>)` | Configures the output postprocessing pipeline for the model using a fluent builder. |
| `ConfigurePostprocessing(IDataTransformer<,,>)` | Configures the output postprocessing pipeline for the model using a single transformer. |
| `ConfigurePostprocessing(PostprocessingPipeline<,,>)` | Configures the output postprocessing pipeline for the model using an existing pipeline. |
| `ConfigurePreprocessing(Action<PreprocessingPipeline<,,>>)` | Configures the data preprocessing pipeline for the model using a fluent builder. |
| `ConfigurePreprocessing(IDataTransformer<,,>)` | Configures the data preprocessing pipeline for the model using a single transformer. |
| `ConfigureProgramSynthesis(ProgramSynthesisOptions)` | Configures built-in Program Synthesis defaults for code tasks. |
| `ConfigureProgramSynthesisServing(ProgramSynthesisServingClientOptions,IProgramSynthesisServingClient)` | Configures Program Synthesis to prefer calling `AiDotNet.Serving` for sandboxed execution and evaluation. |
| `ConfigureQuantization(QuantizationConfig)` | Configures model quantization for reducing model size and improving inference speed. |
| `ConfigureQueryStrategy(IQueryStrategy<,,>)` | Configures a query strategy for active learning sample selection. |
| `ConfigureRLAgent(IRLAgent<>)` | Configures a reinforcement learning agent for learning through interaction with an environment. |
| `ConfigureRadialBasisFunction(IRadialBasisFunction<>)` | Configures a radial basis function for RBF networks and interpolation. |
| `ConfigureReasoning(ReasoningConfig)` | Configures advanced reasoning capabilities for the model using Chain-of-Thought, Tree-of-Thoughts, and Self-Consistency strategies. |
| `ConfigureRegression(IRegression<>)` | Configures a regression algorithm for predicting continuous numeric values. |
| `ConfigureRegressionMetric(IRegressionMetric<>)` | Configures a regression metric for evaluating regression model performance. |
| `ConfigureRegularization(IRegularization<,,>)` | Configures the regularization component for the model. |
| `ConfigureReinforcementLearning(RLTrainingOptions<>)` | Configures reinforcement learning options for training an RL agent. |
| `ConfigureRetrievalAugmentedGeneration(IRetriever<>,IReranker<>,IGenerator<>,IEnumerable<IQueryProcessor>,IGraphStore<>,KnowledgeGraph<>,IDocumentStore<>)` | Configures the retrieval-augmented generation (RAG) components for use during model inference. |
| `ConfigureSSLMethod(ISSLMethod<>)` | Configures a self-supervised learning method for learning representations without labeled data. |
| `ConfigureSafety(Action<SafetyConfig>)` | Configures the comprehensive safety pipeline for input validation and output filtering. |
| `ConfigureScoringRule(IScoringRule<>)` | Configures a scoring rule for evaluating probabilistic predictions. |
| `ConfigureSelfSupervisedLearning(Action<SSLConfig>)` | Configures self-supervised pretraining (configuration-only — SSL settings are stored but no pretraining stage runs). |
| `ConfigureSelfSupervisedLearning(Action<SSLConfig>,Func<IFullModel<,,>,SSLConfig,CancellationToken,Task<IFullModel<,,>>>)` | Configures self-supervised pretraining with a user-supplied typed hook (AiDotNet#1361 wire-up). |
| `ConfigureSimilarityMetric(ISimilarityMetric<>)` | Configures a similarity metric for vector similarity search operations. |
| `ConfigureSpeechRecognizer(ISpeechRecognizer<>)` | Configures a speech recognition model for converting spoken audio to text. |
| `ConfigureStoppingCriterion(IStoppingCriterion<>)` | Configures a stopping criterion for active learning loops. |
| `ConfigureSurvivalAnalysis(ISurvivalModel<>)` | Configures a survival analysis model for time-to-event prediction. |
| `ConfigureTargetScaling(PreprocessingPipeline<,,>)` | Configures TARGET (label) scaling for regression: targets are scaled before training (fit on the TRAINING split only; default z-score via `TargetStandardScaler`) and `Predict` automatically inverse-transforms outputs back to the ORIGINAL ta… |
| `ConfigureTelemetry(TelemetryConfig)` | Configures telemetry for tracking and monitoring model inference metrics. |
| `ConfigureTextToSpeech(ITextToSpeech<>)` | Configures a text-to-speech model for converting written text to spoken audio. |
| `ConfigureTextVectorizer(ITextVectorizer<>)` | Configures a text vectorizer for converting text data into numeric feature vectors. |
| `ConfigureTimeSeriesDecomposition(ITimeSeriesDecomposition<>)` | Configures a time series decomposition method for separating time series into components. |
| `ConfigureTool(IAgentTool)` | Configures a tool for agent-based systems and function calling. |
| `ConfigureTrainingGroups(IReadOnlyList<IReadOnlyList<Int32>>)` | Configures GROUPED training (one fit per query group per epoch) for ranking-style objectives — see the builder method for semantics. |
| `ConfigureTrainingMonitor(ITrainingMonitor<>)` | Configures training monitoring for real-time visibility into training progress. |
| `ConfigureTrainingPipeline(TrainingPipelineConfiguration<,,>)` | Configures a multi-stage training pipeline for advanced training workflows. |
| `ConfigureUncertaintyQuantification(UncertaintyQuantificationOptions,UncertaintyCalibrationData<,>)` | Configures uncertainty quantification (UQ) for inference-time uncertainty estimates. |
| `ConfigureVersioning(VersioningConfig)` | Configures model versioning for managing multiple versions of the same model. |
| `ConfigureVideoModel(IVideoModel<>)` | Configures a video model for video understanding and generation tasks. |
| `ConfigureWaveletFunction(IWaveletFunction<>)` | Configures a wavelet function for time-frequency analysis and signal decomposition. |
| `ConfigureWindowFunction(IWindowFunction<>)` | Configures a window function for signal processing and spectral analysis. |
| `DeserializeModel(Byte[])` | Reconstructs a model from a previously serialized byte array. |
| `EnableTensorsOpProfiling` | Enables low-level per-tensor-op profiling via Tensors' `PerformanceProfiler.Instance`. |
| `LoadModel(String)` | Loads a previously saved model from a file. |
| `Predict(,AiModelResult<,,>)` | Uses a trained model to make predictions on new data. |
| `ReportAccelerationStatus(Action<String>)` | Captures SIMD/GPU/native-BLAS acceleration status at build time, logs it, and surfaces a structured snapshot on `PredictionModelResult.AccelerationSnapshot`. |
| `SaveModel(AiModelResult<,,>,String)` | Saves a trained model to a file. |
| `SerializeModel(AiModelResult<,,>)` | Converts a trained model into a byte array for storage or transmission. |

