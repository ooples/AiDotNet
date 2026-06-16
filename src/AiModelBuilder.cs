

global using AiDotNet.Configuration;
global using AiDotNet.Deployment.Configuration;
global using AiDotNet.Diagnostics;
global using AiDotNet.DistributedTraining;
global using AiDotNet.Enums;
global using AiDotNet.Extensions;
global using AiDotNet.FitDetectors;
global using AiDotNet.FitnessCalculators;
global using AiDotNet.Helpers;
global using AiDotNet.KnowledgeDistillation;
global using AiDotNet.LinearAlgebra;
global using AiDotNet.LossFunctions;
global using AiDotNet.MetaLearning;
global using AiDotNet.MixedPrecision;
global using AiDotNet.Models;
global using AiDotNet.Models.Inputs;
global using AiDotNet.Models.Options;
global using AiDotNet.Optimizers;
global using AiDotNet.AnomalyDetection;
global using AiDotNet.ProgramSynthesis.Interfaces;
global using AiDotNet.ProgramSynthesis.Serving;
global using AiDotNet.PromptEngineering.FewShot;
global using AiDotNet.PromptEngineering.Optimization;
global using AiDotNet.PromptEngineering.Templates;
global using AiDotNet.Reasoning.Models;
global using AiDotNet.Regularization;
global using AiDotNet.RetrievalAugmentedGeneration.Graph;
global using AiDotNet.Tensors.Helpers;
global using AiDotNet.Tokenization.Configuration;
global using AiDotNet.Tokenization.HuggingFace;
global using AiDotNet.Tokenization.Interfaces;
global using AiDotNet.UncertaintyQuantification.Layers;
using AiDotNet.Augmentation;
using AiDotNet.AutoML.NAS;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;
using AiDotNet.AutoML.Policies;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Postprocessing;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet;

/// <summary>
/// A builder class that helps create and configure machine learning prediction models.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class uses the builder pattern to configure various components of a machine learning model
/// before building and using it for predictions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this class as a recipe builder for creating AI models.
/// You add different ingredients (like data normalization, feature selection, etc.)
/// and then "cook" (build) the final model. This approach makes it easy to customize
/// your model without having to understand all the complex details at once.
/// </para>
/// <para>
/// <b>Training Infrastructure Example:</b> Complete example showing experiment tracking,
/// checkpointing, model registry, and hyperparameter optimization working together:
/// </para>
/// <code>
/// // 1. Create training infrastructure components
/// var experimentTracker = new ExperimentTracker&lt;double&gt;("./mlruns");
/// var checkpointManager = new CheckpointManager&lt;double, double[], double&gt;("./checkpoints");
/// var modelRegistry = new ModelRegistry&lt;double, double[], double&gt;("./models");
/// var bayesianOptimizer = new BayesianOptimizer&lt;double, double[], double&gt;(
///     maximize: false,  // Minimize loss
///     acquisitionFunction: AcquisitionFunctionType.ExpectedImprovement,
///     nInitialPoints: 5,
///     seed: 42);
///
/// // 2. Create an experiment and start a run
/// var experimentId = experimentTracker.CreateExperiment(
///     "image-classification",
///     description: "CNN training with hyperparameter tuning",
///     tags: new Dictionary&lt;string, string&gt; { ["team"] = "ml-research" });
///
/// var run = experimentTracker.StartRun(experimentId, "baseline-run");
/// run.LogParameters(new Dictionary&lt;string, object&gt;
/// {
///     ["learning_rate"] = 0.001,
///     ["batch_size"] = 32,
///     ["epochs"] = 100
/// });
///
/// // 3. Configure the builder with training infrastructure
/// var builder = new AiModelBuilder&lt;double, double[], double&gt;()
///     .ConfigureModel(neuralNetwork)
///     .ConfigureOptimizer(adamOptimizer)
///     .ConfigureExperimentTracker(experimentTracker)
///     .ConfigureCheckpointManager(checkpointManager)
///     .ConfigureModelRegistry(modelRegistry)
///     .ConfigureHyperparameterOptimizer(bayesianOptimizer);
///
/// // 4. Train and track progress
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     // Train epoch...
///     var loss = TrainEpoch(model, data);
///     var accuracy = Evaluate(model, validationData);
///
///     // Log metrics to experiment tracker
///     run.LogMetric("loss", loss, step: epoch);
///     run.LogMetric("accuracy", accuracy, step: epoch);
///
///     // Save checkpoint periodically
///     if (epoch % 10 == 0)
///     {
///         checkpointManager.SaveCheckpoint(
///             model, optimizer, epoch, totalSteps,
///             new Dictionary&lt;string, double&gt; { ["loss"] = loss, ["accuracy"] = accuracy });
///     }
/// }
///
/// // 5. Complete the run and register the model
/// run.Complete();
///
/// var modelVersion = modelRegistry.RegisterModel(
///     model, "cnn-classifier", ModelType.NeuralNetwork,
///     new Dictionary&lt;string, object&gt; { ["final_accuracy"] = 0.95 });
///
/// // 6. Promote model to production
/// modelRegistry.TransitionStage(modelVersion.ModelId, modelVersion.Version, ModelStage.Production);
/// </code>
/// </remarks>
public partial class AiModelBuilder<T, TInput, TOutput> : IAiModelBuilder<T, TInput, TOutput>, IWeightStreamingCapableBuilder<T, TInput, TOutput>, AiDotNet.Configuration.IConfiguredView<T, TInput, TOutput>
{
    private static IEngine Engine => AiDotNetEngine.Current;

    // audit-2026-05 phase 2a slice 1 — data-pipeline concern extracted into a separately-testable
    // component. The Configure{Preprocessing,Postprocessing,DataLoader,DataPreparation,Augmentation}
    // methods + SetPostprocessingFitMaxRows delegate here; the legacy private fields below stay as
    // synced caches that BuildAsync and partial-class siblings continue to read until slice 2
    // migrates those callsites to the component's properties. See
    // docs/internal/audit-2026-05-phase2a-aimodelbuilder-refactor.md for the full migration plan.
    private readonly AiDotNet.Configuration.IAiModelDataPipeline<T, TInput, TOutput> _dataPipeline
        = new AiDotNet.Configuration.AiModelDataPipeline<T, TInput, TOutput>();

    // audit-2026-05 phase 2a slice 2 — training-core concern extracted similarly. The
    // Configure{Model,Optimizer,Regularization,FitnessCalculator,FitDetector,TrainingPipeline,
    // CheckpointManager,MemoryManagement,TrainingMonitor} methods delegate here. Legacy fields
    // below stay as synced caches that BuildAsync and partial-class siblings read from.
    private readonly AiDotNet.Configuration.IAiModelTrainingCore<T, TInput, TOutput> _trainingCore
        = new AiDotNet.Configuration.AiModelTrainingCore<T, TInput, TOutput>();

    // audit-2026-05 phase 2a slice 3 — cross-validation concern.
    private readonly AiDotNet.Configuration.IAiModelCrossValidation<T, TInput, TOutput> _crossValidation
        = new AiDotNet.Configuration.AiModelCrossValidation<T, TInput, TOutput>();

    // audit-2026-05 phase 2a slice 4 — compliance concern (bias, fairness, interpretability,
    // adversarial robustness, safety filtering).
    private readonly AiDotNet.Configuration.IAiModelCompliance<T, TInput, TOutput> _compliance
        = new AiDotNet.Configuration.AiModelCompliance<T, TInput, TOutput>();

    // audit-2026-05 phase 2a slice 12 — license / enterprise-gate concern. Holds both the user-
    // supplied AiDotNetLicenseKey and the cached LicenseValidator; ConfigureLicenseKey resets
    // the validator any time the key changes.
    private AiDotNet.Configuration.IAiModelLicensing? _licensing;

    // audit-2026-05 phase 2a slice 9 — storage / artifact-management concern.
    private readonly AiDotNet.Configuration.IAiModelStorage<T, TInput, TOutput> _storage
        = new AiDotNet.Configuration.AiModelStorage<T, TInput, TOutput>();

    // audit-2026-05 phase 2a slice 10 — observability concern (benchmarking, profiling,
    // telemetry, GPU diagnostics).
    private readonly AiDotNet.Configuration.IAiModelObservability _observability
        = new AiDotNet.Configuration.AiModelObservability();

    private PreprocessingPipeline<T, TInput, TInput>? _preprocessingPipeline;
    private PreprocessingPipeline<T, TOutput, TOutput>? _targetPipeline;
    private IReadOnlyList<IReadOnlyList<int>>? _trainingGroups;
    private PostprocessingPipeline<T, TOutput, TOutput>? _postprocessingPipeline;

    /// <summary>
    /// Optional cap on the number of training rows fed into the
    /// post-train pipeline-fit Predict call (review #1368 C7HAu). Default
    /// is null = no cap = full <c>XTrain</c> tensor. Set via
    /// <see cref="ConfigurePostprocessingFitMaxRows"/> when the user's
    /// pipeline transformers stabilise on a subsample and the doubled
    /// build-time inference cost matters.
    /// </summary>
    private int? _postprocessingFitMaxRows;
    private IRegularization<T, TInput, TOutput>? _regularization;
    private IFitnessCalculator<T, TInput, TOutput>? _fitnessCalculator;
    private IFitDetector<T, TInput, TOutput>? _fitDetector;
    private IFullModel<T, TInput, TOutput>? _model;

    /// <summary>
    /// Gets the configured model instance for use by domain-specific partial class methods.
    /// </summary>
    internal IFullModel<T, TInput, TOutput>? ConfiguredModel => _model;

    private IOptimizer<T, TInput, TOutput>? _optimizer;
    private IDataLoader<T>? _dataLoader;
    private DataPreparationPipeline<T>? _dataPreparationPipeline;
    private IBiasDetector<T>? _biasDetector;
    private IFairnessEvaluator<T>? _fairnessEvaluator;
    private InterpretabilityOptions? _interpretabilityOptions;
    private AdversarialRobustnessConfiguration<T, TInput, TOutput>? _adversarialRobustnessConfiguration;
    private FineTuningConfiguration<T, TInput, TOutput>? _fineTuningConfiguration;
    private ILoRAConfiguration<T>? _loraConfiguration;
    private IRetriever<T>? _ragRetriever;
    private IReranker<T>? _ragReranker;
    private IGenerator<T>? _ragGenerator;
    private IEnumerable<IQueryProcessor>? _queryProcessors;

    // Graph RAG components for knowledge graph-enhanced retrieval
    private KnowledgeGraph<T>? _knowledgeGraph;
    private IGraphStore<T>? _graphStore;
    private HybridGraphRetriever<T>? _hybridGraphRetriever;
    private KnowledgeGraphOptions? _knowledgeGraphOptions;
    private IMetaLearner<T, TInput, TOutput>? _metaLearner;
    private ICommunicationBackend<T>? _distributedBackend;
    private DistributedStrategy _distributedStrategy = DistributedStrategy.DDP;
    private IShardingConfiguration<T>? _distributedConfiguration;
    private IPipelinePartitionStrategy<T>? _pipelinePartitionStrategy;
    private IPipelineSchedule<T>? _pipelineSchedule;
    private ActivationCheckpointConfig? _pipelineCheckpointConfig;
    private int _pipelineMicroBatchCount = 1;
    private ICrossValidator<T, TInput, TOutput>? _crossValidator;
    private KnowledgeDistillationOptions<T, TInput, TOutput>? _knowledgeDistillationOptions;
    private MixedPrecisionConfig? _mixedPrecisionConfig;
    // #1632 — inference optimizations ON BY DEFAULT (opt-out). The stack (layer fusion +
    // flash/cached attention) is only ENGAGED if this is non-null, and previously it was null
    // unless the user called ConfigureInferenceOptimizations() — so the whole built-and-verified
    // inference stack sat dormant for every model. Default it to the sensible Default config; the
    // Predict path applies only the STATELESS subset on a CLONE (original untouched) inside a
    // try/catch fallback, and OptimizeForInference no-ops for models without foldable BatchNorm /
    // optimizable attention. Opt out by calling ConfigureInferenceOptimizations with disabled flags.
    private AiDotNet.Configuration.InferenceOptimizationConfig? _inferenceOptimizationConfig
        = AiDotNet.Configuration.InferenceOptimizationConfig.Default;
    private AiDotNet.Configuration.JitCompilationConfig? _jitCompilationConfig;
    private bool _reportAccelerationAtBuild;
    private Action<string>? _accelerationLogger;
    private AiDotNet.Diagnostics.AccelerationSnapshot? _accelerationSnapshot;
    private bool _tensorsOpProfilingEnabled;
    private AiDotNet.Diagnostics.TensorsOperationProfile? _tensorsOperationProfile;

    /// <summary>
    /// When <c>true</c>, <see cref="BuildAsync"/> does NOT force deterministic
    /// math on the engine.
    /// </summary>
    private bool _allowNondeterminism;
    private RLTrainingOptions<T>? _rlOptions;
    private IAutoMLModel<T, TInput, TOutput>? _autoMLModel;
    private AutoMLOptions<T, TInput, TOutput>? _autoMLOptions;

    // Curriculum learning configuration
    private CurriculumLearningOptions<T, TInput, TOutput>? _curriculumLearningOptions;

    // Unified augmentation configuration
    private Augmentation.AugmentationConfig? _augmentationConfig;

    // Process-wide once-per-run latch for the ConfigureAugmentation
    // informational messages (review #1368 C4TPM: warnings were firing on
    // every successful Build, polluting traces in production / CI).
    // Mutated via Interlocked.Exchange on the NON-GENERIC
    // AugmentationWarningLatch helper class — without that indirection,
    // each closed-generic instantiation of AiModelBuilder<T,TIn,TOut>
    // gets its own static field (review #1368 C7HAP) and the once-per-
    // run guarantee silently breaks for mixed-generic test runs.

    // Self-supervised learning configuration
    private SelfSupervisedLearning.SSLConfig? _sslConfig;
    // Optional user-supplied pretraining hook invoked BEFORE main training when
    // ConfigureSelfSupervisedLearning is used with the action overload. Receives
    // the current base model + SSLConfig + cancellation token; returns the model
    // that should feed into main training. See #1361.
    private Func<IFullModel<T, TInput, TOutput>, SelfSupervisedLearning.SSLConfig, CancellationToken,
        Task<IFullModel<T, TInput, TOutput>>>? _sslPretrainAction;

    // Federated learning configuration (facade-first: orchestration is internal)
    private FederatedLearningOptions? _federatedLearningOptions;
    private IAggregationStrategy<IFullModel<T, TInput, TOutput>>? _federatedAggregationStrategy;
    private IClientSelectionStrategy? _federatedClientSelectionStrategy;
    private IFederatedServerOptimizer<T>? _federatedServerOptimizer;
    private IFederatedHeterogeneityCorrection<T>? _federatedHeterogeneityCorrection;
    private IHomomorphicEncryptionProvider<T>? _federatedHomomorphicEncryptionProvider;
    private FederatedLearning.PSI.IPrivateSetIntersection? _federatedPrivateSetIntersection;
    private FederatedLearning.MPC.ISecureComputationProtocol<T>? _federatedSecureComputationProtocol;
    private FederatedLearning.TEE.ITeeProvider<T>? _federatedTeeProvider;
    private FederatedLearning.Verification.IZkProofSystem? _federatedZkProofSystem;
    private FederatedLearning.Unlearning.IFederatedUnlearner<T>? _federatedUnlearner;
    private FederatedLearning.DriftDetection.IFederatedDriftDetector<T>? _federatedDriftDetector;
    private FederatedLearning.Fairness.IClientContributionEvaluator<T>? _federatedContributionEvaluator;
    private FederatedLearning.Fairness.IFairnessConstraint<T>? _federatedFairnessConstraint;

    // Deployment configuration fields
    private QuantizationConfig? _quantizationConfig;
    private CompressionConfig? _compressionConfig;
    private CacheConfig? _cacheConfig;
    private VersioningConfig? _versioningConfig;
    private ABTestingConfig? _abTestingConfig;
    private TelemetryConfig? _telemetryConfig;
    private ExportConfig? _exportConfig;
    private GpuAccelerationConfig? _gpuAccelerationConfig;
    private ReasoningConfig? _reasoningConfig;
    private ProfilingConfig? _profilingConfig;
    private BenchmarkingOptions? _benchmarkingOptions;
    private AiDotNet.Safety.SafetyConfig? _safetyPipelineConfig;

    // License key configuration
    private AiDotNetLicenseKey? _licenseKey;
    private LicenseValidator? _licenseValidator;

    // Tokenization configuration
    private ITokenizer? _tokenizer;
    private TokenizationConfig? _tokenizationConfig;

    // Program synthesis Serving configuration
    private ProgramSynthesisServingClientOptions? _programSynthesisServingClientOptions;
    private IProgramSynthesisServingClient? _programSynthesisServingClient;
    private IFullModel<T, Tensor<T>, Tensor<T>>? _programSynthesisModel;

    // Training infrastructure configuration
    private IExperimentTracker<T>? _experimentTracker;
    private ICheckpointManager<T, TInput, TOutput>? _checkpointManager;
    private ITrainingMonitor<T>? _trainingMonitor;
    private IModelRegistry<T, TInput, TOutput>? _modelRegistry;
    private IDataVersionControl<T>? _dataVersionControl;
    private IHyperparameterOptimizer<T, TInput, TOutput>? _hyperparameterOptimizer;
    private HyperparameterSearchSpace? _hyperparameterSearchSpace;
    private int _hyperparameterTrials = 10;

    // Uncertainty quantification configuration
    private UncertaintyQuantificationOptions? _uncertaintyQuantificationOptions;
    private UncertaintyCalibrationData<TInput, TOutput>? _uncertaintyCalibrationData;

    // Training pipeline configuration
    private TrainingPipelineConfiguration<T, TInput, TOutput>? _trainingPipelineConfiguration;

    // Memory management configuration for gradient checkpointing, activation pooling, and model sharding
    private Training.Memory.TrainingMemoryConfig? _memoryConfig;

    /// <summary>
    /// Decides whether BuildSupervisedInternalAsync should route through
    /// the model's own <c>Train</c> method instead of the outer
    /// optimizer's clone-evaluate-select loop.
    /// </summary>
    /// <remarks>
    /// Direct-training kicks in when ANY of these conditions hold:
    /// <list type="number">
    ///   <item>Model isn't <see cref="IParameterizable{T,TInput,TOutput}"/>
    ///   (time-series, density-based clustering, etc.) — no parameters to
    ///   optimize, the model handles its own training.</item>
    ///   <item>Model is a <see cref="Clustering.Base.ClusteringBase{T}"/>
    ///   — clustering uses K-means EM not gradient updates; the outer
    ///   optimizer's random search runs hundreds of unrelated trials
    ///   then leaves the model untrained (the bug pattern that caused
    ///   25 clustering builder tests to time out in #1224 cluster B).</item>
    ///   <item>Model is a <see cref="NeuralNetworks.NeuralNetworkBase{T}"/>
    ///   with LoRA wrapping — NormalOptimizer.SpawnIndividual
    ///   Clone-serialize round-trip throws on LoRA-wrapped layers
    ///   because the frozen base + LoRA delta parameter counts get out
    ///   of sync. Routing through model.Train uses the NN's own
    ///   gradient path which handles LoRA adapters correctly via
    ///   Forward dispatch.</item>
    /// </list>
    /// </remarks>
    private bool UseDirectTrainingPath(IFullModel<T, TInput, TOutput> model)
    {
        // The `model` parameter is the resolved model at the call site
        // (possibly post-wrapping), while the other two clauses read
        // the builder's _model field for the predicates that need the
        // original (non-wrapped) instance (clustering-base check and
        // LoRA-wrapped detection both look at the user-supplied model
        // type, not whatever wrapper is now in `model`). Reviewer
        // (#1368) noted the asymmetry — documented inline so future
        // edits don't accidentally swap `model` ↔ `_model` in one of
        // these clauses without realising the intent (the
        // IParameterizable check follows the wrapped chain; the other
        // two follow the original user choice).
        bool modelLacksParameterizableInit =
            model is not IParameterizable<T, TInput, TOutput> { SupportsParameterInitialization: true };
        bool isClusteringBase = _model is Clustering.Base.ClusteringBase<T>;
        bool isLoraWrappedNeuralNetwork =
            _loraConfiguration is not null && _model is NeuralNetworks.NeuralNetworkBase<T>;
        return modelLacksParameterizableInit || isClusteringBase || isLoraWrappedNeuralNetwork;
    }

    /// <summary>
    /// Carves a 1-sample probe off the training input for LoRA warmup
    /// forwards. Returns the full input unchanged if the type doesn't
    /// expose a recognised slicing pattern — better to do a full forward
    /// than to error out on shape-resolution.
    /// </summary>
    /// <remarks>
    /// <para><b>Layout assumption</b> (review #1368): for Tensor&lt;T&gt;,
    /// the slice loop assumes <see cref="Tensor{T}.GetFlat"/> /
    /// <see cref="Tensor{T}.SetFlat"/> address a contiguous batch-first
    /// row-major buffer (i.e. the first <c>perSample</c> flat positions
    /// are the first sample's elements in row-major order). All
    /// AiDotNet.Tensors tensor allocations satisfy this — the storage
    /// is a contiguous <c>T[]</c> with row-major strides — but if a
    /// future tensor backend exposes non-contiguous views (e.g. stride
    /// tricks for slicing without copy), this loop would silently copy
    /// the wrong elements. <see cref="Tensor{T}.GetFlat"/>'s contract
    /// is "flat index across the contiguous storage in row-major
    /// order" which holds today; revisit if that contract relaxes.</para>
    /// <para><b>Future direction</b>: the proper fix is to eliminate
    /// the warmup forward entirely via a layer-side
    /// <c>TryDeclareShape()</c> oracle that lets lazy-init layers
    /// declare their shapes from constructor / config args without
    /// needing a forward pass. Tracked at
    /// <see href="https://github.com/ooples/AiDotNet/issues/1370">#1370</see>.
    /// Until that ships, this 1-sample slice is the perf-stopgap for the
    /// warmup cost (one row, one forward, one-time at Build).</para>
    /// </remarks>
    private static TInput TrySliceFirstSampleForLoRAWarmup(TInput x)
    {
        // Tensor<T>: take the first sample along axis 0.
        if (x is Tensor<T> tensor && tensor.Shape.Length > 0 && tensor.Shape[0] > 1)
        {
            var sliceShape = new int[tensor.Shape.Length];
            sliceShape[0] = 1;
            for (int i = 1; i < tensor.Shape.Length; i++) sliceShape[i] = tensor.Shape[i];

            int perSample = 1;
            for (int i = 1; i < tensor.Shape.Length; i++) perSample *= tensor.Shape[i];
            var slice = new Tensor<T>(sliceShape);
            // Bulk Span<T>.CopyTo over the first `perSample` elements of
            // the source tensor's storage into the slice tensor's
            // storage. Both are contiguous row-major Span<T> views over
            // managed T[] arrays (AiDotNet.Tensors tensor allocation
            // contract); CopyTo dispatches to the JIT's vectorized
            // memmove. Replaces the prior per-element GetFlat/SetFlat
            // loop, which made two virtual calls per element on the
            // training hot path (review #1368 C6WM9). One call per
            // Build instead of perSample calls.
            //
            // Defense-in-depth (review #1368 C7mpf): verify the storage
            // contract before issuing the bulk copy. If a future
            // AiDotNet.Tensors refactor introduces strided / non-
            // contiguous backing or a smaller Span view, the Slice below
            // would silently copy garbage or throw an opaque
            // ArgumentOutOfRangeException. Debug.Assert catches the
            // contract break in debug builds without the runtime cost
            // in release (the bulk copy is the LoRA warmup's perf hot
            // path).
            // Long-typed product so a large tensor (e.g. [1024, 1024,
            // 64, 64] = 4.3 GiB at fp32) doesn't silently overflow int —
            // the assert below would otherwise fire spuriously on a
            // legitimate tensor whose Data.Span.Length is correct (review
            // #1368 C88RH). The slicing path only copies the first
            // perSample elements anyway, so the assert is purely a
            // contract sanity-check on the FULL backing storage.
            long totalElements = (long)perSample * tensor.Shape[0];
            System.Diagnostics.Debug.Assert(
                tensor.Data.Span.Length >= totalElements,
                $"Tensor<T>.Data.Span ({tensor.Data.Span.Length}) shorter than logical " +
                $"shape ({string.Join("x", tensor.Shape)} = {totalElements}); the row-major " +
                "contiguous-storage contract this slicing relies on no longer holds. " +
                "Update TrySliceFirstSampleForLoRAWarmup before this Span<T>.CopyTo can run.");
            tensor.Data.Span.Slice(0, perSample).CopyTo(slice.Data.Span);
            if (slice is TInput typedSlice) return typedSlice;
        }
        // Fallback: full forward (non-Tensor TInput, or single-sample
        // input where slicing is unnecessary).
        return x;
    }

    // Explicit-interface implementation of IConfiguredView<T,TInput,TOutput>:
    // these accessors exist solely for the integration-test bucket suite to
    // verify post-Configure*() state, and previously sat on the AiModelBuilder
    // internal surface. Moving them behind an explicit interface keeps them
    // off the regular type surface entirely — test code casts to
    // IConfiguredView<T,TInput,TOutput> to read; production callers never
    // see (or accidentally bind against) the accessors (review #1368 C6WRW).
    IOptimizer<T, TInput, TOutput>? AiDotNet.Configuration.IConfiguredView<T, TInput, TOutput>.ConfiguredOptimizer => _optimizer;
    CacheConfig? AiDotNet.Configuration.IConfiguredView<T, TInput, TOutput>.ConfiguredCaching => _cacheConfig;
    AiDotNet.Configuration.InferenceOptimizationConfig? AiDotNet.Configuration.IConfiguredView<T, TInput, TOutput>.ConfiguredInferenceOptimizations => _inferenceOptimizationConfig;
    AiDotNet.Configuration.JitCompilationConfig? AiDotNet.Configuration.IConfiguredView<T, TInput, TOutput>.ConfiguredJitCompilation => _jitCompilationConfig;
    InterpretabilityOptions? AiDotNet.Configuration.IConfiguredView<T, TInput, TOutput>.ConfiguredInterpretability => _interpretabilityOptions;
    Training.Memory.TrainingMemoryConfig? AiDotNet.Configuration.IConfiguredView<T, TInput, TOutput>.ConfiguredMemoryManagement => _memoryConfig;
    AiDotNetLicenseKey? AiDotNet.Configuration.IConfiguredView<T, TInput, TOutput>.ConfiguredLicenseKey => _licenseKey;

    /// <summary>
    /// Creates a new <see cref="AiModelBuilder{T, TInput, TOutput}"/> with configuration loaded from a YAML file.
    /// </summary>
    /// <param name="configFilePath">The path to a YAML configuration file.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instead of configuring everything in C# code, you can write your
    /// configuration in a YAML file and load it automatically. The YAML file sets the base
    /// configuration, and you can still override any setting using the fluent <c>.Configure*()</c>
    /// methods afterwards.
    /// </para>
    /// <para>
    /// <b>Example YAML file (training-recipe.yaml):</b>
    /// </para>
    /// <code>
    /// optimizer:
    ///   type: "Adam"
    ///
    /// quantization:
    ///   mode: Int8
    ///   targetBitWidth: 4
    ///
    /// caching:
    ///   enabled: true
    ///   maxCacheSize: 1000
    ///
    /// jitCompilation:
    ///   enabled: true
    ///   throwOnFailure: false
    /// </code>
    /// <para>
    /// <b>Usage:</b>
    /// </para>
    /// <code>
    /// // YAML base + optional code overrides
    /// var result = await new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;("config.yaml")
    ///     .ConfigureOptimizer(customOptimizer)  // Override YAML optimizer
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when <paramref name="configFilePath"/> is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the YAML file does not exist.</exception>
    public AiModelBuilder(string configFilePath, AiDotNetLicenseKey? licenseKey = null)
    {
        if (string.IsNullOrWhiteSpace(configFilePath))
        {
            throw new ArgumentException("Config file path cannot be null or empty.", nameof(configFilePath));
        }

        _licensing = new AiDotNet.Configuration.AiModelLicensing(licenseKey);
        _licenseKey = _licensing.LicenseKey;

        var fullPath = Path.GetFullPath(configFilePath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("YAML config file not found.", fullPath);
        }

        var config = YamlConfigLoader.LoadFromFile(fullPath);
        YamlConfigApplier<T, TInput, TOutput>.Apply(config, this);
    }

    /// <summary>
    /// Creates a new <see cref="AiModelBuilder{T, TInput, TOutput}"/> with default (empty) configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you want to configure everything via the fluent
    /// <c>.Configure*()</c> methods in C# code.
    /// </para>
    /// </remarks>
    public AiModelBuilder(AiDotNetLicenseKey? licenseKey = null)
    {
        _licensing = new AiDotNet.Configuration.AiModelLicensing(licenseKey);
        _licenseKey = _licensing.LicenseKey;
    }

}
