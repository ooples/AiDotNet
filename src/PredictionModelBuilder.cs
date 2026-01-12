

global using AiDotNet.Agents;
global using AiDotNet.Configuration;
global using AiDotNet.DataProcessor;
global using AiDotNet.Deployment.Configuration;
global using AiDotNet.Diagnostics;
global using AiDotNet.DistributedTraining;
global using AiDotNet.Enums;
global using AiDotNet.Extensions;
global using AiDotNet.FeatureSelectors;
global using AiDotNet.FitDetectors;
global using AiDotNet.FitnessCalculators;
global using AiDotNet.Helpers;
global using AiDotNet.KnowledgeDistillation;
global using AiDotNet.LanguageModels;
global using AiDotNet.LinearAlgebra;
global using AiDotNet.LossFunctions;
global using AiDotNet.MetaLearning;
global using AiDotNet.MixedPrecision;
global using AiDotNet.Models;
global using AiDotNet.Models.Inputs;
global using AiDotNet.Models.Options;
global using AiDotNet.Normalizers;
global using AiDotNet.Optimizers;
global using AiDotNet.OutlierRemoval;
global using AiDotNet.ProgramSynthesis.Interfaces;
global using AiDotNet.ProgramSynthesis.Serving;
global using AiDotNet.PromptEngineering.Chains;
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
global using AiDotNet.Tools;
global using AiDotNet.UncertaintyQuantification.Layers;
using AiDotNet.Augmentation;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.Policies;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Postprocessing;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;

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
/// var builder = new PredictionModelBuilder&lt;double, double[], double&gt;()
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
public partial class PredictionModelBuilder<T, TInput, TOutput> : IPredictionModelBuilder<T, TInput, TOutput>
{
    private IFeatureSelector<T, TInput>? _featureSelector;
    private INormalizer<T, TInput, TOutput>? _normalizer;
    private PreprocessingPipeline<T, TInput, TInput>? _preprocessingPipeline;
    private PostprocessingPipeline<T, TOutput, TOutput>? _postprocessingPipeline;
    private IRegularization<T, TInput, TOutput>? _regularization;
    private IFitnessCalculator<T, TInput, TOutput>? _fitnessCalculator;
    private IFitDetector<T, TInput, TOutput>? _fitDetector;
    private IFullModel<T, TInput, TOutput>? _model;
    private IOptimizer<T, TInput, TOutput>? _optimizer;
    private IDataPreprocessor<T, TInput, TOutput>? _dataPreprocessor;
    private IDataLoader<T>? _dataLoader;
    private IOutlierRemoval<T, TInput, TOutput>? _outlierRemoval;
    private IBiasDetector<T>? _biasDetector;
    private IFairnessEvaluator<T>? _fairnessEvaluator;
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
    private IMetaLearner<T, TInput, TOutput>? _metaLearner;
    private ICommunicationBackend<T>? _distributedBackend;
    private DistributedStrategy _distributedStrategy = DistributedStrategy.DDP;
    private IShardingConfiguration<T>? _distributedConfiguration;
    private IModelEvaluator<T, TInput, TOutput>? _modelEvaluator;
    private ICrossValidator<T, TInput, TOutput>? _crossValidator;
    private AgentConfiguration<T>? _agentConfig;
    private AgentAssistanceOptions _agentOptions = AgentAssistanceOptions.Default;
    private KnowledgeDistillationOptions<T, TInput, TOutput>? _knowledgeDistillationOptions;
    private MixedPrecisionConfig? _mixedPrecisionConfig;
    private AiDotNet.Configuration.JitCompilationConfig? _jitCompilationConfig;
    private AiDotNet.Configuration.InferenceOptimizationConfig? _inferenceOptimizationConfig;
    private RLTrainingOptions<T>? _rlOptions;
    private IAutoMLModel<T, TInput, TOutput>? _autoMLModel;
    private AutoMLOptions<T, TInput, TOutput>? _autoMLOptions;

    // Curriculum learning configuration
    private CurriculumLearningOptions<T, TInput, TOutput>? _curriculumLearningOptions;

    // Unified augmentation configuration
    private Augmentation.AugmentationConfig? _augmentationConfig;

    // Self-supervised learning configuration
    private SelfSupervisedLearning.SSLConfig? _sslConfig;

    // Federated learning configuration (facade-first: orchestration is internal)
    private FederatedLearningOptions? _federatedLearningOptions;
    private IAggregationStrategy<IFullModel<T, TInput, TOutput>>? _federatedAggregationStrategy;
    private IClientSelectionStrategy? _federatedClientSelectionStrategy;
    private IFederatedServerOptimizer<T>? _federatedServerOptimizer;
    private IFederatedHeterogeneityCorrection<T>? _federatedHeterogeneityCorrection;
    private IHomomorphicEncryptionProvider<T>? _federatedHomomorphicEncryptionProvider;

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

    // Tokenization configuration
    private ITokenizer? _tokenizer;
    private TokenizationConfig? _tokenizationConfig;

    // Program synthesis Serving configuration
    private ProgramSynthesisServingClientOptions? _programSynthesisServingClientOptions;
    private IProgramSynthesisServingClient? _programSynthesisServingClient;
    private IFullModel<T, Tensor<T>, Tensor<T>>? _programSynthesisModel;

    // Prompt engineering configuration
    private IPromptTemplate? _promptTemplate;
    private IChain<string, string>? _promptChain;
    private IPromptOptimizer<T>? _promptOptimizer;
    private IFewShotExampleSelector<T>? _fewShotExampleSelector;
    private IPromptAnalyzer? _promptAnalyzer;
    private IPromptCompressor? _promptCompressor;

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
    /// Configures which features (input variables) should be used in the model.
    /// </summary>
    /// <param name="selector">The feature selection strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Sometimes, not all of your data is useful for making predictions.
    /// Feature selection helps pick out which parts of your data are most important.
    /// For example, when predicting house prices, the number of bedrooms might be important,
    /// but the house's street number probably isn't.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFeatureSelector(IFeatureSelector<T, TInput> selector)
    {
        _featureSelector = selector;
        return this;
    }

    /// <summary>
    /// Configures how the input data should be normalized (scaled).
    /// </summary>
    /// <param name="normalizer">The normalization strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Normalization makes sure all your data is on a similar scale.
    /// For example, if you have data about people's ages (0-100) and incomes ($0-$1,000,000),
    /// normalization might scale both to ranges like 0-1 so the model doesn't think
    /// income is 10,000 times more important than age just because the numbers are bigger.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureNormalizer(INormalizer<T, TInput, TOutput> normalizer)
    {
        _normalizer = normalizer;
        return this;
    }

    /// <summary>
    /// Configures a preprocessing pipeline using a builder action.
    /// </summary>
    /// <param name="pipelineBuilder">An action that configures the preprocessing pipeline.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// The new preprocessing pipeline replaces the legacy normalizer system with a more flexible,
    /// composable approach supporting scalers, encoders, imputers, and feature generators.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you chain multiple preprocessing steps together:
    /// <code>
    /// builder.ConfigurePreprocessing(pipeline => pipeline
    ///     .Add(new SimpleImputer&lt;double&gt;(ImputationStrategy.Mean))
    ///     .Add(new StandardScaler&lt;double&gt;())
    ///     .Add(new PolynomialFeatures&lt;double&gt;(degree: 2)));
    /// </code>
    /// The pipeline will apply these transformations in order during training,
    /// and remember them for predictions on new data.
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigurePreprocessing(
        Action<PreprocessingPipeline<T, TInput, TInput>>? pipelineBuilder = null)
    {
        _preprocessingPipeline = new PreprocessingPipeline<T, TInput, TInput>();

        if (pipelineBuilder is not null)
        {
            pipelineBuilder(_preprocessingPipeline);
        }
        else
        {
            // Industry standard AutoML-style defaults:
            // 1. Handle missing values with mean imputation (most common strategy)
            // 2. Standardize features to zero mean and unit variance
            // 3. These are the minimum recommended preprocessing steps for most ML algorithms
            _preprocessingPipeline.Add((IDataTransformer<T, TInput, TInput>)(object)
                new SimpleImputer<T>(ImputationStrategy.Mean));
            _preprocessingPipeline.Add((IDataTransformer<T, TInput, TInput>)(object)
                new StandardScaler<T>());
        }

        // Set global registry so all models automatically use this pipeline
        PreprocessingRegistry<T, TInput>.Current = _preprocessingPipeline;

        return this;
    }

    /// <summary>
    /// Configures a single preprocessing transformer.
    /// </summary>
    /// <param name="transformer">The transformer to use for preprocessing.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Use this overload when you only need a single transformer. For multiple transformers,
    /// use the overload that takes an Action to build a pipeline.
    /// </para>
    /// <para><b>For Beginners:</b> This is a simple way to add just one preprocessing step:
    /// <code>
    /// builder.ConfigurePreprocessing(new StandardScaler&lt;double&gt;());
    /// </code>
    /// If you need multiple steps, use the pipeline builder overload instead.
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigurePreprocessing(
        IDataTransformer<T, TInput, TInput>? transformer = null)
    {
        _preprocessingPipeline = new PreprocessingPipeline<T, TInput, TInput>();

        if (transformer is not null)
        {
            _preprocessingPipeline.Add(transformer);
        }
        else
        {
            // Industry standard AutoML-style defaults:
            // 1. Handle missing values with mean imputation (most common strategy)
            // 2. Standardize features to zero mean and unit variance
            // 3. These are the minimum recommended preprocessing steps for most ML algorithms
            _preprocessingPipeline.Add((IDataTransformer<T, TInput, TInput>)(object)
                new SimpleImputer<T>(ImputationStrategy.Mean));
            _preprocessingPipeline.Add((IDataTransformer<T, TInput, TInput>)(object)
                new StandardScaler<T>());
        }

        // Set global registry so all models automatically use this pipeline
        PreprocessingRegistry<T, TInput>.Current = _preprocessingPipeline;

        return this;
    }

    /// <summary>
    /// Configures a pre-built preprocessing pipeline.
    /// </summary>
    /// <param name="pipeline">The preprocessing pipeline to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Use this overload when you have a pre-configured pipeline that you want to reuse
    /// across multiple model builders.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when you've already created a pipeline elsewhere:
    /// <code>
    /// var myPipeline = new PreprocessingPipeline&lt;double, Matrix&lt;double&gt;, Matrix&lt;double&gt;&gt;()
    ///     .Add(new StandardScaler&lt;double&gt;());
    ///
    /// builder.ConfigurePreprocessing(myPipeline);
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigurePreprocessing(
        PreprocessingPipeline<T, TInput, TInput>? pipeline = null)
    {
        if (pipeline is not null)
        {
            _preprocessingPipeline = pipeline;
        }
        else
        {
            // Industry standard AutoML-style defaults:
            // 1. Handle missing values with mean imputation (most common strategy)
            // 2. Standardize features to zero mean and unit variance
            // 3. These are the minimum recommended preprocessing steps for most ML algorithms
            _preprocessingPipeline = new PreprocessingPipeline<T, TInput, TInput>();
            _preprocessingPipeline.Add((IDataTransformer<T, TInput, TInput>)(object)
                new SimpleImputer<T>(ImputationStrategy.Mean));
            _preprocessingPipeline.Add((IDataTransformer<T, TInput, TInput>)(object)
                new StandardScaler<T>());
        }

        // Set global registry so all models automatically use this pipeline
        PreprocessingRegistry<T, TInput>.Current = _preprocessingPipeline;

        return this;
    }

    /// <summary>
    /// Configures the output postprocessing pipeline for the model using a fluent builder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The postprocessing pipeline transforms model outputs into the desired format.
    /// This includes operations like applying softmax, decoding labels, filtering results,
    /// and formatting outputs for specific use cases.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you chain multiple postprocessing steps together:
    /// <code>
    /// builder.ConfigurePostprocessing(pipeline => pipeline
    ///     .Add(new SoftmaxTransformer&lt;double&gt;())
    ///     .Add(new LabelDecoder&lt;double&gt;(labels)));
    /// </code>
    /// The pipeline will apply these transformations in order to model outputs,
    /// and can reverse them if needed.
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigurePostprocessing(
        Action<PostprocessingPipeline<T, TOutput, TOutput>>? pipelineBuilder = null)
    {
        _postprocessingPipeline = new PostprocessingPipeline<T, TOutput, TOutput>();

        if (pipelineBuilder is not null)
        {
            pipelineBuilder(_postprocessingPipeline);
        }
        // Note: Unlike preprocessing, postprocessing doesn't have universal defaults
        // because the appropriate postprocessing depends heavily on the model type
        // (classification vs regression vs generation, etc.)

        // Set global registry so all models automatically use this pipeline
        PostprocessingRegistry<T, TOutput>.Current = _postprocessingPipeline;

        return this;
    }

    /// <summary>
    /// Configures the output postprocessing using a single transformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this overload when you only need a single postprocessing transformer.
    /// For multiple transformers, use the overload that takes an Action to build a pipeline.
    /// </para>
    /// <para><b>For Beginners:</b> This is a simple way to add just one postprocessing step:
    /// <code>
    /// builder.ConfigurePostprocessing(new SoftmaxTransformer&lt;double&gt;());
    /// </code>
    /// If you need multiple steps, use the pipeline builder overload instead.
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigurePostprocessing(
        IDataTransformer<T, TOutput, TOutput>? transformer = null)
    {
        _postprocessingPipeline = new PostprocessingPipeline<T, TOutput, TOutput>();

        if (transformer is not null)
        {
            _postprocessingPipeline.Add(transformer);
        }
        // Note: Unlike preprocessing, postprocessing doesn't have universal defaults
        // because the appropriate postprocessing depends heavily on the model type

        // Set global registry so all models automatically use this pipeline
        PostprocessingRegistry<T, TOutput>.Current = _postprocessingPipeline;

        return this;
    }

    /// <summary>
    /// Configures the output postprocessing using an existing pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this overload when you have a pre-configured PostprocessingPipeline instance.
    /// If null is passed, an empty postprocessing pipeline will be created.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when you've already created a pipeline elsewhere:
    /// <code>
    /// var myPipeline = new PostprocessingPipeline&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .Add(new SoftmaxTransformer&lt;double&gt;());
    ///
    /// builder.ConfigurePostprocessing(myPipeline);
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigurePostprocessing(
        PostprocessingPipeline<T, TOutput, TOutput>? pipeline = null)
    {
        if (pipeline is not null)
        {
            _postprocessingPipeline = pipeline;
        }
        else
        {
            // Create empty pipeline - no default postprocessing
            // because appropriate postprocessing depends on model type
            _postprocessingPipeline = new PostprocessingPipeline<T, TOutput, TOutput>();
        }

        // Set global registry so all models automatically use this pipeline
        PostprocessingRegistry<T, TOutput>.Current = _postprocessingPipeline;

        return this;
    }

    /// <summary>
    /// Configures regularization to prevent overfitting in the model.
    /// </summary>
    /// <param name="regularization">The regularization strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Regularization helps prevent your model from "memorizing" the training data
    /// instead of learning general patterns. It's like teaching a student to understand the concepts
    /// rather than just memorizing answers to specific questions. This helps the model perform better
    /// on new, unseen data.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureRegularization(IRegularization<T, TInput, TOutput> regularization)
    {
        _regularization = regularization;
        return this;
    }

    /// <summary>
    /// Configures how to measure the model's performance.
    /// </summary>
    /// <param name="calculator">The fitness calculation strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This determines how we score how well our model is doing.
    /// Different problems might need different scoring methods. For example, when predicting house prices,
    /// we might care about the average error in dollars, but when predicting if an email is spam,
    /// we might care more about the percentage of emails correctly classified.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFitnessCalculator(IFitnessCalculator<T, TInput, TOutput> calculator)
    {
        _fitnessCalculator = calculator;
        return this;
    }

    /// <summary>
    /// Configures how to detect if the model is overfitting or underfitting.
    /// </summary>
    /// <param name="detector">The fit detection strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This helps detect if your model is learning too much from the training data
    /// (overfitting) or not learning enough (underfitting). It's like having a teacher who can tell
    /// if a student is just memorizing answers or not studying enough.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFitDetector(IFitDetector<T, TInput, TOutput> detector)
    {
        _fitDetector = detector;
        return this;
    }

    /// <summary>
    /// Configures the core algorithm to use for predictions.
    /// </summary>
    /// <param name="model">The prediction algorithm to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is the main "brain" of your AI model - the algorithm that will
    /// learn patterns from your data and make predictions. Different algorithms work better for
    /// different types of problems, so you can choose the one that fits your needs.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureModel(IFullModel<T, TInput, TOutput> model)
    {
        _model = model;
        return this;
    }

    /// <summary>
    /// Configures the optimization algorithm to find the best model parameters.
    /// </summary>
    /// <param name="optimizationAlgorithm">The optimization algorithm to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The optimizer helps find the best settings for your model.
    /// It's like having someone adjust the knobs on a radio to get the clearest signal.
    /// The optimizer tries different settings and keeps the ones that work best.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureOptimizer(IOptimizer<T, TInput, TOutput> optimizationAlgorithm)
    {
        _optimizer = optimizationAlgorithm;
        return this;
    }

    /// <summary>
    /// Enables federated learning training using the provided options.
    /// </summary>
    /// <param name="options">Federated learning configuration options.</param>
    /// <param name="aggregationStrategy">Optional aggregation strategy override (null uses defaults based on options).</param>
    /// <param name="clientSelectionStrategy">Optional client selection strategy override (null uses defaults based on options).</param>
    /// <param name="serverOptimizer">Optional server-side optimizer override (null uses defaults based on options).</param>
    /// <returns>This builder instance for method chaining.</returns>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFederatedLearning(
        FederatedLearningOptions options,
        IAggregationStrategy<IFullModel<T, TInput, TOutput>>? aggregationStrategy = null,
        IClientSelectionStrategy? clientSelectionStrategy = null,
        IFederatedServerOptimizer<T>? serverOptimizer = null,
        IFederatedHeterogeneityCorrection<T>? heterogeneityCorrection = null,
        IHomomorphicEncryptionProvider<T>? homomorphicEncryptionProvider = null)
    {
        _federatedLearningOptions = options ?? throw new ArgumentNullException(nameof(options));
        _federatedAggregationStrategy = aggregationStrategy;
        _federatedClientSelectionStrategy = clientSelectionStrategy;
        _federatedServerOptimizer = serverOptimizer;
        _federatedHeterogeneityCorrection = heterogeneityCorrection;
        _federatedHomomorphicEncryptionProvider = homomorphicEncryptionProvider;
        return this;
    }

    /// <summary>
    /// Enables mixed-precision training with optional configuration.
    /// </summary>
    /// <param name="config">Mixed-precision configuration (optional, uses defaults if null).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mixed-precision training uses a combination of 16-bit (FP16) and 32-bit (FP32)
    /// floating-point numbers during training. This provides:
    ///
    /// Benefits:
    /// - **2-3x faster training** on modern GPUs with Tensor Cores (V100, A100, RTX 3000+)
    /// - **~50% memory reduction** allows training larger models or using bigger batches
    /// - **Maintained accuracy** through careful precision management and loss scaling
    ///
    /// <b>Requirements:</b>
    /// Mixed-precision training has specific technical requirements:
    ///
    /// 1. **Type Constraint: float only**
    ///    - Type parameter T must be float (FP32)
    ///    - Cannot use double, decimal, or integer types
    ///    - Reason: Mixed-precision converts between FP32 (float) and FP16 (Half) representations
    ///
    /// 2. **Gradient-Based Optimizers Only**
    ///    - Requires optimizers that compute gradients (SGD, Adam, RMSProp, etc.)
    ///    - Does NOT work with non-gradient methods (genetic algorithms, random search, Bayesian optimization)
    ///    - Reason: Core techniques require gradient computation:
    ///      * Loss scaling: Multiplies gradients to prevent underflow in FP16
    ///      * Master weights: FP32 copy for accurate incremental gradient updates
    ///      * Gradient accumulation: Accumulates tiny updates in FP32 precision
    ///
    /// 3. **Neural Networks (Recommended)**
    ///    - Best suited for neural networks with large parameter counts
    ///    - Can technically work with other gradient-based models, but benefits are minimal
    ///    - Reason: Neural networks benefit from:
    ///      * GPU Tensor Core acceleration for matrix operations (2-3x speedup)
    ///      * Massive parameter counts (millions/billions) where 50% memory reduction matters
    ///      * Robustness to small FP16 precision losses during training
    ///
    /// When to use:
    /// - ✅ Neural networks trained with gradient-based optimizers (SGD, Adam, etc.)
    /// - ✅ Large models (>100M parameters) on modern GPUs with Tensor Cores
    /// - ✅ Memory-constrained scenarios where you need bigger batch sizes
    /// - ❌ Non-gradient optimizers (evolutionary algorithms, random search)
    /// - ❌ CPU-only training (minimal benefit without Tensor Cores)
    /// - ❌ Very small models (<1M parameters) where memory isn't a concern
    /// - ❌ Non-float types (double, decimal, int)
    ///
    /// <b>Technical Details:</b>
    /// Mixed-precision maintains two copies of model parameters:
    /// - Working weights (FP16): Used for forward/backward passes to save memory and increase speed
    /// - Master weights (FP32): Used for optimizer updates to maintain precision over many iterations
    ///
    /// Loss scaling prevents gradient underflow by multiplying the loss by a large factor (e.g., 65536)
    /// before backpropagation, then dividing gradients by the same factor before applying updates.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Enable with default settings (recommended)
    /// var result = await new PredictionModelBuilder&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;()
    ///     .ConfigureModel(network)
    ///     .ConfigureOptimizer(optimizer)
    ///     .ConfigureMixedPrecision()  // Enable mixed-precision
    ///     .BuildAsync();
    ///
    /// // Or with custom configuration
    /// builder.ConfigureMixedPrecision(MixedPrecisionConfig.Conservative());
    /// </code>
    /// </example>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureMixedPrecision(MixedPrecisionConfig? config = null)
    {
        _mixedPrecisionConfig = config ?? new MixedPrecisionConfig();
        return this;
    }

    /// <summary>
    /// Configures advanced reasoning capabilities for the model using Chain-of-Thought, Tree-of-Thoughts, and Self-Consistency strategies.
    /// </summary>
    /// <param name="config">The reasoning configuration (optional, uses defaults if null).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Reasoning capabilities make AI models "think step by step" instead of
    /// giving quick answers that might be wrong. Just like a student showing their work on a math test,
    /// reasoning strategies help the AI:
    /// - Break down complex problems into manageable steps
    /// - Explore multiple solution approaches
    /// - Verify and refine its answers
    /// - Provide transparent, explainable reasoning
    ///
    /// After building your model, use the reasoning methods on PredictionModelResult:
    /// - ReasonAsync(): Solve problems with configurable reasoning strategies
    /// - QuickReasonAsync(): Fast answers for simple problems
    /// - DeepReasonAsync(): Thorough analysis for complex problems
    ///
    /// Example:
    /// <code>
    /// // Configure reasoning during model building
    /// var agentConfig = new AgentConfiguration&lt;double&gt;
    /// {
    ///     ApiKey = "sk-...",
    ///     Provider = LLMProvider.OpenAI,
    ///     IsEnabled = true
    /// };
    ///
    /// var result = await new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureAgentAssistance(agentConfig)
    ///     .ConfigureReasoning()
    ///     .BuildAsync();
    ///
    /// // Use reasoning on the trained model
    /// var reasoningResult = await result.ReasonAsync(
    ///     "Explain why this prediction was made and what factors contributed most?",
    ///     ReasoningMode.ChainOfThought
    /// );
    /// Console.WriteLine(reasoningResult.FinalAnswer);
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureReasoning(ReasoningConfig? config = null)
    {
        _reasoningConfig = config ?? new ReasoningConfig();
        return this;
    }

    /// <summary>
    /// Configures JIT (Just-In-Time) compilation for accelerated model inference.
    /// </summary>
    /// <param name="config">The JIT compilation configuration. If null, uses default settings with JIT enabled.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// JIT compilation converts your model's computation graph into optimized native code, providing
    /// significant performance improvements (5-10x faster) for inference. The compilation happens once
    /// during model building, then the optimized code is reused for all predictions.
    /// </para>
    /// <para><b>For Beginners:</b> JIT compilation makes your model's predictions much faster by
    /// "pre-compiling" the calculations into optimized code before you start using it.
    ///
    /// <b>Benefits:</b>
    /// - 2-3x faster for simple operations
    /// - 5-10x faster for complex models
    /// - Automatic operation fusion and optimization
    /// - Near-zero overhead for cached compilations
    ///
    /// <b>When to use JIT:</b>
    /// - Production inference (maximize speed)
    /// - Batch processing (repeated predictions)
    /// - Large or complex models (more optimization opportunities)
    ///
    /// <b>When NOT to use JIT:</b>
    /// - Training (JIT is for inference only)
    /// - Very simple models (compilation overhead exceeds benefits)
    /// - Models with dynamic structure
    ///
    /// <b>Important:</b> Your model must implement IJitCompilable to support JIT compilation.
    /// Currently, models built with TensorOperations computation graphs are supported.
    /// Neural networks using layer-based architecture will be supported in a future update.
    ///
    /// <b>Example usage:</b>
    /// <code>
    /// var result = await new PredictionModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
    ///     .ConfigureModel(myModel)
    ///     .ConfigureJitCompilation(new JitCompilationConfig
    ///     {
    ///         Enabled = true,
    ///         CompilerOptions = new JitCompilerOptions
    ///         {
    ///             EnableOperationFusion = true,     // Biggest performance gain
    ///             EnableDeadCodeElimination = true,
    ///             EnableConstantFolding = true,
    ///             EnableCaching = true
    ///         },
    ///         ThrowOnFailure = false  // Graceful fallback if JIT not supported
    ///     })
    ///     .BuildAsync();
    ///
    /// // Predictions now use JIT-compiled code (5-10x faster!)
    /// var prediction = result.Predict(newData);
    /// </code>
    ///
    /// <b>Simple usage (uses defaults):</b>
    /// <code>
    /// var result = await new PredictionModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
    ///     .ConfigureModel(myModel)
    ///     .ConfigureJitCompilation()  // Enables JIT with default settings
    ///     .BuildAsync();
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureJitCompilation(AiDotNet.Configuration.JitCompilationConfig? config = null)
    {
        _jitCompilationConfig = config ?? new AiDotNet.Configuration.JitCompilationConfig { Enabled = true };
        return this;
    }

    /// <summary>
    /// Configures inference-time optimizations for faster predictions.
    /// </summary>
    /// <param name="config">Inference optimization configuration (optional, uses defaults if null).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Inference optimization makes your model's predictions faster and more efficient.
    ///
    /// Key features enabled:
    /// - <b>KV Cache:</b> Speeds up transformer/attention models by 2-10x
    /// - <b>Batching:</b> Groups predictions for higher throughput
    /// - <b>Speculative Decoding:</b> Speeds up text generation by 1.5-3x
    ///
    /// Example:
    /// <code>
    /// var result = await new PredictionModelBuilder&lt;double, ...&gt;()
    ///     .ConfigureModel(myModel)
    ///     .ConfigureInferenceOptimizations()  // Uses sensible defaults
    ///     .BuildAsync();
    ///
    /// // Or with custom settings:
    /// var config = new InferenceOptimizationConfig
    /// {
    ///     EnableKVCache = true,
    ///     MaxBatchSize = 64,
    ///     EnableSpeculativeDecoding = true
    /// };
    /// 
    /// var result = await builder
    ///     .ConfigureInferenceOptimizations(config)
    ///     .BuildAsync();
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureInferenceOptimizations(AiDotNet.Configuration.InferenceOptimizationConfig? config = null)
    {
        _inferenceOptimizationConfig = config ?? AiDotNet.Configuration.InferenceOptimizationConfig.Default;
        return this;
    }

    // Uncertainty quantification configuration lives in PredictionModelBuilder.UncertaintyQuantification.cs to keep this file focused.

    /// <summary>
    /// Enables GPU acceleration for training and inference with optional configuration.
    /// </summary>
    /// <param name="config">GPU acceleration configuration (optional, uses defaults if null).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GPU acceleration makes your model train **10-100x faster** on large datasets
    /// by using your computer's graphics card (GPU) for parallel computation. This is one of the most
    /// impactful optimizations you can make!
    ///
    /// Benefits:
    /// - **10-100x faster training** for large neural networks and matrix operations
    /// - **Automatic optimization** - GPU is only used when beneficial
    /// - **Zero code changes** - works with existing models transparently
    /// - **Cross-platform** - supports NVIDIA (CUDA), AMD/Intel (OpenCL), and CPU fallback
    ///
    /// <b>Requirements:</b>
    ///
    /// 1. **GPU Support (Recommended but Optional)**
    ///    - Works best with NVIDIA GPUs (CUDA support)
    ///    - Also supports AMD/Intel GPUs via OpenCL
    ///    - Automatically falls back to CPU if GPU unavailable
    ///    - No GPU? No problem - just slower performance
    ///
    /// 2. **Works with All Models**
    ///    - Neural networks get the biggest speedup (10-100x)
    ///    - Other gradient-based models also benefit
    ///    - Automatically decides which operations benefit from GPU
    ///
    /// 3. **Type Compatibility**
    ///    - Recommended with T = float for best performance
    ///    - Supports other numeric types with some overhead
    ///
    /// When to use:
    /// - ✅ Training neural networks (massive speedup!)
    /// - ✅ Large datasets (>10,000 samples)
    /// - ✅ Matrix-heavy operations (linear regression, etc.)
    /// - ✅ When you have a GPU available
    /// - ⚠️ Small datasets (<1,000 samples) - minimal benefit
    /// - ⚠️ Simple models with no matrix operations - no benefit
    ///
    /// <b>Performance Expectations:</b>
    ///
    /// Operation speedups (depends on GPU and data size):
    /// - Large matrix multiplication: **50-100x faster**
    /// - Neural network training: **10-50x faster**
    /// - Element-wise operations: **5-20x faster**
    /// - Small operations (<100K elements): Similar or slower (transfer overhead)
    ///
    /// The system automatically uses CPU for small operations and GPU for large ones,
    /// so you get optimal performance without any manual tuning!
    ///
    /// <b>Memory Considerations:</b>
    /// - GPU has separate memory from CPU (typically 4-24GB)
    /// - Data is automatically transferred between CPU ↔ GPU as needed
    /// - Transfers are minimized by batching operations
    /// - If GPU runs out of memory, automatically falls back to CPU
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Enable with default settings (recommended for most cases)
    /// var result = await new PredictionModelBuilder&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;()
    ///     .ConfigureModel(network)
    ///     .ConfigureOptimizer(optimizer)
    ///     .ConfigureGpuAcceleration()  // Enable GPU acceleration with sensible defaults
    ///     .BuildAsync();
    ///
    /// // Or with custom configuration for high-end GPUs
    /// builder.ConfigureGpuAcceleration(new GpuAccelerationConfig
    /// {
    ///     UsageLevel = GpuUsageLevel.Aggressive,
    ///     DeviceType = GpuDeviceType.CUDA
    /// });
    ///
    /// // Or conservative settings for older/slower GPUs
    /// builder.ConfigureGpuAcceleration(new GpuAccelerationConfig
    /// {
    ///     UsageLevel = GpuUsageLevel.Conservative
    /// });
    ///
    /// // Or force CPU-only (for debugging or deployment to CPU servers)
    /// builder.ConfigureGpuAcceleration(new GpuAccelerationConfig
    /// {
    ///     UsageLevel = GpuUsageLevel.AlwaysCpu
    /// });
    /// </code>
    /// </example>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureGpuAcceleration(GpuAccelerationConfig? config = null)
    {
        _gpuAccelerationConfig = config ?? new GpuAccelerationConfig();
        return this;
    }

    /// <summary>
    /// Configures how the data should be preprocessed before training.
    /// </summary>
    /// <param name="dataPreprocessor">The data preprocessing strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Data preprocessing cleans and prepares your raw data before feeding it to the model.
    /// It's like washing and cutting vegetables before cooking. This might include handling missing values,
    /// converting text to numbers, or combining related features.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureDataPreprocessor(IDataPreprocessor<T, TInput, TOutput> dataPreprocessor)
    {
        _dataPreprocessor = dataPreprocessor;
        return this;
    }

    /// <summary>
    /// Configures the data loader for providing training data.
    /// </summary>
    /// <param name="dataLoader">The data loader that provides training data.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A data loader handles loading your data from various sources
    /// (files, databases, memory, URLs) and provides it in a format suitable for model training.
    ///
    /// You can use:
    /// - IInputOutputDataLoader for standard supervised learning (features + labels)
    /// - IGraphDataLoader for graph neural networks
    /// - IEpisodicDataLoader for meta-learning
    ///
    /// Example:
    /// <code>
    /// // Simple in-memory data
    /// var loader = DataLoaders.FromArrays(features, labels);
    ///
    /// // Or graph data
    /// var graphLoader = new CitationNetworkLoader("cora");
    ///
    /// var result = await builder
    ///     .ConfigureDataLoader(loader)
    ///     .ConfigureModel(model)
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureDataLoader(IDataLoader<T> dataLoader)
    {
        _dataLoader = dataLoader;
        return this;
    }

    /// <summary>
    /// Configures how to detect and handle outliers in the data.
    /// </summary>
    /// <param name="outlierRemoval">The outlier removal strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Outliers are unusual data points that are very different from the rest of your data.
    /// For example, if you're analyzing house prices and most are between $100,000-$500,000,
    /// a $10,000,000 mansion would be an outlier. These unusual points can sometimes confuse the model,
    /// so we might want to handle them specially.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureOutlierRemoval(IOutlierRemoval<T, TInput, TOutput> outlierRemoval)
    {
        _outlierRemoval = outlierRemoval;
        return this;
    }

    /// <summary>
    /// Builds a predictive model using data from ConfigureDataLoader() or meta-learning from ConfigureMetaLearning().
    /// </summary>
    /// <returns>A task that represents the asynchronous operation, containing the trained model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when neither ConfigureDataLoader() nor ConfigureMetaLearning() has been called.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> Use this method when you've configured either:
    /// - A data loader (via ConfigureDataLoader) - the loader provides the training data
    /// - Meta-learning (via ConfigureMetaLearning) - trains your model to learn NEW tasks quickly
    ///
    /// **Data Loader Path**:
    /// - LoadAsync() is called on the data loader
    /// - Features and Labels are extracted from the loader
    /// - Training proceeds using the loaded data
    ///
    /// **Meta-Learning Path**:
    /// - Trains a model that can quickly adapt to new tasks
    /// - Uses episodic data from the meta-learner configuration
    ///
    /// Example with data loader:
    /// <code>
    /// var result = await new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureDataLoader(DataLoaders.FromArrays(features, labels))
    ///     .ConfigureModel(model)
    ///     .BuildAsync();
    /// </code>
    ///
    /// Example with meta-learning:
    /// <code>
    /// var result = await new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureMetaLearning(metaLearner)
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    public async Task<PredictionModelResult<T, TInput, TOutput>> BuildAsync()
    {
        PredictionModelResult<T, TInput, TOutput> result;

        // RL TRAINING PATH - check if RL options are configured with an environment
        if (_rlOptions?.Environment is not null)
        {
            // Use episodes from options (default: 1000)
            int episodes = _rlOptions.Episodes;
            bool verbose = _rlOptions.LogFrequency > 0;
            result = await BuildRLInternalAsync(episodes, verbose);
            await RunBenchmarksIfConfiguredAsync(result).ConfigureAwait(false);
            return result;
        }

        // DATA LOADER PATH - check if data loader is configured and provides input/output data
        if (_dataLoader is IInputOutputDataLoader<T, TInput, TOutput> inputOutputLoader)
        {
            // Load data if not already loaded
            if (!_dataLoader.IsLoaded)
            {
                await _dataLoader.LoadAsync();
            }

            // Get features and labels from the typed loader
            var features = inputOutputLoader.Features;
            var labels = inputOutputLoader.Labels;

            // Delegate to the internal supervised training method
            result = await BuildSupervisedInternalAsync(features, labels);
            await RunBenchmarksIfConfiguredAsync(result).ConfigureAwait(false);
            return result;
        }

        // STREAMING DATA LOADER PATH - check if data loader is a streaming loader
        if (_dataLoader is IStreamingDataLoader<T, TInput, TOutput> streamingLoader)
        {
            // Load/prepare the streaming loader if not already loaded
            if (!_dataLoader.IsLoaded)
            {
                await _dataLoader.LoadAsync();
            }

            // True streaming training - train on batches without materializing all data
            result = await BuildStreamingSupervisedAsync(streamingLoader);
            await RunBenchmarksIfConfiguredAsync(result).ConfigureAwait(false);
            return result;
        }

        // META-LEARNING PATH - check if meta-learner is configured
        if (_metaLearner is not null)
        {
            result = BuildMetaLearningInternalAsync();
            await RunBenchmarksIfConfiguredAsync(result).ConfigureAwait(false);
            return result;
        }

        // PROGRAM SYNTHESIS INFERENCE PATH - allow inference-only builds when a code model is configured.
        // This supports code-task workflows that do not require a training dataset, while keeping other
        // training paths explicit via ConfigureDataLoader/ConfigureReinforcementLearning/ConfigureMetaLearning.
        if (_programSynthesisModel is not null && _model is not null)
        {
            return BuildProgramSynthesisInferenceOnlyResult();
        }

        // No training path configured
        throw new InvalidOperationException(
            "BuildAsync() requires one of the following to be configured first:\n" +
            "- ConfigureReinforcementLearning() for RL training\n" +
            "- ConfigureDataLoader() for supervised learning\n" +
            "- ConfigureMetaLearning() for meta-learning\n" +
            "For supervised learning, configure a data loader via ConfigureDataLoader() and then call BuildAsync().");
    }

    private PredictionModelResult<T, TInput, TOutput> BuildProgramSynthesisInferenceOnlyResult()
    {
        // Ensure inference-only builds still honor configured GPU acceleration.
        ApplyGpuConfiguration();

        var optimizationResult = new OptimizationResult<T, TInput, TOutput>
        {
            BestSolution = _model!
        };

        var deploymentConfig = DeploymentConfiguration.Create(
            _quantizationConfig,
            _cacheConfig,
            _versioningConfig,
            _abTestingConfig,
            _telemetryConfig,
            _exportConfig,
            _gpuAccelerationConfig,
            _compressionConfig,
            _profilingConfig);

        var options = new PredictionModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = optimizationResult,
            NormalizationInfo = new NormalizationInfo<T, TInput, TOutput> { Normalizer = new NoNormalizer<T, TInput, TOutput>() },
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            ProgramSynthesisModel = _programSynthesisModel,
            ProgramSynthesisServingClient = _programSynthesisServingClient,
            ProgramSynthesisServingClientOptions = _programSynthesisServingClientOptions,
            InferenceOptimizationConfig = _inferenceOptimizationConfig,
            AugmentationConfig = _augmentationConfig,
            ReasoningConfig = _reasoningConfig,
            DeploymentConfiguration = deploymentConfig,
            BiasDetector = _biasDetector,
            FairnessEvaluator = _fairnessEvaluator,
            RagRetriever = _ragRetriever,
            RagReranker = _ragReranker,
            RagGenerator = _ragGenerator,
            QueryProcessors = _queryProcessors,
            KnowledgeGraph = _knowledgeGraph,
            GraphStore = _graphStore,
            HybridGraphRetriever = _hybridGraphRetriever,
            AgentConfig = _agentConfig,
            PromptTemplate = _promptTemplate,
            PromptChain = _promptChain,
            PromptOptimizer = _promptOptimizer,
            FewShotExampleSelector = _fewShotExampleSelector,
            PromptAnalyzer = _promptAnalyzer,
            PromptCompressor = _promptCompressor,
            MemoryConfig = _memoryConfig
        };

        return new PredictionModelResult<T, TInput, TOutput>(options);
    }

    private Task RunBenchmarksIfConfiguredAsync(PredictionModelResult<T, TInput, TOutput> result)
    {
        if (_benchmarkingOptions is null || _benchmarkingOptions.Suites is null || _benchmarkingOptions.Suites.Length == 0)
        {
            return Task.CompletedTask;
        }

        return result.EvaluateBenchmarksAsync(_benchmarkingOptions);
    }

    /// <summary>
    /// Performs true streaming supervised training without materializing all data in memory.
    /// </summary>
    /// <param name="streamingLoader">The streaming data loader to train from.</param>
    /// <returns>The result of training including the trained model.</returns>
    /// <remarks>
    /// <para>
    /// This method implements true streaming training by iterating through the streaming loader's
    /// batches and training on each batch individually. This allows training on datasets that
    /// are too large to fit in memory.
    /// </para>
    /// <para><b>For Beginners:</b> Unlike the regular training path which loads all data into memory,
    /// this method processes one batch at a time, trains on it, then moves to the next batch.
    /// This is essential for large datasets like ImageNet or large text corpora.
    /// </para>
    /// </remarks>
    private async Task<PredictionModelResult<T, TInput, TOutput>> BuildStreamingSupervisedAsync(
        IStreamingDataLoader<T, TInput, TOutput> streamingLoader)
    {
        // Apply GPU configuration first
        ApplyGpuConfiguration();

        // Apply memory management configuration (gradient checkpointing, etc.)
        ApplyMemoryConfiguration();

        // Ensure we have a model configured
        if (_model is null)
        {
            throw new InvalidOperationException(
                "Streaming training requires a model to be configured. Use ConfigureModel() before calling BuildAsync().");
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        // Get optimizer options for training parameters
        var optimizerOptions = _optimizer?.GetOptions();
        int epochs = optimizerOptions?.MaxIterations ?? 100;
        T learningRate = optimizerOptions is not null
            ? numOps.FromDouble(optimizerOptions.InitialLearningRate)
            : numOps.FromDouble(0.01);
        T learningRateDecay = optimizerOptions is not null
            ? numOps.FromDouble(optimizerOptions.LearningRateDecay)
            : numOps.FromDouble(0.99);
        T minLearningRate = optimizerOptions is not null
            ? numOps.FromDouble(optimizerOptions.MinLearningRate)
            : numOps.FromDouble(1e-6);

        // Get loss function
        var lossFunction = _model.DefaultLossFunction;

        // Training metrics
        T totalLoss = numOps.Zero;
        int totalBatches = 0;

        // Train for the specified number of epochs
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            T epochLoss = numOps.Zero;
            int epochBatches = 0;

            // Iterate through all batches in the streaming loader
            await foreach (var (inputs, outputs) in streamingLoader.GetBatchesAsync(shuffle: true))
            {
                // Process each sample in the batch
                for (int i = 0; i < inputs.Length; i++)
                {
                    var input = inputs[i];
                    var target = outputs[i];

                    // Compute gradients without updating parameters
                    var gradients = _model.ComputeGradients(input, target, lossFunction);

                    // Apply gradients with current learning rate
                    _model.ApplyGradients(gradients, learningRate);

                    // Accumulate loss for monitoring (optional - compute prediction loss)
                    var prediction = _model.Predict(input);
                    var predictionVector = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
                    var targetVector = ConversionsHelper.ConvertToVector<T, TOutput>(target);
                    var loss = lossFunction.CalculateLoss(predictionVector, targetVector);
                    epochLoss = numOps.Add(epochLoss, loss);
                    epochBatches++;
                }
            }

            totalLoss = numOps.Add(totalLoss, epochLoss);
            totalBatches += epochBatches;

            // Decay learning rate
            if (optimizerOptions is not null && optimizerOptions.UseAdaptiveLearningRate)
            {
                learningRate = numOps.Multiply(learningRate, learningRateDecay);
                if (numOps.Compare(learningRate, minLearningRate) < 0)
                {
                    learningRate = minLearningRate;
                }
            }

            // Check for early stopping if configured
            if (_optimizer is not null && _optimizer.ShouldEarlyStop())
            {
                break;
            }
        }

        // Calculate average loss
        T avgLoss = totalBatches > 0
            ? numOps.Divide(totalLoss, numOps.FromDouble(totalBatches))
            : numOps.Zero;

        // Build the result
        var optimizationResult = new OptimizationResult<T, TInput, TOutput>
        {
            BestSolution = _model,
            BestFitnessScore = avgLoss,
            FitnessHistory = new Vector<T>(new[] { avgLoss }),
            Iterations = totalBatches
        };

        // Create deployment configuration from individual configs
        var deploymentConfig = DeploymentConfiguration.Create(
            _quantizationConfig,
            _cacheConfig,
            _versioningConfig,
            _abTestingConfig,
            _telemetryConfig,
            _exportConfig,
            _gpuAccelerationConfig,
            _compressionConfig,
            _profilingConfig);

        // Build result using options pattern like other Build methods
        var options = new PredictionModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = optimizationResult,
            NormalizationInfo = new NormalizationInfo<T, TInput, TOutput> { Normalizer = new NoNormalizer<T, TInput, TOutput>() },
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            ProgramSynthesisModel = _programSynthesisModel,
            ProgramSynthesisServingClient = _programSynthesisServingClient,
            ProgramSynthesisServingClientOptions = _programSynthesisServingClientOptions,
            InferenceOptimizationConfig = _inferenceOptimizationConfig,
            AugmentationConfig = _augmentationConfig,
            ReasoningConfig = _reasoningConfig,
            DeploymentConfiguration = deploymentConfig,
            BiasDetector = _biasDetector,
            FairnessEvaluator = _fairnessEvaluator,
            RagRetriever = _ragRetriever,
            RagReranker = _ragReranker,
            RagGenerator = _ragGenerator,
            QueryProcessors = _queryProcessors,
            KnowledgeGraph = _knowledgeGraph,
            GraphStore = _graphStore,
            HybridGraphRetriever = _hybridGraphRetriever,
            LoRAConfiguration = _loraConfiguration,
            AgentConfig = _agentConfig,
            MemoryConfig = _memoryConfig
        };

        return new PredictionModelResult<T, TInput, TOutput>(options);
    }

    /// <summary>
    /// Collects all data from a streaming data loader into aggregated features and labels.
    /// </summary>
    /// <param name="streamingLoader">The streaming data loader to collect from.</param>
    /// <returns>A tuple containing the aggregated features and labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reads all batches from a streaming data loader
    /// and combines them into single feature and label collections. This allows streaming
    /// loaders to work with the existing training infrastructure while preserving the
    /// benefit of on-demand data loading from files or other sources.
    /// </para>
    /// </remarks>
    private async Task<(TInput Features, TOutput Labels)> CollectStreamingDataAsync(
        IStreamingDataLoader<T, TInput, TOutput> streamingLoader)
    {
        var allInputs = new List<TInput>();
        var allOutputs = new List<TOutput>();

        // Collect all batches asynchronously
        await foreach (var (inputs, outputs) in streamingLoader.GetBatchesAsync(shuffle: false))
        {
            foreach (var input in inputs)
            {
                allInputs.Add(input);
            }
            foreach (var output in outputs)
            {
                allOutputs.Add(output);
            }
        }

        if (allInputs.Count == 0)
        {
            throw new InvalidOperationException("Streaming data loader returned no data.");
        }

        // Aggregate the collected samples into single feature/label structures
        var aggregatedFeatures = AggregateStreamingInputs(allInputs);
        var aggregatedLabels = AggregateStreamingOutputs(allOutputs);

        return (aggregatedFeatures, aggregatedLabels);
    }

    /// <summary>
    /// Aggregates a list of input samples into a single TInput structure.
    /// </summary>
    private TInput AggregateStreamingInputs(List<TInput> inputs) =>
        DataAggregationHelper.Aggregate<T, TInput>(inputs, "input");

    /// <summary>
    /// Aggregates a list of output samples into a single TOutput structure.
    /// </summary>
    private TOutput AggregateStreamingOutputs(List<TOutput> outputs) =>
        DataAggregationHelper.Aggregate<T, TOutput>(outputs, "output");

    /// <summary>
    /// Internal method that performs supervised training with the provided input features and output values.
    /// This contains all the core supervised learning logic.
    /// </summary>
    /// <param name="x">Matrix of input features.</param>
    /// <param name="y">Vector of output values.</param>
    /// <returns>A task that represents the asynchronous operation, containing the trained model.</returns>
    private async Task<PredictionModelResult<T, TInput, TOutput>> BuildSupervisedInternalAsync(TInput x, TOutput y)
    {
        // SUPERVISED TRAINING PATH

        // Create profiler session if profiling is enabled
        var profilerSession = CreateProfilerSession();
        using var _ = profilerSession?.Scope("BuildSupervisedInternalAsync");

        // Apply GPU configuration first (before any operations that might use GPU)
        ApplyGpuConfiguration();

        // ============================================================================
        // Training Infrastructure Initialization
        // ============================================================================

        // Variables to track training infrastructure state
        string? experimentRunId = null;
        string? experimentId = null;
        IExperimentRun<T>? experimentRun = null;
        string? monitorSessionId = null;
        string? checkpointPath = null;
        string? registeredModelName = null;
        int? modelVersion = null;
        string? dataVersionHash = null;
        var trainingStartTime = DateTime.UtcNow;

        // If a federated-aware data loader is configured, prefer its natural client partitions (e.g., LEAF users).
        // This keeps federated simulations faithful to benchmark/client boundaries and avoids leaking partitioning
        // concerns into the public facade API.
        List<(int ClientId, int StartRow, int SampleCount)>? federatedClientRanges = null;
        if (_federatedLearningOptions != null &&
            _dataLoader is IFederatedClientDataLoader<T, TInput, TOutput> federatedClientDataLoader)
        {
            var aggregated = BuildAggregatedDatasetFromClientData(federatedClientDataLoader.ClientData);
            x = aggregated.X;
            y = aggregated.Y;
            federatedClientRanges = aggregated.ClientRanges;
        }

        // Convert and validate inputs
        int xSamples = ConversionsHelper.GetSampleCount<T, TInput>(x);
        int ySamples = ConversionsHelper.GetSampleCount<T, TOutput>(y);

        if (xSamples != ySamples)
            throw new ArgumentException("Number of rows in features must match length of actual values", nameof(x));

        // Convert inputs to Matrix/Vector for internal processing
        var convertedX = ConversionsHelper.ConvertToMatrix<T, TInput>(x);
        var convertedY = ConversionsHelper.ConvertToVector<T, TOutput>(y);

        // AGENT ASSISTANCE (if enabled)
        AgentRecommendation<T, TInput, TOutput>? agentRecommendation = null;
        if (_agentConfig != null && _agentConfig.IsEnabled)
        {
            try
            {
                agentRecommendation = await GetAgentRecommendationsAsync(x, y);
                ApplyAgentRecommendations(agentRecommendation);
            }
            catch (Exception ex)
            {
                // Log warning but don't fail the build if agent assistance fails
                // The build can proceed without agent recommendations
                Console.WriteLine($"Warning: Agent assistance failed: {ex.Message}");
                Console.WriteLine("Proceeding with model building without agent recommendations.");
            }
        }

        // AUTOML SEARCH (if configured and no model explicitly set)
        // AutoML finds the best model type and hyperparameters automatically
        AutoMLRunSummary? autoMLSummary = null;
        if (_autoMLModel != null && _model == null)
        {
            Console.WriteLine("AutoML configured - starting model search...");
            var searchStartedUtc = DateTimeOffset.UtcNow;

            // Set up preprocessing for AutoML search
            var autoMLNormalizer = _normalizer ?? new NoNormalizer<T, TInput, TOutput>();
            var autoMLFeatureSelector = _featureSelector ?? new NoFeatureSelector<T, TInput>();
            var autoMLOutlierRemoval = _outlierRemoval ?? new NoOutlierRemoval<T, TInput, TOutput>();

            var autoMLDataProcessorOptions = new DataProcessorOptions();
            if (_autoMLOptions?.TaskFamilyOverride == AutoMLTaskFamily.TimeSeriesForecasting
                || _autoMLOptions?.TaskFamilyOverride == AutoMLTaskFamily.TimeSeriesAnomalyDetection)
            {
                autoMLDataProcessorOptions.ShuffleBeforeSplit = false;
            }

            var autoMLPreprocessor = _dataPreprocessor ?? new DefaultDataPreprocessor<T, TInput, TOutput>(
                autoMLNormalizer, autoMLFeatureSelector, autoMLOutlierRemoval, autoMLDataProcessorOptions);

            // Preprocess and split data for AutoML search
            var (autoMLPreprocessedX, autoMLPreprocessedY, _) = autoMLPreprocessor.PreprocessData(x, y);
            var (autoMLXTrain, autoMLYTrain, autoMLXVal, autoMLYVal, _, _) = autoMLPreprocessor.SplitData(
                autoMLPreprocessedX, autoMLPreprocessedY);

            // Configure AutoML with model evaluator if available
            if (_modelEvaluator != null)
            {
                _autoMLModel.SetModelEvaluator(_modelEvaluator);
            }

            if (_autoMLOptions?.TaskFamilyOverride is AutoMLTaskFamily taskFamilyOverride)
            {
                int featureCount = InputHelper<T, TInput>.GetInputSize(autoMLXTrain);
                var candidates = AutoMLDefaultCandidateModelsPolicy.GetDefaultCandidates(taskFamilyOverride, featureCount, _autoMLOptions.Budget.Preset);
                if (candidates.Count > 0)
                {
                    _autoMLModel.SetCandidateModels(candidates.ToList());
                }

                if (!_autoMLOptions.OptimizationMetricOverride.HasValue)
                {
                    var (metric, maximize) = AutoMLDefaultMetricPolicy.GetDefault(taskFamilyOverride);
                    _autoMLModel.SetOptimizationMetric(metric, maximize);
                }
            }

            // Run AutoML search to find the best model
            var bestModel = await _autoMLModel.SearchAsync(
                autoMLXTrain,
                autoMLYTrain,
                autoMLXVal,
                autoMLYVal,
                _autoMLModel.TimeLimit,
                CancellationToken.None);

            _model = bestModel;

            var searchEndedUtc = DateTimeOffset.UtcNow;
            autoMLSummary = CreateAutoMLRunSummary(searchStartedUtc, searchEndedUtc);

            Console.WriteLine("AutoML search complete.");
            Console.WriteLine($"Best score: {_autoMLModel.BestScore}");
            Console.WriteLine($"Trials completed: {_autoMLModel.GetTrialHistory().Count}");
        }

        // Validate model is set (either by user, agent, or AutoML)
        if (_model == null)
            throw new InvalidOperationException("Model implementation must be specified. Use ConfigureModel() to set a model, ConfigureAutoML() for automatic model selection, or enable agent assistance.");

        // Use defaults for these interfaces if they aren't set
        var normalizer = _normalizer ?? new NoNormalizer<T, TInput, TOutput>();
        var optimizer = _optimizer ?? new NormalOptimizer<T, TInput, TOutput>(_model);
        var featureSelector = _featureSelector ?? new NoFeatureSelector<T, TInput>();
        var outlierRemoval = _outlierRemoval ?? new NoOutlierRemoval<T, TInput, TOutput>();
        var dataPreprocessor = _dataPreprocessor ?? new DefaultDataPreprocessor<T, TInput, TOutput>(normalizer, featureSelector, outlierRemoval);

        // LORA ADAPTATION (if configured)
        // Apply LoRA adapters to neural network layers for parameter-efficient fine-tuning
        if (_loraConfiguration != null && _model is NeuralNetworks.NeuralNetworkBase<T> neuralNetForLoRA)
        {
            Console.WriteLine("Applying LoRA adapters to neural network layers...");

            int adaptedCount = 0;
            for (int i = 0; i < neuralNetForLoRA.Layers.Count; i++)
            {
                var originalLayer = neuralNetForLoRA.Layers[i];
                var adaptedLayer = _loraConfiguration.ApplyLoRA(originalLayer);

                // If the layer was adapted (wrapped with LoRA), update the list
                if (!ReferenceEquals(originalLayer, adaptedLayer))
                {
                    neuralNetForLoRA.Layers[i] = adaptedLayer;
                    adaptedCount++;
                }
            }

            Console.WriteLine($"LoRA applied to {adaptedCount} layers (rank={_loraConfiguration.Rank}, alpha={_loraConfiguration.Alpha})");
        }


        // Wrap model and optimizer for distributed training if configured
        IFullModel<T, TInput, TOutput> model = _model;
        IOptimizer<T, TInput, TOutput> finalOptimizer = optimizer;

        // Enable mixed-precision training BEFORE distributed training wrapping (if configured)
        // This ensures mixed-precision is applied to the base model/optimizer before any wrapping
        if (_mixedPrecisionConfig != null)
        {
            // Verify T is float
            if (typeof(T) != typeof(float))
            {
                throw new InvalidOperationException(
                    $"Mixed-precision training requires T = float, got T = {typeof(T).Name}. " +
                    $"Use PredictionModelBuilder<float, ...> to enable mixed-precision training.");
            }

            // Enable on neural network model if applicable
            if (_model is NeuralNetworkBase<T> neuralNet)
            {
                neuralNet.EnableMixedPrecision(_mixedPrecisionConfig);
            }

            // Enable on gradient-based optimizer if applicable
            if (optimizer is GradientBasedOptimizerBase<T, TInput, TOutput> gradOptimizer)
            {
                gradOptimizer.EnableMixedPrecision(_mixedPrecisionConfig);
            }
        }

        // Enable distributed training if backend or configuration was explicitly provided
        if (_distributedBackend != null || _distributedConfiguration != null)
        {
            // Use provided backend or default to InMemory for single-process
            var backend = _distributedBackend ?? new DistributedTraining.InMemoryCommunicationBackend<T>(rank: 0, worldSize: 1);

            // Use provided configuration or create default from backend
            var shardingConfig = _distributedConfiguration ?? new DistributedTraining.ShardingConfiguration<T>(backend);

            // Check if model/optimizer are already sharded to avoid double-wrapping
            bool isModelAlreadySharded = _model is DistributedTraining.IShardedModel<T, TInput, TOutput>;
            bool isOptimizerAlreadySharded = optimizer is DistributedTraining.IShardedOptimizer<T, TInput, TOutput>;

            // Only wrap if not already sharded
            if (isModelAlreadySharded || isOptimizerAlreadySharded)
            {
                // Model or optimizer already sharded - skip wrapping to avoid double-wrapping
                model = _model;
                finalOptimizer = optimizer;
            }
            else
            {
                // Switch on strategy to create appropriate model/optimizer pair
                (model, finalOptimizer) = _distributedStrategy switch
                {
                    DistributedStrategy.DDP => CreateDistributedPair(
                        new DistributedTraining.DDPModel<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.DDPOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.FSDP => CreateDistributedPair(
                        new DistributedTraining.FSDPModel<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.FSDPOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.ZeRO1 => CreateDistributedPair(
                        new DistributedTraining.ZeRO1Model<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.ZeRO1Optimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.ZeRO2 => CreateDistributedPair(
                        new DistributedTraining.ZeRO2Model<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.ZeRO2Optimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.ZeRO3 => CreateDistributedPair(
                        new DistributedTraining.ZeRO3Model<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.ZeRO3Optimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.PipelineParallel => CreateDistributedPair(
                        new DistributedTraining.PipelineParallelModel<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.PipelineParallelOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.TensorParallel => CreateDistributedPair(
                        new DistributedTraining.TensorParallelModel<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.TensorParallelOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.Hybrid => CreateDistributedPair(
                        new DistributedTraining.HybridShardedModel<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.HybridShardedOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    _ => throw new InvalidOperationException($"Unsupported distributed strategy: {_distributedStrategy}")
                };
            }
        }

        bool usePartitionedFederatedData = _federatedLearningOptions != null && federatedClientRanges != null;
        int expectedFederatedSampleCount = 0;
        if (usePartitionedFederatedData)
        {
            foreach (var range in federatedClientRanges!)
            {
                expectedFederatedSampleCount += range.SampleCount;
            }
        }

        // Preprocess the data
        TInput preprocessedX;
        TOutput preprocessedY;
        NormalizationInfo<T, TInput, TOutput>? normInfo = null;
        PreprocessingInfo<T, TInput, TOutput>? preprocessingInfo = null;

        if (_preprocessingPipeline is not null)
        {
            // Use new preprocessing pipeline
            preprocessedX = _preprocessingPipeline.FitTransform(x);
            preprocessedY = y; // Target preprocessing handled separately if needed

            // Create PreprocessingInfo with fitted pipeline
            preprocessingInfo = new PreprocessingInfo<T, TInput, TOutput>(
                _preprocessingPipeline,
                targetPipeline: null // Target pipeline can be added later if needed
            );

            // Create a legacy normInfo for backward compatibility with NoNormalizer
            normInfo = new NormalizationInfo<T, TInput, TOutput> { Normalizer = new NoNormalizer<T, TInput, TOutput>() };
        }
        else
        {
            // Use legacy dataPreprocessor
            (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);
        }

        if (usePartitionedFederatedData)
        {
            var preprocessedMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(preprocessedX);
            var preprocessedVector = ConversionsHelper.ConvertToVector<T, TOutput>(preprocessedY);

            if (preprocessedMatrix.Rows != preprocessedVector.Length)
            {
                throw new InvalidOperationException(
                    "Federated learning with partitioned client data requires preprocessing to preserve X/y row alignment. " +
                    $"Got X rows={preprocessedMatrix.Rows} and y length={preprocessedVector.Length}.");
            }

            if (preprocessedMatrix.Rows != expectedFederatedSampleCount)
            {
                throw new InvalidOperationException(
                    "Federated learning with partitioned client data requires preprocessing to preserve the total number of samples and row ordering. " +
                    $"Expected {expectedFederatedSampleCount} samples from the data loader partitions but preprocessing produced {preprocessedMatrix.Rows}. " +
                    "If you are using outlier removal, filtering, or other preprocessing that drops/reorders rows, disable it for partitioned federated learning or " +
                    "apply preprocessing at the per-client level before aggregating client datasets.");
            }
        }

        TInput XTrain;
        TOutput yTrain;
        TInput XVal = default!;
        TOutput yVal = default!;
        TInput XTest = default!;
        TOutput yTest = default!;

        if (usePartitionedFederatedData)
        {
            // For natural per-client datasets (e.g., LEAF), avoid re-splitting at the sample level so that we can
            // preserve the client boundaries through preprocessing.
            XTrain = preprocessedX;
            yTrain = preprocessedY;
        }
        else
        {
            // Standard supervised learning path: split into train/validation/test.
            (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);
        }

        // Perform cross-validation on training data BEFORE final model training (if configured)
        // This follows industry standard patterns from H2O and caret where CV is integrated into model building
        CrossValidationResult<T, TInput, TOutput>? cvResults = null;
        if (_crossValidator != null && _modelEvaluator != null)
        {
            // Cross-validation uses only the training data to prevent data leakage
            // It trains multiple models on different folds to assess generalization
            cvResults = _modelEvaluator.PerformCrossValidation(
                model: _model,
                X: XTrain,
                y: yTrain,
                optimizer: optimizer,
                crossValidator: _crossValidator
            );
        }

        // Reset optimizer state after cross-validation to ensure final model training starts fresh
        // This prevents state contamination from CV (accumulated fitness lists, cache, learning rates)
        optimizer.Reset();

        // ============================================================================
        // Start Training Infrastructure (before optimization)
        // ============================================================================

        // Track data version for reproducibility
        if (_dataVersionControl is not null)
        {
            // Compute a hash of the training data for lineage tracking
            // This enables reproducibility by recording exactly which data was used
            var dataVersionNumOps = MathHelper.GetNumericOperations<T>();
            dataVersionHash = ComputeDataVersionHash(convertedX, convertedY, dataVersionNumOps);

            // Note: For in-memory data, we cannot use CreateDatasetVersion (requires dataPath).
            // Instead, we track the data characteristics via the experiment run parameters below.
            // The data_version_hash is logged to experiment run parameters (line ~1122) and linked
            // via _dataVersionControl.LinkDatasetToRun (line ~1131) for full traceability.
            // Key metadata tracked: rows, columns, target_length, feature_count, training/validation/test samples.
        }

        // Start experiment tracking run
        if (_experimentTracker is not null)
        {
            experimentId = _experimentTracker.CreateExperiment(
                name: "supervised-training",
                description: $"Supervised learning with {model.GetType().Name}",
                tags: new Dictionary<string, string>
                {
                    ["model_type"] = model.GetType().Name,
                    ["optimizer_type"] = finalOptimizer.GetType().Name,
                    ["framework"] = "AiDotNet"
                });

            experimentRun = _experimentTracker.StartRun(
                experimentId: experimentId,
                runName: $"run-{trainingStartTime:yyyyMMdd-HHmmss}",
                tags: new Dictionary<string, string>
                {
                    ["start_time"] = trainingStartTime.ToString("O")
                });

            experimentRunId = experimentRun.RunId;

            // Log hyperparameters from optimizer options
            var optimizerOptions = finalOptimizer.GetOptions();
            experimentRun.LogParameters(new Dictionary<string, object>
            {
                ["model_type"] = model.GetType().FullName ?? model.GetType().Name,
                ["optimizer_type"] = finalOptimizer.GetType().FullName ?? finalOptimizer.GetType().Name,
                ["max_iterations"] = optimizerOptions.MaxIterations,
                ["use_early_stopping"] = optimizerOptions.UseEarlyStopping,
                ["early_stopping_patience"] = optimizerOptions.EarlyStoppingPatience,
                ["training_samples"] = XTrain is Matrix<T> trainMatrix ? trainMatrix.Rows : 0,
                ["validation_samples"] = XVal is Matrix<T> valMatrix ? valMatrix.Rows : 0,
                ["test_samples"] = XTest is Matrix<T> testMatrix ? testMatrix.Rows : 0,
                ["feature_count"] = convertedX.Columns,
                ["target_length"] = convertedY.Length,
                ["data_version_hash"] = dataVersionHash ?? "not_tracked"
            });

            // Link data version to experiment run for full traceability
            if (_dataVersionControl is not null && dataVersionHash is not null)
            {
                try
                {
                    var datasetName = $"training-data-{model.GetType().Name}";
                    _dataVersionControl.LinkDatasetToRun(
                        datasetName: datasetName,
                        versionHash: dataVersionHash,
                        runId: experimentRunId,
                        modelId: null); // Model ID will be set after registry
                }
                catch
                {
                    // Data version control linkage is optional - don't fail training
                }
            }
        }

        // Start training monitor session
        if (_trainingMonitor is not null)
        {
            monitorSessionId = _trainingMonitor.StartSession(
                sessionName: $"training-{model.GetType().Name}",
                metadata: new Dictionary<string, object>
                {
                    ["model_type"] = model.GetType().Name,
                    ["optimizer_type"] = finalOptimizer.GetType().Name,
                    ["experiment_run_id"] = experimentRunId ?? string.Empty,
                    ["start_time"] = trainingStartTime
                });
        }

        // ============================================================================
        // Hyperparameter Optimization (if configured)
        // ============================================================================

        HyperparameterOptimizationResult<T>? hyperparameterOptimizationResult = null;
        int? bestHyperparameterTrialId = null;
        Dictionary<string, object>? bestHyperparameters = null;

        if (_hyperparameterOptimizer is not null && _hyperparameterSearchSpace is not null)
        {
            var numOps = MathHelper.GetNumericOperations<T>();

            try
            {
                // Log hyperparameter optimization start
                if (_trainingMonitor is not null && monitorSessionId is not null)
                {
                    _trainingMonitor.LogMessage(monitorSessionId, LogLevel.Info, "Starting hyperparameter optimization...");
                }

                // Create objective function that trains the model and returns validation loss
                T ObjectiveFunction(Dictionary<string, object> trialHyperparameters)
                {
                    // Reset optimizer for fresh training
                    finalOptimizer.Reset();

                    // Apply trial hyperparameters to the optimizer
                    var optimizerOptions = finalOptimizer.GetOptions();
                    ApplyTrialHyperparameters(optimizerOptions, trialHyperparameters);

                    // Log hyperparameters for this trial to experiment tracker
                    if (experimentRun is not null)
                    {
                        foreach (var kvp in trialHyperparameters)
                        {
                            experimentRun.LogParameter($"hpo_trial_{kvp.Key}", kvp.Value);
                        }
                    }

                    // Train with current hyperparameters
                    var trialResult = finalOptimizer.Optimize(
                        OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(
                            XTrain, yTrain, XVal, yVal, XTest, yTest));

                    // Return validation MSE as objective (minimizing)
                    if (trialResult.ValidationResult.ErrorStats is not null)
                    {
                        return trialResult.ValidationResult.ErrorStats.MSE;
                    }

                    // Fallback to training loss if validation unavailable
                    if (trialResult.TrainingResult.ErrorStats is not null)
                    {
                        return trialResult.TrainingResult.ErrorStats.MSE;
                    }

                    // Return maximum value for failed trials to penalize them heavily
                    // when minimizing (zero would incorrectly indicate perfect performance)
                    return numOps.MaxValue;
                }

                // Run hyperparameter optimization
                hyperparameterOptimizationResult = _hyperparameterOptimizer.Optimize(
                    ObjectiveFunction,
                    _hyperparameterSearchSpace,
                    _hyperparameterTrials);

                // Extract best trial information
                if (hyperparameterOptimizationResult.BestTrial is not null)
                {
                    bestHyperparameterTrialId = hyperparameterOptimizationResult.BestTrial.TrialNumber;
                    bestHyperparameters = hyperparameterOptimizationResult.BestParameters;

                    // Log best hyperparameters to experiment tracker
                    if (experimentRun is not null && bestHyperparameters is not null)
                    {
                        foreach (var kvp in bestHyperparameters)
                        {
                            experimentRun.LogParameter($"best_{kvp.Key}", kvp.Value);
                        }

                        var bestValue = hyperparameterOptimizationResult.BestTrial.ObjectiveValue ?? numOps.Zero;
                        experimentRun.LogMetric("best_trial_objective", bestValue);
                    }

                    if (_trainingMonitor is not null && monitorSessionId is not null)
                    {
                        _trainingMonitor.LogMessage(monitorSessionId, LogLevel.Info,
                            $"HPO complete: best trial={bestHyperparameterTrialId}, completed={hyperparameterOptimizationResult.CompletedTrials}");
                    }
                }

                // Reset optimizer for final training
                finalOptimizer.Reset();
            }
            catch (Exception ex)
            {
                // Hyperparameter optimization is optional - log warning and continue
                if (_trainingMonitor is not null && monitorSessionId is not null)
                {
                    _trainingMonitor.LogMessage(monitorSessionId, LogLevel.Warning, $"HPO failed: {ex.Message}");
                }
            }
        }

        // Uncertainty quantification: create deep ensemble template before optimization
        var deepEnsembleTemplate = _uncertaintyQuantificationOptions is { Enabled: true, Method: UncertaintyQuantificationMethod.DeepEnsemble }
            ? _model.DeepCopy()
            : null;

        OptimizationResult<T, TInput, TOutput> optimizationResult;
        FederatedLearningMetadata? federatedLearningMetadata = null;
        var optimizationInputData = OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest);

        // FEDERATED LEARNING PATH (facade-first: orchestration stays internal)
        if (_federatedLearningOptions != null)
        {
            if (_knowledgeDistillationOptions != null)
            {
                throw new InvalidOperationException("Federated learning cannot be combined with knowledge distillation in the same Build() call.");
            }

            if (_distributedBackend != null || _distributedConfiguration != null)
            {
                throw new InvalidOperationException("Federated learning is not currently compatible with distributed training configuration. Use either federated learning or distributed training per build.");
            }

            var flOptions = _federatedLearningOptions;

            Dictionary<int, FederatedClientDataset<TInput, TOutput>> clientPartitions;
            int effectiveClientCount;

            if (usePartitionedFederatedData)
            {
                clientPartitions = CreateFederatedClientPartitionsFromClientRanges(
                    XTrain,
                    yTrain,
                    federatedClientRanges!);

                effectiveClientCount = clientPartitions.Count;

                if (effectiveClientCount <= 0)
                {
                    throw new InvalidOperationException("Federated client data resolved from the data loader is empty.");
                }

                if (flOptions.NumberOfClients != effectiveClientCount)
                {
                    Console.WriteLine(
                        $"[AiDotNet] Warning: FederatedLearningOptions.NumberOfClients={flOptions.NumberOfClients} does not match the data loader client count {effectiveClientCount}. Using the data loader client count.");
                }
            }
            else
            {
                clientPartitions = CreateFederatedClientPartitions(XTrain, yTrain, flOptions.NumberOfClients, flOptions.RandomSeed);
                effectiveClientCount = clientPartitions.Count;
            }

            var trainer = new AiDotNet.FederatedLearning.Trainers.InMemoryFederatedTrainer<T, TInput, TOutput>(
                optimizerPrototype: finalOptimizer,
                learningRateOverride: flOptions.LearningRate,
                randomSeed: flOptions.RandomSeed,
                convergenceThreshold: flOptions.ConvergenceThreshold,
                minRoundsBeforeConvergence: flOptions.MinRoundsBeforeConvergence,
                federatedLearningOptions: flOptions,
                clientSelectionStrategy: _federatedClientSelectionStrategy,
                serverOptimizer: _federatedServerOptimizer,
                heterogeneityCorrection: _federatedHeterogeneityCorrection,
                homomorphicEncryptionProvider: _federatedHomomorphicEncryptionProvider);

            var aggregationStrategy = _federatedAggregationStrategy ?? CreateDefaultFederatedAggregationStrategy(flOptions);
            trainer.SetAggregationStrategy(aggregationStrategy);
            trainer.Initialize(model, effectiveClientCount);

            federatedLearningMetadata = trainer.Train(
                clientData: clientPartitions,
                rounds: flOptions.MaxRounds,
                clientSelectionFraction: flOptions.ClientSelectionFraction,
                localEpochs: flOptions.LocalEpochs);

            optimizationResult = new OptimizationResult<T, TInput, TOutput>
            {
                BestSolution = trainer.GetGlobalModel(),
                Iterations = federatedLearningMetadata.RoundsCompleted
            };
        }
        else
        {
            // REGULAR TRAINING PATH
            // Optimize the final model on the full training set (optionally using knowledge distillation)
            optimizationResult = _knowledgeDistillationOptions != null
                ? await PerformKnowledgeDistillationAsync(
                    model,
                    finalOptimizer,
                    XTrain,
                    yTrain,
                    XVal,
                    yVal,
                    XTest,
                    yTest)
                : finalOptimizer.Optimize(optimizationInputData);
        }

        var trainingEndTime = DateTime.UtcNow;
        var trainingDuration = trainingEndTime - trainingStartTime;

        // ============================================================================
        // Finalize Training Infrastructure (after optimization)
        // ============================================================================

        // Collect final metrics from optimization result
        var finalMetrics = new Dictionary<string, T>();
        if (optimizationResult.TrainingResult.ErrorStats is not null)
        {
            finalMetrics["training_rmse"] = optimizationResult.TrainingResult.ErrorStats.RMSE;
            finalMetrics["training_mae"] = optimizationResult.TrainingResult.ErrorStats.MAE;
            finalMetrics["training_mse"] = optimizationResult.TrainingResult.ErrorStats.MSE;
        }
        if (optimizationResult.ValidationResult.ErrorStats is not null)
        {
            finalMetrics["validation_rmse"] = optimizationResult.ValidationResult.ErrorStats.RMSE;
            finalMetrics["validation_mae"] = optimizationResult.ValidationResult.ErrorStats.MAE;
            finalMetrics["validation_mse"] = optimizationResult.ValidationResult.ErrorStats.MSE;
        }
        if (optimizationResult.TestResult.ErrorStats is not null)
        {
            finalMetrics["test_rmse"] = optimizationResult.TestResult.ErrorStats.RMSE;
            finalMetrics["test_mae"] = optimizationResult.TestResult.ErrorStats.MAE;
            finalMetrics["test_mse"] = optimizationResult.TestResult.ErrorStats.MSE;
        }

        // Log final metrics to experiment run
        if (experimentRun is not null)
        {
            experimentRun.LogMetrics(finalMetrics, step: 1);
            experimentRun.LogParameter("training_duration_seconds", trainingDuration.TotalSeconds);
            experimentRun.Complete();
        }

        // Log final metrics to training monitor
        if (_trainingMonitor is not null && monitorSessionId is not null)
        {
            _trainingMonitor.LogMetrics(monitorSessionId, finalMetrics, step: 1);
            _trainingMonitor.EndSession(monitorSessionId);
        }

        // Save checkpoint if checkpoint manager is configured
        if (_checkpointManager is not null && optimizationResult.BestSolution is not null)
        {
            var checkpointMetrics = new Dictionary<string, T>(finalMetrics);
            var checkpointMetadata = new Dictionary<string, object>
            {
                ["experiment_run_id"] = experimentRunId ?? string.Empty,
                ["training_duration_seconds"] = trainingDuration.TotalSeconds,
                ["model_type"] = optimizationResult.BestSolution.GetType().Name
            };

            checkpointPath = _checkpointManager.SaveCheckpoint(
                model: optimizationResult.BestSolution,
                optimizer: finalOptimizer,
                epoch: 1,
                step: 1,
                metrics: checkpointMetrics,
                metadata: checkpointMetadata);
        }

        // Register model in model registry if configured
        if (_modelRegistry is not null && optimizationResult.BestSolution is not null)
        {
            registeredModelName = $"{model.GetType().Name}-{trainingStartTime:yyyyMMdd-HHmmss}";

            // Determine model type from the actual model class
            var modelTypeName = optimizationResult.BestSolution.GetType().Name;
            ModelType derivedModelType = ModelType.None;
            if (Enum.TryParse<ModelType>(modelTypeName, ignoreCase: true, out var parsedType))
            {
                derivedModelType = parsedType;
            }
            else if (modelTypeName.Contains("Regression", StringComparison.OrdinalIgnoreCase))
            {
                derivedModelType = ModelType.SimpleRegression;
            }
            else if (modelTypeName.Contains("Neural", StringComparison.OrdinalIgnoreCase))
            {
                derivedModelType = ModelType.NeuralNetwork;
            }

            var modelMetadata = new ModelMetadata<T>
            {
                Name = registeredModelName,
                Version = "1.0",
                TrainingDate = trainingStartTime,
                ModelType = derivedModelType,
                FeatureCount = convertedX.Columns,
                Complexity = optimizationResult.BestSolution.GetType().GetProperties().Length,
                Description = $"Model trained via PredictionModelBuilder on {trainingStartTime:yyyy-MM-dd HH:mm:ss} UTC",
                AdditionalInfo = new Dictionary<string, object>
                {
                    ["experiment_run_id"] = experimentRunId ?? string.Empty,
                    ["training_duration_seconds"] = trainingDuration.TotalSeconds,
                    ["optimizer_type"] = finalOptimizer.GetType().Name
                }
            };

            // Add final metrics to AdditionalInfo (only non-null values)
            foreach (var metric in finalMetrics)
            {
                if (metric.Value is not null)
                {
                    modelMetadata.AdditionalInfo[$"metric_{metric.Key}"] = metric.Value;
                }
            }

            modelVersion = _modelRegistry.CreateModelVersion(
                modelName: registeredModelName,
                model: optimizationResult.BestSolution,
                metadata: modelMetadata,
                description: $"Auto-registered from training run {experimentRunId ?? "unknown"}");
        }

        // Apply uncertainty quantification if configured
        ApplyUncertaintyQuantificationIfConfigured(optimizationResult.BestSolution, _uncertaintyQuantificationOptions);

        // Create deployment configuration from individual configs
        var deploymentConfig = DeploymentConfiguration.Create(
            _quantizationConfig,
            _cacheConfig,
            _versioningConfig,
            _abTestingConfig,
            _telemetryConfig,
            _exportConfig,
            _gpuAccelerationConfig,
            _compressionConfig,
            _profilingConfig);

        // JIT COMPILATION (if configured and supported)
        Func<Tensor<T>[], Tensor<T>[]>? jitCompiledFunction = null;
        if (_jitCompilationConfig?.Enabled == true)
        {
            try
            {
                // Check if the model supports JIT compilation
                if (optimizationResult.BestSolution is IJitCompilable<T> jitModel &&
                    jitModel.SupportsJitCompilation)
                {
                    // Export computation graph from model
                    var inputNodes = new List<Autodiff.ComputationNode<T>>();
                    var outputNode = jitModel.ExportComputationGraph(inputNodes);

                    // Compile the graph with configured options
                    var jitCompiler = new AiDotNet.JitCompiler.JitCompiler(_jitCompilationConfig.CompilerOptions);
                    jitCompiledFunction = jitCompiler.Compile(outputNode, inputNodes);

                    Console.WriteLine($"JIT compilation successful for model {optimizationResult.BestSolution?.GetType().Name}");
                }
                else if (_jitCompilationConfig.ThrowOnFailure)
                {
                    throw new InvalidOperationException(
                        $"JIT compilation requested but model type {optimizationResult.BestSolution?.GetType().Name ?? "null"} " +
                        $"does not implement IJitCompilable<T> or does not support JIT compilation. " +
                        $"To use JIT compilation, the model must implement IJitCompilable and set SupportsJitCompilation = true.");
                }
                else
                {
                    // Graceful fallback - log warning
                    Console.WriteLine($"Warning: JIT compilation requested but model type {optimizationResult.BestSolution?.GetType().Name ?? "null"} does not support it. " +
                                      $"Proceeding without JIT acceleration.");
                }
            }
            catch (Exception ex) when (!_jitCompilationConfig.ThrowOnFailure)
            {
                // Graceful fallback - log warning and continue without JIT
                Console.WriteLine($"Warning: JIT compilation failed: {ex.Message}");
                Console.WriteLine("Proceeding without JIT acceleration.");
                jitCompiledFunction = null;
            }
        }

        // Build hyperparameters dictionary from optimizer options for result tracking
        var hyperparameters = new Dictionary<string, object>();
        try
        {
            var opts = finalOptimizer.GetOptions();
            hyperparameters["max_iterations"] = opts.MaxIterations;
            hyperparameters["use_early_stopping"] = opts.UseEarlyStopping;
            hyperparameters["early_stopping_patience"] = opts.EarlyStoppingPatience;
            hyperparameters["model_type"] = model.GetType().Name;
            hyperparameters["optimizer_type"] = finalOptimizer.GetType().Name;
        }
        catch
        {
            // Ignore errors collecting hyperparameters - they are optional
        }

        // Build training metrics history from optimization result
        var trainingMetricsHistory = new Dictionary<string, List<double>>();
        if (optimizationResult.FitnessHistory is not null && optimizationResult.FitnessHistory.Length > 0)
        {
            var fitnessHistoryAsDouble = new List<double>();
            for (int i = 0; i < optimizationResult.FitnessHistory.Length; i++)
            {
                // Use Convert.ToDouble for generic type conversion (standard pattern in this codebase)
                fitnessHistoryAsDouble.Add(Convert.ToDouble(optimizationResult.FitnessHistory[i]));
            }
            trainingMetricsHistory["fitness"] = fitnessHistoryAsDouble;
        }

        // Return PredictionModelResult with CV results, agent data, JIT compilation, reasoning config, and training infrastructure
        var options = new PredictionModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = optimizationResult,
            NormalizationInfo = normInfo,
            PreprocessingInfo = preprocessingInfo,
            AutoMLSummary = autoMLSummary,
            BiasDetector = _biasDetector,
            FairnessEvaluator = _fairnessEvaluator,
            RagRetriever = _ragRetriever,
            RagReranker = _ragReranker,
            RagGenerator = _ragGenerator,
            QueryProcessors = _queryProcessors,
            LoRAConfiguration = _loraConfiguration,
            CrossValidationResult = cvResults,
            AgentConfig = _agentConfig,
            AgentRecommendation = agentRecommendation,
            DeploymentConfiguration = deploymentConfig,
            JitCompiledFunction = jitCompiledFunction,
            InferenceOptimizationConfig = _inferenceOptimizationConfig,
            AugmentationConfig = _augmentationConfig,
            ReasoningConfig = _reasoningConfig,
            KnowledgeGraph = _knowledgeGraph,
            GraphStore = _graphStore,
            HybridGraphRetriever = _hybridGraphRetriever,
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            ProgramSynthesisModel = _programSynthesisModel,
            ProgramSynthesisServingClient = _programSynthesisServingClient,
            ProgramSynthesisServingClientOptions = _programSynthesisServingClientOptions,
            PromptTemplate = _promptTemplate,
            PromptChain = _promptChain,
            PromptOptimizer = _promptOptimizer,
            FewShotExampleSelector = _fewShotExampleSelector,
            PromptAnalyzer = _promptAnalyzer,
            PromptCompressor = _promptCompressor,
            // Diagnostics Properties
            ProfilingReport = profilerSession?.GetReport(),

            // Training Infrastructure Properties
            MemoryConfig = _memoryConfig,
            ExperimentRunId = experimentRunId,
            ExperimentId = experimentId,
            ModelVersion = modelVersion,
            RegisteredModelName = registeredModelName,
            CheckpointPath = checkpointPath,
            DataVersionHash = dataVersionHash,
            Hyperparameters = hyperparameters.Count > 0 ? hyperparameters : null,
            TrainingMetricsHistory = trainingMetricsHistory.Count > 0 ? trainingMetricsHistory : null,

            // Hyperparameter Optimization Properties
            HyperparameterOptimizationResult = hyperparameterOptimizationResult,
            HyperparameterTrialId = bestHyperparameterTrialId
        };

        var finalResult = new PredictionModelResult<T, TInput, TOutput>(options);

        finalResult.SetUncertaintyQuantificationOptions(_uncertaintyQuantificationOptions);
        TryComputeAndAttachDeepEnsembleModels(finalResult, deepEnsembleTemplate, optimizationInputData, optimizer, _uncertaintyQuantificationOptions);
        TryComputeAndAttachUncertaintyCalibrationArtifacts(finalResult);

        if (federatedLearningMetadata != null)
        {
            finalResult.GetModelMetadata().SetProperty(FederatedLearningMetadata.MetadataKey, federatedLearningMetadata);
        }

        return finalResult;
    }

    private AutoMLRunSummary CreateAutoMLRunSummary(DateTimeOffset startedUtc, DateTimeOffset endedUtc)
    {
        if (_autoMLModel is null)
        {
            throw new InvalidOperationException("AutoML summary requested but AutoML is not configured.");
        }

        MetricType metric = MetricType.Accuracy;
        bool maximize = true;

        if (_autoMLModel is AiDotNet.AutoML.AutoMLModelBase<T, TInput, TOutput> baseAutoML)
        {
            metric = baseAutoML.OptimizationMetric;
            maximize = baseAutoML.MaximizeOptimizationMetric;
        }
        else if (_autoMLOptions?.OptimizationMetricOverride is MetricType overrideMetric)
        {
            metric = overrideMetric;
            maximize = IsHigherBetter(metric);
        }

        var summary = new AutoMLRunSummary
        {
            SearchStrategy = _autoMLOptions?.SearchStrategy,
            TimeLimit = _autoMLModel.TimeLimit,
            TrialLimit = _autoMLModel.TrialLimit,
            OptimizationMetric = metric,
            MaximizeMetric = maximize,
            BestScore = _autoMLModel.BestScore,
            UsedEnsemble = _autoMLModel.BestModel is AiDotNet.AutoML.AutoMLEnsembleModel<T>,
            EnsembleSize = _autoMLModel.BestModel is AiDotNet.AutoML.AutoMLEnsembleModel<T> ensemble ? ensemble.Members.Count : null,
            SearchStartedUtc = startedUtc,
            SearchEndedUtc = endedUtc
        };

        foreach (var trial in _autoMLModel.GetTrialHistory())
        {
            summary.Trials.Add(new AutoMLTrialSummary
            {
                TrialId = trial.TrialId,
                Score = trial.Score,
                Duration = trial.Duration,
                CompletedUtc = trial.Timestamp,
                Success = trial.Success,
                ErrorMessage = trial.ErrorMessage
            });
        }

        return summary;
    }

    /// <summary>
    /// Internal method that performs meta-learning training.
    /// This contains all the core meta-learning logic.
    /// </summary>
    /// <returns>A task that represents the asynchronous operation, containing the trained meta-learner result.</returns>
    private PredictionModelResult<T, TInput, TOutput> BuildMetaLearningInternalAsync()
    {
        // META-LEARNING TRAINING PATH

        // Create profiler session if profiling is enabled
        var profilerSession = CreateProfilerSession();
        using var _ = profilerSession?.Scope("BuildMetaLearningInternalAsync");

        // Apply GPU configuration first
        ApplyGpuConfiguration();

        // Apply memory management configuration (gradient checkpointing, etc.)
        ApplyMemoryConfiguration();

        // Validate meta-learner is configured (should be checked by caller, but defensive)
        if (_metaLearner is null)
        {
            throw new InvalidOperationException(
                "BuildMetaLearningInternalAsync requires ConfigureMetaLearning() to be called first.");
        }

        // Perform meta-training using parameters from config (specified during meta-learner construction)
        var metaResult = _metaLearner.Train();

        // Create deployment configuration from individual configs
        var deploymentConfig = DeploymentConfiguration.Create(
            _quantizationConfig,
            _cacheConfig,
            _versioningConfig,
            _abTestingConfig,
            _telemetryConfig,
            _exportConfig,
            _gpuAccelerationConfig,
            _compressionConfig,
            _profilingConfig);

        // Create PredictionModelResult with meta-learning options
        var metaOptions = new PredictionModelResultOptions<T, TInput, TOutput>
        {
            MetaLearner = _metaLearner,
            MetaTrainingResult = metaResult,
            LoRAConfiguration = _loraConfiguration,
            BiasDetector = _biasDetector,
            FairnessEvaluator = _fairnessEvaluator,
            RagRetriever = _ragRetriever,
            RagReranker = _ragReranker,
            RagGenerator = _ragGenerator,
            QueryProcessors = _queryProcessors,
            AgentConfig = _agentConfig,
            DeploymentConfiguration = deploymentConfig,
            ReasoningConfig = _reasoningConfig,
            KnowledgeGraph = _knowledgeGraph,
            GraphStore = _graphStore,
            HybridGraphRetriever = _hybridGraphRetriever,
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            ProgramSynthesisModel = _programSynthesisModel,
            ProgramSynthesisServingClient = _programSynthesisServingClient,
            ProgramSynthesisServingClientOptions = _programSynthesisServingClientOptions,
            PromptTemplate = _promptTemplate,
            PromptChain = _promptChain,
            PromptOptimizer = _promptOptimizer,
            FewShotExampleSelector = _fewShotExampleSelector,
            PromptAnalyzer = _promptAnalyzer,
            PromptCompressor = _promptCompressor,
            ProfilingReport = profilerSession?.GetReport(),
            MemoryConfig = _memoryConfig
        };

        var result = new PredictionModelResult<T, TInput, TOutput>(metaOptions);

        return result;
    }

    private (IRLAgent<T> Agent, AutoMLRunSummary Summary) SelectRLAgentWithAutoML(RLTrainingOptions<T> rlTrainingOptions)
    {
        if (_autoMLOptions is null || _autoMLOptions.TaskFamilyOverride != AutoMLTaskFamily.ReinforcementLearning)
        {
            throw new InvalidOperationException("RL AutoML was requested but AutoML options are not configured for reinforcement learning.");
        }

        if (rlTrainingOptions.Environment is null)
        {
            throw new ArgumentException("RL training options must include a valid Environment.", nameof(rlTrainingOptions));
        }

        if (typeof(TInput) != typeof(AiDotNet.Tensors.LinearAlgebra.Vector<T>) || typeof(TOutput) != typeof(AiDotNet.Tensors.LinearAlgebra.Vector<T>))
        {
            throw new InvalidOperationException(
                $"RL AutoML requires PredictionModelBuilder<T, Vector<T>, Vector<T>>. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }

        if (_autoMLOptions.SearchStrategy != AutoMLSearchStrategy.RandomSearch)
        {
            throw new NotSupportedException(
                $"RL AutoML currently supports only '{AutoMLSearchStrategy.RandomSearch}'. Received '{_autoMLOptions.SearchStrategy}'.");
        }

        var (defaultTimeLimit, defaultTrialLimit) = AiDotNet.AutoML.AutoMLBudgetDefaults.Resolve(_autoMLOptions.Budget.Preset);
        var timeLimit = _autoMLOptions.Budget.TimeLimitOverride ?? defaultTimeLimit;
        var trialLimit = _autoMLOptions.Budget.TrialLimitOverride ?? defaultTrialLimit;

        var rlAutoOptions = _autoMLOptions.ReinforcementLearning ?? new RLAutoMLOptions<T>();
        var maxSteps = rlAutoOptions.MaxStepsPerEpisodeOverride ?? rlTrainingOptions.MaxStepsPerEpisode;

        if (rlTrainingOptions.Seed.HasValue)
        {
            rlTrainingOptions.Environment.Seed(rlTrainingOptions.Seed.Value);
        }

        var search = new AiDotNet.AutoML.RL.RandomSearchRLAutoML<T>(
            rlTrainingOptions.Environment,
            rlAutoOptions,
            timeLimit,
            trialLimit,
            maxSteps,
            rlTrainingOptions.Seed);

        return search.Search();
    }

    /// <summary>
    /// Internal method that performs reinforcement learning training.
    /// This contains all the core RL training logic.
    /// </summary>
    /// <param name="episodes">Number of episodes to train for.</param>
    /// <param name="verbose">Whether to print training progress.</param>
    /// <returns>A task that represents the asynchronous operation, containing the trained RL agent result.</returns>
#pragma warning disable CS1998
    private async Task<PredictionModelResult<T, TInput, TOutput>> BuildRLInternalAsync(int episodes, bool verbose)
    {
        // RL TRAINING PATH

        // Create profiler session if profiling is enabled
        var profilerSession = CreateProfilerSession();
        using var _ = profilerSession?.Scope("BuildRLInternalAsync");

        // Apply GPU configuration first (before any operations that might use GPU)
        ApplyGpuConfiguration();

        // Apply memory management configuration (gradient checkpointing, etc.)
        ApplyMemoryConfiguration();

        // Validate RL options are configured
        if (_rlOptions?.Environment is null)
        {
            throw new InvalidOperationException(
                "BuildRLInternalAsync requires ConfigureReinforcementLearning() with a valid Environment.");
        }

        AutoMLRunSummary? autoMLSummary = null;

        // AutoML can optionally select an RL agent when TaskFamilyOverride is set to ReinforcementLearning.
        if (_model is null)
        {
            if (_autoMLOptions?.TaskFamilyOverride == AutoMLTaskFamily.ReinforcementLearning)
            {
                var (selectedAgent, summary) = SelectRLAgentWithAutoML(_rlOptions);
                _model = (IFullModel<T, TInput, TOutput>)selectedAgent;
                autoMLSummary = summary;
            }
            else
            {
                throw new InvalidOperationException("Model (RL agent) must be specified using ConfigureModel().");
            }
        }

        if (_model is not IRLAgent<T> rlAgent)
            throw new InvalidOperationException(
                "The configured model must implement IRLAgent<T> for RL training. " +
                "Use RL agent types like DQNAgent, PPOAgent, SACAgent, etc.");

        // Track training metrics
        var episodeRewards = new List<T>();
        var episodeLengths = new List<int>();
        var losses = new List<T>();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        int totalStepsAcrossEpisodes = 0;

        var numOps = MathHelper.GetNumericOperations<T>();

        // Invoke OnTrainingStart callback
        _rlOptions.OnTrainingStart?.Invoke();

        if (verbose)
        {
            Console.WriteLine($"Starting RL training for {episodes} episodes...");
            Console.WriteLine($"Environment: {_rlOptions.Environment.GetType().Name}");
            Console.WriteLine($"Agent: {rlAgent.GetType().Name}");
            Console.WriteLine();
        }

        // Training loop
        for (int episode = 0; episode < episodes; episode++)
        {
            var state = _rlOptions.Environment.Reset();
            rlAgent.ResetEpisode();

            T episodeReward = numOps.Zero;
            int steps = 0;
            bool done = false;

            // Episode loop
            while (!done && steps < _rlOptions.MaxStepsPerEpisode)
            {
                // Select action
                var action = rlAgent.SelectAction(state, explore: true);

                // Take step in environment
                var (nextState, reward, isDone, info) = _rlOptions.Environment.Step(action);

                // Store experience
                rlAgent.StoreExperience(state, action, reward, nextState, isDone);

                // Train agent
                var loss = rlAgent.Train();
                bool didTrain = numOps.ToDouble(loss) > 0;
                if (didTrain)
                {
                    losses.Add(loss);
                }

                // Update for next step
                state = nextState;
                episodeReward = numOps.Add(episodeReward, reward);
                steps++;
                totalStepsAcrossEpisodes++;
                done = isDone;

                // Invoke OnStepComplete callback
                if (_rlOptions.OnStepComplete is not null)
                {
                    var stepMetrics = new RLStepMetrics<T>
                    {
                        Episode = episode + 1,
                        Step = steps,
                        TotalSteps = totalStepsAcrossEpisodes,
                        Reward = reward,
                        Loss = didTrain ? loss : default,
                        DidTrain = didTrain
                    };
                    _rlOptions.OnStepComplete(stepMetrics);
                }
            }

            episodeRewards.Add(episodeReward);
            episodeLengths.Add(steps);

            // Calculate metrics for this episode
            var recentRewards = episodeRewards.Skip(Math.Max(0, episodeRewards.Count - 100)).Take(100).ToList();
            var avgRewardRecent = recentRewards.Count > 0
                ? StatisticsHelper<T>.CalculateMean(recentRewards)
                : numOps.Zero;

            var recentLosses = losses.Skip(Math.Max(0, losses.Count - 100)).Take(100).ToList();
            var avgLoss = recentLosses.Count > 0
                ? StatisticsHelper<T>.CalculateMean(recentLosses)
                : numOps.Zero;

            // Print progress
            if (verbose && (episode + 1) % Math.Max(1, episodes / 10) == 0)
            {
                Console.WriteLine($"Episode {episode + 1}/{episodes} | " +
                                  $"Avg Reward (last 100): {numOps.ToDouble(avgRewardRecent):F2} | " +
                                  $"Avg Loss: {numOps.ToDouble(avgLoss):F6} | " +
                                  $"Steps: {steps}");
            }

            // Invoke OnEpisodeComplete callback
            if (_rlOptions.OnEpisodeComplete is not null)
            {
                var episodeMetrics = new RLEpisodeMetrics<T>
                {
                    Episode = episode + 1,
                    TotalReward = episodeReward,
                    Steps = steps,
                    AverageLoss = avgLoss,
                    TerminatedNaturally = done,
                    AverageRewardRecent = avgRewardRecent,
                    ElapsedTime = stopwatch.Elapsed
                };
                _rlOptions.OnEpisodeComplete(episodeMetrics);
            }
        }

        // Stop timing
        stopwatch.Stop();

        // Calculate final summary metrics
        var finalRecentRewards = episodeRewards.Skip(Math.Max(0, episodeRewards.Count - 100)).Take(100).ToList();
        var finalAvgReward = finalRecentRewards.Count > 0
            ? StatisticsHelper<T>.CalculateMean(finalRecentRewards)
            : numOps.Zero;

        var overallAvgReward = episodeRewards.Count > 0
            ? StatisticsHelper<T>.CalculateMean(episodeRewards)
            : numOps.Zero;

        var bestReward = episodeRewards.Count > 0
            ? episodeRewards.Aggregate(numOps.Zero, (max, r) => numOps.GreaterThan(r, max) ? r : max)
            : numOps.Zero;

        var overallAvgLoss = losses.Count > 0
            ? StatisticsHelper<T>.CalculateMean(losses)
            : numOps.Zero;

        if (verbose)
        {
            Console.WriteLine();
            Console.WriteLine("Training completed!");
            Console.WriteLine($"Final average reward (last 100 episodes): {numOps.ToDouble(finalAvgReward):F2}");
        }

        // Invoke OnTrainingComplete callback
        if (_rlOptions.OnTrainingComplete is not null)
        {
            var summary = new RLTrainingSummary<T>
            {
                TotalEpisodes = episodes,
                TotalSteps = totalStepsAcrossEpisodes,
                AverageReward = overallAvgReward,
                BestReward = bestReward,
                FinalAverageReward = finalAvgReward,
                AverageLoss = overallAvgLoss,
                TotalTime = stopwatch.Elapsed,
                EarlyStopTriggered = false
            };
            _rlOptions.OnTrainingComplete(summary);
        }

        // Create optimization result for RL training
        var optimizationResult = new OptimizationResult<T, TInput, TOutput>
        {
            BestSolution = _model
        };

        // Create normalization info with NoNormalizer (RL doesn't use normalization like supervised learning)
        var normInfo = new NormalizationInfo<T, TInput, TOutput>
        {
            Normalizer = new NoNormalizer<T, TInput, TOutput>()
        };
        PreprocessingInfo<T, TInput, TOutput>? preprocessingInfo = null;

        // Create deployment configuration from individual configs
        var deploymentConfig = DeploymentConfiguration.Create(
            _quantizationConfig,
            _cacheConfig,
            _versioningConfig,
            _abTestingConfig,
            _telemetryConfig,
            _exportConfig,
            _gpuAccelerationConfig,
            _compressionConfig,
            _profilingConfig);

        // Return standard PredictionModelResult
        // Note: This Build() overload doesn't perform JIT compilation (only the main Build() does),
        // so JitCompiledFunction is not set
        var rlOptions = new PredictionModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = optimizationResult,
            NormalizationInfo = normInfo,
            PreprocessingInfo = preprocessingInfo,
            AutoMLSummary = autoMLSummary,
            BiasDetector = _biasDetector,
            FairnessEvaluator = _fairnessEvaluator,
            RagRetriever = _ragRetriever,
            RagReranker = _ragReranker,
            RagGenerator = _ragGenerator,
            QueryProcessors = _queryProcessors,
            LoRAConfiguration = _loraConfiguration,
            AgentConfig = _agentConfig,
            DeploymentConfiguration = deploymentConfig,
            InferenceOptimizationConfig = _inferenceOptimizationConfig,
            ReasoningConfig = _reasoningConfig,
            KnowledgeGraph = _knowledgeGraph,
            GraphStore = _graphStore,
            HybridGraphRetriever = _hybridGraphRetriever,
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            ProgramSynthesisModel = _programSynthesisModel,
            ProgramSynthesisServingClient = _programSynthesisServingClient,
            ProgramSynthesisServingClientOptions = _programSynthesisServingClientOptions,
            PromptTemplate = _promptTemplate,
            PromptChain = _promptChain,
            PromptOptimizer = _promptOptimizer,
            FewShotExampleSelector = _fewShotExampleSelector,
            PromptAnalyzer = _promptAnalyzer,
            PromptCompressor = _promptCompressor,
            ProfilingReport = profilerSession?.GetReport(),
            MemoryConfig = _memoryConfig
        };

        var result = new PredictionModelResult<T, TInput, TOutput>(rlOptions);

        return result;
    }
#pragma warning restore CS1998

    /// <summary>
    /// Uses a trained model to make predictions on new data.
    /// </summary>
    /// <param name="newData">The matrix of new input features to predict outcomes for.</param>
    /// <param name="modelResult">The trained predictive model to use for making predictions.</param>
    /// <returns>A vector containing the predicted output values for each row in the input matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> After training your model with the Build method, you can use this method to get
    /// predictions for new data. For example, if you trained a model to predict house prices based on features
    /// like size and location, you can now give it details of houses currently for sale (without knowing their prices)
    /// and the model will predict what their prices should be.
    ///
    /// The input matrix should have the same number of columns (features) as the data you used to train the model.
    /// </remarks>
    public TOutput Predict(TInput newData, PredictionModelResult<T, TInput, TOutput> modelResult)
    {
        return modelResult.Predict(newData);
    }

    /// <summary>
    /// Saves a trained model to a file so it can be used later without retraining.
    /// </summary>
    /// <param name="modelResult">The trained predictive model to save.</param>
    /// <param name="filePath">The file path where the model should be saved.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Training a model can take time, so once you have a good model,
    /// you'll want to save it. This method lets you store your trained model in a file on your computer.
    /// Later, you can load this saved model and use it to make predictions without having to train it again.
    /// 
    /// Think of it like saving a document in a word processor - you can close the program and come back later
    /// to continue where you left off.
    /// </remarks>
    public void SaveModel(PredictionModelResult<T, TInput, TOutput> modelResult, string filePath)
    {
        File.WriteAllBytes(filePath, SerializeModel(modelResult));
    }

    /// <summary>
    /// Loads a previously saved model from a file.
    /// </summary>
    /// <param name="filePath">The file path where the model was saved.</param>
    /// <returns>The loaded predictive model that can be used to make predictions.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method lets you load a model that you previously saved using the SaveModel method.
    /// Once loaded, you can immediately use the model to make predictions without having to train it again.
    /// 
    /// This is useful when you want to use your model in different applications or at different times
    /// without the time and computational cost of retraining.
    /// </remarks>
    public PredictionModelResult<T, TInput, TOutput> LoadModel(string filePath)
    {
        byte[] modelData = File.ReadAllBytes(filePath);
        return DeserializeModel(modelData);
    }

    /// <summary>
    /// Converts a trained model into a byte array for storage or transmission.
    /// </summary>
    /// <param name="modelResult">The trained predictive model to serialize.</param>
    /// <returns>A byte array representing the serialized model.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Serialization converts your model into a format (a series of bytes) that can be
    /// easily stored or sent over a network. This is the underlying mechanism that makes saving models possible.
    /// 
    /// You might use this directly if you want to store the model in a database or send it over a network,
    /// rather than saving it to a file.
    /// </remarks>
    public byte[] SerializeModel(PredictionModelResult<T, TInput, TOutput> modelResult)
    {
        return modelResult.Serialize();
    }

    /// <summary>
    /// Converts a byte array back into a usable predictive model.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <returns>The deserialized predictive model that can be used to make predictions.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Deserialization is the opposite of serialization - it takes the byte array
    /// representation of your model and converts it back into a usable model object. This is what happens
    /// behind the scenes when you load a model from a file.
    /// 
    /// You might use this directly if you retrieved a serialized model from a database or received it over a network.
    /// </remarks>
    public PredictionModelResult<T, TInput, TOutput> DeserializeModel(byte[] modelData)
    {
        var result = new PredictionModelResult<T, TInput, TOutput>();
        result.Deserialize(modelData);

        // Automatically reattach Graph RAG components if they were configured on this builder
        // Graph RAG components cannot be serialized (file handles, WAL, etc.), so we reattach
        // them from the builder's configuration to provide a seamless experience for users
        if (_knowledgeGraph != null || _graphStore != null || _hybridGraphRetriever != null)
        {
            result.AttachGraphComponents(_knowledgeGraph, _graphStore, _hybridGraphRetriever);
        }

        // Reattach tokenizer if configured
        if (_tokenizer != null)
        {
            result.AttachTokenizer(_tokenizer, _tokenizationConfig);
        }

        // Reattach prompt engineering components if configured
        AttachPromptEngineeringConfiguration(result);

        return result;
    }

    /// <summary>
    /// Configures the bias detector component for ethical AI evaluation.
    /// </summary>
    /// <param name="detector">The bias detector implementation to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Bias detection helps ensure your model treats different groups fairly.
    /// You can choose from different bias detection strategies like Disparate Impact (80% rule),
    /// Demographic Parity, or Equal Opportunity. This component will be used to evaluate your
    /// trained model's fairness across demographic groups.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureBiasDetector(IBiasDetector<T> detector)
    {
        _biasDetector = detector;
        return this;
    }

    /// <summary>
    /// Configures the fairness evaluator component for ethical AI evaluation.
    /// </summary>
    /// <param name="evaluator">The fairness evaluator implementation to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Fairness evaluation measures how equitably your model performs.
    /// You can choose evaluators that compute different sets of fairness metrics, from basic
    /// (just key metrics) to comprehensive (all fairness measures). This helps ensure your
    /// AI system is not only accurate but also ethical.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFairnessEvaluator(IFairnessEvaluator<T> evaluator)
    {
        _fairnessEvaluator = evaluator;
        return this;
    }

    /// <summary>
    /// Configures adversarial robustness and AI safety features for the model.
    /// </summary>
    /// <param name="configuration">The adversarial robustness configuration. When null, uses industry-standard defaults.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This unified configuration provides comprehensive control over all aspects of adversarial robustness and AI safety:
    /// </para>
    /// <list type="bullet">
    /// <item><term>Safety Filtering</term><description>Input validation and output filtering for harmful content</description></item>
    /// <item><term>Adversarial Attacks</term><description>FGSM, PGD, CW, AutoAttack for robustness testing</description></item>
    /// <item><term>Adversarial Defenses</term><description>Adversarial training, input preprocessing, ensemble methods</description></item>
    /// <item><term>Certified Robustness</term><description>Randomized smoothing, IBP, CROWN for provable guarantees</description></item>
    /// <item><term>Content Moderation</term><description>Prompt injection detection, PII filtering for LLMs</description></item>
    /// <item><term>Red Teaming</term><description>Automated adversarial prompt generation for evaluation</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> This is your one-stop configuration for making your model safe and robust.
    /// When called with no parameters (null), industry-standard defaults are applied automatically.
    /// You can use factory methods like <c>AdversarialRobustnessConfiguration.BasicSafety()</c> for common setups,
    /// or customize individual options for your specific needs.</para>
    /// <example>
    /// <code>
    /// // Use industry-standard defaults
    /// builder.ConfigureAdversarialRobustness();
    ///
    /// // Basic safety filtering
    /// builder.ConfigureAdversarialRobustness(AdversarialRobustnessConfiguration&lt;double, Vector&lt;double&gt;, int&gt;.BasicSafety());
    ///
    /// // Comprehensive robustness with certified guarantees
    /// builder.ConfigureAdversarialRobustness(AdversarialRobustnessConfiguration&lt;double, Vector&lt;double&gt;, int&gt;.Comprehensive());
    ///
    /// // LLM safety with content moderation
    /// builder.ConfigureAdversarialRobustness(AdversarialRobustnessConfiguration&lt;double, string, string&gt;.ForLLM());
    ///
    /// // Custom configuration
    /// builder.ConfigureAdversarialRobustness(new AdversarialRobustnessConfiguration&lt;double, Vector&lt;double&gt;, int&gt;
    /// {
    ///     Enabled = true,
    ///     Options = new AdversarialRobustnessOptions&lt;double&gt;
    ///     {
    ///         EnableSafetyFiltering = true,
    ///         EnableAdversarialTraining = true,
    ///         EnableCertifiedRobustness = true
    ///     },
    ///     UseCertifiedInference = true
    /// });
    /// </code>
    /// </example>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureAdversarialRobustness(
        AdversarialRobustnessConfiguration<T, TInput, TOutput>? configuration = null)
    {
        _adversarialRobustnessConfiguration = configuration ?? new AdversarialRobustnessConfiguration<T, TInput, TOutput>();
        return this;
    }

    /// <summary>
    /// Configures fine-tuning for the model using preference learning, RLHF, or other alignment methods.
    /// </summary>
    /// <param name="configuration">The fine-tuning configuration including training data. When null, uses industry-standard defaults.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This configuration enables post-training fine-tuning using various alignment techniques:
    /// </para>
    /// <list type="bullet">
    /// <item><term>Supervised Fine-Tuning (SFT)</term><description>Traditional fine-tuning on labeled examples</description></item>
    /// <item><term>Direct Preference Optimization (DPO)</term><description>Learn from human preferences without reward models</description></item>
    /// <item><term>Simple Preference Optimization (SimPO)</term><description>Reference-free, length-normalized preference learning</description></item>
    /// <item><term>Group Relative Policy Optimization (GRPO)</term><description>Memory-efficient RL without critic models</description></item>
    /// <item><term>Odds Ratio Preference Optimization (ORPO)</term><description>Combined SFT + preference in one step</description></item>
    /// <item><term>Identity Preference Optimization (IPO)</term><description>Regularized preference optimization</description></item>
    /// <item><term>Kahneman-Tversky Optimization (KTO)</term><description>Utility-maximizing preference learning</description></item>
    /// <item><term>Contrastive Preference Optimization (CPO)</term><description>Contrastive learning for preferences</description></item>
    /// <item><term>Constitutional AI (CAI)</term><description>Self-improvement with constitutional principles</description></item>
    /// <item><term>Reinforcement Learning from Human Feedback (RLHF)</term><description>Classic PPO-based alignment</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> Fine-tuning helps align your model with human preferences.
    /// When called with no parameters (null), industry-standard defaults are applied automatically.
    /// Training data should be set in the configuration's TrainingData property.
    /// Use factory methods like <c>FineTuningConfiguration.ForDPO(data)</c> for quick setup.
    /// DPO and SimPO are simpler (no reward model needed), while RLHF and GRPO provide more control.</para>
    /// <example>
    /// <code>
    /// // Use industry-standard defaults (training data set separately)
    /// builder.ConfigureFineTuning();
    ///
    /// // DPO fine-tuning with preference pairs
    /// var preferenceData = new FineTuningData&lt;double, string, string&gt;
    /// {
    ///     Inputs = prompts,
    ///     ChosenOutputs = preferredResponses,
    ///     RejectedOutputs = rejectedResponses
    /// };
    /// builder.ConfigureFineTuning(FineTuningConfiguration&lt;double, string, string&gt;.ForDPO(preferenceData));
    ///
    /// // GRPO for RL-based alignment
    /// var rlData = new FineTuningData&lt;double, string, string&gt;
    /// {
    ///     Inputs = prompts,
    ///     Rewards = rewardScores
    /// };
    /// builder.ConfigureFineTuning(FineTuningConfiguration&lt;double, string, string&gt;.ForGRPO(rlData));
    ///
    /// // Custom fine-tuning configuration
    /// builder.ConfigureFineTuning(new FineTuningConfiguration&lt;double, Vector&lt;double&gt;, int&gt;
    /// {
    ///     Enabled = true,
    ///     Options = new FineTuningOptions&lt;double&gt;
    ///     {
    ///         MethodType = FineTuningMethodType.SimPO,
    ///         LearningRate = 1e-5,
    ///         Epochs = 3,
    ///         SimPOGamma = 1.0
    ///     },
    ///     TrainingData = myPreferenceData
    /// });
    /// </code>
    /// </example>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFineTuning(
        FineTuningConfiguration<T, TInput, TOutput>? configuration = null)
    {
        _fineTuningConfiguration = configuration ?? new FineTuningConfiguration<T, TInput, TOutput>();
        return this;
    }

    /// <summary>
    /// Configures a multi-stage training pipeline for advanced training workflows.
    /// </summary>
    /// <param name="configuration">
    /// The training pipeline configuration defining the stages to execute.
    /// When null, uses the default single-stage training based on other configured settings.
    /// </param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// ConfigureTrainingPipeline enables advanced multi-stage training workflows where each stage
    /// can have its own training method, optimizer, learning rate, and dataset. Stages execute
    /// sequentially, with each stage's output model becoming the next stage's input.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as a recipe with multiple cooking steps.
    /// Just like you might marinate, then sear, then bake - training can have multiple
    /// phases where each phase teaches the model something different.</para>
    /// <para>
    /// <b>Common Training Pipelines:</b>
    /// <list type="bullet">
    /// <item><term>Standard Alignment</term><description>SFT → DPO (most common for chat models)</description></item>
    /// <item><term>Full RLHF</term><description>SFT → Reward Model → PPO</description></item>
    /// <item><term>Constitutional AI</term><description>SFT → CAI critique/revision → preference</description></item>
    /// <item><term>Curriculum Learning</term><description>Easy data → Medium → Hard (progressive difficulty)</description></item>
    /// <item><term>Iterative Refinement</term><description>Multiple DPO rounds with decreasing beta</description></item>
    /// </list>
    /// </para>
    /// <example>
    /// <code>
    /// // Standard alignment pipeline (SFT → DPO)
    /// builder.ConfigureTrainingPipeline(
    ///     TrainingPipelineConfiguration&lt;double, string, string&gt;.StandardAlignment(sftData, preferenceData));
    ///
    /// // Automatic pipeline based on available data
    /// builder.ConfigureTrainingPipeline(
    ///     TrainingPipelineConfiguration&lt;double, string, string&gt;.Auto(myData));
    ///
    /// // Custom multi-stage pipeline with builder pattern
    /// var pipeline = new TrainingPipelineConfiguration&lt;double, string, string&gt;()
    ///     .AddSFTStage(stage => {
    ///         stage.TrainingData = sftData;
    ///         stage.Options = new FineTuningOptions&lt;double&gt; { Epochs = 3 };
    ///     })
    ///     .AddPreferenceStage(FineTuningMethodType.DPO, stage => {
    ///         stage.TrainingData = preferenceData;
    ///         stage.Options = new FineTuningOptions&lt;double&gt; { Beta = 0.1 };
    ///     })
    ///     .AddEvaluationStage();
    /// builder.ConfigureTrainingPipeline(pipeline);
    ///
    /// // Iterative refinement with multiple DPO rounds
    /// builder.ConfigureTrainingPipeline(
    ///     TrainingPipelineConfiguration&lt;double, string, string&gt;.IterativeRefinement(3, sftData, preferenceData));
    ///
    /// // Custom stage with user-defined training logic
    /// var customPipeline = new TrainingPipelineConfiguration&lt;double, string, string&gt;()
    ///     .AddSFTStage()
    ///     .AddCustomStage("My Custom Training", async (model, data, ct) => {
    ///         // Custom training logic
    ///         return model;
    ///     });
    /// builder.ConfigureTrainingPipeline(customPipeline);
    /// </code>
    /// </example>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureTrainingPipeline(
        TrainingPipelineConfiguration<T, TInput, TOutput>? configuration = null)
    {
        _trainingPipelineConfiguration = configuration;
        return this;
    }

    /// <summary>
    /// Configures LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    /// </summary>
    /// <param name="loraConfiguration">The LoRA configuration implementation to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> LoRA enables parameter-efficient fine-tuning by adding small "correction layers"
    /// to your neural network. This lets you adapt large pre-trained models with 100x fewer parameters,
    /// making fine-tuning much faster and more memory-efficient. The configuration determines which layers
    /// get LoRA adaptations and how they behave (rank, scaling, freezing).
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureLoRA(ILoRAConfiguration<T> loraConfiguration)
    {
        _loraConfiguration = loraConfiguration;
        return this;
    }

    /// <summary>
    /// Configures the retrieval-augmented generation (RAG) components for use during model inference.
    /// </summary>
    /// <param name="retriever">Optional retriever for finding relevant documents. If not provided, standard RAG won't be available.</param>
    /// <param name="reranker">Optional reranker for improving document ranking quality. If not provided, a default reranker will be used if RAG is configured.</param>
    /// <param name="generator">Optional generator for producing grounded answers. If not provided, a default generator will be used if RAG is configured.</param>
    /// <param name="queryProcessors">Optional query processors for improving search quality.</param>
    /// <param name="graphStore">Optional graph storage backend for Graph RAG (e.g., MemoryGraphStore, FileGraphStore).</param>
    /// <param name="knowledgeGraph">Optional pre-configured knowledge graph. If null but graphStore is provided, a new one is created.</param>
    /// <param name="documentStore">Optional document store for hybrid vector + graph retrieval.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RAG combines retrieval and generation to create answers backed by real documents.
    /// Configure it with:
    /// <list type="bullet">
    /// <item><description>A retriever (finds relevant documents from your collection) - required for standard RAG</description></item>
    /// <item><description>A reranker (improves the ordering of retrieved documents) - optional, defaults provided</description></item>
    /// <item><description>A generator (creates answers based on the documents) - optional, defaults provided</description></item>
    /// <item><description>Optional query processors (improve search queries before retrieval)</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Graph RAG:</b> When graphStore or knowledgeGraph is provided, enables knowledge graph-based
    /// retrieval that finds related entities and their relationships, providing richer context than
    /// vector similarity alone. Traditional RAG finds similar documents using vectors. Graph RAG goes further by
    /// also exploring relationships between entities. For example, if you ask about "Paris", it can find
    /// not just documents mentioning Paris, but also related concepts like France, Eiffel Tower, and Seine River.
    /// </para>
    /// <para>
    /// <b>Hybrid Retrieval:</b> When both knowledgeGraph and documentStore are provided, creates a
    /// HybridGraphRetriever that combines vector search and graph traversal for optimal results.
    /// </para>
    /// <para>
    /// <b>Disabling RAG:</b> Call with all parameters as null to disable RAG functionality completely.
    /// </para>
    /// <para>
    /// RAG operations are performed during inference (after model training) via the PredictionModelResult.
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureRetrievalAugmentedGeneration(
        IRetriever<T>? retriever = null,
        IReranker<T>? reranker = null,
        IGenerator<T>? generator = null,
        IEnumerable<IQueryProcessor>? queryProcessors = null,
        IGraphStore<T>? graphStore = null,
        KnowledgeGraph<T>? knowledgeGraph = null,
        IDocumentStore<T>? documentStore = null)
    {
        // Configure standard RAG components
        _ragRetriever = retriever;
        _ragReranker = reranker;
        _ragGenerator = generator;
        _queryProcessors = queryProcessors;

        // Configure Graph RAG components
        // If all Graph RAG parameters are null, clear Graph RAG fields
        if (graphStore == null && knowledgeGraph == null && documentStore == null)
        {
            _graphStore = null;
            _knowledgeGraph = null;
            _hybridGraphRetriever = null;
            return this;
        }

        _graphStore = graphStore;

        // Use provided knowledge graph or create one from the store
        if (knowledgeGraph != null)
        {
            _knowledgeGraph = knowledgeGraph;
        }
        else if (graphStore != null)
        {
            _knowledgeGraph = new KnowledgeGraph<T>(graphStore);
        }
        else
        {
            // No knowledge graph source provided, clear the field
            _knowledgeGraph = null;
        }

        // Create or clear hybrid retriever based on available components
        if (_knowledgeGraph != null && documentStore != null)
        {
            _hybridGraphRetriever = new HybridGraphRetriever<T>(_knowledgeGraph, documentStore);
        }
        else
        {
            // Clear hybrid retriever if dependencies are missing
            _hybridGraphRetriever = null;
        }

        return this;
    }

    /// <summary>
    /// Configures the model evaluator component for comprehensive model evaluation and cross-validation.
    /// </summary>
    /// <param name="evaluator">The model evaluator implementation to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The model evaluator helps you understand how well your model performs.
    /// If you configure both a model evaluator and cross-validator (via ConfigureCrossValidation),
    /// cross-validation will automatically run during Build() on your training data, and the results
    /// will be included in your trained model.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureModelEvaluator(IModelEvaluator<T, TInput, TOutput> evaluator)
    {
        _modelEvaluator = evaluator;
        return this;
    }

    /// <summary>
    /// Configures the cross-validation strategy for automatic model evaluation during training.
    /// </summary>
    /// <param name="crossValidator">The cross-validation strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Cross-validation tests how well your model will perform on new data
    /// by training and testing it multiple times on different subsets of your training data.
    /// If you configure both a cross-validator and model evaluator (via ConfigureModelEvaluator),
    /// cross-validation will automatically run during Build() and the results will be included
    /// in your trained model.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureCrossValidation(ICrossValidator<T, TInput, TOutput> crossValidator)
    {
        _crossValidator = crossValidator;
        return this;
    }

    /// <summary>
    /// Configures an AutoML model for automatic machine learning optimization.
    /// </summary>
    /// <param name="autoMLModel">The AutoML model instance to use for hyperparameter search and model selection.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML (Automated Machine Learning) automatically searches for the best
    /// model and hyperparameters for your problem. Instead of manually trying different models and settings,
    /// AutoML does this for you.
    /// </para>
    /// <para>
    /// When you configure an AutoML model:
    /// - The Build() method will run the AutoML search process
    /// - AutoML will try different models and hyperparameters
    /// - The best model found will be returned as your trained model
    /// - You can configure search time limits, candidate models, and optimization metrics
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// // Advanced usage: plug in your own AutoML implementation.
    /// // Most users should prefer the ConfigureAutoML(AutoMLOptions&lt;...&gt;) overload instead.
    /// var autoML = new RandomSearchAutoML&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;();
    /// autoML.SetTimeLimit(TimeSpan.FromMinutes(30));
    /// autoML.SetCandidateModels(new List&lt;ModelType&gt; { ModelType.RandomForest, ModelType.GradientBoosting });
    ///
    /// var builder = new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureAutoML(autoML)
    ///     .Build(trainingData, trainingLabels);
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureAutoML(IAutoMLModel<T, TInput, TOutput> autoMLModel)
    {
        _autoMLModel = autoMLModel;
        _autoMLOptions = null;
        return this;
    }

    /// <summary>
    /// Configures AutoML using facade-style options (recommended for most users).
    /// </summary>
    /// <param name="options">AutoML options (budget, strategy, and optional overrides). If null, defaults are used.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML automatically tries different models/settings to find a strong result.
    /// With this overload you only choose a budget (how much time to spend), and AiDotNet handles the rest.
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureAutoML(AutoMLOptions<T, TInput, TOutput>? options = null)
    {
        _autoMLOptions = options ?? new AutoMLOptions<T, TInput, TOutput>();

        var (defaultTimeLimit, defaultTrialLimit) = AiDotNet.AutoML.AutoMLBudgetDefaults.Resolve(_autoMLOptions.Budget.Preset);
        var timeLimit = _autoMLOptions.Budget.TimeLimitOverride ?? defaultTimeLimit;
        var trialLimit = _autoMLOptions.Budget.TrialLimitOverride ?? defaultTrialLimit;

        if (_autoMLOptions.TaskFamilyOverride == AutoMLTaskFamily.ReinforcementLearning)
        {
            if (_autoMLOptions.SearchStrategy != AutoMLSearchStrategy.RandomSearch)
            {
                throw new NotSupportedException(
                    $"RL AutoML currently supports only '{AutoMLSearchStrategy.RandomSearch}'. Received '{_autoMLOptions.SearchStrategy}'.");
            }

            // RL AutoML runs through the RL training path and selects an IRLAgent; it does not use the supervised IAutoMLModel pipeline.
            _autoMLModel = null;
            return this;
        }

        if (_autoMLOptions.TaskFamilyOverride is AutoMLTaskFamily taskFamilyOverride
            && !IsBuiltInSupervisedTaskFamilySupported(taskFamilyOverride))
        {
            throw new NotSupportedException(
                $"Facade AutoML options currently support only Regression/Binary/MultiClass/TimeSeriesForecasting/TimeSeriesAnomalyDetection/Ranking/Recommendation task families. " +
                $"Received '{taskFamilyOverride}'. Use {nameof(ConfigureAutoML)}(IAutoMLModel<...>) to plug in a custom implementation.");
        }

        _autoMLModel = CreateBuiltInAutoMLModel(_autoMLOptions.SearchStrategy);
        _autoMLModel.TimeLimit = timeLimit;
        _autoMLModel.TrialLimit = trialLimit;

        if (_autoMLModel is AiDotNet.AutoML.SupervisedAutoMLModelBase<T, TInput, TOutput> supervised)
        {
            supervised.EnsembleOptions = _autoMLOptions.Ensembling ?? ResolveDefaultEnsembling(_autoMLOptions.Budget.Preset);
            supervised.BudgetPreset = _autoMLOptions.Budget.Preset;
        }

        if (_autoMLOptions.OptimizationMetricOverride.HasValue)
        {
            var metric = _autoMLOptions.OptimizationMetricOverride.Value;
            _autoMLModel.SetOptimizationMetric(metric, maximize: IsHigherBetter(metric));
        }
        else if (_autoMLOptions.TaskFamilyOverride is AutoMLTaskFamily familyOverride)
        {
            var (metric, maximize) = AutoMLDefaultMetricPolicy.GetDefault(familyOverride);
            _autoMLModel.SetOptimizationMetric(metric, maximize);
        }

        return this;
    }

    /// <summary>
    /// Configures curriculum learning for training with ordered sample difficulty.
    /// </summary>
    /// <param name="options">Curriculum learning options (schedule type, phases, difficulty estimation).
    /// If null, sensible defaults are used (Linear schedule, 5 phases, loss-based difficulty).</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Curriculum Learning trains models by presenting samples in order of difficulty,
    /// starting with easy examples and gradually introducing harder ones. This often leads to faster
    /// convergence and better final performance compared to random training order.</para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureCurriculumLearning(
        CurriculumLearningOptions<T, TInput, TOutput>? options = null)
    {
        _curriculumLearningOptions = options ?? new CurriculumLearningOptions<T, TInput, TOutput>();
        return this;
    }

    private static AutoMLEnsembleOptions ResolveDefaultEnsembling(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.Standard => new AutoMLEnsembleOptions { Enabled = true, MaxModelCount = 3 },
            AutoMLBudgetPreset.Thorough => new AutoMLEnsembleOptions { Enabled = true, MaxModelCount = 5 },
            _ => new AutoMLEnsembleOptions { Enabled = false, MaxModelCount = 3 }
        };
    }

    private static bool IsHigherBetter(MetricType metric)
    {
        return metric switch
        {
            MetricType.MeanSquaredError => false,
            MetricType.RootMeanSquaredError => false,
            MetricType.MeanAbsoluteError => false,
            MetricType.MSE => false,
            MetricType.RMSE => false,
            MetricType.MAE => false,
            MetricType.MAPE => false,
            MetricType.SMAPE => false,
            MetricType.MeanSquaredLogError => false,
            MetricType.CrossEntropyLoss => false,
            MetricType.AIC => false,
            MetricType.BIC => false,
            MetricType.AICAlt => false,
            MetricType.Perplexity => false,
            _ => true
        };
    }

    private static bool IsBuiltInSupervisedTaskFamilySupported(AutoMLTaskFamily taskFamily)
    {
        return taskFamily == AutoMLTaskFamily.Regression
               || taskFamily == AutoMLTaskFamily.BinaryClassification
               || taskFamily == AutoMLTaskFamily.MultiClassClassification
               || taskFamily == AutoMLTaskFamily.TimeSeriesForecasting
               || taskFamily == AutoMLTaskFamily.TimeSeriesAnomalyDetection
               || taskFamily == AutoMLTaskFamily.Ranking
               || taskFamily == AutoMLTaskFamily.Recommendation;
    }

    private IAutoMLModel<T, TInput, TOutput> CreateBuiltInAutoMLModel(AutoMLSearchStrategy strategy)
    {
        return strategy switch
        {
            AutoMLSearchStrategy.RandomSearch => new AiDotNet.AutoML.RandomSearchAutoML<T, TInput, TOutput>(_modelEvaluator, RandomHelper.CreateSecureRandom()),
            AutoMLSearchStrategy.BayesianOptimization => new AiDotNet.AutoML.BayesianOptimizationAutoML<T, TInput, TOutput>(_modelEvaluator, RandomHelper.CreateSecureRandom()),
            AutoMLSearchStrategy.Evolutionary => new AiDotNet.AutoML.EvolutionaryAutoML<T, TInput, TOutput>(_modelEvaluator, RandomHelper.CreateSecureRandom()),
            AutoMLSearchStrategy.MultiFidelity => new AiDotNet.AutoML.MultiFidelityAutoML<T, TInput, TOutput>(_modelEvaluator, RandomHelper.CreateSecureRandom(), _autoMLOptions?.MultiFidelity),
            AutoMLSearchStrategy.NeuralArchitectureSearch or
            AutoMLSearchStrategy.DARTS or
            AutoMLSearchStrategy.GDAS or
            AutoMLSearchStrategy.OnceForAll => CreateNasAutoMLModel(strategy),
            _ => throw new NotSupportedException(
                $"AutoML search strategy '{strategy}' is not available via the facade options overload. " +
                $"Use {nameof(ConfigureAutoML)}(IAutoMLModel<...>) to plug in a custom implementation.")
        };
    }

    /// <summary>
    /// Creates a NAS-based AutoML model with the specified strategy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// NAS strategies require TInput and TOutput to be <see cref="Tensor{T}"/> types.
    /// If the types don't match, this method throws a helpful exception.
    /// </para>
    /// <para>
    /// <b>Industry Defaults:</b> When NAS options are not specified, sensible defaults are used:
    /// <list type="bullet">
    /// <item><description>SearchSpace: <see cref="MobileNetSearchSpace{T}"/> (efficient for most use cases)</description></item>
    /// <item><description>NumNodes: 4 (balanced complexity)</description></item>
    /// <item><description>GDAS temperature: 5.0 initial, 0.1 final (proven values from research)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private IAutoMLModel<T, TInput, TOutput> CreateNasAutoMLModel(AutoMLSearchStrategy strategy)
    {
        // NAS models specifically work with Tensor<T> inputs/outputs.
        // Validate type compatibility at runtime.
        if (typeof(TInput) != typeof(Tensor<T>) || typeof(TOutput) != typeof(Tensor<T>))
        {
            throw new NotSupportedException(
                $"Neural Architecture Search strategies ({strategy}) require TInput and TOutput to be Tensor<T>. " +
                $"Current types are TInput={typeof(TInput).Name}, TOutput={typeof(TOutput).Name}. " +
                $"Consider using PredictionModelBuilder<{typeof(T).Name}, Tensor<{typeof(T).Name}>, Tensor<{typeof(T).Name}>> for NAS.");
        }

        // Resolve NAS options with industry-standard defaults.
        var nasOptions = _autoMLOptions?.NAS;
        var searchSpace = nasOptions?.SearchSpace ?? new MobileNetSearchSpace<T>();
        var numNodes = Math.Max(searchSpace.MaxNodes, 4);

        // Create the appropriate NAS model based on strategy.
        NasAutoMLModelBase<T> nasModel = strategy switch
        {
            AutoMLSearchStrategy.DARTS => new GDAS<T>(
                searchSpace,
                numNodes,
                nasOptions?.ArchitectureLearningRate ?? 5.0,    // Initial temperature (GDAS uses temp, not LR)
                0.1),                                            // Final temperature

            AutoMLSearchStrategy.GDAS => new GDAS<T>(
                searchSpace,
                numNodes,
                nasOptions?.ArchitectureLearningRate ?? 5.0,    // Initial temperature
                0.1),                                            // Final temperature

            AutoMLSearchStrategy.OnceForAll => new OnceForAll<T>(
                searchSpace,
                nasOptions?.ElasticDepths,
                nasOptions?.ElasticWidths,
                nasOptions?.ElasticKernelSizes,
                nasOptions?.ElasticExpansionRatios),

            AutoMLSearchStrategy.NeuralArchitectureSearch => SelectBestNasStrategy(searchSpace, numNodes, nasOptions),

            _ => throw new NotSupportedException($"NAS strategy '{strategy}' is not implemented.")
        };

        // Configure NAS model with time/trial limits from parent options.
        // The SearchAsync will receive proper limits when called.

        // Cast via object to satisfy generic constraints (we validated types above).
        return (IAutoMLModel<T, TInput, TOutput>)(object)nasModel;
    }

    /// <summary>
    /// Auto-selects the best NAS strategy based on task characteristics.
    /// </summary>
    /// <remarks>
    /// <para><b>Selection Heuristics:</b></para>
    /// <list type="bullet">
    /// <item><description>Mobile/Edge platforms: OnceForAll (elastic deployment)</description></item>
    /// <item><description>Quick search (&lt;2 hours): GDAS (fast gradient-based)</description></item>
    /// <item><description>Default: GDAS (proven balance of speed and quality)</description></item>
    /// </list>
    /// </remarks>
    private NasAutoMLModelBase<T> SelectBestNasStrategy(SearchSpaceBase<T> searchSpace, int numNodes, NASOptions<T>? nasOptions)
    {
        // If targeting mobile/edge, use OFA for elastic deployment.
        if (nasOptions?.TargetPlatform is HardwarePlatform.Mobile or HardwarePlatform.EdgeTPU)
        {
            return new OnceForAll<T>(
                searchSpace,
                nasOptions?.ElasticDepths,
                nasOptions?.ElasticWidths,
                nasOptions?.ElasticKernelSizes,
                nasOptions?.ElasticExpansionRatios);
        }

        // Default to GDAS - good balance of speed and architecture quality.
        return new GDAS<T>(
            searchSpace,
            numNodes,
            nasOptions?.ArchitectureLearningRate ?? 5.0,
            0.1);
    }

    /// <summary>
    /// Configures a meta-learning algorithm for training models that can quickly adapt to new tasks.
    /// </summary>
    /// <param name="metaLearner">The meta-learning algorithm to use (e.g., ReptileTrainer).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> If you configure this, Build() will do meta-training instead of regular training.
    /// The meta-learner should be created with all its dependencies (model, loss function, episodic data loader).
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureMetaLearning(IMetaLearner<T, TInput, TOutput> metaLearner)
    {
        _metaLearner = metaLearner;
        return this;
    }

    /// <summary>
    /// Configures distributed training across multiple GPUs or machines.
    /// </summary>
    /// <param name="backend">Communication backend to use. If null, uses InMemoryCommunicationBackend.</param>
    /// <param name="strategy">Distributed training strategy. Default is DDP.</param>
    /// <param name="configuration">Optional sharding configuration for advanced settings like gradient compression, parameter grouping, etc.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// When distributed training is configured, the Build() method will automatically wrap
    /// the model and optimizer with their distributed counterparts based on the chosen strategy.
    /// This enables training across multiple GPUs or machines with automatic parameter
    /// sharding and gradient synchronization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This enables distributed training across multiple GPUs or machines.
    /// You can call it with no parameters for sensible defaults, or customize as needed.
    ///
    /// When you configure this, the builder automatically handles all the complexity:
    /// - Your model gets split across GPUs (parameter sharding)
    /// - Gradients are synchronized automatically
    /// - Training is coordinated across all processes
    ///
    /// You just train as normal - the distributed magic happens behind the scenes!
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureDistributedTraining(
        ICommunicationBackend<T>? backend = null,
        DistributedStrategy strategy = DistributedStrategy.DDP,
        IShardingConfiguration<T>? configuration = null)
    {
        _distributedBackend = backend;
        _distributedStrategy = strategy;
        _distributedConfiguration = configuration;
        return this;
    }

    /// <summary>
    /// Enables AI agent assistance during the model building process.
    /// </summary>
    /// <param name="configuration">The agent configuration containing API key, provider, and assistance options.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This enables an AI agent to help you during model building.
    /// By default, the agent will:
    /// - Analyze your data characteristics
    /// - Suggest appropriate model types (if you haven't chosen one)
    /// - Recommend hyperparameter values
    /// - Provide insights on feature importance
    ///
    /// The API key is stored securely and will be reused during inference if you call AskAsync() on the trained model.
    ///
    /// Example with defaults:
    /// <code>
    /// var agentConfig = new AgentConfiguration&lt;double&gt;
    /// {
    ///     ApiKey = "sk-...",
    ///     Provider = LLMProvider.OpenAI,
    ///     IsEnabled = true
    /// };
    ///
    /// var result = await new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureAgentAssistance(agentConfig)
    ///     .BuildAsync();
    /// </code>
    ///
    /// Example with customization:
    /// <code>
    /// var agentConfig = new AgentConfiguration&lt;double&gt;
    /// {
    ///     ApiKey = "sk-...",
    ///     Provider = LLMProvider.OpenAI,
    ///     IsEnabled = true,
    ///     AssistanceOptions = AgentAssistanceOptions.Create()
    ///         .EnableModelSelection()
    ///         .EnableHyperparameterTuning()
    ///         .DisableFeatureAnalysis()
    /// };
    ///
    /// var result = await new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureAgentAssistance(agentConfig)
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureAgentAssistance(AgentConfiguration<T> configuration)
    {
        _agentConfig = configuration;
        _agentOptions = configuration.AssistanceOptions ?? AgentAssistanceOptions.Default;
        return this;
    }

    /// <summary>
    /// Configures reinforcement learning options for training an RL agent.
    /// </summary>
    /// <param name="options">The reinforcement learning configuration options.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Reinforcement learning trains an agent through trial and error
    /// in an environment. This method configures all aspects of RL training:
    /// - The environment (simulation/game for the agent to learn from)
    /// - Training parameters (episodes, steps, batch size)
    /// - Exploration strategies (how to balance trying new things vs using learned behavior)
    /// - Replay buffers (how to store and sample past experiences)
    /// - Callbacks for monitoring training progress
    ///
    /// After configuring RL options, use BuildAsync(episodes) to train the agent.
    ///
    /// Example:
    /// <code>
    /// var options = new RLTrainingOptions&lt;double&gt;
    /// {
    ///     Environment = new CartPoleEnvironment&lt;double&gt;(),
    ///     Episodes = 1000,
    ///     MaxStepsPerEpisode = 500,
    ///     OnEpisodeComplete = (metrics) =&gt; Console.WriteLine($"Episode {metrics.Episode}: {metrics.TotalReward}")
    /// };
    ///
    /// var result = await new PredictionModelBuilder&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureReinforcementLearning(options)
    ///     .ConfigureModel(new DQNAgent&lt;double&gt;())
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureReinforcementLearning(RLTrainingOptions<T> options)
    {
        _rlOptions = options;
        return this;
    }

    /// <summary>
    /// Asks the agent a question about your model building process.
    /// Only available after calling ConfigureAgentAssistance().
    /// </summary>
    /// <param name="question">Natural language question to ask the agent.</param>
    /// <returns>The agent's answer based on your current configuration.</returns>
    /// <exception cref="InvalidOperationException">Thrown if ConfigureAgentAssistance() hasn't been called.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> Use this to get AI-powered advice during model building.
    ///
    /// Example:
    /// <code>
    /// var builder = new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureAgentAssistance(apiKey: "sk-...");
    ///
    /// var advice = await builder.AskAgentAsync(
    ///     "Should I use Ridge or Lasso regression for my dataset with 50 features?");
    /// Console.WriteLine(advice);
    /// </code>
    /// </remarks>
    public async Task<string> AskAgentAsync(string question)
    {
        if (_agentConfig == null || !_agentConfig.IsEnabled)
        {
            throw new InvalidOperationException(
                "Agent assistance not enabled. Call ConfigureAgentAssistance() first.");
        }

        // Create a simple agent
        var chatModel = CreateChatModel(_agentConfig);
        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<T>(chatModel, tools);

        return await agent.RunAsync(question);
    }

    /// <summary>
    /// Configures knowledge distillation to train a smaller, faster student model from a larger teacher model.
    /// </summary>
    /// <param name="options">The knowledge distillation configuration options.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Knowledge distillation is a technique to compress a large, accurate "teacher" model
    /// into a smaller, faster "student" model while preserving most of the teacher's accuracy. Think of it like
    /// an expert (teacher) teaching a student - the student learns not just the answers, but also the reasoning process.</para>
    ///
    /// <para><b>Benefits:</b>
    /// - **Model Compression**: 40-90% size reduction with 90-97% accuracy preserved
    /// - **Faster Inference**: Smaller models run 2-10x faster
    /// - **Edge Deployment**: Deploy on mobile devices, IoT, browsers
    /// - **Cost Reduction**: Lower compute and memory costs</para>
    ///
    /// <para><b>Common Use Cases:</b>
    /// - Deploy BERT/GPT models on mobile devices (DistilBERT is 40% smaller, 60% faster)
    /// - Run vision models on edge devices (MobileNet distilled from ResNet)
    /// - Reduce cloud compute costs for inference
    /// - Multi-teacher ensembles distilled into single student</para>
    ///
    /// <para><b>Quick Start Example:</b>
    /// <code>
    /// // Configure knowledge distillation with default settings (good for most cases)
    /// var distillationOptions = new KnowledgeDistillationOptions&lt;Vector&lt;double&gt;, Vector&lt;double&gt;, double&gt;
    /// {
    ///     TeacherModelType = TeacherModelType.NeuralNetwork,
    ///     StrategyType = DistillationStrategyType.ResponseBased,
    ///     Temperature = 3.0,      // Soften predictions (2-5 typical)
    ///     Alpha = 0.3,            // 30% hard labels, 70% teacher knowledge
    ///     Epochs = 20,
    ///     BatchSize = 32,
    ///     LearningRate = 0.001
    /// };
    ///
    /// var result = await new PredictionModelBuilder&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(studentModel)
    ///     .ConfigureKnowledgeDistillation(distillationOptions)
    ///     .BuildAsync();
    /// </code>
    /// </para>
    ///
    /// <para><b>Advanced Techniques:</b>
    /// - **Response-Based**: Standard Hinton distillation (recommended start)
    /// - **Feature-Based**: Match intermediate layer representations
    /// - **Attention-Based**: For transformers (BERT, GPT)
    /// - **Relational**: Preserve relationships between samples
    /// - **Self-Distillation**: Model teaches itself for better calibration
    /// - **Ensemble**: Multiple teachers for richer knowledge</para>
    ///
    /// <para><b>Key Parameters:</b>
    /// - **Temperature** (2-5): Higher = softer predictions, more knowledge transfer
    /// - **Alpha** (0.2-0.5): Lower = rely more on teacher, higher = rely more on labels
    /// - **Strategy**: ResponseBased (standard), FeatureBased (deeper), AttentionBased (transformers)
    /// - **Teacher Type**: NeuralNetwork (single), Ensemble (multiple), Self (no separate teacher)</para>
    ///
    /// <para><b>Success Stories:</b>
    /// - DistilBERT: 40% smaller than BERT, 97% performance, 60% faster
    /// - TinyBERT: 7.5x smaller than BERT for mobile deployment
    /// - MobileNet: Distilled from ResNet, 10x fewer parameters
    /// - SqueezeNet: AlexNet-level accuracy at 50x smaller size</para>
    ///
    /// <para><b>References:</b>
    /// - Hinton et al. (2015). Distilling the Knowledge in a Neural Network
    /// - Sanh et al. (2019). DistilBERT
    /// - Park et al. (2019). Relational Knowledge Distillation</para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureKnowledgeDistillation(
        KnowledgeDistillationOptions<T, TInput, TOutput>? options = null)
    {
        _knowledgeDistillationOptions = options ?? new KnowledgeDistillationOptions<T, TInput, TOutput>();
        return this;
    }

    // ============================================================================
    // Deployment Configuration Methods
    // ============================================================================

    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureQuantization(QuantizationConfig? config = null)
    {
        _quantizationConfig = config;
        return this;
    }

    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureCompression(CompressionConfig? config = null)
    {
        _compressionConfig = config ?? new CompressionConfig();
        return this;
    }

    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureCaching(CacheConfig? config = null)
    {
        _cacheConfig = config;
        return this;
    }

    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureVersioning(VersioningConfig? config = null)
    {
        _versioningConfig = config;
        return this;
    }

    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureABTesting(ABTestingConfig? config = null)
    {
        _abTestingConfig = config;
        return this;
    }

    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureTelemetry(TelemetryConfig? config = null)
    {
        _telemetryConfig = config;
        return this;
    }

    /// <summary>
    /// Configures benchmarking to run standardized benchmark suites and attach a structured report to the built model.
    /// </summary>
    /// <param name="options">Benchmarking options (suites, sampling, failure policy). If null, sensible defaults are used.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This integrates benchmarking into the facade flow: users select suites via enums and receive a structured report,
    /// without wiring benchmark implementations manually.
    /// </para>
    /// <para><b>For Beginners:</b> This is like running a standardized test after building your model to see how it performs.</para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureBenchmarking(BenchmarkingOptions? options = null)
    {
        _benchmarkingOptions = options ?? new BenchmarkingOptions();
        return this;
    }

    /// <summary>
    /// Configures performance profiling for training and inference operations.
    /// </summary>
    /// <param name="config">The profiling configuration, or null to use industry-standard defaults.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Profiling measures how long different parts of your ML code take to run.
    /// Think of it like a stopwatch for your code - it helps you find bottlenecks and optimize performance.
    ///
    /// The profiling report will be available in the result after training:
    /// <code>
    /// var result = await builder
    ///     .ConfigureProfiling() // Enable with defaults
    ///     .Build(features, labels);
    ///
    /// // Access the profiling report
    /// var report = result.ProfilingReport;
    /// Console.WriteLine(report?.GetFormattedSummary());
    /// </code>
    ///
    /// Features tracked:
    /// - Operation timing: How long each training step, forward pass, backward pass takes
    /// - Memory allocations: How much memory is used during training
    /// - Call hierarchy: Which operations call which other operations
    /// - Percentiles: P50 (median), P95, P99 timing for statistical analysis
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureProfiling(ProfilingConfig? config = null)
    {
        _profilingConfig = config ?? new ProfilingConfig { Enabled = true };
        return this;
    }

    /// <summary>
    /// Creates a ProfilerSession if profiling is enabled; otherwise returns null.
    /// </summary>
    private ProfilerSession? CreateProfilerSession()
    {
        if (_profilingConfig?.Enabled != true)
        {
            return null;
        }

        return new ProfilerSession(_profilingConfig);
    }

    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureExport(ExportConfig? config = null)
    {
        _exportConfig = config;
        return this;
    }

    /// <summary>
    /// Configures experiment tracking for logging and organizing training runs.
    /// </summary>
    /// <param name="tracker">The experiment tracker implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> An experiment tracker is like a lab notebook for your ML experiments.
    /// It logs parameters, metrics, and artifacts so you can compare runs and reproduce results.</para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureExperimentTracker(IExperimentTracker<T> tracker)
    {
        _experimentTracker = tracker;
        return this;
    }

    /// <summary>
    /// Configures checkpoint management for saving and restoring training state.
    /// </summary>
    /// <param name="manager">The checkpoint manager implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Checkpoints are like save points in a video game.
    /// They let you pause training and resume later, or go back to an earlier state if something goes wrong.</para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureCheckpointManager(ICheckpointManager<T, TInput, TOutput> manager)
    {
        _checkpointManager = manager;
        return this;
    }

    /// <summary>
    /// Configures memory management for training including gradient checkpointing,
    /// activation pooling, and model sharding.
    /// </summary>
    /// <param name="configuration">The memory configuration to use. If null, uses default settings.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training large neural networks requires a lot of memory.
    /// Memory management helps you train bigger models by:
    /// </para>
    /// <list type="bullet">
    /// <item><description><b>Gradient Checkpointing:</b> Trades compute for memory by recomputing
    /// activations during backpropagation instead of storing them all.</description></item>
    /// <item><description><b>Activation Pooling:</b> Reuses memory buffers to reduce garbage collection.</description></item>
    /// <item><description><b>Model Sharding:</b> Splits large models across multiple GPUs.</description></item>
    /// </list>
    /// <para>
    /// <b>Available Presets:</b>
    /// <list type="bullet">
    /// <item><description><c>TrainingMemoryConfig.MemoryEfficient()</c> - Maximum memory savings</description></item>
    /// <item><description><c>TrainingMemoryConfig.SpeedOptimized()</c> - Maximum speed</description></item>
    /// <item><description><c>TrainingMemoryConfig.MultiGpu(n)</c> - Multi-GPU training</description></item>
    /// <item><description><c>TrainingMemoryConfig.ForTransformers()</c> - Optimized for transformers</description></item>
    /// <item><description><c>TrainingMemoryConfig.ForConvNets()</c> - Optimized for CNNs</description></item>
    /// </list>
    /// </para>
    /// <example>
    /// <code>
    /// // Using a preset configuration
    /// builder.ConfigureMemoryManagement(TrainingMemoryConfig.MemoryEfficient());
    ///
    /// // Using a custom configuration
    /// builder.ConfigureMemoryManagement(new TrainingMemoryConfig
    /// {
    ///     UseGradientCheckpointing = true,
    ///     CheckpointEveryNLayers = 2,
    ///     UseActivationPooling = true,
    ///     MaxPoolMemoryMB = 2048
    /// });
    ///
    /// // Multi-GPU training
    /// builder.ConfigureMemoryManagement(TrainingMemoryConfig.MultiGpu(4));
    /// </code>
    /// </example>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureMemoryManagement(
        Training.Memory.TrainingMemoryConfig? configuration = null)
    {
        _memoryConfig = configuration;
        return this;
    }

    /// <summary>
    /// Configures training monitoring for real-time visibility into training progress.
    /// </summary>
    /// <param name="monitor">The training monitor implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A training monitor is like a dashboard for your model training.
    /// It shows you how training is progressing, what resources are being used, and if there are any problems.</para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureTrainingMonitor(ITrainingMonitor<T> monitor)
    {
        _trainingMonitor = monitor;
        return this;
    }

    /// <summary>
    /// Configures model registry for centralized model storage and versioning.
    /// </summary>
    /// <param name="registry">The model registry implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A model registry is like a library for your trained models.
    /// It keeps track of all your models, their versions, and which ones are in production.</para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureModelRegistry(IModelRegistry<T, TInput, TOutput> registry)
    {
        _modelRegistry = registry;
        return this;
    }

    /// <summary>
    /// Configures data version control for tracking dataset changes.
    /// </summary>
    /// <param name="dataVersionControl">The data version control implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Data version control is like Git, but for your datasets.
    /// It tracks what data was used for training each model and lets you reproduce experiments.</para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureDataVersionControl(IDataVersionControl<T> dataVersionControl)
    {
        _dataVersionControl = dataVersionControl;
        return this;
    }

    /// <summary>
    /// Configures hyperparameter optimization for automatic tuning of model settings.
    /// </summary>
    /// <param name="optimizer">The hyperparameter optimizer implementation to use.</param>
    /// <param name="searchSpace">The hyperparameter search space defining parameter ranges. If null, hyperparameter optimization is disabled.</param>
    /// <param name="nTrials">Number of trials to run. Default is 10.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Hyperparameter optimization automatically finds the best settings
    /// for your model (like learning rate, number of layers, etc.) instead of you having to guess.</para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureHyperparameterOptimizer(
        IHyperparameterOptimizer<T, TInput, TOutput> optimizer,
        HyperparameterSearchSpace? searchSpace = null,
        int nTrials = 10)
    {
        _hyperparameterOptimizer = optimizer;
        _hyperparameterSearchSpace = searchSpace;
        _hyperparameterTrials = nTrials;
        return this;
    }

    /// <summary>
    /// Configures data augmentation for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Data augmentation creates variations of training data on-the-fly to help models
    /// generalize better. This configuration covers both training-time augmentation
    /// and Test-Time Augmentation (TTA) for improved inference accuracy.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Augmentation is like showing the model many variations of
    /// the same data. For images, this might include rotations, flips, and color changes.
    /// The model learns to recognize objects regardless of these variations.
    /// </para>
    /// <para><b>Key features:</b>
    /// <list type="bullet">
    /// <item>Automatic data-type detection (image, tabular, audio, text, video)</item>
    /// <item>Industry-standard defaults that work well out-of-the-box</item>
    /// <item>Test-Time Augmentation (TTA) enabled by default for better predictions</item>
    /// </list>
    /// </para>
    /// <para>
    /// Example - Simple usage with defaults:
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(myModel)
    ///     .ConfigureAugmentation()  // Uses auto-detected defaults
    ///     .Build(X, y);
    /// </code>
    /// </para>
    /// <para>
    /// Example - Custom configuration:
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(myModel)
    ///     .ConfigureAugmentation(new AugmentationConfig
    ///     {
    ///         EnableTTA = true,
    ///         TTANumAugmentations = 8,
    ///         ImageSettings = new ImageAugmentationSettings
    ///         {
    ///             EnableFlips = true,
    ///             EnableRotation = true,
    ///             RotationRange = 20.0
    ///         }
    ///     })
    ///     .Build(images, labels);
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="config">
    /// Augmentation configuration. If null, uses industry-standard defaults
    /// with automatic data-type detection.
    /// </param>
    /// <returns>The builder instance for method chaining.</returns>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureAugmentation(
        Augmentation.AugmentationConfig? config = null)
    {
        _augmentationConfig = config ?? CreateDefaultAugmentationConfig();
        return this;
    }

    /// <summary>
    /// Creates a default augmentation configuration with auto-detected modality settings.
    /// </summary>
    private Augmentation.AugmentationConfig CreateDefaultAugmentationConfig()
    {
        var config = new Augmentation.AugmentationConfig();

        // Auto-detect data type from TInput and apply appropriate defaults
        var dataType = Augmentation.DataModalityDetector.Detect<TInput>();

        switch (dataType)
        {
            case Augmentation.DataModality.Image:
                config.ImageSettings = new Augmentation.ImageAugmentationSettings();
                break;
            case Augmentation.DataModality.Tabular:
                config.TabularSettings = new Augmentation.TabularAugmentationSettings();
                break;
            case Augmentation.DataModality.Audio:
                config.AudioSettings = new Augmentation.AudioAugmentationSettings();
                break;
            case Augmentation.DataModality.Text:
                config.TextSettings = new Augmentation.TextAugmentationSettings();
                break;
            case Augmentation.DataModality.Video:
                config.VideoSettings = new Augmentation.VideoAugmentationSettings();
                break;
            default:
                // Unknown type - use generic settings, user can configure manually
                break;
        }

        return config;
    }

    /// <summary>
    /// Configures self-supervised learning for unsupervised representation learning.
    /// </summary>
    /// <param name="configure">Optional action to configure SSL settings.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Self-supervised learning (SSL) allows training powerful representations from unlabeled data.
    /// The learned representations can then be fine-tuned on smaller labeled datasets, often
    /// achieving better results than training from scratch.
    /// </para>
    /// <para><b>For Beginners:</b> SSL is like teaching a model to understand patterns in data
    /// without needing human labels. Think of it as the model learning to "see" or "understand"
    /// images/text before being taught specific tasks. This makes it much better at learning
    /// new tasks with less labeled data.</para>
    ///
    /// <para><b>Supported Methods:</b></para>
    /// <list type="bullet">
    /// <item><b>SimCLR:</b> Contrastive learning with in-batch negatives (large batch sizes)</item>
    /// <item><b>MoCo/MoCoV2/MoCoV3:</b> Momentum contrastive with memory queue (efficient)</item>
    /// <item><b>BYOL:</b> No negatives required, uses momentum teacher</item>
    /// <item><b>SimSiam:</b> Simple Siamese networks with stop-gradient</item>
    /// <item><b>BarlowTwins:</b> Decorrelation-based, no negatives needed</item>
    /// <item><b>DINO:</b> Self-distillation for Vision Transformers</item>
    /// <item><b>MAE:</b> Masked autoencoding for ViT pretraining</item>
    /// </list>
    ///
    /// <para><b>Example - Basic SSL pretraining:</b></para>
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(encoder)
    ///     .ConfigureSelfSupervisedLearning()  // Uses SimCLR by default
    ///     .Build(unlabeledImages);
    /// </code>
    ///
    /// <para><b>Example - Custom SSL configuration:</b></para>
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(encoder)
    ///     .ConfigureSelfSupervisedLearning(ssl =>
    ///     {
    ///         ssl.Method = SSLMethodType.MoCoV3;
    ///         ssl.PretrainingEpochs = 300;
    ///         ssl.Temperature = 0.2;
    ///         ssl.ProjectorOutputDimension = 256;
    ///         ssl.MoCo = new MoCoConfig { Momentum = 0.99 };
    ///     })
    ///     .Build(unlabeledImages);
    /// </code>
    ///
    /// <para><b>Example - BYOL without negative samples:</b></para>
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(encoder)
    ///     .ConfigureSelfSupervisedLearning(ssl =>
    ///     {
    ///         ssl.Method = SSLMethodType.BYOL;
    ///         ssl.BYOL = new BYOLConfig { Momentum = 0.996 };
    ///     })
    ///     .Build(unlabeledImages);
    /// </code>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureSelfSupervisedLearning(
        Action<SelfSupervisedLearning.SSLConfig>? configure = null)
    {
        _sslConfig = new SelfSupervisedLearning.SSLConfig();
        configure?.Invoke(_sslConfig);
        return this;
    }

    /// <summary>
    /// Configures tokenization for text-based input processing.
    /// </summary>
    /// <param name="tokenizer">The tokenizer to use for text processing.</param>
    /// <param name="config">Optional tokenization configuration. If null, default settings are used.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Tokenization is the process of breaking text into smaller pieces (tokens) that can be processed
    /// by machine learning models. This is essential for NLP and text-based models.
    /// </para>
    /// <para><b>For Beginners:</b> Tokenization converts human-readable text into numbers that AI models understand.
    ///
    /// Different tokenization strategies include:
    /// - BPE (Byte Pair Encoding): Used by GPT models, learns subword units from data
    /// - WordPiece: Used by BERT, splits unknown words into known subwords
    /// - SentencePiece: Language-independent tokenization used by many multilingual models
    ///
    /// Example:
    /// <code>
    /// // Using BPE tokenizer
    /// var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 32000);
    /// var builder = new PredictionModelBuilder&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;()
    ///     .ConfigureTokenizer(tokenizer)
    ///     .ConfigureModel(new TransformerModel())
    ///     .Build(trainingData);
    ///
    /// // Or use AutoTokenizer for HuggingFace models
    /// var tokenizer = AutoTokenizer.FromPretrained("bert-base-uncased");
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureTokenizer(
        ITokenizer? tokenizer = null,
        TokenizationConfig? config = null)
    {
        _tokenizer = tokenizer;
        _tokenizationConfig = config ?? new TokenizationConfig();
        return this;
    }

    /// <summary>
    /// Configures program synthesis (code generation / repair) settings with sensible defaults.
    /// </summary>
    /// <param name="options">Optional configuration options. If null, safe industry-standard defaults are used.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Program synthesis focuses on code-oriented tasks such as generation, completion, and repair.
    /// This method wires up the default program-synthesis components and chooses safe default values when
    /// options are not provided (for example, a safe maximum sequence length and vocabulary size).
    /// </para>
    /// <para>
    /// Tokenizer selection:
    /// - If <paramref name="options"/> provides a tokenizer, it is used.
    /// - Otherwise, if a tokenizer was configured earlier via <see cref="ConfigureTokenizer"/>, that tokenizer is reused.
    /// - If no tokenizer is available, a code-aware tokenizer is created automatically based on the target language.
    /// </para>
    /// <para>
    /// Model selection:
    /// - The builder creates a program-synthesis model based on the configured model kind (for example CodeBERT / GraphCodeBERT / CodeT5).
    /// - If the created model is compatible with this builder’s <typeparamref name="TInput"/> and <typeparamref name="TOutput"/>, the model is applied.
    ///   If not compatible, the tokenizer/options are still configured, but the existing model is left unchanged.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when you want a ready-to-use setup for code tasks and you do not want to
    /// manually choose every low-level component (tokenizer, defaults, and model configuration).
    ///
    /// Simple usage (defaults):
    /// <code>
    /// var result = await new PredictionModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureProgramSynthesis()
    ///     .BuildAsync();
    /// </code>
    ///
    /// Custom usage:
    /// <code>
    /// var result = await new PredictionModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureProgramSynthesis(new ProgramSynthesisOptions
    ///     {
    ///         TargetLanguage = ProgramLanguage.CSharp,
    ///         ModelKind = ProgramSynthesisModelKind.CodeT5,
    ///         MaxSequenceLength = 1024
    ///     })
    ///     .BuildAsync();
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureProgramSynthesis(
        AiDotNet.ProgramSynthesis.Options.ProgramSynthesisOptions? options = null)
    {
        options ??= new AiDotNet.ProgramSynthesis.Options.ProgramSynthesisOptions();

        // Defaults-first: clamp invalid user inputs to safe industry-standard values.
        var maxSequenceLength = options.MaxSequenceLength > 0 ? options.MaxSequenceLength : 512;
        var vocabularySize = options.VocabularySize > 0 ? options.VocabularySize : 50000;
        var numEncoderLayers = Math.Max(0, options.NumEncoderLayers);
        var numDecoderLayersConfigured = Math.Max(0, options.NumDecoderLayers);

        var tokenizer = options.Tokenizer ?? _tokenizer;

        if (tokenizer is null)
        {
            var baseTokenizer = AiDotNet.Tokenization.Algorithms.CharacterTokenizer.CreateAscii(
                AiDotNet.Tokenization.Models.SpecialTokens.Bert(),
                lowercase: false);

            var codeLanguage = options.TargetLanguage switch
            {
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.CSharp => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.CSharp,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.Python => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Python,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.Java => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Java,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.JavaScript => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.JavaScript,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.TypeScript => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.TypeScript,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.C => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.C,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.CPlusPlus => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Cpp,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.Go => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Go,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.Rust => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Rust,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.SQL => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.SQL,
                _ => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Generic
            };

            tokenizer = new AiDotNet.Tokenization.CodeTokenization.CodeTokenizer(
                baseTokenizer,
                codeLanguage,
                splitIdentifiers: true);
        }

        _tokenizer = tokenizer;
        _tokenizationConfig ??= new TokenizationConfig();

        bool useDataFlow = options.ModelKind == AiDotNet.ProgramSynthesis.Options.ProgramSynthesisModelKind.GraphCodeBERT;

        int numDecoderLayers = options.ModelKind == AiDotNet.ProgramSynthesis.Options.ProgramSynthesisModelKind.CodeT5
            ? Math.Max(1, numDecoderLayersConfigured)
            : 0;

        var architecture = new AiDotNet.ProgramSynthesis.Models.CodeSynthesisArchitecture<T>(
            synthesisType: options.SynthesisType,
            targetLanguage: options.TargetLanguage,
            codeTaskType: options.DefaultTask,
            numEncoderLayers: numEncoderLayers,
            numDecoderLayers: numDecoderLayers,
            maxSequenceLength: maxSequenceLength,
            vocabularySize: vocabularySize,
            useDataFlow: useDataFlow);

        AiDotNet.ProgramSynthesis.Interfaces.ICodeModel<T> codeModel = options.ModelKind switch
        {
            AiDotNet.ProgramSynthesis.Options.ProgramSynthesisModelKind.CodeBERT =>
                new AiDotNet.ProgramSynthesis.Engines.CodeBERT<T>(architecture, tokenizer: tokenizer),
            AiDotNet.ProgramSynthesis.Options.ProgramSynthesisModelKind.GraphCodeBERT =>
                new AiDotNet.ProgramSynthesis.Engines.GraphCodeBERT<T>(architecture, tokenizer: tokenizer),
            _ =>
                new AiDotNet.ProgramSynthesis.Engines.CodeT5<T>(architecture, tokenizer: tokenizer)
        };

        // Store the program-synthesis model separately so it is available regardless of the primary model's TInput/TOutput types.
        _programSynthesisModel = codeModel;

        // If compatible, also apply as the primary model (supports code-only workflows).
        if (codeModel is IFullModel<T, TInput, TOutput> fullModel)
        {
            _model = fullModel;
        }

        return this;
    }

    /// <summary>
    /// Configures program synthesis to use <c>AiDotNet.Serving</c> for program execution and evaluation (optional).
    /// </summary>
    /// <param name="options">
    /// Serving client options. If null (and <paramref name="client"/> is null), a default configuration is used that targets
    /// <c>http://localhost:52432/</c>.
    /// </param>
    /// <param name="client">Optional custom client implementation. When provided, this takes precedence over <paramref name="options"/>.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Some program-synthesis workflows need to run or evaluate generated programs (for example, execute code against test cases).
    /// This method lets you route those operations through a Serving endpoint (or a custom client), which is useful for centralized
    /// execution, resource control, and isolation.
    /// </para>
    /// <para>
    /// Precedence rules:
    /// - If <paramref name="client"/> is provided, it is used.
    /// - Otherwise, if <paramref name="options"/> is provided it is used.
    /// - Otherwise, a default configuration is used that targets <c>http://localhost:52432/</c>.
    /// </para>
    /// <para><b>For Beginners:</b> If you only want the model to generate code, you can skip this.
    /// If you want to automatically execute or evaluate generated code, configure Serving. If you're running
    /// <c>AiDotNet.Serving</c> locally with default settings, calling this method with no parameters is enough.
    ///
    /// Example:
    /// <code>
    /// var result = await new PredictionModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureProgramSynthesis()
    ///     .ConfigureProgramSynthesisServing() // Defaults to http://localhost:52432/
    ///     .BuildAsync();
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureProgramSynthesisServing(
        ProgramSynthesisServingClientOptions? options = null,
        IProgramSynthesisServingClient? client = null)
    {
        // Defaults-first: calling this method opts into Serving using the standard local endpoint unless overridden.
        _programSynthesisServingClientOptions = options ?? (client is null
            ? new ProgramSynthesisServingClientOptions { BaseAddress = new Uri("http://localhost:52432/") }
            : null);

        _programSynthesisServingClient = client;
        return this;
    }

    /// <summary>
    /// Configures tokenization using a pretrained tokenizer from HuggingFace Hub.
    /// </summary>
    /// <param name="model">The pretrained tokenizer model to use. Defaults to BertBaseUncased.</param>
    /// <param name="config">Optional tokenization configuration.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the easiest and most type-safe way to use industry-standard tokenizers.
    /// Using the enum ensures you always specify a valid model name.
    ///
    /// Simply call without parameters for sensible defaults:
    /// <code>
    /// var builder = new PredictionModelBuilder&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;()
    ///     .ConfigureTokenizerFromPretrained()  // Uses BertBaseUncased by default
    ///     .ConfigureModel(new BertModel())
    ///     .Build(trainingData);
    /// </code>
    ///
    /// Or specify a model using the enum:
    /// <code>
    /// builder.ConfigureTokenizerFromPretrained(PretrainedTokenizerModel.Gpt2)
    /// </code>
    ///
    /// Available models include:
    /// - BertBaseUncased: BERT tokenizer for English text (default)
    /// - Gpt2, Gpt2Medium, Gpt2Large: GPT-2 tokenizers for text generation
    /// - RobertaBase, RobertaLarge: RoBERTa tokenizers (improved BERT)
    /// - T5Small, T5Base, T5Large: T5 tokenizers for text-to-text tasks
    /// - DistilBertBaseUncased: Faster, smaller BERT
    /// - CodeBertBase: For code understanding tasks
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureTokenizerFromPretrained(
        PretrainedTokenizerModel model = PretrainedTokenizerModel.BertBaseUncased,
        TokenizationConfig? config = null)
    {
        _tokenizer = AutoTokenizer.FromPretrained(model.ToModelId());
        _tokenizationConfig = config ?? new TokenizationConfig();
        return this;
    }

    /// <summary>
    /// Configures tokenization using a pretrained tokenizer from a custom HuggingFace model name or local path.
    /// </summary>
    /// <param name="modelNameOrPath">The HuggingFace model name or local path. Defaults to "bert-base-uncased" if not specified.</param>
    /// <param name="config">Optional tokenization configuration.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this overload when you need to specify a custom model name or path
    /// that isn't in the PretrainedTokenizerModel enum. For common models, prefer the enum-based overload
    /// for type safety.
    ///
    /// Example with custom model:
    /// <code>
    /// // Use a custom or community model from HuggingFace
    /// builder.ConfigureTokenizerFromPretrained("sentence-transformers/all-MiniLM-L6-v2")
    /// </code>
    ///
    /// If null or empty, defaults to "bert-base-uncased".
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureTokenizerFromPretrained(
        string? modelNameOrPath = null,
        TokenizationConfig? config = null)
    {
        // Default to bert-base-uncased, the most widely-used pretrained tokenizer
        // Use null-coalescing to ensure a non-null model name
        string defaultModel = PretrainedTokenizerModel.BertBaseUncased.ToModelId();
        string modelName = modelNameOrPath is not null && !string.IsNullOrWhiteSpace(modelNameOrPath)
            ? modelNameOrPath
            : defaultModel;
        _tokenizer = AutoTokenizer.FromPretrained(modelName);
        _tokenizationConfig = config ?? new TokenizationConfig();
        return this;
    }

    /// <inheritdoc />
    /// <summary>
    /// Asynchronously configures the tokenizer by loading a pretrained model from HuggingFace Hub.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the async version of ConfigureTokenizerFromPretrained.
    /// Use this when you want to avoid blocking the thread while downloading tokenizer files
    /// from HuggingFace Hub. This is especially important in UI applications or web servers.
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// await builder.ConfigureTokenizerFromPretrainedAsync(PretrainedTokenizerModel.BertBaseUncased);
    /// </code>
    /// </para>
    /// </remarks>
    public async Task<IPredictionModelBuilder<T, TInput, TOutput>> ConfigureTokenizerFromPretrainedAsync(
        PretrainedTokenizerModel model = PretrainedTokenizerModel.BertBaseUncased,
        TokenizationConfig? config = null)
    {
        _tokenizer = await AutoTokenizer.FromPretrainedAsync(model.ToModelId());
        _tokenizationConfig = config ?? new TokenizationConfig();
        return this;
    }

    /// <inheritdoc />
    /// <summary>
    /// Asynchronously configures the tokenizer by loading a pretrained model from HuggingFace Hub using a model name or path.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the async version that accepts a custom model name or path.
    /// Use this when loading custom or community models without blocking the thread.
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// await builder.ConfigureTokenizerFromPretrainedAsync("sentence-transformers/all-MiniLM-L6-v2");
    /// </code>
    /// </para>
    /// </remarks>
    public async Task<IPredictionModelBuilder<T, TInput, TOutput>> ConfigureTokenizerFromPretrainedAsync(
        string? modelNameOrPath = null,
        TokenizationConfig? config = null)
    {
        // Default to bert-base-uncased, the most widely-used pretrained tokenizer
        string defaultModel = PretrainedTokenizerModel.BertBaseUncased.ToModelId();
        string modelName = modelNameOrPath is not null && !string.IsNullOrWhiteSpace(modelNameOrPath)
            ? modelNameOrPath
            : defaultModel;
        _tokenizer = await AutoTokenizer.FromPretrainedAsync(modelName);
        _tokenizationConfig = config ?? new TokenizationConfig();
        return this;
    }

    // ============================================================================
    // Prompt Engineering Configuration Methods
    // ============================================================================

    /// <summary>
    /// Configures the prompt template for language model interactions.
    /// </summary>
    /// <param name="template">The prompt template to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// A prompt template provides a structured way to create prompts for language models by combining
    /// a template string with runtime variables.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A prompt template is like a form with blanks to fill in. You define the
    /// structure once and fill in different values each time you use it.
    ///
    /// Example:
    /// <code>
    /// var template = new SimplePromptTemplate("Translate {text} from {source} to {target}");
    /// var builder = new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigurePromptTemplate(template)
    ///     .ConfigureModel(model);
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigurePromptTemplate(IPromptTemplate? template = null)
    {
        _promptTemplate = template;
        return this;
    }

    /// <summary>
    /// Configures the prompt chain for composing multiple language model operations.
    /// </summary>
    /// <param name="chain">The chain to use for processing prompts.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// A chain orchestrates multiple language model calls, tools, and transformations into a cohesive
    /// workflow. Chains can be sequential, conditional, or parallel.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A chain connects multiple steps into a complete workflow, like a recipe
    /// where each step builds on the previous one.
    ///
    /// Example:
    /// <code>
    /// var chain = new SequentialChain&lt;string, string&gt;("CustomerSupport")
    ///     .AddStep("classify", ClassifyEmail)
    ///     .AddStep("respond", GenerateResponse);
    ///
    /// var builder = new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigurePromptChain(chain)
    ///     .ConfigureModel(model);
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigurePromptChain(IChain<string, string>? chain = null)
    {
        _promptChain = chain;
        return this;
    }

    /// <summary>
    /// Configures the prompt optimizer for automatically improving prompts.
    /// </summary>
    /// <param name="optimizer">The prompt optimizer to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// A prompt optimizer automatically refines prompts to achieve better performance on a specific task.
    /// Optimization strategies include discrete search, gradient-based methods, and evolutionary algorithms.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A prompt optimizer automatically improves your prompts by testing variations
    /// and keeping the best-performing ones.
    ///
    /// Example:
    /// <code>
    /// var optimizer = new DiscreteSearchOptimizer&lt;double&gt;();
    ///
    /// var builder = new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigurePromptOptimizer(optimizer)
    ///     .ConfigureModel(model);
    ///
    /// // Later, optimize a prompt
    /// var optimized = optimizer.Optimize(
    ///     initialPrompt: "Classify sentiment:",
    ///     evaluationFunction: EvaluatePrompt,
    ///     maxIterations: 50
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigurePromptOptimizer(IPromptOptimizer<T>? optimizer = null)
    {
        _promptOptimizer = optimizer;
        return this;
    }

    /// <summary>
    /// Configures the few-shot example selector for selecting examples to include in prompts.
    /// </summary>
    /// <param name="selector">The few-shot example selector to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// A few-shot example selector chooses the most relevant examples to include in prompts based
    /// on the current query. Different strategies include random selection, fixed order, and
    /// similarity-based selection.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Few-shot learning teaches the model by showing it examples. The selector
    /// picks which examples to show for each new query.
    ///
    /// Example:
    /// <code>
    /// var selector = new RandomExampleSelector&lt;double&gt;(seed: 42);
    /// selector.AddExample(new FewShotExample { Input = "Hello", Output = "Hola" });
    /// selector.AddExample(new FewShotExample { Input = "Goodbye", Output = "Adiós" });
    ///
    /// var builder = new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureFewShotExampleSelector(selector)
    ///     .ConfigureModel(model);
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFewShotExampleSelector(IFewShotExampleSelector<T>? selector = null)
    {
        _fewShotExampleSelector = selector;
        return this;
    }

    /// <summary>
    /// Configures a prompt analyzer for analyzing prompt quality, metrics, and potential issues.
    /// </summary>
    /// <param name="analyzer">The prompt analyzer implementation, or null to use default.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// A prompt analyzer examines prompts to provide metrics like token count, estimated cost,
    /// complexity scores, and can detect potential issues like prompt injection or unclear instructions.
    /// </para>
    /// <para><b>For Beginners:</b> The prompt analyzer is like a "spell checker" for your prompts.
    ///
    /// It helps you understand:
    /// - How many tokens your prompt uses (affects cost)
    /// - How complex your prompt is
    /// - Whether there might be issues with your prompt
    ///
    /// Example:
    /// <code>
    /// var analyzer = new PromptAnalyzer();
    ///
    /// var builder = new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigurePromptAnalyzer(analyzer)
    ///     .ConfigureModel(model);
    ///
    /// // After building, the trained model can analyze prompts
    /// var metrics = trainedModel.AnalyzePrompt("Your prompt text...");
    /// Console.WriteLine($"Token count: {metrics.TokenCount}");
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigurePromptAnalyzer(IPromptAnalyzer? analyzer = null)
    {
        _promptAnalyzer = analyzer;
        return this;
    }

    /// <summary>
    /// Configures a prompt compressor for reducing prompt token counts while preserving meaning.
    /// </summary>
    /// <param name="compressor">The prompt compressor implementation, or null to use default.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// A prompt compressor reduces the length of prompts to save on API costs and fit within
    /// context windows. Different compression strategies include removing redundancy,
    /// summarizing sections, and caching repeated content.
    /// </para>
    /// <para><b>For Beginners:</b> The prompt compressor makes your prompts shorter without losing meaning.
    ///
    /// Benefits:
    /// - Lower API costs (fewer tokens = less money)
    /// - Faster responses (shorter prompts process faster)
    /// - Fit within model limits (some models have token limits)
    ///
    /// Example:
    /// <code>
    /// var compressor = new RedundancyCompressor();
    ///
    /// var builder = new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigurePromptCompressor(compressor)
    ///     .ConfigureModel(model);
    ///
    /// // After building, the trained model can compress prompts
    /// var result = trainedModel.CompressPrompt("Your long verbose prompt...");
    /// Console.WriteLine($"Original: {result.OriginalTokenCount}, Compressed: {result.CompressedTokenCount}");
    /// </code>
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigurePromptCompressor(IPromptCompressor? compressor = null)
    {
        _promptCompressor = compressor;
        return this;
    }

    // ============================================================================
    // Private Prompt Engineering Helper Methods
    // ============================================================================

    /// <summary>
    /// Attaches the configured prompt engineering components to a PredictionModelResult.
    /// </summary>
    /// <param name="result">The result to attach configuration to.</param>
    private void AttachPromptEngineeringConfiguration(PredictionModelResult<T, TInput, TOutput> result)
    {
        result.AttachPromptEngineering(
            _promptTemplate,
            _promptChain,
            _promptOptimizer,
            _fewShotExampleSelector,
            _promptAnalyzer,
            _promptCompressor);
    }

    // ============================================================================
    // Private Knowledge Distillation Helper Methods
    // ============================================================================

    /// <summary>
    /// Performs knowledge distillation training using the configured options.
    /// </summary>
    private Task<OptimizationResult<T, TInput, TOutput>> PerformKnowledgeDistillationAsync(
        IFullModel<T, TInput, TOutput> studentModel,
        IOptimizer<T, TInput, TOutput> optimizer,
        TInput XTrain,
        TOutput yTrain,
        TInput XVal,
        TOutput yVal,
        TInput XTest,
        TOutput yTest)
    {
        if (_knowledgeDistillationOptions == null)
            throw new InvalidOperationException("Knowledge distillation options not configured");

        var options = _knowledgeDistillationOptions;



        var NumOps = MathHelper.GetNumericOperations<T>();

        // Get a reference input for shape conversions (needed for Matrix<T> and Tensor<T>)
        // Use InputHelper to extract a single sample from the training data
        TInput referenceInput = InputHelper<T, TInput>.GetItem(XTrain, 0);

        try
        {
            // Step 1: Create teacher model using factory
            // Trainer expects Vector<T> for single samples, but options.TeacherModel uses TInput/TOutput (dataset types)
            ITeacherModel<Vector<T>, Vector<T>> teacher;
            if (options.TeacherModel != null)
            {
                // Wrap IFullModel as teacher - requires explicit output dimension
                if (!options.OutputDimension.HasValue)
                    throw new InvalidOperationException(
                        "OutputDimension is required when using TeacherModel. " +
                        "Please specify options.OutputDimension explicitly.");

                // Adapter function: IFullModel<T, TInput, TOutput>.Predict -> Func<Vector<T>, Vector<T>>
                Func<Vector<T>, Vector<T>> adaptedTeacherPredict = inputSampleVector =>
                {
                    // Convert trainer's Vector<T> (single sample) to teacher's TInput type
                    TInput teacherInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(inputSampleVector, referenceInput);
                    // Call teacher's predict method with TInput
                    TOutput teacherOutput = options.TeacherModel.Predict(teacherInput);
                    // Convert teacher's TOutput back to trainer's Vector<T>
                    return ConversionsHelper.ConvertToVector<T, TOutput>(teacherOutput);
                };

                teacher = new KnowledgeDistillation.TeacherModelWrapper<T>(
                    adaptedTeacherPredict,
                    options.OutputDimension.Value);
            }
            else if (options.Teachers != null && options.Teachers.Length > 0)
            {
                // Adapt each teacher in the ensemble to work with Vector<T> samples
                var adaptedTeachers = options.Teachers.Select(t =>
                {
                    Func<Vector<T>, Vector<T>> adaptedPredict = inputSampleVector =>
                    {
                        TInput teacherInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(inputSampleVector, referenceInput);
                        TOutput teacherOutput = t.GetLogits(teacherInput);
                        return ConversionsHelper.ConvertToVector<T, TOutput>(teacherOutput);
                    };
                    return new KnowledgeDistillation.TeacherModelWrapper<T>(adaptedPredict, t.OutputDimension);
                }).ToArray();

                teacher = KnowledgeDistillation.TeacherModelFactory<T>.CreateTeacher(
                    TeacherModelType.Ensemble,
                    ensembleModels: adaptedTeachers,
                    ensembleWeights: options.EnsembleWeights != null ? (double[])options.EnsembleWeights : null);
            }
            else if (options.TeacherForward != null)
            {
                if (!options.OutputDimension.HasValue)
                    throw new InvalidOperationException(
                        "OutputDimension is required when using TeacherForward. " +
                        "Please specify options.OutputDimension explicitly.");

                // Adapter function: Func<TInput, TOutput> -> Func<Vector<T>, Vector<T>>
                Func<Vector<T>, Vector<T>> adaptedTeacherForward = inputSampleVector =>
                {
                    // Convert trainer's Vector<T> (single sample) to teacher's TInput type
                    TInput teacherInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(inputSampleVector, referenceInput);
                    // Call teacher's forward function with TInput
                    TOutput teacherOutput = options.TeacherForward(teacherInput);
                    // Convert teacher's TOutput back to trainer's Vector<T>
                    return ConversionsHelper.ConvertToVector<T, TOutput>(teacherOutput);
                };

                teacher = new KnowledgeDistillation.TeacherModelWrapper<T>(
                    adaptedTeacherForward,
                    options.OutputDimension.Value);
            }
            else
            {
                throw new InvalidOperationException(
                    "No teacher model configured. Please set TeacherModel, Teachers, or TeacherForward in KnowledgeDistillationOptions.");
            }

            // Step 2: Create distillation strategy using factory
            var strategy = KnowledgeDistillation.DistillationStrategyFactory<T>.CreateStrategy(
                options.StrategyType,
                temperature: options.Temperature,
                alpha: options.Alpha);

            // Step 3: Create checkpoint configuration from options
            DistillationCheckpointConfig? checkpointConfig = null;
            if (options.SaveCheckpoints)
            {
                checkpointConfig = new DistillationCheckpointConfig
                {
                    CheckpointDirectory = options.CheckpointDirectory ?? "./checkpoints",
                    SaveEveryEpochs = options.CheckpointFrequency,
                    KeepBestN = options.SaveOnlyBestCheckpoint ? 1 : 0,
                    SaveStudent = true,
                    BestMetric = "validation_loss",
                    LowerIsBetter = true
                };
            }

            // Step 4: Create trainer with early stopping and checkpointing configuration
            var trainer = new KnowledgeDistillation.KnowledgeDistillationTrainer<T>(
                teacher,
                strategy,
                checkpointConfig: checkpointConfig,
                useEarlyStopping: options.UseEarlyStopping,
                earlyStoppingMinDelta: options.EarlyStoppingMinDelta,
                earlyStoppingPatience: options.EarlyStoppingPatience);

            Console.WriteLine($"Starting Knowledge Distillation:");
            Console.WriteLine($"  Strategy: {options.StrategyType}");
            Console.WriteLine($"  Temperature: {options.Temperature}");
            Console.WriteLine($"  Alpha: {options.Alpha}");
            Console.WriteLine($"  Epochs: {options.Epochs}");
            Console.WriteLine($"  Batch Size: {options.BatchSize}");
            Console.WriteLine();

            // Step 4: Prepare training data - convert to Vector<Vector<T>>
            var trainMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(XTrain);
            var trainVector = ConversionsHelper.ConvertToVector<T, TOutput>(yTrain);
            var valMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(XVal);
            var valVector = ConversionsHelper.ConvertToVector<T, TOutput>(yVal);

            var trainInputs = new Vector<Vector<T>>(trainMatrix.Rows);
            var trainLabels = new Vector<Vector<T>>(trainMatrix.Rows);
            for (int i = 0; i < trainMatrix.Rows; i++)
            {
                trainInputs[i] = trainMatrix.GetRow(i);
                // Create one-hot encoded labels
                var oneHot = new Vector<T>(teacher.OutputDimension);
                int labelIdx = (int)Convert.ToDouble(trainVector[i]);
                if (labelIdx >= 0 && labelIdx < teacher.OutputDimension)
                    oneHot[labelIdx] = NumOps.One;
                trainLabels[i] = oneHot;
            }

            Vector<Vector<T>>? valInputs = null;
            Vector<Vector<T>>? valLabels = null;
            if (valMatrix.Rows > 0)
            {
                valInputs = new Vector<Vector<T>>(valMatrix.Rows);
                valLabels = new Vector<Vector<T>>(valMatrix.Rows);
                for (int i = 0; i < valMatrix.Rows; i++)
                {
                    valInputs[i] = valMatrix.GetRow(i);
                    var oneHot = new Vector<T>(teacher.OutputDimension);
                    int labelIdx = (int)Convert.ToDouble(valVector[i]);
                    if (labelIdx >= 0 && labelIdx < teacher.OutputDimension)
                        oneHot[labelIdx] = NumOps.One;
                    valLabels[i] = oneHot;
                }
            }

            // Step 5: Define forward and backward functions
            // Storage for per-sample inputs to enable forward pass replay during backprop
            // Use a queue to match forward inputs with backward gradients in FIFO order
            var inputQueue = new Queue<Vector<T>>();

            // Forward function must save activations for backprop AND capture inputs for replay
            // Convert Vector<T> (from KD trainer) → TInput → model.Predict → TOutput → Vector<T>
            Func<Vector<T>, Vector<T>> studentForwardCapturing = input =>
            {
                // Capture input for forward replay in backward pass (FIFO queue)
                var capturedInput = new Vector<T>(input.Length);
                for (int i = 0; i < input.Length; i++)
                    capturedInput[i] = input[i];
                inputQueue.Enqueue(capturedInput);

                // Convert KD trainer's Vector<T> to model's TInput type using reference for shape
                TInput modelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(input, referenceInput);

                if (studentModel is INeuralNetwork<T> nnModel)
                {
                    // Use ForwardWithMemory() to save activations for backpropagation
                    var output = nnModel.ForwardWithMemory(Tensor<T>.FromVector(input));
                    return output.ToVector();
                }

                // Fallback for non-NeuralNetworkModel: call Predict and convert result
                TOutput modelOutput = studentModel.Predict(modelInput);
                return ConversionsHelper.ConvertToVector<T, TOutput>(modelOutput);
            };

            // Prepare backward function for parameter updates during distillation training
            // This function receives output gradients from distillation strategy and applies them to the model
            Action<Vector<T>> studentBackward = gradient =>
            {
                // Cast to INeuralNetwork to access backpropagation methods
                if (studentModel is not INeuralNetwork<T> nnModel)
                {
                    throw new InvalidOperationException(
                        "Knowledge distillation requires a neural network (INeuralNetwork<T>) for gradient backpropagation. " +
                        $"Current model type: {studentModel.GetType().Name}");
                }

                try
                {
                    // CRITICAL FIX: Replay forward pass to restore correct activations before backprop
                    // The KD trainer calls forward for all batch samples first, which overwrites
                    // the activation memory. We must dequeue and rerun forward with the matching input
                    // to ensure Backpropagate uses the correct activations for this specific sample.
                    if (inputQueue.Count > 0)
                    {
                        var matchingInput = inputQueue.Dequeue();
                        nnModel.ForwardWithMemory(Tensor<T>.FromVector(matchingInput));
                    }

                    // Step 1: Backpropagate output gradient through network to compute parameter gradients
                    nnModel.Backpropagate(Tensor<T>.FromVector(gradient));

                    // Step 2: Get parameter gradients from backpropagation
                    var paramGradients = nnModel.GetParameterGradients();

                    // Step 3: Apply gradient-based optimizer update if available
                    if (optimizer is IGradientBasedOptimizer<T, Vector<T>, Vector<T>> gradOptimizer)
                    {
                        // Use optimizer's UpdateParameters to apply gradients with proper state management
                        // This preserves momentum, ADAM state, and uses configured learning rate
                        var currentParams = nnModel.GetParameters();
                        var updatedParams = gradOptimizer.UpdateParameters(currentParams, paramGradients);
                        nnModel.UpdateParameters(updatedParams);
                    }
                    else
                    {
                        // Fallback: Simple gradient descent with configured learning rate
                        // This doesn't preserve optimizer state but respects the learning rate
                        var currentParams = nnModel.GetParameters();
                        var learningRate = NumOps.FromDouble(options.LearningRate);
                        var newParams = new Vector<T>(currentParams.Length);

                        for (int i = 0; i < currentParams.Length; i++)
                        {
                            // Apply gradient descent: params = params - learningRate * gradient
                            newParams[i] = NumOps.Subtract(currentParams[i],
                                NumOps.Multiply(learningRate, paramGradients[i]));
                        }

                        nnModel.UpdateParameters(newParams);
                    }
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        "Failed to apply gradient updates during knowledge distillation. " +
                        "Ensure the model supports backpropagation.", ex);
                }
            };

            // Step 5: Run knowledge distillation training
            Console.WriteLine("Training student model with knowledge distillation...");
            trainer.Train(
                studentForwardCapturing,
                studentBackward,
                trainInputs,
                trainLabels,
                epochs: options.Epochs,
                batchSize: options.BatchSize,
                validationInputs: valInputs,
                validationLabels: valLabels);

            // Step 7: Return result from KD-trained model (don't re-optimize)
            // Model is already trained via knowledge distillation, just wrap it in result
            var result = new OptimizationResult<T, TInput, TOutput>
            {
                BestSolution = studentModel,
                BestFitnessScore = NumOps.FromDouble(0.0) // Score tracking happened during KD training
            };
            return Task.FromResult(result);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error setting up knowledge distillation: {ex.Message}");
            Console.WriteLine("Falling back to standard training.");
            return Task.FromResult(optimizer.Optimize(OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(
                XTrain, yTrain, XVal, yVal, XTest, yTest)));
        }
    }

    // ============================================================================
    // Private Agent Helper Methods
    // ============================================================================

    /// <summary>
    /// Analyzes dataset and generates AI agent recommendations for model selection, hyperparameters, and training strategy.
    /// </summary>
    /// <remarks>
    /// ARCHITECTURE NOTES AND KNOWN LIMITATIONS:
    ///
    /// 1. Generic Type Conversion (Convert.ToDouble):
    ///    - Statistical calculations require floating-point arithmetic (mean, std, etc.)
    ///    - Generic type T is converted to double for analysis purposes only
    ///    - Model training continues to use the original generic type T throughout
    ///    - Limitation: Custom numeric types that can't convert to double won't work with agent analysis
    ///    - Future: Could be improved by using INumericOperations<T> for all calculations
    ///
    /// 2. Method Length (253 lines):
    ///    - This method orchestrates multiple agent analysis phases:
    ///      * Data analysis (statistics, distributions, correlations)
    ///      * Model selection (algorithm recommendation)
    ///      * Hyperparameter tuning (parameter optimization)
    ///      * Feature importance analysis
    ///      * Cross-validation strategy
    ///    - Each phase involves LLM calls and result parsing
    ///    - Breaking into smaller methods would require passing many parameters
    ///    - Trade-off: Single coherent workflow vs. method length guidelines
    ///    - Future: Could extract phases to separate analyzer classes
    ///
    /// 3. Hardcoded Assumptions:
    ///    - Assumes regression for continuous targets (line ~747)
    ///    - Assumes no outliers/missing values initially (line ~775-777)
    ///    - These are safe defaults; actual data analysis overrides them
    ///    - Future: Could infer problem type from TOutput constraints
    ///
    /// 4. Error Handling:
    ///    - LLM failures gracefully degrade (skip that analysis phase)
    ///    - Partial recommendations are still useful
    ///    - Future: Could add retry logic for transient failures
    /// </remarks>
    private async Task<AgentRecommendation<T, TInput, TOutput>> GetAgentRecommendationsAsync(
        TInput x, TOutput y)
    {
        var chatModel = CreateChatModel(_agentConfig!);
        var recommendation = new AgentRecommendation<T, TInput, TOutput>();

        var convertedX = ConversionsHelper.ConvertToMatrix<T, TInput>(x);
        var convertedY = ConversionsHelper.ConvertToVector<T, TOutput>(y);

        var nSamples = convertedX.Rows;
        var nFeatures = convertedX.Columns;

        // Instantiate all specialized agent tools
        var allTools = new ITool[]
        {
            new DataAnalysisTool(),
            new ModelSelectionTool(),
            new HyperparameterTool(),
            new FeatureImportanceTool(),
            new CrossValidationTool(),
            new RegularizationTool()
        };

        // Create agent with all specialized tools
        var agent = new ChainOfThoughtAgent<T>(chatModel, allTools);

        var reasoningTrace = new System.Text.StringBuilder();
        reasoningTrace.AppendLine("=== AGENT ASSISTANCE ANALYSIS ===\n");

        // 1. DATA ANALYSIS
        if (_agentOptions.EnableDataAnalysis)
        {
            reasoningTrace.AppendLine("STEP 1: Analyzing dataset characteristics...\n");

            // Calculate basic statistics for data analysis tool
            // Note: Convert.ToDouble is used here because statistical calculations (mean, std, etc.)
            // require floating-point arithmetic. This is a known limitation where the generic type T
            // is converted to double for agent analysis purposes only. The actual model training
            // continues to use the generic type T throughout.
            var statistics = new Newtonsoft.Json.Linq.JObject();
            for (int col = 0; col < nFeatures; col++)
            {
                var featureData = new List<double>();
                for (int row = 0; row < nSamples; row++)
                {
                    featureData.Add(Convert.ToDouble(convertedX[row, col]));
                }

                var mean = featureData.Average();
                var variance = featureData.Select(x => Math.Pow(x - mean, 2)).Average();
                var std = Math.Sqrt(variance);

                statistics[$"feature_{col}"] = new Newtonsoft.Json.Linq.JObject
                {
                    ["mean"] = mean,
                    ["std"] = std,
                    ["min"] = featureData.Min(),
                    ["max"] = featureData.Max(),
                    ["missing_pct"] = 0.0  // Assume no missing values for now
                };
            }

            var dataAnalysisInput = new Newtonsoft.Json.Linq.JObject
            {
                ["dataset_info"] = new Newtonsoft.Json.Linq.JObject
                {
                    ["n_samples"] = nSamples,
                    ["n_features"] = nFeatures,
                    ["target_type"] = "continuous"  // Assume regression for now
                },
                ["statistics"] = statistics
            }.ToString(Formatting.None);

            var dataAnalysisResult = await agent.RunAsync(
                $@"Use the DataAnalysisTool to analyze this dataset.

                Input for DataAnalysisTool:
                {dataAnalysisInput}

                Provide comprehensive data analysis.");

            recommendation.DataAnalysis = dataAnalysisResult;
            reasoningTrace.AppendLine($"Data Analysis Results:\n{dataAnalysisResult}\n");
        }

        // 2. MODEL SELECTION
        if (_agentOptions.EnableModelSelection && _model == null)
        {
            reasoningTrace.AppendLine("STEP 2: Selecting optimal model type...\n");

            var modelSelectionInput = new Newtonsoft.Json.Linq.JObject
            {
                ["problem_type"] = "regression",  // Assuming regression for now
                ["n_samples"] = nSamples,
                ["n_features"] = nFeatures,
                ["is_linear"] = false,  // Conservative default
                ["has_outliers"] = false,  // Would need analysis
                ["has_missing_values"] = false,
                ["requires_interpretability"] = false,
                ["computational_constraints"] = "moderate"
            }.ToString(Formatting.None);

            var modelSelectionResult = await agent.RunAsync(
                $@"Use the ModelSelectionTool to recommend the best model for this dataset.

                Input for ModelSelectionTool:
                {modelSelectionInput}

                Based on the tool's recommendation, suggest ONE specific ModelType.");

            recommendation.ModelSelectionReasoning = modelSelectionResult;

            // Try to extract model type from agent response
            var agentResponse = modelSelectionResult.ToLower();
            recommendation.SuggestedModelType = agentResponse switch
            {
                var r when r.Contains("random forest") || r.Contains("randomforest") => ModelType.RandomForest,
                var r when r.Contains("neural network") || r.Contains("neuralnetwork") => ModelType.NeuralNetworkRegression,
                var r when r.Contains("polynomial") => ModelType.PolynomialRegression,
                var r when r.Contains("ridge") => ModelType.SimpleRegression,
                var r when r.Contains("multiple") => ModelType.MultipleRegression,
                var r when r.Contains("simple") || r.Contains("linear") => ModelType.SimpleRegression,
                _ => null
            };

            reasoningTrace.AppendLine($"Model Selection:\n{modelSelectionResult}\n");
            if (recommendation.SuggestedModelType.HasValue)
            {
                reasoningTrace.AppendLine($"Selected Model: {recommendation.SuggestedModelType.Value}\n");
            }
        }

        // 3. HYPERPARAMETER TUNING
        if (_agentOptions.EnableHyperparameterTuning)
        {
            reasoningTrace.AppendLine("STEP 3: Recommending hyperparameter values...\n");

            var modelTypeStr = recommendation.SuggestedModelType?.ToString() ?? _model?.GetType().Name ?? "RandomForest";

            var hyperparameterInput = new Newtonsoft.Json.Linq.JObject
            {
                ["model_type"] = modelTypeStr,
                ["n_samples"] = nSamples,
                ["n_features"] = nFeatures,
                ["problem_type"] = "regression",
                ["data_complexity"] = "moderate"
            }.ToString(Formatting.None);

            var hyperparameterResult = await agent.RunAsync(
                $@"Use the HyperparameterTool to suggest optimal hyperparameters.

                Input for HyperparameterTool:
                {hyperparameterInput}

                Provide specific hyperparameter recommendations.");

            recommendation.TuningReasoning = hyperparameterResult;
            reasoningTrace.AppendLine($"Hyperparameter Recommendations:\n{hyperparameterResult}\n");

            // Try to extract hyperparameters (simplified - could be enhanced)
            recommendation.SuggestedHyperparameters = new Dictionary<string, object>
            {
                ["info"] = "See TuningReasoning for detailed hyperparameter recommendations"
            };
        }

        // 4. FEATURE ANALYSIS
        if (_agentOptions.EnableFeatureAnalysis)
        {
            reasoningTrace.AppendLine("STEP 4: Analyzing feature importance...\n");

            // Build feature analysis input using lightweight statistics (correlation-based proxies).
            var numOps = MathHelper.GetNumericOperations<T>();
            var features = new Newtonsoft.Json.Linq.JObject();
            for (int col = 0; col < Math.Min(nFeatures, 20); col++)  // Limit to first 20 features
            {
                int missingCount = 0;
                int count = 0;
                double sumX = 0.0;
                double sumY = 0.0;
                double sumXX = 0.0;
                double sumYY = 0.0;
                double sumXY = 0.0;

                for (int row = 0; row < nSamples; row++)
                {
                    double xValue = numOps.ToDouble(convertedX[row, col]);
                    double yValue = numOps.ToDouble(convertedY[row]);

                    if (double.IsNaN(xValue) || double.IsInfinity(xValue) || double.IsNaN(yValue) || double.IsInfinity(yValue))
                    {
                        missingCount++;
                        continue;
                    }

                    count++;
                    sumX += xValue;
                    sumY += yValue;
                    sumXX += xValue * xValue;
                    sumYY += yValue * yValue;
                    sumXY += xValue * yValue;
                }

                double correlation = 0.0;
                if (count >= 2)
                {
                    double cov = sumXY - (sumX * sumY / count);
                    double varX = sumXX - (sumX * sumX / count);
                    double varY = sumYY - (sumY * sumY / count);

                    if (varX > 0.0 && varY > 0.0)
                    {
                        correlation = cov / Math.Sqrt(varX * varY);
                        if (correlation > 1.0) correlation = 1.0;
                        if (correlation < -1.0) correlation = -1.0;
                    }
                }

                double missingPct = nSamples > 0 ? (double)missingCount / nSamples : 0.0;
                double importance = Math.Abs(correlation);

                features[$"feature_{col}"] = new Newtonsoft.Json.Linq.JObject
                {
                    ["target_correlation"] = correlation,
                    ["importance_score"] = importance,
                    ["missing_pct"] = missingPct,
                    ["correlations"] = new Newtonsoft.Json.Linq.JObject()
                };
            }

            var featureAnalysisInput = new Newtonsoft.Json.Linq.JObject
            {
                ["features"] = features,
                ["target_name"] = "target",
                ["n_samples"] = nSamples
            }.ToString(Formatting.None);

            var featureAnalysisResult = await agent.RunAsync(
                $@"Use the FeatureImportanceTool to analyze features and suggest improvements.

                Input for FeatureImportanceTool:
                {featureAnalysisInput}

                Provide feature importance analysis and engineering suggestions.");

            recommendation.FeatureRecommendations = featureAnalysisResult;
            reasoningTrace.AppendLine($"Feature Analysis:\n{featureAnalysisResult}\n");
        }

        // 5. CROSS-VALIDATION STRATEGY (part of meta-learning advice)
        if (_agentOptions.EnableMetaLearningAdvice)
        {
            reasoningTrace.AppendLine("STEP 5: Recommending validation strategy...\n");

            var cvInput = new Newtonsoft.Json.Linq.JObject
            {
                ["n_samples"] = nSamples,
                ["n_features"] = nFeatures,
                ["problem_type"] = "regression",
                ["is_time_series"] = false,
                ["is_imbalanced"] = false,
                ["has_groups"] = false,
                ["computational_budget"] = "moderate"
            }.ToString(Formatting.None);

            var cvResult = await agent.RunAsync(
                $@"Use the CrossValidationTool to recommend the best validation strategy.

                Input for CrossValidationTool:
                {cvInput}

                Suggest optimal cross-validation approach.");

            reasoningTrace.AppendLine($"Cross-Validation Strategy:\n{cvResult}\n");

            // Regularization recommendations
            var regularizationInput = new Newtonsoft.Json.Linq.JObject
            {
                ["model_type"] = recommendation.SuggestedModelType?.ToString() ?? "RandomForest",
                ["n_samples"] = nSamples,
                ["n_features"] = nFeatures,
                ["training_score"] = 0.0,
                ["validation_score"] = 0.0,
                ["is_overfitting"] = false,
                ["current_regularization"] = "none"
            }.ToString(Formatting.None);

            var regularizationResult = await agent.RunAsync(
                $@"Use the RegularizationTool to recommend regularization techniques.

                Input for RegularizationTool:
                {regularizationInput}

                Provide regularization recommendations for preventing overfitting.");

            reasoningTrace.AppendLine($"Regularization Recommendations:\n{regularizationResult}\n");
        }

        // Store complete reasoning trace
        recommendation.ReasoningTrace = reasoningTrace.ToString();

        return recommendation;
    }

    /// <summary>
    /// Applies agent recommendations to the model builder where possible, or provides user guidance.
    /// </summary>
    /// <remarks>
    /// ARCHITECTURE DECISIONS AND LIMITATIONS:
    ///
    /// This method provides INFORMATIONAL GUIDANCE rather than full auto-configuration because:
    ///
    /// 1. Model Auto-Creation Complexity:
    ///    - The library has 80+ model types with different constructor signatures
    ///    - Creating a universal model factory would require:
    ///      * Mapping model types to constructors
    ///      * Determining appropriate default parameters for each model
    ///      * Handling model-specific dependencies and configurations
    ///    - This complexity outweighs the benefit of auto-creation
    ///    - Solution: Provide clear console guidance for manual configuration
    ///
    /// 2. Hyperparameter Auto-Application:
    ///    - Tracked in Issue #460: "Auto-Apply Agent Hyperparameter Recommendations"
    ///    - Requires reflection-based property setting
    ///    - Needs validation that hyperparameters are compatible with model
    ///    - Future enhancement with HyperparameterApplicator service
    ///
    /// 3. User Control:
    ///    - Developers may want to review recommendations before applying
    ///    - Explicit configuration prevents unexpected model changes
    ///    - Recommendation details available in result.AgentRecommendation for review
    ///
    /// 4. Current Functionality:
    ///    - Displays model type recommendation with reasoning via console
    ///    - Provides code example for manual configuration
    ///    - Stores full recommendations in result object for programmatic access
    ///    - Future: Will auto-apply hyperparameters when model is already set
    /// </remarks>
    private void ApplyAgentRecommendations(AgentRecommendation<T, TInput, TOutput> recommendation)
    {
        ApplyAgentRecommendationsCore(recommendation);
    }

    /// <summary>
    /// Applies trial hyperparameters from HPO to the optimizer options.
    /// </summary>
    /// <param name="options">The optimizer options to modify.</param>
    /// <param name="trialHyperparameters">Dictionary of hyperparameter names to values.</param>
    /// <remarks>
    /// <para>
    /// This method applies hyperparameters discovered during HPO to the optimizer options.
    /// Common hyperparameters that can be tuned include:
    /// - learning_rate: The learning rate for gradient-based optimizers
    /// - max_iterations: Maximum number of training iterations/epochs
    /// - tolerance: Convergence tolerance
    /// - beta1, beta2: Adam optimizer momentum parameters
    /// - momentum: Momentum for SGD-style optimizers
    /// </para>
    /// <para><b>For Beginners:</b> Hyperparameter optimization (HPO) tries different combinations
    /// of settings to find what works best for your specific problem. This method takes those
    /// discovered settings and applies them to the optimizer before training.
    /// </para>
    /// </remarks>
    private static void ApplyTrialHyperparameters(object options, Dictionary<string, object> trialHyperparameters)
    {
        if (options is null || trialHyperparameters is null || trialHyperparameters.Count == 0)
        {
            return;
        }

        var optionsType = options.GetType();

        foreach (var kvp in trialHyperparameters)
        {
            var paramName = kvp.Key;
            var paramValue = kvp.Value;

            if (paramValue is null)
            {
                continue;
            }

            // Map common hyperparameter names to property names
            var propertyName = MapHyperparameterToProperty(paramName);
            if (string.IsNullOrEmpty(propertyName))
            {
                continue;
            }

            // Find the property on the options object
            var property = optionsType.GetProperty(propertyName,
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);

            if (property is null || !property.CanWrite)
            {
                // Try base types for inherited properties
                var baseType = optionsType.BaseType;
                while (baseType is not null && property is null)
                {
                    property = baseType.GetProperty(propertyName,
                        System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                    baseType = baseType.BaseType;
                }

                if (property is null || !property.CanWrite)
                {
                    continue;
                }
            }

            // Convert and set the value
            try
            {
                var convertedValue = ConvertHyperparameterValue(paramValue, property.PropertyType);
                if (convertedValue is not null)
                {
                    property.SetValue(options, convertedValue);
                }
            }
            catch
            {
                // Skip hyperparameters that cannot be converted - this is non-fatal
                // as the optimizer will use its default value
            }
        }
    }

    /// <summary>
    /// Maps common hyperparameter names from search spaces to property names on optimizer options.
    /// </summary>
    private static string MapHyperparameterToProperty(string hyperparameterName)
    {
        // Normalize the name to lowercase for matching
        var normalizedName = hyperparameterName.ToLowerInvariant().Replace("_", "").Replace("-", "");

        return normalizedName switch
        {
            // Learning rate variations
            "learningrate" or "lr" or "initiallearningrate" => "InitialLearningRate",

            // Iteration/epoch settings
            "maxiterations" or "iterations" or "epochs" or "maxepochs" => "MaxIterations",

            // Convergence settings
            "tolerance" or "tol" or "convergencetolerance" => "Tolerance",

            // Early stopping
            "earlystoppingpatience" or "patience" => "EarlyStoppingPatience",

            // Adam-specific parameters (check optimizer-specific options first)
            "beta1" or "b1" => "Beta1",
            "beta2" or "b2" => "Beta2",
            "epsilon" or "eps" => "Epsilon",

            // Momentum
            "momentum" or "initialmomentum" => "InitialMomentum",

            // Learning rate scheduling
            "learningratedecay" or "lrdecay" or "decay" => "LearningRateDecay",
            "minlearningrate" or "minlr" => "MinLearningRate",
            "maxlearningrate" or "maxlr" => "MaxLearningRate",

            // Regularization strength
            "l2regularization" or "weightdecay" or "regularization" => "RegularizationStrength",

            // Batch size (for applicable optimizers)
            "batchsize" or "batch" => "BatchSize",

            // Gradient clipping
            "maxgradientnorm" or "clipnorm" or "gradientclipnorm" => "MaxGradientNorm",
            "maxgradientvalue" or "clipvalue" or "gradientclipvalue" => "MaxGradientValue",

            // Unknown hyperparameter - try using the name directly
            _ => ToPascalCase(hyperparameterName)
        };
    }

    /// <summary>
    /// Converts a hyperparameter name to PascalCase for property lookup.
    /// </summary>
    private static string ToPascalCase(string name)
    {
        if (string.IsNullOrEmpty(name))
        {
            return name;
        }

        var parts = name.Split(new[] { '_', '-' }, StringSplitOptions.RemoveEmptyEntries);
        var result = new System.Text.StringBuilder();

        foreach (var part in parts)
        {
            if (part.Length > 0)
            {
                result.Append(char.ToUpperInvariant(part[0]));
                if (part.Length > 1)
                {
                    result.Append(part.Substring(1).ToLowerInvariant());
                }
            }
        }

        return result.ToString();
    }

    /// <summary>
    /// Converts a hyperparameter value to the target property type.
    /// </summary>
    private static object? ConvertHyperparameterValue(object value, Type targetType)
    {
        if (value is null)
        {
            return null;
        }

        var valueType = value.GetType();

        // If already the correct type, return as-is
        if (targetType.IsAssignableFrom(valueType))
        {
            return value;
        }

        // Handle numeric conversions
        if (targetType == typeof(double))
        {
            return Convert.ToDouble(value);
        }
        if (targetType == typeof(float))
        {
            return Convert.ToSingle(value);
        }
        if (targetType == typeof(int))
        {
            return Convert.ToInt32(value);
        }
        if (targetType == typeof(long))
        {
            return Convert.ToInt64(value);
        }
        if (targetType == typeof(bool))
        {
            return Convert.ToBoolean(value);
        }

        // Handle nullable types
        var underlyingType = Nullable.GetUnderlyingType(targetType);
        if (underlyingType is not null)
        {
            return ConvertHyperparameterValue(value, underlyingType);
        }

        // Try using Convert.ChangeType as a last resort
        try
        {
            return Convert.ChangeType(value, targetType);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Core implementation of ApplyAgentRecommendations.
    /// </summary>
    private void ApplyAgentRecommendationsCore(AgentRecommendation<T, TInput, TOutput> recommendation)
    {
        // Apply agent recommendations where possible
        if (_model == null && recommendation.SuggestedModelType.HasValue)
        {
            // Agent recommended a model type
            // Note: Auto-creation of model instances is not implemented to avoid the complexity
            // of a model factory with correct constructor parameters for all ~80+ model types.
            // Instead, the recommendation is available in result.AgentRecommendation for the user
            // to review and manually configure using the builder's UseModel() method.

            Console.WriteLine($"\n=== AGENT RECOMMENDATION ===");
            Console.WriteLine($"The AI agent recommends using: {recommendation.SuggestedModelType.Value}");

            var reasoning = recommendation.ModelSelectionReasoning ?? string.Empty;
            if (reasoning.Length > 0)
            {
                var maxLength = Math.Min(200, reasoning.Length);
                Console.WriteLine($"Reason: {reasoning.Substring(0, maxLength)}...");
            }

            Console.WriteLine($"\nTo use this recommendation, configure your builder:");
            Console.WriteLine($"  builder.UseModel(/* create {recommendation.SuggestedModelType.Value} instance */);");
            Console.WriteLine($"\nFull recommendation details available in result.AgentRecommendation");
            Console.WriteLine("===========================\n");
        }

        // Note: Hyperparameter recommendations are currently stored in recommendation.SuggestedHyperparameters
        // but not auto-applied. Future enhancement: Apply hyperparameters to compatible models.
    }

    /// <summary>
    /// Computes a robust hash of the training data for version control and lineage tracking.
    /// </summary>
    /// <param name="features">The feature matrix (X).</param>
    /// <param name="targets">The target vector (y).</param>
    /// <param name="numOps">Numeric operations for type conversion.</param>
    /// <returns>A 16-character hex hash representing the data version.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a hash that captures the essential characteristics of the training data:
    /// - Dataset dimensions (rows, columns)
    /// - Sample of feature values from first, middle, and last rows
    /// - Sample of target values from first, middle, and last positions
    /// - Statistical summary (sum of sampled values for collision resistance)
    /// </para>
    /// <para><b>For Beginners:</b> This hash is like a fingerprint for your training data.
    /// If the data changes, the hash will change too, allowing you to track exactly which
    /// version of the data was used to train a model. This is essential for reproducibility.
    /// </para>
    /// </remarks>
    private static string ComputeDataVersionHash(Matrix<T> features, Vector<T> targets, INumericOperations<T> numOps)
    {
        var hashBuilder = new StringBuilder();

        // Include dimensions for basic structure identification
        hashBuilder.Append($"X:{features.Rows}x{features.Columns};");
        hashBuilder.Append($"y:{targets.Length};");

        // Sample feature values from first, middle, and last rows for better coverage
        // This catches changes anywhere in the dataset, not just at boundaries
        int maxCols = Math.Min(features.Columns, 10);
        int[] sampleRows = GetSampleRowIndices(features.Rows);

        foreach (int row in sampleRows)
        {
            if (row >= 0 && row < features.Rows)
            {
                hashBuilder.Append($"r{row}:");
                for (int col = 0; col < maxCols; col++)
                {
                    hashBuilder.Append($"{numOps.ToDouble(features[row, col]):G6},");
                }
                hashBuilder.Append(';');
            }
        }

        // Include target values from sampled positions
        // This ensures the hash changes if targets are modified
        int[] sampleTargetIndices = GetSampleRowIndices(targets.Length);
        hashBuilder.Append("y:");
        foreach (int idx in sampleTargetIndices)
        {
            if (idx >= 0 && idx < targets.Length)
            {
                hashBuilder.Append($"{numOps.ToDouble(targets[idx]):G6},");
            }
        }
        hashBuilder.Append(';');

        // Add a statistical fingerprint for additional collision resistance
        // Sum of sampled values helps detect subtle changes
        double featureSum = 0.0;
        double targetSum = 0.0;

        foreach (int row in sampleRows)
        {
            if (row >= 0 && row < features.Rows)
            {
                for (int col = 0; col < maxCols; col++)
                {
                    featureSum += numOps.ToDouble(features[row, col]);
                }
            }
        }

        foreach (int idx in sampleTargetIndices)
        {
            if (idx >= 0 && idx < targets.Length)
            {
                targetSum += numOps.ToDouble(targets[idx]);
            }
        }

        hashBuilder.Append($"fsum:{featureSum:G6};tsum:{targetSum:G6};");

        // Compute SHA256 hash and return first 16 hex characters
        using var sha256 = System.Security.Cryptography.SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(hashBuilder.ToString()));

        // Convert bytes to hex string (compatible with net471 which doesn't have Convert.ToHexString)
        var fullHex = BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
        return fullHex.Substring(0, Math.Min(16, fullHex.Length));
    }

    /// <summary>
    /// Gets sample row indices for data hashing (first, middle, last).
    /// </summary>
    private static int[] GetSampleRowIndices(int totalRows)
    {
        if (totalRows <= 0)
        {
            return Array.Empty<int>();
        }

        if (totalRows == 1)
        {
            return new[] { 0 };
        }

        if (totalRows == 2)
        {
            return new[] { 0, 1 };
        }

        // Sample first, middle, and last rows
        int middle = totalRows / 2;
        return new[] { 0, middle, totalRows - 1 };
    }

    /// <summary>
    /// Applies GPU acceleration configuration to the global AiDotNetEngine based on user settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method configures the AiDotNetEngine (internal GPU/CPU engine) according to the user's
    /// GPU acceleration preferences set via ConfigureGpuAcceleration(). This is an internal method
    /// called automatically during BuildAsync() and is not part of the public facade API.
    /// </para>
    /// <para>
    /// The facade pattern is maintained: users configure GPU via ConfigureGpuAcceleration() with
    /// nullable defaults (null = industry standard behavior), and this method translates those
    /// settings into internal engine configuration.
    /// </para>
    /// <para><b>GPU Usage Level Behaviors:</b>
    /// - <b>Null config (default)</b>: Auto-detect GPU with CPU fallback (industry standard)
    /// - <b>Default</b>: Balanced GPU usage, good for most desktop GPUs (recommended)
    /// - <b>Conservative</b>: Auto-detect GPU, use it only for very large operations, frequent CPU fallback
    /// - <b>Aggressive</b>: Force GPU, throw exception if not available, use GPU for smaller operations
    /// - <b>AlwaysGpu</b>: Force all operations to GPU (maximize GPU utilization)
    /// - <b>AlwaysCpu</b>: Force CPU-only execution, never use GPU
    /// </para>
    /// <para><b>GPU Device Type Behaviors:</b>
    /// - <b>Auto</b>: Use DirectGpu backend order (CUDA → OpenCL → HIP) with CPU fallback
    /// - <b>CUDA</b>: Force NVIDIA CUDA backend (throws if NVIDIA GPU not available)
    /// - <b>OpenCL</b>: Force OpenCL backend (works with NVIDIA, AMD, Intel, throws if no GPU)
    /// - <b>CPU</b>: Force CPU-only execution (equivalent to UsageLevel.AlwaysCpu)
    /// </para>
    /// </remarks>
    private void ApplyGpuConfiguration()
    {
        // Skip if no GPU configuration was provided (null = default = auto-detect with CPU fallback)
        if (_gpuAccelerationConfig == null)
        {
            // Industry standard default: Try to auto-detect GPU, use CPU fallback if not available
            // This is silent and non-intrusive - if GPU exists, use it; if not, use CPU
            try
            {
                AiDotNetEngine.AutoDetectAndConfigureGpu();
            }
            catch
            {
                // Silently fall back to CPU if GPU detection fails
                // This ensures the library works out of the box on any hardware
            }
            return;
        }

        if (_gpuAccelerationConfig.UsageLevel == AiDotNet.Engines.GpuUsageLevel.AlwaysCpu)
        {
            AiDotNetEngine.ResetToCpu();
            return;
        }

        if (_gpuAccelerationConfig.DeviceType == AiDotNet.Engines.GpuDeviceType.CPU)
        {
            AiDotNetEngine.ResetToCpu();
            return;
        }

        if (_gpuAccelerationConfig.DeviceType == AiDotNet.Engines.GpuDeviceType.CUDA)
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "cuda");
        }
        else if (_gpuAccelerationConfig.DeviceType == AiDotNet.Engines.GpuDeviceType.OpenCL)
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "opencl");
        }

        // Apply configuration based on usage level
        switch (_gpuAccelerationConfig.UsageLevel)
        {
            case AiDotNet.Engines.GpuUsageLevel.AlwaysCpu:
                // Force CPU-only execution (useful for debugging, testing, or CPU-only servers)
                AiDotNetEngine.ResetToCpu();
                break;

            case AiDotNet.Engines.GpuUsageLevel.Default:
                // Balanced GPU usage - recommended mode for most users
                // Auto-detect GPU with intelligent fallback for typical desktop GPUs
                try
                {
                    bool gpuDetected = AiDotNetEngine.AutoDetectAndConfigureGpu();
                    if (!gpuDetected)
                    {
                        // No GPU detected - system already fell back to CPU
                        // No error needed, CPU fallback is expected behavior
                    }
                }
                catch (Exception ex)
                {
                    // GPU initialization failed - fall back to CPU
                    Console.WriteLine($"[AiDotNet] GPU initialization failed: {ex.Message}");
                    Console.WriteLine("[AiDotNet] Falling back to CPU execution");
                    AiDotNetEngine.ResetToCpu();
                }
                break;

            case AiDotNet.Engines.GpuUsageLevel.Conservative:
                // Use GPU conservatively - auto-detect but use higher thresholds and more frequent CPU fallback
                // This is for older/slower GPUs or systems where GPU reliability is a concern
                try
                {
                    bool gpuDetected = AiDotNetEngine.AutoDetectAndConfigureGpu();
                    if (gpuDetected)
                    {
                        Console.WriteLine($"[AiDotNet] Conservative GPU mode enabled: {AiDotNetEngine.Current.Name}");
                        Console.WriteLine("[AiDotNet] GPU will be used only for very large operations (100K+ elements)");
                    }
                    else
                    {
                        // No GPU detected - fall back to CPU (expected behavior in Conservative mode)
                        Console.WriteLine("[AiDotNet] No GPU detected - using CPU (Conservative mode)");
                    }
                }
                catch (Exception ex)
                {
                    // GPU initialization failed in Conservative mode - fall back to CPU silently
                    Console.WriteLine($"[AiDotNet] GPU initialization failed in Conservative mode: {ex.Message}");
                    Console.WriteLine("[AiDotNet] Falling back to CPU execution");
                    AiDotNetEngine.ResetToCpu();
                }
                break;

            case AiDotNet.Engines.GpuUsageLevel.Aggressive:
                // Force GPU with minimal fallback - throw exception if GPU is not available
                // This is for users with high-end GPUs who want maximum performance and need to know if GPU fails
                try
                {
                    bool gpuDetected = AiDotNetEngine.AutoDetectAndConfigureGpu();
                    if (!gpuDetected)
                    {
                        throw new InvalidOperationException(
                            "GPU acceleration is set to Aggressive mode but no compatible GPU was detected. " +
                            "Aggressive mode requires a GPU to be available. " +
                            "Options: (1) Install a compatible GPU (NVIDIA/AMD/Intel), " +
                            "(2) Install GPU drivers, " +
                            "(3) Use GpuUsageLevel.Default for automatic CPU fallback, " +
                            "(4) Use GpuUsageLevel.AlwaysCpu for CPU-only execution.");
                    }

                    // Verify GPU is actually being used
                    if (!AiDotNetEngine.Current.SupportsGpu)
                    {
                        throw new InvalidOperationException(
                            "GPU acceleration is set to Aggressive mode but the current engine does not support GPU. " +
                            "This may indicate a GPU initialization failure. Check GPU drivers and compatibility.");
                    }

                    Console.WriteLine($"[AiDotNet] Aggressive GPU mode enabled: {AiDotNetEngine.Current.Name}");
                }
                catch (InvalidOperationException)
                {
                    // Re-throw our explicit exceptions
                    throw;
                }
                catch (Exception ex)
                {
                    // GPU initialization failed in Aggressive mode - this is an error
                    throw new InvalidOperationException(
                        $"GPU acceleration is set to Aggressive mode but GPU initialization failed: {ex.Message}. " +
                        $"Aggressive mode requires a working GPU. " +
                        $"Options: (1) Fix GPU drivers/setup, " +
                        $"(2) Use GpuUsageLevel.Default for automatic CPU fallback, " +
                        $"(3) Use GpuUsageLevel.AlwaysCpu for CPU-only execution.",
                        ex);
                }
                break;

            case AiDotNet.Engines.GpuUsageLevel.AlwaysGpu:
                // Force all operations to GPU - maximize GPU utilization
                // Similar to Aggressive but even more strict
                try
                {
                    bool gpuDetected = AiDotNetEngine.AutoDetectAndConfigureGpu();
                    if (!gpuDetected)
                    {
                        throw new InvalidOperationException(
                            "GPU acceleration is set to AlwaysGpu mode but no compatible GPU was detected. " +
                            "AlwaysGpu mode requires a GPU to be available. " +
                            "Options: (1) Install a compatible GPU (NVIDIA/AMD/Intel), " +
                            "(2) Install GPU drivers, " +
                            "(3) Use GpuUsageLevel.Default for automatic CPU fallback, " +
                            "(4) Use GpuUsageLevel.AlwaysCpu for CPU-only execution.");
                    }

                    // Verify GPU is actually being used
                    if (!AiDotNetEngine.Current.SupportsGpu)
                    {
                        throw new InvalidOperationException(
                            "GPU acceleration is set to AlwaysGpu mode but the current engine does not support GPU. " +
                            "This may indicate a GPU initialization failure. Check GPU drivers and compatibility.");
                    }

                    Console.WriteLine($"[AiDotNet] AlwaysGpu mode enabled: {AiDotNetEngine.Current.Name}");
                    Console.WriteLine("[AiDotNet] All operations will run on GPU for maximum GPU utilization");
                }
                catch (InvalidOperationException)
                {
                    // Re-throw our explicit exceptions
                    throw;
                }
                catch (Exception ex)
                {
                    // GPU initialization failed in AlwaysGpu mode - this is an error
                    throw new InvalidOperationException(
                        $"GPU acceleration is set to AlwaysGpu mode but GPU initialization failed: {ex.Message}. " +
                        $"AlwaysGpu mode requires a working GPU. " +
                        $"Options: (1) Fix GPU drivers/setup, " +
                        $"(2) Use GpuUsageLevel.Default for automatic CPU fallback, " +
                        $"(3) Use GpuUsageLevel.AlwaysCpu for CPU-only execution.",
                        ex);
                }
                break;

            default:
                throw new ArgumentException($"Unknown GPU usage level: {_gpuAccelerationConfig.UsageLevel}");
        }

        if (_gpuAccelerationConfig.DeviceType != AiDotNet.Engines.GpuDeviceType.Auto)
        {
            Console.WriteLine($"[AiDotNet] DirectGpu backend order forced to {_gpuAccelerationConfig.DeviceType}.");
        }

        if (_gpuAccelerationConfig.DeviceIndex != 0)
        {
            Console.WriteLine("[AiDotNet] Warning: GPU DeviceIndex selection is not implemented for DirectGpu backends.");
        }

        // Apply advanced GPU execution options (Phase 2-3)
        ApplyAdvancedGpuExecutionOptions();
    }

    /// <summary>
    /// Applies advanced GPU execution options from the configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>Phase 2-3: GPU Execution Optimization</b></para>
    /// <para>
    /// Configures advanced execution features:
    /// - Execution mode (eager, deferred, scoped-deferred)
    /// - Graph compilation and kernel fusion
    /// - Multi-stream compute/transfer overlap
    /// - Memory management and prefetching
    /// </para>
    /// </remarks>
    private void ApplyAdvancedGpuExecutionOptions()
    {
        if (_gpuAccelerationConfig == null)
        {
            return;
        }

        // Convert to internal execution options
        var executionOptions = _gpuAccelerationConfig.ToExecutionOptions();

        // Validate the options
        try
        {
            executionOptions.Validate();
        }
        catch (ArgumentException ex)
        {
            Console.WriteLine($"[AiDotNet] Warning: Invalid GPU execution option: {ex.Message}");
            return;
        }

        // Log advanced options if verbose logging is enabled
        if (_gpuAccelerationConfig.VerboseLogging)
        {
            Console.WriteLine($"[AiDotNet] Advanced GPU Execution Options:");
            Console.WriteLine($"  Execution Mode: {_gpuAccelerationConfig.ExecutionMode}");
            Console.WriteLine($"  Graph Compilation: {(_gpuAccelerationConfig.EnableGraphCompilation ? "Enabled" : "Disabled")}");
            Console.WriteLine($"  Auto Fusion: {(_gpuAccelerationConfig.EnableAutoFusion ? "Enabled" : "Disabled")}");
            Console.WriteLine($"  Compute/Transfer Overlap: {(_gpuAccelerationConfig.EnableComputeTransferOverlap ? "Enabled" : "Disabled")}");
            Console.WriteLine($"  Max Compute Streams: {_gpuAccelerationConfig.MaxComputeStreams}");
            Console.WriteLine($"  Min GPU Elements: {_gpuAccelerationConfig.MinGpuElements}");
            Console.WriteLine($"  Max GPU Memory: {_gpuAccelerationConfig.MaxGpuMemoryUsage:P0}");
            Console.WriteLine($"  Prefetch: {(_gpuAccelerationConfig.EnablePrefetch ? "Enabled" : "Disabled")}");
            Console.WriteLine($"  Graph Caching: {(_gpuAccelerationConfig.CacheCompiledGraphs ? "Enabled" : "Disabled")}");
            Console.WriteLine($"  Profiling: {(_gpuAccelerationConfig.EnableProfiling ? "Enabled" : "Disabled")}");
        }

        // Set environment variables for GPU execution options
        // These are read by GpuExecutionOptions.FromEnvironment() when creating execution contexts
        SetGpuExecutionEnvironmentVariables(executionOptions);
    }

    /// <summary>
    /// Sets environment variables for GPU execution options.
    /// </summary>
    /// <param name="options">The execution options to set.</param>
    private static void SetGpuExecutionEnvironmentVariables(AiDotNet.Tensors.Engines.Gpu.GpuExecutionOptions options)
    {
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_MIN_ELEMENTS", options.MinGpuElements.ToString());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_STREAMS", options.MaxComputeStreams.ToString());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_FORCE_GPU", options.ForceGpu.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_FORCE_CPU", options.ForceCpu.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_ENABLE_GRAPH", options.EnableGraphCompilation.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_ENABLE_FUSION", options.EnableAutoFusion.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_MAX_MEMORY", options.MaxMemoryUsage.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_ENABLE_PREFETCH", options.EnablePrefetch.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_ENABLE_OVERLAP", options.EnableComputeTransferOverlap.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_EXECUTION_MODE", options.ExecutionMode.ToString());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_RESIDENT", options.EnableGpuResidency.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_PROFILING", options.EnableProfiling.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_CACHE_GRAPHS", options.CacheCompiledGraphs.ToString().ToLowerInvariant());
    }

    /// <summary>
    /// Applies memory management configuration to models that support it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Memory management helps train larger models by:
    /// - Gradient checkpointing: Trade compute for memory (recompute activations instead of storing all)
    /// - Activation pooling: Reuse tensor memory to reduce garbage collection
    /// </para>
    /// </remarks>
    private void ApplyMemoryConfiguration()
    {
        // Skip if no memory configuration was provided
        if (_memoryConfig is null)
            return;

        // Apply to models that support memory management
        if (_model is NeuralNetworks.NeuralNetworkBase<T> neuralNetwork)
        {
            neuralNetwork.EnableMemoryManagement(_memoryConfig);
        }
    }

    private static (TInput X, TOutput Y, List<(int ClientId, int StartRow, int SampleCount)> ClientRanges)
        BuildAggregatedDatasetFromClientData(IReadOnlyDictionary<int, FederatedClientDataset<TInput, TOutput>> clientData)
    {
        if (clientData is null)
        {
            throw new ArgumentNullException(nameof(clientData));
        }

        if (clientData.Count == 0)
        {
            throw new ArgumentException("Federated client data cannot be empty.", nameof(clientData));
        }

        var orderedClients = clientData.OrderBy(kvp => kvp.Key).ToList();

        int featureCount = -1;
        foreach (var dataset in orderedClients.Select(kvp => kvp.Value))
        {
            if (dataset is null)
            {
                throw new ArgumentException("Federated client data cannot contain null datasets.", nameof(clientData));
            }

            if (dataset.SampleCount <= 0)
            {
                continue;
            }

            var featuresMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(dataset.Features);
            featureCount = featuresMatrix.Columns;
            break;
        }

        if (featureCount < 0)
        {
            featureCount = 0;
        }

        int totalSamples = 0;
        foreach (var kvp in orderedClients)
        {
            var dataset = kvp.Value;
            if (dataset is null)
            {
                throw new ArgumentException("Federated client data cannot contain null datasets.", nameof(clientData));
            }

            if (dataset.SampleCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(clientData), "Federated client datasets must have non-negative SampleCount values.");
            }

            totalSamples += dataset.SampleCount;
        }

        if (totalSamples == 0)
        {
            throw new ArgumentException(
                "Federated client data contains no samples. At least one client must provide SampleCount > 0.",
                nameof(clientData));
        }

        var aggregatedMatrix = new Matrix<T>(totalSamples, featureCount);
        var aggregatedVector = new Vector<T>(totalSamples);
        var ranges = new List<(int ClientId, int StartRow, int SampleCount)>(orderedClients.Count);

        int row = 0;
        foreach (var kvp in orderedClients)
        {
            int clientId = kvp.Key;
            var dataset = kvp.Value;

            if (dataset is null)
            {
                throw new ArgumentException("Federated client data cannot contain null datasets.", nameof(clientData));
            }

            int sampleCount = dataset.SampleCount;
            if (sampleCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(clientData), "Federated client datasets must have non-negative SampleCount values.");
            }

            ranges.Add((clientId, row, sampleCount));

            if (sampleCount == 0)
            {
                continue;
            }

            var featuresMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(dataset.Features);
            var labelsVector = ConversionsHelper.ConvertToVector<T, TOutput>(dataset.Labels);

            if (featuresMatrix.Rows != sampleCount || labelsVector.Length != sampleCount)
            {
                throw new InvalidOperationException(
                    $"Federated client dataset for clientId={clientId} has inconsistent dimensions. " +
                    $"Expected SampleCount={sampleCount}, got X rows={featuresMatrix.Rows} and y length={labelsVector.Length}.");
            }

            if (featuresMatrix.Columns != featureCount)
            {
                throw new InvalidOperationException(
                    $"Federated client dataset for clientId={clientId} has inconsistent feature count. " +
                    $"Expected {featureCount} but found {featuresMatrix.Columns}.");
            }

            for (int i = 0; i < sampleCount; i++)
            {
                aggregatedMatrix.SetRow(row, featuresMatrix.GetRow(i));
                aggregatedVector[row] = labelsVector[i];
                row++;
            }
        }

        if (row != totalSamples)
        {
            throw new InvalidOperationException("Aggregated federated dataset construction produced an unexpected sample count.");
        }

        return (ConvertMatrixToInputType(aggregatedMatrix), ConvertVectorToOutputType(aggregatedVector), ranges);
    }

    private static Dictionary<int, FederatedClientDataset<TInput, TOutput>> CreateFederatedClientPartitionsFromClientRanges(
        TInput xAll,
        TOutput yAll,
        IReadOnlyList<(int ClientId, int StartRow, int SampleCount)> clientRanges)
    {
        if (clientRanges is null)
        {
            throw new ArgumentNullException(nameof(clientRanges));
        }

        if (clientRanges.Count == 0)
        {
            throw new ArgumentException("Federated client ranges cannot be empty.", nameof(clientRanges));
        }

        var xMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(xAll);
        var yVector = ConversionsHelper.ConvertToVector<T, TOutput>(yAll);

        if (xMatrix.Rows != yVector.Length)
        {
            throw new ArgumentException("Federated client range partitioning requires X row count to match y length.");
        }

        int expectedTotal = 0;
        foreach (var range in clientRanges)
        {
            expectedTotal += range.SampleCount;
        }

        if (expectedTotal != xMatrix.Rows)
        {
            throw new InvalidOperationException(
                "Federated client range partitioning requires preprocessing to preserve the total number of samples. " +
                $"Expected {expectedTotal} samples from client partitions but preprocessing produced {xMatrix.Rows}.");
        }

        var partitions = new Dictionary<int, FederatedClientDataset<TInput, TOutput>>(clientRanges.Count);

        foreach (var (clientId, startRow, sampleCount) in clientRanges)
        {
            if (sampleCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(clientRanges), "SampleCount must be non-negative.");
            }

            if (startRow < 0 || startRow + sampleCount > xMatrix.Rows)
            {
                throw new ArgumentOutOfRangeException(nameof(clientRanges), "Client range is outside the dataset bounds.");
            }

            var xClientMatrix = new Matrix<T>(sampleCount, xMatrix.Columns);
            var yClientVector = new Vector<T>(sampleCount);

            for (int i = 0; i < sampleCount; i++)
            {
                int sourceRow = startRow + i;
                xClientMatrix.SetRow(i, xMatrix.GetRow(sourceRow));
                yClientVector[i] = yVector[sourceRow];
            }

            var xClient = ConvertMatrixToInputType(xClientMatrix);
            var yClient = ConvertVectorToOutputType(yClientVector);

            partitions[clientId] = new FederatedClientDataset<TInput, TOutput>(xClient, yClient, sampleCount);
        }

        return partitions;
    }

    private static Dictionary<int, FederatedClientDataset<TInput, TOutput>> CreateFederatedClientPartitions(
        TInput xTrain,
        TOutput yTrain,
        int numberOfClients,
        int? randomSeed)
    {
        if (numberOfClients <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfClients), "Number of clients must be positive.");
        }

        var xMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(xTrain);
        var yVector = ConversionsHelper.ConvertToVector<T, TOutput>(yTrain);

        if (xMatrix.Rows != yVector.Length)
        {
            throw new ArgumentException("Federated partitioning requires X row count to match y length.");
        }

        if (numberOfClients > xMatrix.Rows)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numberOfClients),
                "NumberOfClients must not exceed the number of training samples when creating federated partitions.");
        }

        var indices = Enumerable.Range(0, xMatrix.Rows).ToList();
        var rng = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();
        ShuffleInPlace(indices, rng);

        var clientIndices = new List<int>[numberOfClients];
        for (int i = 0; i < numberOfClients; i++)
        {
            clientIndices[i] = new List<int>();
        }

        for (int i = 0; i < indices.Count; i++)
        {
            int clientId = i % numberOfClients;
            clientIndices[clientId].Add(indices[i]);
        }

        var partitions = new Dictionary<int, FederatedClientDataset<TInput, TOutput>>(numberOfClients);
        for (int clientId = 0; clientId < numberOfClients; clientId++)
        {
            var rows = clientIndices[clientId];
            rows.Sort();

            var xClientMatrix = xMatrix.GetRows(rows);
            var yClientVector = new Vector<T>(rows.Count);
            for (int i = 0; i < rows.Count; i++)
            {
                yClientVector[i] = yVector[rows[i]];
            }

            var xClient = ConvertMatrixToInputType(xClientMatrix);
            var yClient = ConvertVectorToOutputType(yClientVector);

            partitions[clientId] = new FederatedClientDataset<TInput, TOutput>(xClient, yClient, rows.Count);
        }

        return partitions;
    }

    private static void ShuffleInPlace<TItem>(IList<TItem> list, Random random)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static TInput ConvertMatrixToInputType(Matrix<T> matrix)
    {
        if (matrix is TInput typedMatrix)
        {
            return typedMatrix;
        }

        if (typeof(TInput) == typeof(Tensor<T>))
        {
            var tensor = Tensor<T>.FromRowMatrix(matrix);
            if (tensor is TInput typedTensor)
            {
                return typedTensor;
            }
        }

        throw new InvalidOperationException($"Federated learning currently supports TInput of Matrix<T> or Tensor<T>. Got {typeof(TInput).Name}.");
    }

    private static TOutput ConvertVectorToOutputType(Vector<T> vector)
    {
        if (vector is TOutput typedVector)
        {
            return typedVector;
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            var tensor = Tensor<T>.FromVector(vector);
            if (tensor is TOutput typedTensor)
            {
                return typedTensor;
            }
        }

        throw new InvalidOperationException($"Federated learning currently supports TOutput of Vector<T> or Tensor<T>. Got {typeof(TOutput).Name}.");
    }

    private static IAggregationStrategy<IFullModel<T, TInput, TOutput>> CreateDefaultFederatedAggregationStrategy(FederatedLearningOptions options)
    {
        switch (options.AggregationStrategy)
        {
            case FederatedAggregationStrategy.FedAvg:
                return new AiDotNet.FederatedLearning.Aggregators.FedAvgFullModelAggregationStrategy<T, TInput, TOutput>();

            case FederatedAggregationStrategy.FedProx:
                return new AiDotNet.FederatedLearning.Aggregators.FedProxFullModelAggregationStrategy<T, TInput, TOutput>(options.ProximalMu);

            case FederatedAggregationStrategy.FedBN:
                return new AiDotNet.FederatedLearning.Aggregators.FedBNFullModelAggregationStrategy<T, TInput, TOutput>();

            case FederatedAggregationStrategy.Median:
                return new AiDotNet.FederatedLearning.Aggregators.MedianFullModelAggregationStrategy<T, TInput, TOutput>();

            case FederatedAggregationStrategy.TrimmedMean:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.TrimmedMeanFullModelAggregationStrategy<T, TInput, TOutput>(robust.TrimFraction);
            }

            case FederatedAggregationStrategy.WinsorizedMean:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.WinsorizedMeanFullModelAggregationStrategy<T, TInput, TOutput>(robust.TrimFraction);
            }

            case FederatedAggregationStrategy.Rfa:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return AiDotNet.FederatedLearning.Aggregators.RfaFullModelAggregationStrategy<T, TInput, TOutput>.FromOptions(robust);
            }

            case FederatedAggregationStrategy.Krum:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.KrumFullModelAggregationStrategy<T, TInput, TOutput>(robust.ByzantineClientCount);
            }

            case FederatedAggregationStrategy.MultiKrum:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.MultiKrumFullModelAggregationStrategy<T, TInput, TOutput>(
                    robust.ByzantineClientCount,
                    robust.MultiKrumSelectionCount,
                    robust.UseClientWeightsWhenAveragingSelectedUpdates);
            }

            case FederatedAggregationStrategy.Bulyan:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.BulyanFullModelAggregationStrategy<T, TInput, TOutput>(
                    robust.ByzantineClientCount,
                    robust.UseClientWeightsWhenAveragingSelectedUpdates);
            }

            default:
                throw new InvalidOperationException($"Unsupported federated aggregation strategy '{options.AggregationStrategy}'.");
        }
    }

    private static (IFullModel<T, TInput, TOutput> Model, IOptimizer<T, TInput, TOutput> Optimizer) CreateDistributedPair(
        IFullModel<T, TInput, TOutput> distributedModel,
        IOptimizer<T, TInput, TOutput> distributedOptimizer)
    {
        return (distributedModel, distributedOptimizer);
    }

    /// <summary>
    /// Computes deep ensemble members and attaches them to the result when deep ensemble uncertainty quantification is enabled.
    /// </summary>
    /// <param name="result">The prediction model result to update.</param>
    /// <param name="deepEnsembleTemplate">A template model used to create additional ensemble members.</param>
    /// <param name="optimizationInputData">Optimization/training data for the ensemble members.</param>
    /// <param name="templateOptimizer">The optimizer used for the main model, used as a template for ensemble member optimizers.</param>
    /// <param name="options">Uncertainty quantification configuration.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> A deep ensemble trains multiple similar models and combines their predictions. If the models disagree,
    /// it usually means the prediction is less certain.</para>
    /// <para>
    /// This method reuses the best solution from the primary optimization run (if available) and then trains additional members using:
    /// - Optional bootstrapping (sampling training rows with replacement)
    /// - Optional parameter perturbation (small Gaussian noise)
    /// </para>
    /// </remarks>
    private static void TryComputeAndAttachDeepEnsembleModels(
        PredictionModelResult<T, TInput, TOutput> result,
        IFullModel<T, TInput, TOutput>? deepEnsembleTemplate,
        OptimizationInputData<T, TInput, TOutput> optimizationInputData,
        IOptimizer<T, TInput, TOutput> templateOptimizer,
        UncertaintyQuantificationOptions? options)
    {
        if (options is not { Enabled: true, Method: UncertaintyQuantificationMethod.DeepEnsemble })
        {
            return;
        }

        if (deepEnsembleTemplate == null)
        {
            return;
        }

        var ensembleSize = Math.Max(2, options.DeepEnsembleSize);
        var members = new List<IFullModel<T, TInput, TOutput>>(capacity: ensembleSize);

        if (result.OptimizationResult.BestSolution != null)
        {
            members.Add(result.OptimizationResult.BestSolution);
        }

        var baseSeed = options.RandomSeed ?? Environment.TickCount;

        for (int memberIndex = members.Count; memberIndex < ensembleSize; memberIndex++)
        {
            var memberModel = deepEnsembleTemplate is Models.NeuralNetworkModel<T> nnTemplate
                ? (IFullModel<T, TInput, TOutput>)(object)new Models.NeuralNetworkModel<T>(nnTemplate.Architecture)
                : deepEnsembleTemplate.DeepCopy();

            PerturbInitialParametersIfSupported(memberModel, baseSeed, memberIndex, options.DeepEnsembleInitialNoiseStdDev);

            var memberOptimizer = CreateOptimizerForEnsembleMember(memberModel, templateOptimizer);
            memberOptimizer.Reset();

            var memberInputData = CreateDeepEnsembleMemberOptimizationInputData(optimizationInputData, baseSeed, memberIndex);
            var memberResult = memberOptimizer.Optimize(memberInputData);
            if (memberResult.BestSolution != null)
            {
                members.Add(memberResult.BestSolution);
            }
        }

        if (members.Count > 0)
        {
            result.SetDeepEnsembleModels(members);
        }
    }

    /// <summary>
    /// Creates per-member optimization input data for deep ensembles.
    /// </summary>
    /// <param name="baseInputData">The baseline input data.</param>
    /// <param name="baseSeed">Seed used to deterministically vary members.</param>
    /// <param name="memberIndex">The ensemble member index.</param>
    /// <returns>Optimization input data for the ensemble member.</returns>
    /// <remarks>
    /// <para>
    /// This optionally bootstraps training data to encourage diversity across members while keeping validation/test stable.
    /// </para>
    /// </remarks>
    private static OptimizationInputData<T, TInput, TOutput> CreateDeepEnsembleMemberOptimizationInputData(
        OptimizationInputData<T, TInput, TOutput> baseInputData,
        int baseSeed,
        int memberIndex)
    {
        var rng = RandomHelper.CreateSeededRandom(unchecked(baseSeed + (memberIndex + 1) * 10007));

        if (TryBootstrapTrainingData(baseInputData.XTrain, baseInputData.YTrain, rng, out var bootstrappedXTrain, out var bootstrappedYTrain))
        {
            return new OptimizationInputData<T, TInput, TOutput>
            {
                XTrain = bootstrappedXTrain,
                YTrain = bootstrappedYTrain,
                XValidation = baseInputData.XValidation,
                YValidation = baseInputData.YValidation,
                XTest = baseInputData.XTest,
                YTest = baseInputData.YTest
            };
        }

        return new OptimizationInputData<T, TInput, TOutput>
        {
            XTrain = baseInputData.XTrain,
            YTrain = baseInputData.YTrain,
            XValidation = baseInputData.XValidation,
            YValidation = baseInputData.YValidation,
            XTest = baseInputData.XTest,
            YTest = baseInputData.YTest
        };
    }

    /// <summary>
    /// Attempts to bootstrap the training data (sample with replacement) for deep-ensemble diversity.
    /// </summary>
    /// <param name="xTrain">Training inputs.</param>
    /// <param name="yTrain">Training targets.</param>
    /// <param name="rng">Random number generator.</param>
    /// <param name="bootstrappedXTrain">Bootstrapped inputs if supported.</param>
    /// <param name="bootstrappedYTrain">Bootstrapped targets if supported.</param>
    /// <returns>True if bootstrapping was applied; otherwise false.</returns>
    /// <remarks>
    /// <para>
    /// This method is best-effort and only supports common vector/matrix/tensor training data representations.
    /// </para>
    /// </remarks>
    private static bool TryBootstrapTrainingData(
        TInput xTrain,
        TOutput yTrain,
        Random rng,
        out TInput bootstrappedXTrain,
        out TOutput bootstrappedYTrain)
    {
        if (xTrain is Matrix<T> xTrainMatrix)
        {
            var sampleCount = xTrainMatrix.Rows;
            if (sampleCount <= 0)
            {
                bootstrappedXTrain = xTrain;
                bootstrappedYTrain = yTrain;
                return false;
            }

            var indices = new int[sampleCount];
            for (int i = 0; i < sampleCount; i++)
            {
                indices[i] = rng.Next(sampleCount);
            }

            var xBoot = xTrainMatrix.GetRows(indices);

            if (yTrain is Vector<T> yTrainVector && yTrainVector.Length == sampleCount)
            {
                var yBoot = yTrainVector.GetElements(indices);
                bootstrappedXTrain = (TInput)(object)xBoot;
                bootstrappedYTrain = (TOutput)(object)yBoot;
                return true;
            }

            if (yTrain is Matrix<T> yTrainMatrix && yTrainMatrix.Rows == sampleCount)
            {
                var yBoot = yTrainMatrix.GetRows(indices);
                bootstrappedXTrain = (TInput)(object)xBoot;
                bootstrappedYTrain = (TOutput)(object)yBoot;
                return true;
            }
        }

        if (xTrain is Tensor<T> xTrainTensor && xTrainTensor.Rank >= 2)
        {
            var sampleCount = xTrainTensor.Shape[0];
            if (sampleCount <= 0)
            {
                bootstrappedXTrain = xTrain;
                bootstrappedYTrain = yTrain;
                return false;
            }

            var indices = new int[sampleCount];
            for (int i = 0; i < sampleCount; i++)
            {
                indices[i] = rng.Next(sampleCount);
            }

            var xSlices = indices.Select(i => xTrainTensor.GetSlice(i)).ToArray();
            var xBoot = Tensor<T>.Stack(xSlices, axis: 0);

            if (yTrain is Tensor<T> yTrainTensor &&
                yTrainTensor.Rank >= 2 &&
                yTrainTensor.Shape[0] == sampleCount)
            {
                var ySlices = indices.Select(i => yTrainTensor.GetSlice(i)).ToArray();
                var yBoot = Tensor<T>.Stack(ySlices, axis: 0);
                bootstrappedXTrain = (TInput)(object)xBoot;
                bootstrappedYTrain = (TOutput)(object)yBoot;
                return true;
            }
        }

        bootstrappedXTrain = xTrain;
        bootstrappedYTrain = yTrain;
        return false;
    }

    /// <summary>
    /// Perturbs initial model parameters to avoid ensemble member collapse to identical solutions.
    /// </summary>
    /// <param name="model">The model to perturb.</param>
    /// <param name="baseSeed">Base seed for deterministic perturbation.</param>
    /// <param name="memberIndex">Ensemble member index.</param>
    /// <param name="noiseStdDev">Standard deviation of Gaussian noise to add.</param>
    /// <remarks>
    /// <para>
    /// The perturbation is only applied when the model supports parameter get/set via <see cref="IParameterizable{T,TInput,TOutput}"/>.
    /// </para>
    /// </remarks>
    private static void PerturbInitialParametersIfSupported(
        IFullModel<T, TInput, TOutput> model,
        int baseSeed,
        int memberIndex,
        double noiseStdDev)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (model is not IParameterizable<T, TInput, TOutput> parameterizable)
        {
            return;
        }

        if (noiseStdDev <= 0)
        {
            return;
        }

        var parameters = parameterizable.GetParameters();
        if (parameters.Length == 0)
        {
            return;
        }

        var rng = RandomHelper.CreateSeededRandom(unchecked(baseSeed + (memberIndex + 1) * 10007));
        var perturbed = new Vector<T>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            var noise = NextGaussian(rng, mean: 0.0, stdDev: noiseStdDev);
            perturbed[i] = numOps.Add(parameters[i], numOps.FromDouble(noise));
        }

        parameterizable.SetParameters(perturbed);
    }

    /// <summary>
    /// Generates a Gaussian random value using the Box-Muller transform.
    /// </summary>
    /// <param name="rng">Random source.</param>
    /// <param name="mean">Gaussian mean.</param>
    /// <param name="stdDev">Gaussian standard deviation.</param>
    /// <returns>A normally distributed random value.</returns>
    private static double NextGaussian(Random rng, double mean, double stdDev)
    {
        return rng.NextGaussian(mean, stdDev);
    }

    /// <summary>
    /// Creates an optimizer instance for an ensemble member based on the template optimizer type and options.
    /// </summary>
    /// <param name="model">The ensemble member model.</param>
    /// <param name="templateOptimizer">The optimizer used as a template.</param>
    /// <returns>An optimizer instance that targets <paramref name="model"/>.</returns>
    /// <remarks>
    /// <para>
    /// This is best-effort and supports common optimizer constructors (model + options, or model only).
    /// </para>
    /// </remarks>
    private static IOptimizer<T, TInput, TOutput> CreateOptimizerForEnsembleMember(
        IFullModel<T, TInput, TOutput> model,
        IOptimizer<T, TInput, TOutput> templateOptimizer)
    {
        var optimizerType = templateOptimizer.GetType();
        var options = templateOptimizer.GetOptions();

        foreach (var ctor in optimizerType.GetConstructors())
        {
            var parameters = ctor.GetParameters();
            if (parameters.Length == 2 &&
                parameters[0].ParameterType.IsInstanceOfType(model) &&
                parameters[1].ParameterType.IsInstanceOfType(options))
            {
                return (IOptimizer<T, TInput, TOutput>)ctor.Invoke([model, options]);
            }

            if (parameters.Length == 1 &&
                parameters[0].ParameterType.IsInstanceOfType(model))
            {
                return (IOptimizer<T, TInput, TOutput>)ctor.Invoke([model]);
            }
        }

        if (templateOptimizer is NormalOptimizer<T, TInput, TOutput>)
        {
            return new NormalOptimizer<T, TInput, TOutput>(model);
        }

        throw new InvalidOperationException(
            $"Unable to construct a deep ensemble optimizer of type '{optimizerType.FullName}'. " +
            $"Expected a constructor with signature ({typeof(IFullModel<T, TInput, TOutput>).Name}, {options?.GetType().Name ?? "null"}) or ({typeof(IFullModel<T, TInput, TOutput>).Name}).");
    }

    private static void ApplyUncertaintyQuantificationIfConfigured(
        IFullModel<T, TInput, TOutput>? model,
        UncertaintyQuantificationOptions? options)
    {
        if (model == null)
        {
            return;
        }

        if (options is not { Enabled: true })
        {
            return;
        }

        var method = options.Method == UncertaintyQuantificationMethod.Auto
            ? UncertaintyQuantificationMethod.MonteCarloDropout
            : options.Method;

        if (method != UncertaintyQuantificationMethod.MonteCarloDropout)
        {
            return;
        }

        if (model is not Models.NeuralNetworkModel<T> neuralNetworkModel)
        {
            throw new InvalidOperationException(
                "Uncertainty quantification is currently supported for neural network models only. " +
                "ConfigureModel(new NeuralNetworkModel<T>(...)) to enable Monte Carlo Dropout uncertainty estimation.");
        }

        var injectedCount = TryInjectMonteCarloDropoutLayers(neuralNetworkModel, options);
        if (injectedCount == 0)
        {
            System.Diagnostics.Debug.WriteLine(
                "Warning: Monte Carlo Dropout was enabled but no dropout layers were injected automatically. " +
                "This can happen if the network has no suitable activation layers or uses a non-standard architecture. " +
                "Consider inserting MCDropoutLayer explicitly in your network definition.");
        }
    }

    private static int TryInjectMonteCarloDropoutLayers(
        Models.NeuralNetworkModel<T> neuralNetworkModel,
        UncertaintyQuantificationOptions options)
    {
        var layers = neuralNetworkModel.Network.LayersReadOnly;
        if (layers.OfType<MCDropoutLayer<T>>().Any())
        {
            return -1;
        }

        if (options.MonteCarloDropoutRate <= 0 || options.MonteCarloDropoutRate >= 1)
        {
            throw new ArgumentException("MonteCarloDropoutRate must be between 0 and 1.", nameof(options));
        }

        var injectedCount = 0;
        for (int i = 0; i < layers.Count - 1; i++)
        {
            if (layers[i] is not ActivationLayer<T>)
            {
                continue;
            }

            if (i >= layers.Count - 2)
            {
                continue;
            }

            if (layers[i + 1] is DropoutLayer<T> || layers[i + 1] is MCDropoutLayer<T>)
            {
                continue;
            }

            int? seed = options.RandomSeed.HasValue ? options.RandomSeed.Value + i : (int?)null;
            neuralNetworkModel.Network.InsertLayerIntoCollection(i + 1, new MCDropoutLayer<T>(options.MonteCarloDropoutRate, mcMode: false, randomSeed: seed));
            injectedCount++;
            i++;
        }

        return injectedCount;
    }

    private IChatModel<T> CreateChatModel(AgentConfiguration<T> config)
    {
        var apiKey = AgentKeyResolver.ResolveApiKey(
            config.ApiKey,
            config,
            config.Provider);

        return config.Provider switch
        {
            LLMProvider.OpenAI => new OpenAIChatModel<T>(apiKey),
            LLMProvider.Anthropic => new AnthropicChatModel<T>(apiKey),
            LLMProvider.AzureOpenAI => new AzureOpenAIChatModel<T>(
                config.AzureEndpoint ?? throw new InvalidOperationException("Azure endpoint required"),
                apiKey,
                config.AzureDeployment ?? "gpt-4"),
            _ => throw new ArgumentException($"Unknown provider: {config.Provider}")
        };
    }
}
