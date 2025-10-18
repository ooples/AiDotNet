global using AiDotNet.FeatureSelectors;
global using AiDotNet.FitnessCalculators;
global using AiDotNet.Regularization;
global using AiDotNet.Optimizers;
global using AiDotNet.Normalizers;
global using AiDotNet.OutlierRemoval;
global using AiDotNet.DataProcessor;
global using AiDotNet.FitDetectors;

namespace AiDotNet;

using System;
using AiDotNet.Logging;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using AiDotNet.OnlineLearning;
using AiDotNet.OnlineLearning.Algorithms;
using AiDotNet.AutoML;
using AiDotNet.Interfaces;
using AiDotNet.ModelSelection;
using AiDotNet.Caching;
using AiDotNet.Factories;

/// <summary>
/// A builder class that helps create and configure machine learning prediction models.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
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
/// </remarks>
public class PredictionModelBuilder<T, TInput, TOutput> : IPredictionModelBuilder<T, TInput, TOutput>
{
    private IFeatureSelector<T, TInput>? _featureSelector;
    private INormalizer<T, TInput, TOutput>? _normalizer;
    private IFullModel<T, TInput, TOutput>? _model;
    private IOptimizer<T, TInput, TOutput>? _optimizer;
    private IDataPreprocessor<T, TInput, TOutput>? _dataPreprocessor;
    private IOutlierRemoval<T, TInput, TOutput>? _outlierRemoval;
    private IModelSelector<T, TInput, TOutput>? _modelSelector;
    private LoggingOptions? _loggingOptions;
    
    // Ensemble-specific fields
    private IEnsembleModel<T, TInput, TOutput>? _ensembleModel;
    private List<IFullModel<T, TInput, TOutput>>? _pendingEnsembleModels;
    private EnsembleOptions<T>? _ensembleOptions;
    
    // Online learning specific fields
    private OnlineModelOptions<T>? _onlineOptions;
    private AdaptiveOnlineModelOptions<T>? _adaptiveOnlineOptions;
    
    // Modern AI fields
    private IMultimodalModel<T, TInput, TOutput>? _multimodalModel;
    private readonly Dictionary<ModalityType, IPipelineStep<T, object, object>> _modalityPreprocessors = new();
    private ModalityFusionStrategy _modalityFusionStrategy = ModalityFusionStrategy.LateFusion;
    private IFoundationModel<T, TInput, TOutput>? _foundationModel;
    private FineTuningOptions<double>? _fineTuningOptions;
    private List<(TInput input, TOutput output)>? _fewShotExamples;
    private IAutoMLModel<T, TInput, TOutput>? _autoMLModel;
    private HyperparameterSearchSpace? _searchSpace;
    private TimeSpan? _autoMLTimeLimit;
    private int? _autoMLTrialLimit;
    private NeuralArchitectureSearchStrategy? _nasStrategy;
    private IInterpretableModel<T, TInput, TOutput>? _interpretableModel;
    private List<InterpretationMethod> _interpretationMethods = new();
    private int[]? _sensitiveFeatures;
    private List<FairnessMetric> _fairnessMetrics = new();
    private IProductionMonitor<T>? _productionMonitor;
    private T? _dataDriftThreshold;
    private T? _conceptDriftThreshold;
    private T? _performanceDropThreshold;
    private TimeSpan? _retrainingInterval;
    private readonly Dictionary<PipelinePosition, List<IPipelineStep<T>>> _pipelineSteps = new();
    private readonly Dictionary<string, PredictionModelBuilder<T, TInput, TOutput>> _branches = new();
    private CloudPlatform? _cloudPlatform;
    private OptimizationLevel _optimizationLevel = OptimizationLevel.Balanced;
    private EdgeDevice? _edgeDevice;
    private int? _memoryLimit;
    private int? _latencyTarget;
    private FederatedAggregationStrategy? _federatedStrategy;
    private T? _privacyBudget;
    private Enums.MetaLearningAlgorithm? _metaLearningAlgorithm;
    private int _innerLoopSteps = 5;
    
    // Foundation model specific fields
    private string? _promptTemplate = null;
    private List<string>? _featureNames = null;
    private int? _generationMaxTokens = null;
    private double? _generationTemperature = null;
    private double? _generationTopP = null;
    private int? _maxConcurrency = null;
    private List<TrainingExample>? _trainingData = null;
    private List<TrainingExample>? _validationData = null;

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
        _logger.Debug("Feature selector configured: {FeatureSelectorType}", selector.GetType().Name);
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
        _logger.Debug("Normalizer configured: {NormalizerType}", normalizer.GetType().Name);
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
        _logger.Debug("Optimizer configured: {OptimizerType}", optimizationAlgorithm.GetType().Name);
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
        _logger.Debug("Data preprocessor configured: {PreprocessorType}", dataPreprocessor.GetType().Name);
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
        _logger.Debug("Outlier removal configured: {OutlierRemovalType}", outlierRemoval.GetType().Name);
        return this;
    }

    /// <summary>
    /// Configures a model selector for generating model recommendations.
    /// </summary>
    /// <param name="modelSelector">The model selector implementation to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This configures the component that analyzes your data and recommends
    /// appropriate models. The model selector doesn't change the model you've already chosen
    /// when creating this builder, but it allows you to:
    /// 
    /// 1. Get recommendations about which models might work well for your data
    /// 2. Compare your chosen model against what would be automatically selected
    /// 
    /// This is useful when you want to explore different options or validate your model choice.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureModelSelector(IModelSelector<T, TInput, TOutput> modelSelector)
    {
        _modelSelector = modelSelector;
        _logger.Debug("Model selector configured: {ModelSelectorType}", modelSelector.GetType().Name);
        return this;
    }

    /// <summary>
    /// Configures logging for the model building and prediction process.
    /// </summary>
    /// <param name="options">Options that control logging behavior.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you control how detailed the logs should be and where they should be saved.
    /// Logs help you understand what's happening inside the AI model and can help diagnose problems.
    /// </para>
    /// <para>
    /// More detailed logs (like Debug level) provide deep insights but create larger files.
    /// In production, you typically use Information level or higher to capture important events
    /// without excessive detail.
    /// </para>
    /// <para>
    /// If you need help from technical support, you can share these log files to help them
    /// understand exactly what's happening in your application.
    /// </para>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureLogging(LoggingOptions options)
    {
        _loggingOptions = options;
        LoggingFactory.Configure(options);
        _logger.Information("Logging configured with minimum level: {LogLevel}, enabled: {IsEnabled}",
            options.MinimumLevel, options.IsEnabled);
        return this;
    }

    /// <summary>
    /// Analyzes the input and output data and provides recommended models for the task.
    /// </summary>
    /// <param name="sampleX">A sample of the input data to analyze its structure.</param>
    /// <param name="sampleY">A sample of the output data to analyze its structure.</param>
    /// <returns>A ranked list of recommended model types with brief explanations.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method analyzes your data and gives you recommendations about
    /// which models might work well for your specific task. It doesn't change the model you've 
    /// already chosen, but gives you insights into alternatives you might want to consider.
    /// 
    /// Each recommendation includes:
    /// - The type of model recommended
    /// - A confidence score for how well it might perform
    /// - An explanation of why this model might be appropriate
    /// - Potential advantages and disadvantages
    /// 
    /// This can help you understand the rationale behind model selection and possibly
    /// discover more effective approaches for your specific data.
    /// </remarks>
    public List<ModelRecommendation<T, TInput, TOutput>> GetModelRecommendations(TInput sampleX, TOutput sampleY)
    {
        _logger.Information("Getting model recommendations for provided data samples");
        _modelSelector = _modelSelector ?? new DefaultModelSelector<T, TInput, TOutput>();
        try
        {
            var recommendations = _modelSelector.GetModelRecommendations(sampleX, sampleY);
            _logger.Information("Retrieved {Count} model recommendations", recommendations.Count);

            if (_logger.IsEnabled(LoggingLevel.Debug))
            {
                foreach (var recommendation in recommendations)
                {
                    _logger.Debug("Model recommendation: {ModelName}, confidence: {ConfidenceScore}",
                        recommendation.ModelName, recommendation.ConfidenceScore);
                        
                    if (_logger.IsEnabled(LoggingLevel.Trace))
                    {
                        _logger.Trace("Recommendation explanation: {Explanation}", recommendation.Explanation);
                    }
                }
            }

            return recommendations;
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Error getting model recommendations");
            throw;
        }
    }

    #region Ensemble Methods
    
    /// <summary>
    /// Configures the builder to use a specific ensemble model type.
    /// </summary>
    /// <typeparam name="TEnsemble">The type of ensemble model to use.</typeparam>
    /// <param name="options">Optional configuration for the ensemble.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This creates an ensemble model of the specified type. For example:
    /// UseEnsemble&lt;VotingEnsemble&lt;float&gt;&gt;() creates a voting ensemble.
    /// 
    /// The ensemble will use models added with WithModels() or AddToEnsemble().
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> UseEnsemble<TEnsemble>(
        EnsembleOptions<T>? options = null) 
        where TEnsemble : IEnsembleModel<T, TInput, TOutput>, new()
    {
        _ensembleModel = new TEnsemble();
        _ensembleOptions = options;
        _model = null; // Clear single model
        _logger.Information("Configured to use {EnsembleType} ensemble", typeof(TEnsemble).Name);
        return this;
    }
    
    /// <summary>
    /// Adds models to be included in an ensemble.
    /// </summary>
    /// <param name="models">The models to add to the ensemble.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Use this to specify which models should be part of your ensemble.
    /// You can mix different types of models (neural networks, regression, etc.) as long as
    /// they all work with Tensor<double> inputs/outputs.
    /// 
    /// Example:
    /// .WithModels(
    ///     new NeuralNetworkModel&lt;float&gt;(),
    ///     new RandomForestModel&lt;float&gt;(),
    ///     new GradientBoostingModel&lt;float&gt;()
    /// )
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> WithModels(
        params IFullModel<T, TInput, TOutput>[] models)
    {
        _pendingEnsembleModels ??= new List<IFullModel<T, TInput, TOutput>>();
        _pendingEnsembleModels.AddRange(models);
        _logger.Debug("Added {Count} models to ensemble", models.Length);
        return this;
    }
    
    /// <summary>
    /// Adds a single model to the ensemble being built.
    /// </summary>
    /// <param name="model">The model to add.</param>
    /// <param name="weight">Optional initial weight for this model.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Use this to add models one at a time to your ensemble.
    /// The weight determines how much influence this model has (higher = more influence).
    /// If you don't specify a weight, a default will be used.
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> AddToEnsemble(
        IFullModel<T, TInput, TOutput> model,
        T? weight = default)
    {
        _pendingEnsembleModels ??= new List<IFullModel<T, TInput, TOutput>>();
        _pendingEnsembleModels.Add(model);
        
        if (weight != null && !weight.Equals(default(T)))
        {
            _logger.Debug("Added {ModelType} to ensemble with weight {Weight}", 
                model.GetType().Name, weight);
        }
        else
        {
            _logger.Debug("Added {ModelType} to ensemble", model.GetType().Name);
        }
        
        return this;
    }
    
    /// <summary>
    /// Creates an auto-ensemble with diverse model types.
    /// </summary>
    /// <param name="modelCount">Number of models to include.</param>
    /// <param name="strategy">The ensemble combination strategy.</param>
    /// <param name="includeDifferentCategories">Whether to include models from different categories.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This automatically creates an ensemble by selecting diverse models
    /// that are likely to work well together. It's a good starting point if you're not sure
    /// which models to combine.
    /// 
    /// The builder will analyze your data and choose appropriate models from different
    /// categories (if requested) to maximize diversity and potential performance.
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> UseAutoEnsemble(
        int modelCount = 5,
        EnsembleStrategy strategy = EnsembleStrategy.WeightedAverage,
        bool includeDifferentCategories = true)
    {
        _ensembleOptions = new EnsembleOptions<T>
        {
            MaxModels = modelCount,
            Strategy = strategy,
            AllowDuplicateModelTypes = false
        };
        
        // Create appropriate ensemble based on strategy
        _ensembleModel = strategy switch
        {
            EnsembleStrategy.Average or 
            EnsembleStrategy.WeightedAverage or
            EnsembleStrategy.MajorityVote or
            EnsembleStrategy.WeightedVote or
            EnsembleStrategy.SoftVote => new VotingEnsemble<T, TInput, TOutput>(new VotingEnsembleOptions<T> 
            { 
                Strategy = strategy,
                MaxModels = modelCount,
                VotingType = strategy == EnsembleStrategy.MajorityVote ? VotingType.Hard :
                            strategy == EnsembleStrategy.SoftVote ? VotingType.Soft :
                            VotingType.Weighted
            }),
            
            // Add more ensemble types as they are implemented
            _ => new VotingEnsemble<T, TInput, TOutput>(new VotingEnsembleOptions<T> 
            { 
                Strategy = strategy,
                MaxModels = modelCount 
            })
        };
        
        // Flag for auto-selection during build
        _pendingEnsembleModels = new List<IFullModel<T, TInput, TOutput>>();
        
        _logger.Information("Configured auto-ensemble with {ModelCount} models using {Strategy} strategy", 
            modelCount, strategy);
        
        return this;
    }
    
    /// <summary>
    /// Configures ensemble-specific options.
    /// </summary>
    /// <param name="options">The ensemble options to apply.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Use this to fine-tune how your ensemble works, such as:
    /// - Training strategy (parallel, sequential, bagging, boosting)
    /// - Weight update methods
    /// - Performance thresholds
    /// - Parallelization settings
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> ConfigureEnsemble(
        EnsembleOptions<T> options)
    {
        _ensembleOptions = options;
        _logger.Debug("Configured ensemble options");
        return this;
    }
    
    #endregion
    
    #region Online Learning Methods
    
    /// <summary>
    /// Configures the builder to use an online learning model.
    /// </summary>
    /// <typeparam name="TOnlineModel">The type of online model to use.</typeparam>
    /// <param name="options">Optional configuration for the online model.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Online learning models update incrementally as new data arrives,
    /// perfect for streaming data or when you can't fit all data in memory. Examples:
    /// - UseOnlineModel&lt;OnlinePerceptron&lt;float&gt;&gt;() for simple classification
    /// - UseOnlineModel&lt;OnlineSGDRegressor&lt;float&gt;&gt;() for regression
    /// - UseOnlineModel&lt;PassiveAggressiveRegressor&lt;float&gt;&gt;() for robust regression
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> UseOnlineModel<TOnlineModel>(
        OnlineModelOptions<T>? options = null)
        where TOnlineModel : IOnlineModel<T, TInput, TOutput>, new()
    {
        _model = new TOnlineModel();
        _onlineOptions = options;
        _logger.Information("Configured to use {OnlineModelType} for online learning", typeof(TOnlineModel).Name);
        return this;
    }
    
    /// <summary>
    /// Configures the builder to use an adaptive online learning model with drift detection.
    /// </summary>
    /// <typeparam name="TAdaptiveModel">The type of adaptive online model to use.</typeparam>
    /// <param name="options">Optional configuration for the adaptive model.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Adaptive online models can detect when data patterns change
    /// (concept drift) and adjust automatically. Use these when:
    /// - Your data patterns might change over time (user preferences, market conditions)
    /// - You need the model to stay accurate as the world changes
    /// - You want automatic adaptation without manual intervention
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> UseAdaptiveOnlineModel<TAdaptiveModel>(
        AdaptiveOnlineModelOptions<T>? options = null)
        where TAdaptiveModel : IAdaptiveOnlineModel<T, TInput, TOutput>, new()
    {
        _model = new TAdaptiveModel();
        _adaptiveOnlineOptions = options;
        _logger.Information("Configured to use {AdaptiveModelType} with drift detection", typeof(TAdaptiveModel).Name);
        return this;
    }
    
    /// <summary>
    /// Creates an online learning model based on the algorithm type.
    /// </summary>
    /// <param name="algorithm">The online learning algorithm to use.</param>
    /// <param name="inputDimension">The number of input features.</param>
    /// <param name="options">Optional configuration for the online model.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This lets you choose an online algorithm by name rather than type.
    /// Common choices:
    /// - OnlineLearningAlgorithm.Perceptron - Simple and fast for classification
    /// - OnlineLearningAlgorithm.PassiveAggressive - Good for regression with outliers
    /// - OnlineLearningAlgorithm.StochasticGradientDescent - Versatile for many problems
    /// - OnlineLearningAlgorithm.AdaptiveRandomForest - Powerful ensemble for changing data
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> UseOnlineAlgorithm(
        OnlineLearningAlgorithm algorithm,
        int inputDimension,
        OnlineModelOptions<T>? options = null)
    {
        _onlineOptions = options;
        
        // Create the appropriate model based on the algorithm
        // Note: This is a simplified example. In practice, you'd need to handle
        // the specific TInput/TOutput types for each algorithm
        _logger.Information("Creating online model for algorithm: {Algorithm}", algorithm);
        
        // Store the algorithm type for later instantiation during Build
        // This allows us to defer model creation until we know the data types
        _logger.Debug("Online algorithm {Algorithm} configured with {InputDim} input dimensions", 
            algorithm, inputDimension);
        
        return this;
    }
    
    /// <summary>
    /// Configures online learning specific options.
    /// </summary>
    /// <param name="options">The online learning options to apply.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Use this to fine-tune online learning behavior:
    /// - Learning rate: How quickly the model adapts (higher = faster but less stable)
    /// - Mini-batch size: How many examples to process together
    /// - Regularization: Prevents overfitting to recent examples
    /// - Stream buffer settings: For efficient streaming data processing
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> ConfigureOnlineOptions(
        OnlineModelOptions<T> options)
    {
        _onlineOptions = options;
        _logger.Debug("Configured online learning options");
        return this;
    }
    
    /// <summary>
    /// Enables adaptive learning with drift detection.
    /// </summary>
    /// <param name="driftMethod">The drift detection method to use.</param>
    /// <param name="sensitivity">Drift sensitivity (0-1, higher = more sensitive).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Drift detection helps your model notice when patterns change.
    /// Common drift detection methods:
    /// - ADWIN: Adaptive sliding window, good general purpose
    /// - DDM: Monitors error rate changes
    /// - PageHinkley: Sequential change detection
    /// - None: No drift detection (standard online learning)
    /// 
    /// Higher sensitivity means the model reacts to smaller changes.
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> EnableDriftDetection(
        DriftDetectionMethod driftMethod = DriftDetectionMethod.ADWIN)
    {
        _adaptiveOnlineOptions = new AdaptiveOnlineModelOptions<T>
        {
            DriftDetectionMethod = driftMethod,
            DriftSensitivity = NumOps.FromDouble(0.5),
            DriftWindowSize = 100,
            ResetOnDrift = false,
            DriftLearningRateBoost = NumOps.FromDouble(2.0)
        };
        
        _logger.Information("Enabled drift detection using {Method} with default sensitivity", 
            driftMethod);
        
        return this;
    }
    
    /// <summary>
    /// Enables adaptive learning with drift detection and custom sensitivity.
    /// </summary>
    /// <param name="driftMethod">The drift detection method to use.</param>
    /// <param name="sensitivity">Drift sensitivity (0-1, higher = more sensitive).</param>
    /// <returns>This builder instance for method chaining.</returns>
    public PredictionModelBuilder<T, TInput, TOutput> EnableDriftDetection(
        DriftDetectionMethod driftMethod,
        T sensitivity)
    {
        _adaptiveOnlineOptions = new AdaptiveOnlineModelOptions<T>
        {
            DriftDetectionMethod = driftMethod,
            DriftSensitivity = sensitivity,
            DriftWindowSize = 100,
            ResetOnDrift = false,
            DriftLearningRateBoost = NumOps.FromDouble(2.0)
        };
        
        _logger.Information("Enabled drift detection using {Method} with sensitivity {Sensitivity}", 
            driftMethod, Convert.ToDouble(sensitivity));
        
        return this;
    }
    
    #endregion
    
    #region Modern AI Methods Implementation
    
    /// <summary>
    /// Configures the builder to use a multimodal model that can process multiple data types.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> UseMultimodalModel(IMultimodalModel<T, TInput, TOutput> multimodalModel)
    {
        _multimodalModel = multimodalModel;
        _logger.Information("Configured multimodal model: {ModelType}", multimodalModel.GetType().Name);
        return this;
    }
    
    /// <summary>
    /// Adds a data modality (type) to the multimodal model.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> AddModality(
        ModalityType modalityType,
        IPipelineStep<T>? preprocessor = null)
    {
        if (preprocessor != null)
        {
            _modalityPreprocessors[modalityType] = preprocessor;
        }
        _logger.Debug("Added modality: {ModalityType}", modalityType);
        return this;
    }
    
    /// <summary>
    /// Configures how different modalities are combined.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureModalityFusion(ModalityFusionStrategy fusionStrategy)
    {
        _modalityFusionStrategy = fusionStrategy;
        _logger.Debug("Configured modality fusion strategy: {Strategy}", fusionStrategy);
        return this;
    }
    
    /// <summary>
    /// Uses a foundation model (like GPT, BERT, etc.) as the base model.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> UseFoundationModel(IFoundationModel<T, TInput, TOutput> foundationModel)
    {
        _foundationModel = foundationModel;
        _logger.Information("Configured foundation model: {ModelType}", foundationModel.GetType().Name);
        return this;
    }
    
    /// <summary>
    /// Configures fine-tuning for a foundation model.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFineTuning(FineTuningOptions<double> fineTuningOptions)
    {
        _fineTuningOptions = fineTuningOptions;
        _logger.Debug("Configured fine-tuning options");
        return this;
    }
    
    /// <summary>
    /// Enables few-shot learning with example prompts.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> WithFewShotExamples(params (TInput input, TOutput output)[] examples)
    {
        _fewShotExamples ??= new List<(TInput input, TOutput output)>();
        _fewShotExamples.AddRange(examples);
        _logger.Debug("Added {Count} few-shot examples", examples.Length);
        return this;
    }
    
    /// <summary>
    /// Enables AutoML to automatically find the best model and hyperparameters.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> EnableAutoML(IAutoMLModel<T, TInput, TOutput> autoMLModel)
    {
        _autoMLModel = autoMLModel;
        _logger.Information("Enabled AutoML with model: {ModelType}", autoMLModel.GetType().Name);
        return this;
    }
    
    /// <summary>
    /// Configures AutoML search space and constraints.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureAutoMLSearch(
        HyperparameterSearchSpace searchSpace,
        TimeSpan? timeLimit = null,
        int? trialLimit = null)
    {
        _searchSpace = searchSpace;
        _autoMLTimeLimit = timeLimit;
        _autoMLTrialLimit = trialLimit;
        _logger?.Debug("Configured AutoML search with time limit: {TimeLimit}, trial limit: {TrialLimit}",
            timeLimit?.TotalMinutes ?? -1, trialLimit ?? -1);
        return this;
    }
    
    /// <summary>
    /// Enables neural architecture search (NAS).
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> EnableNeuralArchitectureSearch(
        NeuralArchitectureSearchStrategy searchStrategy)
    {
        _nasStrategy = searchStrategy;
        _logger.Information("Enabled neural architecture search with strategy: {Strategy}", searchStrategy);
        return this;
    }
    
    /// <summary>
    /// Adds interpretability features to the model.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> WithInterpretability(
        IInterpretableModel<T, TInput, TOutput> interpretableModel)
    {
        _interpretableModel = interpretableModel;
        _logger.Information("Added interpretability with model: {ModelType}", interpretableModel.GetType().Name);
        return this;
    }
    
    /// <summary>
    /// Enables specific interpretation methods.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> EnableInterpretationMethods(params InterpretationMethod[] methods)
    {
        _interpretationMethods.AddRange(methods);
        _logger.Debug("Enabled interpretation methods: {Methods}", string.Join(", ", methods));
        return this;
    }
    
    /// <summary>
    /// Configures fairness constraints and monitoring.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureFairness(
        int[] sensitiveFeatures,
        params FairnessMetric[] fairnessMetrics)
    {
        _sensitiveFeatures = sensitiveFeatures;
        _fairnessMetrics.AddRange(fairnessMetrics);
        _logger.Debug("Configured fairness monitoring for {Count} sensitive features", sensitiveFeatures.Length);
        return this;
    }
    
    /// <summary>
    /// Adds production monitoring capabilities to the model.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> WithProductionMonitoring(
        IProductionMonitor<T> monitor)
    {
        _productionMonitor = monitor;
        _logger.Information("Added production monitoring: {MonitorType}", monitor.GetType().Name);
        return this;
    }
    
    /// <summary>
    /// Configures drift detection for production monitoring.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureDriftDetection(
        T dataDriftThreshold,
        T conceptDriftThreshold)
    {
        _dataDriftThreshold = dataDriftThreshold;
        _conceptDriftThreshold = conceptDriftThreshold;
        _logger.Debug("Configured drift detection thresholds");
        return this;
    }
    
    /// <summary>
    /// Sets up automatic retraining triggers.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureAutoRetraining(
        T performanceDropThreshold,
        TimeSpan? timeBasedRetraining = null)
    {
        _performanceDropThreshold = performanceDropThreshold;
        _retrainingInterval = timeBasedRetraining;
        _logger?.Debug("Configured auto-retraining with performance threshold and interval: {Interval}",
            timeBasedRetraining?.TotalDays ?? -1);
        return this;
    }
    
    /// <summary>
    /// Adds a custom pipeline step to the model building process.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> AddPipelineStep(
        IPipelineStep<T> step,
        PipelinePosition position = PipelinePosition.Preprocessing)
    {
        if (!_pipelineSteps.ContainsKey(position))
        {
            _pipelineSteps[position] = new List<IPipelineStep<T>>();
        }
        _pipelineSteps[position].Add(step);
        _logger.Debug("Added pipeline step at position: {Position}", position);
        return this;
    }
    
    /// <summary>
    /// Creates a branching pipeline for A/B testing or ensemble approaches.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> CreateBranch(
        string branchName,
        Action<IPredictionModelBuilder<T, TInput, TOutput>> branchBuilder)
    {
        var branch = new PredictionModelBuilder<T, TInput, TOutput>();
        branchBuilder(branch);
        _branches[branchName] = branch;
        _logger.Debug("Created pipeline branch: {BranchName}", branchName);
        return this;
    }
    
    /// <summary>
    /// Merges multiple pipeline branches.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> MergeBranches(
        BranchMergeStrategy mergeStrategy,
        params string[] branchNames)
    {
        _logger.Debug("Merging branches {Branches} with strategy: {Strategy}",
            string.Join(", ", branchNames), mergeStrategy);
        // Implementation would merge the specified branches according to the strategy
        return this;
    }
    
    /// <summary>
    /// Optimizes the model for cloud deployment.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> OptimizeForCloud(
        CloudPlatform cloudPlatform,
        OptimizationLevel optimizationLevel = OptimizationLevel.Balanced)
    {
        _cloudPlatform = cloudPlatform;
        _optimizationLevel = optimizationLevel;
        _logger.Information("Optimizing for cloud platform: {Platform} with level: {Level}",
            cloudPlatform, optimizationLevel);
        return this;
    }
    
    /// <summary>
    /// Optimizes the model for edge deployment.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> OptimizeForEdge(
        EdgeDevice edgeDevice,
        int? memoryLimit = null,
        int? latencyTarget = null)
    {
        _edgeDevice = edgeDevice;
        _memoryLimit = memoryLimit;
        _latencyTarget = latencyTarget;
        _logger?.Information("Optimizing for edge device: {Device} with memory limit: {Memory}MB, latency target: {Latency}ms",
            edgeDevice.ToString(), memoryLimit ?? -1, latencyTarget ?? -1);
        return this;
    }
    
    /// <summary>
    /// Enables federated learning capabilities.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> EnableFederatedLearning(
        FederatedAggregationStrategy aggregationStrategy,
        T? privacyBudget = default)
    {
        _federatedStrategy = aggregationStrategy;
        _privacyBudget = privacyBudget;
        _logger.Information("Enabled federated learning with strategy: {Strategy}", aggregationStrategy);
        return this;
    }
    
    /// <summary>
    /// Configures meta-learning for quick adaptation to new tasks.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureMetaLearning(
        Enums.MetaLearningAlgorithm metaLearningAlgorithm,
        int innerLoopSteps = 5)
    {
        _metaLearningAlgorithm = metaLearningAlgorithm;
        _innerLoopSteps = innerLoopSteps;
        _logger.Information("Configured meta-learning with algorithm: {Algorithm} and {Steps} inner loop steps",
            metaLearningAlgorithm, innerLoopSteps);
        return this;
    }
    
    #endregion
    
    /// <summary>
    /// Builds a predictive model using the provided input features and output values.
    /// </summary>
    /// <param name="x">The matrix of input features where each row is a data point and each column is a feature.</param>
    /// <param name="y">The vector of output values corresponding to each row in the input matrix.</param>
    /// <returns>A trained predictive model that can be used to make predictions.</returns>
    /// <exception cref="ArgumentNullException">Thrown when input features or output values are null.</exception>
    /// <exception cref="ArgumentException">Thrown when the number of rows in the features matrix doesn't match the length of the output vector.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes your data (inputs and known outputs) and creates a trained AI model.
    /// Think of it like teaching a student: you provide examples (your data) and the student (the model) learns
    /// patterns from these examples. After building, your model is ready to make predictions on new data.
    /// 
    /// The input matrix 'x' contains your features (like house size, number of bedrooms, etc. if predicting house prices),
    /// and the vector 'y' contains the known answers (actual house prices) for those examples.
    /// </remarks>
    public IPredictiveModel<T, TInput, TOutput> Build(TInput x, TOutput y)
    {
        _logger.Information("Starting model build process");

        try
        {
            // Apply logging configuration if it was set
            if (_loggingOptions != null)
            {
                LoggingFactory.Configure(_loggingOptions);
            }

        // Use defaults for these interfaces if they aren't set
        var normalizer = _normalizer ?? new NoNormalizer<T, TInput, TOutput>();
        var optimizer = _optimizer ?? new NormalOptimizer<T, TInput, TOutput>();
        var featureSelector = _featureSelector ?? new NoFeatureSelector<T, TInput>();
        var outlierRemoval = _outlierRemoval ?? new NoOutlierRemoval<T, TInput, TOutput>();
        var dataPreprocessor = _dataPreprocessor ?? new DefaultDataPreprocessor<T, TInput, TOutput>(normalizer, featureSelector, outlierRemoval);

            // Log input and output information if debug is enabled
            if (_logger.IsEnabled(LoggingLevel.Debug))
            {
                _logger.Debug("Input shape: {InputShape}, Output shape: {OutputShape}",
                    InputHelper<T, TInput>.GetInputSize(x),
                    InputHelper<T, TInput>.GetBatchSize(y));
            }

            // Use defaults for these interfaces if they aren't set
            IFullModel<T, TInput, TOutput> model;
            
            // Check if we're building an ensemble
            if (_ensembleModel != null)
            {
                _logger.Information("Building ensemble model");
                
                // Handle ensemble model creation
                if (_pendingEnsembleModels != null && _pendingEnsembleModels.Count > 0)
                {
                    // Add manually specified models to the ensemble
                    _logger.Debug("Adding {Count} manually specified models to ensemble", _pendingEnsembleModels.Count);
                    
                    var defaultWeight = NumOps.One;
                    foreach (var ensembleModel in _pendingEnsembleModels)
                    {
                        _ensembleModel.AddModel(ensembleModel, defaultWeight);
                    }
                }
                else if (_ensembleOptions != null && _ensembleOptions.MaxModels > 0)
                {
                    // Auto-create ensemble with diverse models
                    _logger.Information("Auto-creating ensemble with {MaxModels} models", _ensembleOptions.MaxModels);
                    
                    var modelSelector = _modelSelector ?? new DefaultModelSelector<T, TInput, TOutput>();
                    var recommendations = modelSelector.GetModelRecommendations(x, y);
                    
                    // Add recommended models to the ensemble
                    var addedCount = 0;
                    foreach (var recommendation in recommendations.Take(_ensembleOptions.MaxModels))
                    {
                        // Create and add the recommended model
                        var recommendedModel = recommendation.ModelFactory();
                        _ensembleModel.AddModel(recommendedModel, NumOps.FromDouble(recommendation.ConfidenceScore));
                        addedCount++;
                        _logger.Debug("Added {ModelName} to ensemble with confidence {Confidence}", 
                            recommendation.ModelName, recommendation.ConfidenceScore);
                    }
                    
                    if (addedCount == 0)
                    {
                        throw new InvalidOperationException(
                            "Could not add any models to the ensemble.");
                    }
                    
                    _logger.Information("Added {Count} models to auto-ensemble", addedCount);
                }
                
                // The ensemble model implements IFullModel<T, TInput, TOutput> directly
                model = _ensembleModel;
                _logger.Information("Using ensemble model with {Count} base models", _ensembleModel.BaseModels.Count);
            }
            else if (_multimodalModel != null)
            {
                // User configured a multimodal model
                if (_modalityPreprocessors.Count > 0)
                {
                    foreach (var kvp in _modalityPreprocessors)
                    {
                        var modalityType = kvp.Key;
                        var preprocessor = kvp.Value;
                        _multimodalModel.AddModality(modalityType, preprocessor);
                    }
                }
                _multimodalModel.SetFusionStrategy(_modalityFusionStrategy);
                
                // Multimodal models need to be wrapped in a model adapter
                // TODO: Implement MultimodalModelAdapter when multimodal models are implemented
                _logger.Information("Using multimodal model with {Count} modalities", _modalityPreprocessors.Count);
                throw new NotImplementedException("Multimodal model integration is pending implementation");
            }
            else if (_foundationModel != null)
            {
                // User configured a foundation model
                _logger.Information("Building model with foundation model: {Architecture}", _foundationModel.Architecture);
                
                // Create adapter configuration
                var adapterConfig = new FoundationModels.FoundationModelAdapter<T>.ModelConfiguration
                {
                    PromptTemplate = _promptTemplate,
                    FeatureNames = _featureNames,
                    MaxTokens = _generationMaxTokens,
                    Temperature = _generationTemperature,
                    TopP = _generationTopP,
                    MaxConcurrency = _maxConcurrency
                };
                
                // Create the adapter to bridge foundation model with IPredictiveModel
                var adapter = new FoundationModels.FoundationModelAdapter<T>(
                    _foundationModel,
                    _predictionType ?? Enums.PredictionType.Classification,
                    adapterConfig,
                    _logger);
                
                // Apply fine-tuning if configured
                if (_fineTuningOptions != null && _trainingData != null)
                {
                    _logger.Information("Fine-tuning foundation model with {Count} examples", _trainingData.Count);
                    
                    var trainingExamples = ConvertToTrainingExamples(_trainingData);
                    var validationExamples = _validationData != null ? ConvertToTrainingExamples(_validationData) : new List<TrainingExample>();
                    
                    var fineTunedModel = await _foundationModel.FineTuneAsync(
                        trainingExamples,
                        validationExamples,
                        _fineTuningOptions,
                        progressCallback: (progress) => 
                        {
                            _logger.Debug("Fine-tuning progress: Epoch {Current}/{Total}, Loss: {Loss:F4}",
                                progress.CurrentEpoch, progress.TotalEpochs, progress.TrainingLoss);
                        });
                    
                    // Update adapter with fine-tuned model
                    adapter = new FoundationModels.FoundationModelAdapter<T>(
                        fineTunedModel,
                        _predictionType ?? Enums.PredictionType.Classification,
                        adapterConfig,
                        _logger);
                }
                
                // Apply few-shot examples if configured
                if (_fewShotExamples != null && _fewShotExamples.Count > 0)
                {
                    _logger.Debug("Configured {Count} few-shot examples for in-context learning", _fewShotExamples.Count);
                    // Few-shot examples will be used during prediction via the adapter
                }
                
                _model = adapter as IPredictiveModel<T, TInput, TOutput> 
                    ?? throw new InvalidOperationException("Foundation model adapter type mismatch");

                // Foundation models need to be wrapped in a model adapter
                // TODO: Implement FoundationModelAdapter when foundation models are implemented
                _logger.Information("Using foundation model: {ModelType}", _foundationModel.GetType().Name);
                throw new NotImplementedException("Foundation model integration is pending implementation");
            }
            else if (_autoMLModel != null)
            {
                // User enabled AutoML
                // Note: HyperparameterSearchSpace needs to be converted to Dictionary<string, ParameterRange>
                // This conversion is not implemented yet, so we skip the ConfigureSearchSpace call
                if (_searchSpace != null)
                {
                    _logger.Debug("Search space configured (conversion to ParameterRange dictionary pending implementation)");
                }
                if (_autoMLTimeLimit.HasValue)
                {
                    _autoMLModel.SetTimeLimit(_autoMLTimeLimit.Value);
                }
                if (_autoMLTrialLimit.HasValue)
                {
                    _autoMLModel.SetTrialLimit(_autoMLTrialLimit.Value);
                }
                if (_nasStrategy.HasValue)
                {
                    // EnableNAS takes a boolean, enable if any strategy except None is selected
                    _autoMLModel.EnableNAS(_nasStrategy.Value != NeuralArchitectureSearchStrategy.None);
                }

                // AutoML will select and optimize the model
                // Split data for AutoML validation
                var (autoMLX, autoMLY, autoMLValX, autoMLValY, _, _) = (_dataPreprocessor ?? new DefaultDataPreprocessor<T, TInput, TOutput>(
                    new NoNormalizer<T, TInput, TOutput>(),
                    new NoFeatureSelector<T, TInput>(),
                    new NoOutlierRemoval<T, TInput, TOutput>()
                )).SplitData(x, y);

                model = _autoMLModel.SearchBestModel(autoMLX, autoMLY, autoMLValX, autoMLValY);
                _logger.Information("AutoML selected model: {ModelType}", model.GetType().Name);
            }
            else if (_model != null)
            {
                // User explicitly configured a model, use it
                model = _model;
                _logger.Information("Using user-configured model: {ModelType}", model.GetType().Name);
            }
            else if (_optimizer != null)
            {
                // Use the model from the optimizer
                model = _optimizer.Model;
                _logger.Information("Using model from optimizer: {ModelType}", model.GetType().Name);
            }
            else
            {
                // No model specified and no optimizer with a model, use default selection
                _logger.Information("No model explicitly configured, using automatic model selection");
                model = new DefaultModelSelector<T, TInput, TOutput>().SelectModel(x, y);
                _logger.Information("Auto-selected model: {ModelType}", model.GetType().Name);
            }
            
            // Apply interpretability wrapper if configured
            if (_interpretableModel != null)
            {
                _interpretableModel.SetBaseModel(model);
                if (_interpretationMethods.Count > 0)
                {
                    foreach (var method in _interpretationMethods)
                    {
                        _interpretableModel.EnableMethod(method);
                    }
                }
                if (_sensitiveFeatures != null && _fairnessMetrics.Count > 0)
                {
                    // ConfigureFairness expects List<int> and single FairnessMetric, not arrays
                    var sensitiveAttrList = new List<int>(_sensitiveFeatures);
                    var fairnessMetric = _fairnessMetrics[0]; // Use first metric
                    _interpretableModel.ConfigureFairness(sensitiveAttrList, fairnessMetric);
                }
                // Note: IInterpretableModel doesn't implement IFullModel, so we can't use it directly as the model
                // The interpretable wrapper needs to be applied differently or the interface needs to be extended
                _logger.Information("Configured interpretability features (model wrapping not yet fully implemented)");
            }
            
            // Apply production monitoring if configured
            if (_productionMonitor != null)
            {
                if (_dataDriftThreshold != null && _conceptDriftThreshold != null)
                {
                    // ConfigureDriftDetection expects: (string method, double threshold, int windowSize)
                    // We only have thresholds, not method, so we use default method and window size
                    var dataDriftValue = Convert.ToDouble(_dataDriftThreshold);
                    _productionMonitor.ConfigureDriftDetection("KS-Test", dataDriftValue, 1000);
                }
                if (_performanceDropThreshold != null)
                {
                    // ConfigureRetraining expects: (bool enabled, double performanceThreshold, double driftThreshold)
                    var perfThresholdValue = Convert.ToDouble(_performanceDropThreshold);
                    var driftThresholdValue = _conceptDriftThreshold != null ? Convert.ToDouble(_conceptDriftThreshold) : 0.3;
                    _productionMonitor.ConfigureRetraining(true, perfThresholdValue, driftThresholdValue);
                }
                // Production monitoring is typically applied after training
                _logger.Debug("Configured production monitoring");
            }

            _logger.Debug("Initializing model components");
            var normalizer = _normalizer ?? new NoNormalizer<T, TInput, TOutput>();
            _logger.Debug("Using normalizer: {NormalizerType}", normalizer.GetType().Name);

            var optimizer = _optimizer ?? new NormalOptimizer<T, TInput, TOutput>(model);
            _logger.Debug("Using optimizer: {OptimizerType}", optimizer.GetType().Name);

            var featureSelector = _featureSelector ?? new NoFeatureSelector<T, TInput>();
            _logger.Debug("Using feature selector: {FeatureSelectorType}", featureSelector.GetType().Name);

            var outlierRemoval = _outlierRemoval ?? new NoOutlierRemoval<T, TInput, TOutput>();
            _logger.Debug("Using outlier removal: {OutlierRemovalType}", outlierRemoval.GetType().Name);

            var dataPreprocessor = _dataPreprocessor ?? new DefaultDataPreprocessor<T, TInput, TOutput>(normalizer, featureSelector, outlierRemoval);
            _logger.Debug("Using data preprocessor: {PreprocessorType}", dataPreprocessor.GetType().Name);
            
            // Apply custom pipeline steps if configured
            TInput processedX = x;
            TOutput processedY = y;
            
            // Apply pipeline steps in order
            var orderedPositions = new[] 
            {
                PipelinePosition.Start,
                PipelinePosition.Preprocessing,
                PipelinePosition.FeatureEngineering,
                PipelinePosition.FeatureSelection,
                PipelinePosition.Training,
                PipelinePosition.Validation,
                PipelinePosition.PostProcessing
            };
            
            // Note: Pipeline steps would need to be applied here if the interface supported synchronous operations
            // Currently IPipelineStep only supports async operations, so this is commented out for compilation
            /*
            foreach (var position in orderedPositions)
            {
                if (_pipelineSteps.ContainsKey(position))
                {
                    foreach (var step in _pipelineSteps[position])
                    {
                        _logger.Debug("Applying pipeline step at position {Position}", position);
                        // Would need async support or sync methods in IPipelineStep
                        // (processedX, processedY) = step.FitTransform(processedX, processedY);
                    }
                }
            }
            */

            // Preprocess the data
            _logger.Information("Starting data preprocessing");
            var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);
            _logger.Information("Data preprocessing completed");

            // Split the data
            _logger.Information("Splitting data into training, validation, and test sets");
            var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);
            _logger.Debug("Data split complete - Training set size: {TrainingSize}", InputHelper<T, TInput>.GetInputSize(XTrain));

            // Optimize the model
            _logger.Information("Starting model optimization");
            var inputData = OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest, preprocessedX, preprocessedY);
            DefaultInputCache.CacheDefaultInputData(inputData);
            var optimizationResult = optimizer.Optimize(inputData);
            _logger.Information("Model optimization completed");

            var result = new PredictionModelResult<T, TInput, TOutput>(optimizationResult, normInfo);
            _logger.Information("Model building completed successfully");

            // Log model metrics if available
            if (_logger.IsEnabled(LoggingLevel.Debug))
            {
                LogModelMetrics(optimizationResult);
            }

            return result;
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Error occurred during model building");
            throw;
        }
    }

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
    public TOutput Predict(TInput newData, IPredictiveModel<T, TInput, TOutput> modelResult)
    {
        try
        {
            _logger.Information("Making predictions on new data");
            _logger.Debug("Input data shape: {InputShape}", InputHelper<T, TInput>.GetInputSize(newData));

            var result = modelResult.Predict(newData);

            _logger.Information("Prediction completed successfully");
            _logger.Debug("Output shape: {OutputShape}", InputHelper<T, TInput>.GetBatchSize(result));

            return result;
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Error occurred during prediction");
            throw;
        }
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
    public void SaveModel(IPredictiveModel<T, TInput, TOutput> modelResult, string filePath)
    {
        try
        {
            _logger.Information("Saving model to file: {FilePath}", filePath);
            File.WriteAllBytes(filePath, SerializeModel(modelResult));
            _logger.Information("Model saved successfully");
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Error saving model to file: {FilePath}", filePath);
            throw;
        }
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
    public IPredictiveModel<T, TInput, TOutput> LoadModel(string filePath)
    {
        try
        {
            _logger.Information("Loading model from file: {FilePath}", filePath);
            if (!File.Exists(filePath))
            {
                var ex = new FileNotFoundException("Model file not found", filePath);
                _logger.Error(ex, "Model file not found: {FilePath}", filePath);
                throw ex;
            }

            byte[] modelData = File.ReadAllBytes(filePath);
            var model = DeserializeModel(modelData);
            _logger.Information("Model loaded successfully");
            return model;
        }
        catch (Exception ex) when (!(ex is FileNotFoundException))
        {
            _logger.Error(ex, "Error loading model from file: {FilePath}", filePath);
            throw;
        }
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
    public byte[] SerializeModel(IPredictiveModel<T, TInput, TOutput> modelResult)
    {
        try
        {
            _logger.Debug("Serializing model");
            var result = modelResult.Serialize();
            _logger.Debug("Model serialized successfully, size: {SizeBytes} bytes", result.Length);
            return result;
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Error serializing model");
            throw;
        }
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
    public IPredictiveModel<T, TInput, TOutput> DeserializeModel(byte[] modelData)
    {
        try
        {
            _logger.Debug("Deserializing model, data size: {SizeBytes} bytes", modelData.Length);
            var result = new PredictionModelResult<T, TInput, TOutput>();
            result.Deserialize(modelData);
            _logger.Debug("Model deserialized successfully");
            return result;
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Error deserializing model");
            throw;
        }
    }

    /// <summary>
    /// Creates a zip file containing all log files for sending to customer support.
    /// </summary>
    /// <param name="destinationPath">Optional path where the zip file should be saved. If not specified, uses the current directory.</param>
    /// <returns>The full path to the created zip file, or null if creation failed.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you encounter problems with your model and need help from technical
    /// support, this method creates a single compressed file containing all log files. This makes
    /// it easy to share the logs with support staff who can help diagnose the issue.
    /// </para>
    /// </remarks>
    public string CreateSupportPackage(string? destinationPath = null)
    {
        try
        {
            _logger.Information("Creating support package");
            var packagePath = LoggingFactory.CreateLogArchive(destinationPath);
            if (packagePath != null)
            {
                _logger.Information("Support package created successfully: {PackagePath}", packagePath);
            }
            else
            {
                _logger.Warning("Support package creation returned null path");
            }

            return packagePath ?? string.Empty;
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Error creating support package");
            return string.Empty;
        }
    }
    
    /// <summary>
    /// Logs metrics from the optimization result in a structured way
    /// </summary>
    /// <param name="optimizationResult">The optimization result containing metrics to log</param>
    private void LogModelMetrics(OptimizationResult<T, TInput, TOutput> optimizationResult)
    {
        // Log overall model information
        _logger.Debug("Overall model information:");
        _logger.Debug("  Best fitness score: {BestFitnessScore}", Convert.ToDouble(optimizationResult.BestFitnessScore));
        _logger.Debug("  Optimization iterations: {Iterations}", optimizationResult.Iterations);
        
        if (optimizationResult.SelectedFeatures != null)
        {
            _logger.Debug("  Selected features count: {FeatureCount}", optimizationResult.SelectedFeatures.Count);
        }
        
        // Log training data metrics
        if (optimizationResult.TrainingResult != null)
        {
            _logger.Debug("Training data metrics:");
            LogDatasetResultMetrics(optimizationResult.TrainingResult, "Training");
        }
        
        // Log validation data metrics
        if (optimizationResult.ValidationResult != null)
        {
            _logger.Debug("Validation data metrics:");
            LogDatasetResultMetrics(optimizationResult.ValidationResult, "Validation");
        }
        
        // Log test data metrics
        if (optimizationResult.TestResult != null)
        {
            _logger.Debug("Test data metrics:");
            LogDatasetResultMetrics(optimizationResult.TestResult, "Test");
        }
        
        // Log fit detection results if available
        if (optimizationResult.FitDetectionResult != null)
        {
            _logger.Debug("Fit detection results:");
            _logger.Debug("  Fit type: {FitType}", optimizationResult.FitDetectionResult.FitType);
            
            if (!EqualityComparer<T>.Default.Equals(optimizationResult.FitDetectionResult.ConfidenceLevel, default(T)))
            {
                _logger.Debug("  Confidence level: {ConfidenceLevel}", Convert.ToDouble(optimizationResult.FitDetectionResult.ConfidenceLevel));
            }
            
            if (optimizationResult.FitDetectionResult.Recommendations != null && 
                optimizationResult.FitDetectionResult.Recommendations.Count > 0)
            {
                _logger.Debug("  Recommendations:");
                foreach (var recommendation in optimizationResult.FitDetectionResult.Recommendations)
                {
                    _logger.Debug("    - {Recommendation}", recommendation);
                }
            }
            
            if (optimizationResult.FitDetectionResult.AdditionalInfo != null && 
                optimizationResult.FitDetectionResult.AdditionalInfo.Count > 0)
            {
                _logger.Debug("  Additional information:");
                foreach (var info in optimizationResult.FitDetectionResult.AdditionalInfo)
                {
                    _logger.Debug("    {Key}: {Value}", info.Key, info.Value);
                }
            }
        }
    }
    
    /// <summary>
    /// Logs metrics from a dataset result
    /// </summary>
    /// <param name="datasetResult">The dataset result containing statistics</param>
    /// <param name="datasetName">Name of the dataset (Training, Validation, Test)</param>
    private void LogDatasetResultMetrics(OptimizationResult<T, TInput, TOutput>.DatasetResult datasetResult, string datasetName)
    {
        // Log error statistics
        if (datasetResult.ErrorStats != null)
        {
            _logger.Debug("  {DatasetName} Error Statistics:", datasetName);
            LogStatisticsMetrics(datasetResult.ErrorStats, datasetName, "Error");
        }
        
        // Log prediction statistics
        if (datasetResult.PredictionStats != null)
        {
            _logger.Debug("  {DatasetName} Prediction Statistics:", datasetName);
            LogStatisticsMetrics(datasetResult.PredictionStats, datasetName, "Prediction");
        }
        
        // Log actual value statistics
        if (datasetResult.ActualBasicStats != null)
        {
            _logger.Debug("  {DatasetName} Actual Value Statistics:", datasetName);
            LogStatisticsMetrics(datasetResult.ActualBasicStats, datasetName, "Actual");
        }
        
        // Log predicted value statistics
        if (datasetResult.PredictedBasicStats != null)
        {
            _logger.Debug("  {DatasetName} Predicted Value Statistics:", datasetName);
            LogStatisticsMetrics(datasetResult.PredictedBasicStats, datasetName, "Predicted");
        }
    }
    
    /// <summary>
    /// Logs metrics from a statistics object
    /// </summary>
    /// <param name="statsObject">The statistics object containing metrics</param>
    /// <param name="datasetName">Name of the dataset (Training, Validation, Test)</param>
    /// <param name="statsType">Type of statistics (Error, Prediction, etc.)</param>
    private void LogStatisticsMetrics(object statsObject, string datasetName, string statsType)
    {
        if (statsObject == null)
        {
            return;
        }
        
        // Use reflection to get metrics from the stats object
        // First try to get a Metrics property or dictionary if it exists
        var metricsProperty = statsObject.GetType().GetProperty("Metrics");
        if (metricsProperty != null)
        {
            var metrics = metricsProperty.GetValue(statsObject) as IDictionary<object, object>;
            if (metrics != null && metrics.Count > 0)
            {
                foreach (var metric in metrics)
                {
                    _logger.Debug("    {MetricName}: {MetricValue}", metric.Key, metric.Value);
                }
                return;
            }
        }
        
        // If Metrics property doesn't exist or doesn't work as expected,
        // log all public property values
        var properties = statsObject.GetType().GetProperties();
        foreach (var property in properties)
        {
            // Skip complex objects and collections to avoid excessive logging
            if (property.PropertyType.IsPrimitive || 
                property.PropertyType == typeof(string) || 
                property.PropertyType == typeof(decimal) ||
                property.PropertyType == typeof(double) ||
                property.PropertyType == typeof(float))
            {
                try
                {
                    var value = property.GetValue(statsObject);
                    if (value != null)
                    {
                        _logger.Debug("    {PropertyName}: {PropertyValue}", property.Name, value);
                    }
                }
                catch
                {
                    // Ignore properties that can't be read
                }
            }
        }
    }
    
    /// <summary>
    /// Converts training data to TrainingExample format for foundation models
    /// </summary>
    private List<TrainingExample> ConvertToTrainingExamples(List<(TInput input, TOutput output)> data)
    {
        var examples = new List<TrainingExample>();
        
        foreach (var tuple in data)
        {
            var input = tuple.Item1;
            var output = tuple.Item2;
            var inputText = ConvertInputToText(input);
            var outputText = ConvertOutputToText(output);
            
            examples.Add(new TrainingExample
            {
                Input = inputText,
                Output = outputText,
                Metadata = new Dictionary<string, object>
                {
                    ["original_input_type"] = typeof(TInput).Name,
                    ["original_output_type"] = typeof(TOutput).Name
                }
            });
        }
        
        return examples;
    }
    
    /// <summary>
    /// Converts input data to text representation
    /// </summary>
    private string ConvertInputToText(TInput input)
    {
        if (input is string str)
        {
            return str;
        }
        
        if (input is Matrix<T> matrix)
        {
            var values = new List<string>();
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    if (_featureNames != null && j < _featureNames.Count)
                    {
                        values.Add($"{_featureNames[j]}: {matrix[i, j]}");
                    }
                    else
                    {
                        values.Add($"feature_{j}: {matrix[i, j]}");
                    }
                }
            }
            return string.Join(", ", values);
        }
        
        if (input is Vector<T> vector)
        {
            var values = new List<string>();
            for (int i = 0; i < vector.Length; i++)
            {
                if (_featureNames != null && i < _featureNames.Count)
                {
                    values.Add($"{_featureNames[i]}: {vector[i]}");
                }
                else
                {
                    values.Add($"feature_{i}: {vector[i]}");
                }
            }
            return string.Join(", ", values);
        }
        
        return input?.ToString() ?? string.Empty;
    }
    
    /// <summary>
    /// Converts output data to text representation
    /// </summary>
    private string ConvertOutputToText(TOutput output)
    {
        if (output is string str)
        {
            return str;
        }
        
        if (output is Vector<T> vector && vector.Length > 0)
        {
            // For classification, might be one-hot encoded
            if (_predictionType == Enums.PredictionType.Classification)
            {
                var maxIndex = 0;
                var maxValue = vector[0];
                for (int i = 1; i < vector.Length; i++)
                {
                    if (Comparer<T>.Default.Compare(vector[i], maxValue) > 0)
                    {
                        maxValue = vector[i];
                        maxIndex = i;
                    }
                }
                return $"class_{maxIndex}";
            }
            
            // For regression, return the first value
            var firstValue = vector[0];
            if (firstValue == null) return "0";
            return firstValue.ToString() ?? "0";
        }
        
        return output?.ToString() ?? string.Empty;
    }
}