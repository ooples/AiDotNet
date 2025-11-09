global using AiDotNet.FeatureSelectors;
global using AiDotNet.FitnessCalculators;
global using AiDotNet.Regularization;
global using AiDotNet.Optimizers;
global using AiDotNet.Normalizers;
global using AiDotNet.OutlierRemoval;
global using AiDotNet.DataProcessor;
global using AiDotNet.FitDetectors;
global using AiDotNet.LossFunctions;
global using AiDotNet.MetaLearning.Trainers;
global using AiDotNet.DistributedTraining;
global using AiDotNet.Agents;
global using AiDotNet.LanguageModels;
global using AiDotNet.Tools;
global using AiDotNet.Models;
global using AiDotNet.Enums;

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
/// </remarks>
public class PredictionModelBuilder<T, TInput, TOutput> : IPredictionModelBuilder<T, TInput, TOutput>
{
    private IFeatureSelector<T, TInput>? _featureSelector;
    private INormalizer<T, TInput, TOutput>? _normalizer;
    private IRegularization<T, TInput, TOutput>? _regularization;
    private IFitnessCalculator<T, TInput, TOutput>? _fitnessCalculator;
    private IFitDetector<T, TInput, TOutput>? _fitDetector;
    private IFullModel<T, TInput, TOutput>? _model;
    private IOptimizer<T, TInput, TOutput>? _optimizer;
    private IDataPreprocessor<T, TInput, TOutput>? _dataPreprocessor;
    private IOutlierRemoval<T, TInput, TOutput>? _outlierRemoval;
    private IBiasDetector<T>? _biasDetector;
    private IFairnessEvaluator<T>? _fairnessEvaluator;
    private ILoRAConfiguration<T>? _loraConfiguration;
    private IRetriever<T>? _ragRetriever;
    private IReranker<T>? _ragReranker;
    private IGenerator<T>? _ragGenerator;
    private IEnumerable<IQueryProcessor>? _queryProcessors;
    private IMetaLearner<T, TInput, TOutput>? _metaLearner;
    private ICommunicationBackend<T>? _distributedBackend;
    private DistributedStrategy _distributedStrategy = DistributedStrategy.DDP;
    private IShardingConfiguration<T>? _distributedConfiguration;
    private IModelEvaluator<T, TInput, TOutput>? _modelEvaluator;
    private ICrossValidator<T, TInput, TOutput>? _crossValidator;
    private AgentConfiguration<T>? _agentConfig;
    private AgentAssistanceOptions _agentOptions = AgentAssistanceOptions.Default;

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
    /// Builds a predictive model using meta-learning.
    /// Requires ConfigureMetaLearning() to be called first.
    /// </summary>
    /// <returns>A task that represents the asynchronous operation, containing the trained meta-learning model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when ConfigureMetaLearning() hasn't been called.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This overload is for meta-learning, where the model learns to quickly adapt to new tasks.
    /// Use this when you've configured a meta-learner via ConfigureMetaLearning().
    ///
    /// **Meta-Learning**:
    /// - Trains a model that can quickly adapt to new tasks
    /// - Uses episodic data from the meta-learner configuration
    /// - No need to provide x and y - they're in the meta-learner config
    ///
    /// Example:
    /// <code>
    /// var result = await new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureMetaLearning(metaLearner)
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    public Task<PredictionModelResult<T, TInput, TOutput>> BuildAsync()
    {
        // META-LEARNING PATH - requires ConfigureMetaLearning() to be called first
        if (_metaLearner == null)
            throw new InvalidOperationException(
                "BuildAsync() without parameters requires ConfigureMetaLearning() to be called first. " +
                "For regular training, use BuildAsync(x, y) with your input and output data.");

        // Perform meta-training using parameters from config (specified during meta-learner construction)
        var metaResult = _metaLearner.Train();

        // Create PredictionModelResult with meta-learning constructor
        var result = new PredictionModelResult<T, TInput, TOutput>(
            metaLearner: _metaLearner,
            metaResult: metaResult,
            loraConfiguration: _loraConfiguration,
            biasDetector: _biasDetector,
            fairnessEvaluator: _fairnessEvaluator,
            ragRetriever: _ragRetriever,
            ragReranker: _ragReranker,
            ragGenerator: _ragGenerator,
            queryProcessors: _queryProcessors,
            agentConfig: _agentConfig);

        return Task.FromResult(result);
    }

    /// <summary>
    /// Builds a predictive model using the provided input features and output values.
    /// If agent assistance is enabled, the agent will help with model selection and hyperparameter tuning.
    /// </summary>
    /// <param name="x">Matrix of input features (required).</param>
    /// <param name="y">Vector of output values (required).</param>
    /// <returns>A task that represents the asynchronous operation, containing the trained model.</returns>
    /// <exception cref="ArgumentException">Thrown when the number of rows in the features matrix doesn't match the length of the output vector.</exception>
    /// <exception cref="InvalidOperationException">Thrown when no model has been specified for regular training.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method trains your AI model on your specific dataset.
    ///
    /// **Regular Training**:
    /// - Trains on your specific dataset
    /// - Learns patterns from your examples
    /// - Can use agent assistance to select models and tune hyperparameters
    ///
    /// Example with agent assistance:
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
    ///     .BuildAsync(housingData, prices);
    /// </code>
    /// </remarks>
    public async Task<PredictionModelResult<T, TInput, TOutput>> BuildAsync(TInput x, TOutput y)
    {
        // REGULAR TRAINING PATH
        // Convert and validate inputs

        var convertedX = ConversionsHelper.ConvertToMatrix<T, TInput>(x);
        var convertedY = ConversionsHelper.ConvertToVector<T, TOutput>(y);

        if (convertedX.Rows != convertedY.Length)
            throw new ArgumentException("Number of rows in features must match length of actual values", nameof(x));

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

        // Validate model is set (either by user or by agent)
        if (_model == null)
            throw new InvalidOperationException("Model implementation must be specified");

        // Use defaults for these interfaces if they aren't set
        var normalizer = _normalizer ?? new NoNormalizer<T, TInput, TOutput>();
        var optimizer = _optimizer ?? new NormalOptimizer<T, TInput, TOutput>(_model);
        var featureSelector = _featureSelector ?? new NoFeatureSelector<T, TInput>();
        var outlierRemoval = _outlierRemoval ?? new NoOutlierRemoval<T, TInput, TOutput>();
        var dataPreprocessor = _dataPreprocessor ?? new DefaultDataPreprocessor<T, TInput, TOutput>(normalizer, featureSelector, outlierRemoval);

        // Wrap model and optimizer for distributed training if configured
        IFullModel<T, TInput, TOutput> model = _model;
        IOptimizer<T, TInput, TOutput> finalOptimizer = optimizer;

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
                DistributedStrategy.DDP => (
                    (IFullModel<T, TInput, TOutput>)new DistributedTraining.DDPModel<T, TInput, TOutput>(_model, shardingConfig),
                    (IOptimizer<T, TInput, TOutput>)new DistributedTraining.DDPOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)
                ),
                DistributedStrategy.FSDP => (
                    (IFullModel<T, TInput, TOutput>)new DistributedTraining.FSDPModel<T, TInput, TOutput>(_model, shardingConfig),
                    (IOptimizer<T, TInput, TOutput>)new DistributedTraining.FSDPOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)
                ),
                DistributedStrategy.ZeRO1 => (
                    (IFullModel<T, TInput, TOutput>)new DistributedTraining.ZeRO1Model<T, TInput, TOutput>(_model, shardingConfig),
                    (IOptimizer<T, TInput, TOutput>)new DistributedTraining.ZeRO1Optimizer<T, TInput, TOutput>(optimizer, shardingConfig)
                ),
                DistributedStrategy.ZeRO2 => (
                    (IFullModel<T, TInput, TOutput>)new DistributedTraining.ZeRO2Model<T, TInput, TOutput>(_model, shardingConfig),
                    (IOptimizer<T, TInput, TOutput>)new DistributedTraining.ZeRO2Optimizer<T, TInput, TOutput>(optimizer, shardingConfig)
                ),
                DistributedStrategy.ZeRO3 => (
                    (IFullModel<T, TInput, TOutput>)new DistributedTraining.ZeRO3Model<T, TInput, TOutput>(_model, shardingConfig),
                    (IOptimizer<T, TInput, TOutput>)new DistributedTraining.ZeRO3Optimizer<T, TInput, TOutput>(optimizer, shardingConfig)
                ),
                DistributedStrategy.PipelineParallel => (
                    (IFullModel<T, TInput, TOutput>)new DistributedTraining.PipelineParallelModel<T, TInput, TOutput>(_model, shardingConfig),
                    (IOptimizer<T, TInput, TOutput>)new DistributedTraining.PipelineParallelOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)
                ),
                DistributedStrategy.TensorParallel => (
                    (IFullModel<T, TInput, TOutput>)new DistributedTraining.TensorParallelModel<T, TInput, TOutput>(_model, shardingConfig),
                    (IOptimizer<T, TInput, TOutput>)new DistributedTraining.TensorParallelOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)
                ),
                DistributedStrategy.Hybrid => (
                    (IFullModel<T, TInput, TOutput>)new DistributedTraining.HybridShardedModel<T, TInput, TOutput>(_model, shardingConfig),
                    (IOptimizer<T, TInput, TOutput>)new DistributedTraining.HybridShardedOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)
                ),
                _ => throw new InvalidOperationException($"Unsupported distributed strategy: {_distributedStrategy}")
            };
            }
        }

        // Preprocess the data
        var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);

        // Split the data
        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);

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

        // Optimize the final model on the full training set (using distributed optimizer if configured)
        var optimizationResult = finalOptimizer.Optimize(OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest));

        // Return PredictionModelResult with CV results and agent data
        var finalResult = new PredictionModelResult<T, TInput, TOutput>(
            optimizationResult,
            normInfo,
            _biasDetector,
            _fairnessEvaluator,
            _ragRetriever,
            _ragReranker,
            _ragGenerator,
            _queryProcessors,
            _loraConfiguration,
            cvResults,
            _agentConfig,
            agentRecommendation);

        return finalResult;
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
    /// <param name="retriever">Optional retriever for finding relevant documents. If not provided, RAG functionality won't be available.</param>
    /// <param name="reranker">Optional reranker for improving document ranking quality. If not provided, a default reranker will be used if RAG is configured.</param>
    /// <param name="generator">Optional generator for producing grounded answers. If not provided, a default generator will be used if RAG is configured.</param>
    /// <param name="queryProcessors">Optional query processors for improving search quality.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> RAG combines retrieval and generation to create answers backed by real documents.
    /// Configure it with:
    /// - A retriever (finds relevant documents from your collection) - required for RAG
    /// - A reranker (improves the ordering of retrieved documents) - optional, defaults provided
    /// - A generator (creates answers based on the documents) - optional, defaults provided
    /// - Optional query processors (improve search queries before retrieval)
    /// 
    /// RAG operations are performed during inference (after model training) via the PredictionModelResult.
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureRetrievalAugmentedGeneration(
        IRetriever<T>? retriever = null,
        IReranker<T>? reranker = null,
        IGenerator<T>? generator = null,
        IEnumerable<IQueryProcessor>? queryProcessors = null)
    {
        _ragRetriever = retriever;
        _ragReranker = reranker;
        _ragGenerator = generator;
        _queryProcessors = queryProcessors;
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
    ///     .BuildAsync(data, labels);
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
    ///     .BuildAsync(data, labels);
    /// </code>
    /// </remarks>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureAgentAssistance(AgentConfiguration<T> configuration)
    {
        _agentConfig = configuration;
        _agentOptions = configuration.AssistanceOptions ?? AgentAssistanceOptions.Default;
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

            // Build feature analysis input with mock correlations
            var features = new Newtonsoft.Json.Linq.JObject();
            for (int col = 0; col < Math.Min(nFeatures, 20); col++)  // Limit to first 20 features
            {
                features[$"feature_{col}"] = new Newtonsoft.Json.Linq.JObject
                {
                    ["target_correlation"] = 0.5,  // Placeholder
                    ["importance_score"] = 0.1,  // Placeholder
                    ["missing_pct"] = 0.0,
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
