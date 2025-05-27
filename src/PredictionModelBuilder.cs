global using AiDotNet.FeatureSelectors;
global using AiDotNet.FitnessCalculators;
global using AiDotNet.Regularization;
global using AiDotNet.Optimizers;
global using AiDotNet.Normalizers;
global using AiDotNet.OutlierRemoval;
global using AiDotNet.DataProcessor;
global using AiDotNet.FitDetectors;

namespace AiDotNet;

using AiDotNet.Logging;

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
    private readonly ILogging _logger;
    private IFeatureSelector<T, TInput>? _featureSelector;
    private INormalizer<T, TInput, TOutput>? _normalizer;
    private IFullModel<T, TInput, TOutput>? _model;
    private IOptimizer<T, TInput, TOutput>? _optimizer;
    private IDataPreprocessor<T, TInput, TOutput>? _dataPreprocessor;
    private IOutlierRemoval<T, TInput, TOutput>? _outlierRemoval;
    private IModelSelector<T, TInput, TOutput>? _modelSelector;
    private LoggingOptions? _loggingOptions;

    /// <summary>
    /// Initializes a new instance of the PredictionModelBuilder class with a required model.
    /// </summary>
    /// <param name="model">The model that will be used for predictions.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor requires you to provide the machine learning model
    /// that will be trained and used for predictions. The model is the core component that learns
    /// patterns from your data and makes predictions based on those patterns.
    /// 
    /// Different types of models are better suited for different types of problems:
    /// - Linear regression for simple numeric predictions
    /// - Decision trees for classification tasks
    /// - Neural networks for complex pattern recognition
    /// 
    /// If you're not sure which model to use, consider using the CreateWithAutoModelSelection
    /// factory method instead, which can automatically select an appropriate model for your data.
    /// </remarks>
    public PredictionModelBuilder(IFullModel<T, TInput, TOutput>? model = null)
    {
        _model = model;
        _logger = LoggingFactory.GetLogger<PredictionModelBuilder<T, TInput, TOutput>>();
        _logger.Information("Creating new PredictionModelBuilder with model: {ModelType}",
            model != null ? model.GetType().Name : "null");
    }
    
    /// <summary>
    /// Sets the model to use for this prediction builder.
    /// </summary>
    /// <param name="model">The model to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This method sets the machine learning model that will be trained and used for predictions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use this to specify which machine learning algorithm you want to use.
    /// Different models are better for different types of problems.
    /// </para>
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> SetModel(IFullModel<T, TInput, TOutput> model)
    {
        _model = model;
        _logger.Information("Model set to: {ModelType}", model.GetType().Name);
        return this;
    }
    
    /// <summary>
    /// Gets the current model from the builder.
    /// </summary>
    /// <returns>The current model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the currently configured model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This gives you access to the machine learning model
    /// that has been configured in this builder.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> GetModel()
    {
        if (_model == null)
        {
            throw new InvalidOperationException("No model has been set. Use SetModel or a configuration method that sets a model.");
        }
        
        return _model;
    }

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

            // Validate inputs and their compatibility
            _logger.Debug("Validating input and output dimensions");
            InputHelper<T, TInput>.ValidateInputOutputDimensions(x, y);

            // Log input and output information if debug is enabled
            if (_logger.IsEnabled(LoggingLevel.Debug))
            {
                _logger.Debug("Input shape: {InputShape}, Output shape: {OutputShape}",
                    InputHelper<T, TInput>.GetInputSize(x),
                    InputHelper<T, TInput>.GetBatchSize(y));
            }

            // Use defaults for these interfaces if they aren't set
            IFullModel<T, TInput, TOutput> model;
            if (_model != null)
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
            
            if (optimizationResult.FitDetectionResult.ConfidenceLevel != null)
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
}