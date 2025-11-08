global using Newtonsoft.Json;
global using Formatting = Newtonsoft.Json.Formatting;
using AiDotNet.Data.Abstractions;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.Serialization;
using AiDotNet.Agents;

namespace AiDotNet.Models.Results;

/// <summary>
/// Represents a complete predictive model with its optimization results, normalization information, and metadata.
/// This class implements the IPredictiveModel interface and provides serialization capabilities.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates a trained predictive model along with all the information needed to use it for making 
/// predictions on new data. It includes the model itself, the results of the optimization process that created the 
/// model, normalization information for preprocessing input data and postprocessing predictions, and metadata about 
/// the model. The class also provides methods for serializing and deserializing the model, allowing it to be saved 
/// to and loaded from files.
/// </para>
/// <para><b>For Beginners:</b> This class represents a complete, ready-to-use predictive model.
/// 
/// When working with machine learning models:
/// - You need to store not just the model itself, but also how to prepare data for it
/// - You want to keep track of how the model was created and how well it performs
/// - You need to be able to save the model and load it later
/// 
/// This class handles all of that by storing:
/// - The actual model that makes predictions
/// - Information about how the model was optimized
/// - How to normalize/scale input data before making predictions
/// - Metadata about the model (like feature names, creation date, etc.)
/// 
/// It also provides methods to:
/// - Make predictions on new data
/// - Save the model to a file
/// - Load a model from a file
/// 
/// This makes it easy to train a model once and then use it many times in different applications.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[Serializable]
public class PredictionModelResult<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the underlying model used for making predictions.
    /// </summary>
    /// <value>An implementation of IFullModel&lt;T&gt; representing the trained model.</value>
    /// <remarks>
    /// <para>
    /// This property contains the actual model that is used to make predictions. The model implements the IFullModel&lt;T&gt; 
    /// interface, which provides methods for predicting outputs based on input features. The specific implementation could 
    /// be a linear regression model, a polynomial model, a neural network, or any other type of model that implements the 
    /// interface. This property is marked as nullable, but must be initialized before the model can be used for predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual trained model that makes predictions.
    /// 
    /// The model:
    /// - Contains the mathematical formula or algorithm for making predictions
    /// - Is the core component that transforms input data into predictions
    /// - Could be a linear model, polynomial model, neural network, etc.
    /// 
    /// This property is marked as nullable (with the ? symbol) because:
    /// - When deserializing a model from storage, it might not be immediately available
    /// - The default constructor creates an empty object that will be populated later
    /// 
    /// However, the model must be initialized before you can use the Predict method,
    /// or you'll get an InvalidOperationException.
    /// </para>
    /// </remarks>
    internal IFullModel<T, TInput, TOutput>? Model { get; private set; }

    /// <summary>
    /// Gets or sets the results of the optimization process that created the model.
    /// </summary>
    /// <value>An OptimizationResult&lt;T&gt; object containing detailed optimization information.</value>
    /// <remarks>
    /// <para>
    /// This property contains the results of the optimization process that was used to create the model. It includes 
    /// information such as the best fitness score achieved, the number of iterations performed, the history of fitness 
    /// scores during optimization, and detailed performance metrics on training, validation, and test datasets. This 
    /// information is useful for understanding how the model was created and how well it performs on different datasets.
    /// </para>
    /// <para><b>For Beginners:</b> This contains information about how the model was created and how well it performs.
    /// 
    /// The optimization result:
    /// - Records how the model was trained/optimized
    /// - Includes performance metrics on different datasets
    /// - Stores information about feature selection
    /// - Contains analysis of potential issues like overfitting
    /// 
    /// This information is valuable because:
    /// - It helps you understand the model's strengths and limitations
    /// - It provides context for interpreting predictions
    /// - It can guide decisions about when to retrain the model
    /// 
    /// For example, you might check the R-squared value on the test dataset
    /// to understand how well the model is likely to perform on new data.
    /// </para>
    /// </remarks>
    internal OptimizationResult<T, TInput, TOutput> OptimizationResult { get; private set; } = new();

    /// <summary>
    /// Gets or sets the normalization information used to preprocess input data and postprocess predictions.
    /// </summary>
    /// <value>A NormalizationInfo&lt;T&gt; object containing normalization parameters and the normalizer.</value>
    /// <remarks>
    /// <para>
    /// This property contains information about how input data should be normalized before being fed into the model, and 
    /// how the model's outputs should be denormalized to obtain the final predictions. Normalization is a preprocessing 
    /// step that scales the input features to a standard range, which can improve the performance and stability of many 
    /// machine learning algorithms. The NormalizationInfo object includes the normalizer object that performs the actual 
    /// normalization and denormalization operations, as well as parameters that describe how the target variable (Y) was 
    /// normalized during training.
    /// </para>
    /// <para><b>For Beginners:</b> This contains information about how to scale data before and after prediction.
    /// 
    /// The normalization info:
    /// - Stores how input features should be scaled before making predictions
    /// - Stores how to convert predictions back to their original scale
    /// - Contains the actual normalizer object that performs these operations
    /// 
    /// Normalization is important because:
    /// - Many models perform better with normalized input data
    /// - The model was trained on normalized data, so new data must be normalized the same way
    /// - Predictions need to be converted back to the original scale to be meaningful
    /// 
    /// For example, if your input features were originally in different units (like dollars, years, and percentages),
    /// normalization might scale them all to a range of 0-1 for the model, and then the predictions
    /// need to be scaled back to the original units.
    /// </para>
    /// </remarks>
    internal NormalizationInfo<T, TInput, TOutput> NormalizationInfo { get; private set; } = new();

    /// <summary>
    /// Gets or sets the metadata associated with the model.
    /// </summary>
    /// <value>A ModelMetaData&lt;T&gt; object containing descriptive information about the model.</value>
    /// <remarks>
    /// <para>
    /// This property contains metadata about the model, such as the names of the input features, the name of the target
    /// variable, the date and time the model was created, the type of model, and any additional descriptive information.
    /// This metadata is useful for understanding what the model does and how it should be used, without having to examine
    /// the model itself. It can also be used for documentation, versioning, and tracking purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This contains descriptive information about the model.
    ///
    /// The model metadata:
    /// - Stores information like feature names and target variable name
    /// - Records when the model was created
    /// - Describes what type of model it is
    /// - May include additional descriptive information
    ///
    /// This information is useful because:
    /// - It helps you understand what the model is predicting and what inputs it needs
    /// - It provides documentation for the model
    /// - It can help with versioning and tracking different models
    ///
    /// For example, the metadata might tell you that this model predicts "house_price"
    /// based on features like "square_footage", "num_bedrooms", and "location_score".
    /// </para>
    /// </remarks>
    internal ModelMetadata<T> ModelMetaData { get; private set; } = new();

    /// <summary>
    /// Gets or sets the bias detector used for ethical AI evaluation.
    /// </summary>
    /// <value>An implementation of IBiasDetector&lt;T&gt; for detecting bias in model predictions, or null if not configured.</value>
    internal IBiasDetector<T>? BiasDetector { get; private set; }

    /// <summary>
    /// Gets or sets the fairness evaluator used for ethical AI evaluation.
    /// </summary>
    /// <value>An implementation of IFairnessEvaluator&lt;T&gt; for evaluating fairness metrics, or null if not configured.</value>
    internal IFairnessEvaluator<T>? FairnessEvaluator { get; private set; }

    /// <summary>
    /// Gets or sets the retriever used for RAG document retrieval during inference.
    /// </summary>
    /// <value>An implementation of IRetriever&lt;T&gt; for retrieving documents, or null if RAG is not configured.</value>
    internal IRetriever<T>? RagRetriever { get; private set; }

    /// <summary>
    /// Gets or sets the reranker used for RAG document reranking during inference.
    /// </summary>
    /// <value>An implementation of IReranker&lt;T&gt; for reranking documents, or null if RAG is not configured.</value>
    internal IReranker<T>? RagReranker { get; private set; }

    /// <summary>
    /// Gets or sets the generator used for RAG answer generation during inference.
    /// </summary>
    /// <value>An implementation of IGenerator&lt;T&gt; for generating answers, or null if RAG is not configured.</value>
    internal IGenerator<T>? RagGenerator { get; private set; }

    /// <summary>
    /// Gets or sets the query processors used for RAG query preprocessing during inference.
    /// </summary>
    /// <value>Query processors for preprocessing queries, or null if not configured.</value>
    internal IEnumerable<IQueryProcessor>? QueryProcessors { get; private set; }

    /// <summary>
    /// Gets or sets the meta-learner used for few-shot adaptation and fine-tuning.
    /// </summary>
    /// <value>An implementation of IMetaLearner for meta-learning capabilities, or null if this is a standard supervised model.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If this model was trained using meta-learning, this property contains
    /// the meta-learner that can quickly adapt the model to new tasks with just a few examples.
    ///
    /// If this is null, the model was trained using standard supervised learning.
    /// </para>
    /// </remarks>
    internal IMetaLearner<T, TInput, TOutput>? MetaLearner { get; private set; }

    /// <summary>
    /// Gets or sets the results from meta-training.
    /// </summary>
    /// <value>Meta-training results containing performance history and statistics, or null if this is a standard supervised model.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If this model was meta-trained, this contains information about
    /// how the meta-training process went - loss curves, accuracy across tasks, etc.
    ///
    /// If this is null, the model was trained using standard supervised learning and you should
    /// check OptimizationResult instead.
    /// </para>
    /// </remarks>
    internal MetaTrainingResult<T>? MetaTrainingResult { get; private set; }

    /// <summary>
    /// Gets or sets the results from cross-validation.
    /// </summary>
    /// <value>Cross-validation results containing fold-by-fold performance metrics and aggregated statistics, or null if cross-validation was not performed.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If cross-validation was configured during model building, this contains
    /// detailed information about how the model performed across different subsets of the training data.
    /// This helps you understand:
    /// - How consistently the model performs across different data splits
    /// - Whether the model is overfitting or underfitting
    /// - The typical performance you can expect on new, unseen data
    ///
    /// The results include:
    /// - Performance metrics for each fold (R², RMSE, MAE, etc.)
    /// - Aggregated statistics across all folds (mean, standard deviation)
    /// - Feature importance scores averaged across folds
    /// - Timing information for training and evaluation
    ///
    /// If this is null, cross-validation was not performed, and you should rely on the
    /// OptimizationResult for performance metrics instead.
    ///
    /// Example usage:
    /// <code>
    /// if (result.CrossValidationResult != null)
    /// {
    ///     var avgR2 = result.CrossValidationResult.R2Stats.Mean;
    ///     var r2StdDev = result.CrossValidationResult.R2Stats.StandardDeviation;
    ///     Console.WriteLine($"R² = {avgR2} ± {r2StdDev}");
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public CrossValidationResult<T, TInput, TOutput>? CrossValidationResult { get; internal set; }

    /// <summary>
    /// Gets or sets the LoRA configuration for parameter-efficient fine-tuning.
    /// </summary>
    /// <value>LoRA configuration for adaptation, or null if not configured.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> LoRA (Low-Rank Adaptation) enables efficient fine-tuning by
    /// adding small "adapter" layers instead of retraining all parameters. This makes Adapt()
    /// and FineTune() much faster and require less memory.
    ///
    /// If null, adaptation will train all model parameters (standard fine-tuning).
    /// </para>
    /// </remarks>
    internal ILoRAConfiguration<T>? LoRAConfiguration { get; private set; }

    /// <summary>
    /// Gets or sets the agent configuration used during model building.
    /// </summary>
    /// <value>Agent configuration containing API keys and settings, or null if agent assistance wasn't used.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you enabled agent assistance during model building with WithAgentAssistance(),
    /// this property stores the configuration. The API key is stored here so you can use AskAsync() on the trained
    /// model without providing the key again.
    ///
    /// Note: API keys are NOT serialized when saving the model to disk for security reasons.
    /// </para>
    /// </remarks>
    [JsonIgnore]
    public AgentConfiguration<T>? AgentConfig { get; set; }

    /// <summary>
    /// Gets or sets the agent's recommendations made during model building.
    /// </summary>
    /// <value>Agent recommendations including suggested models and reasoning, or null if agent assistance wasn't used.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you used agent assistance during building, this contains all the recommendations
    /// the agent made, such as:
    /// - Which model type to use (e.g., "RidgeRegression")
    /// - Why that model was chosen
    /// - Suggested hyperparameter values
    ///
    /// You can examine these recommendations to understand why the agent made certain choices.
    ///
    /// Example:
    /// <code>
    /// if (result.AgentRecommendation != null)
    /// {
    ///     Console.WriteLine($"Agent selected: {result.AgentRecommendation.SuggestedModelType}");
    ///     Console.WriteLine($"Reasoning: {result.AgentRecommendation.ModelSelectionReasoning}");
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public AgentRecommendation<T, TInput, TOutput>? AgentRecommendation { get; set; }

    /// <summary>
    /// Initializes a new instance of the PredictionModelResult class with the specified model, optimization results, and normalization information.
    /// </summary>
    /// <param name="model">The underlying model used for making predictions.</param>
    /// <param name="optimizationResult">The results of the optimization process that created the model.</param>
    /// <param name="normalizationInfo">The normalization information used to preprocess input data and postprocess predictions.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new PredictionModelResult instance with the specified model, optimization results, and
    /// normalization information. It also initializes the ModelMetadata property by calling the GetModelMetadata method on
    /// the provided model. This constructor is typically used when a new model has been trained and needs to be packaged
    /// with all the necessary information for making predictions and for later serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new prediction model result with all the necessary components.
    ///
    /// When creating a new PredictionModelResult:
    /// - You provide the trained model that will make predictions
    /// - You provide the optimization results that describe how the model was created
    /// - You provide the normalization information needed to process data
    /// - The constructor automatically extracts metadata from the model
    ///
    /// This constructor is typically used when:
    /// - You've just finished training a model
    /// - You want to package it with all the information needed to use it
    /// - You plan to save it for later use or deploy it in an application
    ///
    /// For example, after training a house price prediction model, you would use this constructor
    /// to create a complete package that can be saved and used for making predictions.
    /// </para>
    /// </remarks>
    public PredictionModelResult(IFullModel<T, TInput, TOutput> model,
        OptimizationResult<T, TInput, TOutput> optimizationResult,
        NormalizationInfo<T, TInput, TOutput> normalizationInfo)
    {
        Model = model;
        OptimizationResult = optimizationResult;
        NormalizationInfo = normalizationInfo;
        ModelMetaData = model?.GetModelMetadata() ?? new();
    }

    /// <summary>
    /// Initializes a new instance of the PredictionModelResult class with optimization results and normalization information.
    /// </summary>
    /// <param name="optimizationResult">The results of the optimization process that created the model.</param>
    /// <param name="normalizationInfo">The normalization information used to preprocess input data and postprocess predictions.</param>
    /// <param name="biasDetector">Optional bias detector for ethical AI evaluation.</param>
    /// <param name="fairnessEvaluator">Optional fairness evaluator for ethical AI evaluation.</param>
    /// <param name="ragRetriever">Optional retriever for RAG functionality during inference.</param>
    /// <param name="ragReranker">Optional reranker for RAG functionality during inference.</param>
    /// <param name="ragGenerator">Optional generator for RAG functionality during inference.</param>
    /// <param name="queryProcessors">Optional query processors for RAG query preprocessing.</param>
    /// <param name="loraConfiguration">Optional LoRA configuration for parameter-efficient fine-tuning.</param>
    /// <param name="crossValidationResult">Optional cross-validation results from training.</param>
    public PredictionModelResult(OptimizationResult<T, TInput, TOutput> optimizationResult,
        NormalizationInfo<T, TInput, TOutput> normalizationInfo,
        IBiasDetector<T>? biasDetector = null,
        IFairnessEvaluator<T>? fairnessEvaluator = null,
        IRetriever<T>? ragRetriever = null,
        IReranker<T>? ragReranker = null,
        IGenerator<T>? ragGenerator = null,
        IEnumerable<IQueryProcessor>? queryProcessors = null,
        ILoRAConfiguration<T>? loraConfiguration = null,
        CrossValidationResult<T, TInput, TOutput>? crossValidationResult = null)
    {
        Model = optimizationResult.BestSolution;
        OptimizationResult = optimizationResult;
        NormalizationInfo = normalizationInfo;
        ModelMetaData = Model?.GetModelMetadata() ?? new();
        BiasDetector = biasDetector;
        FairnessEvaluator = fairnessEvaluator;
        RagRetriever = ragRetriever;
        RagReranker = ragReranker;
        RagGenerator = ragGenerator;
        QueryProcessors = queryProcessors;
        LoRAConfiguration = loraConfiguration;
        CrossValidationResult = crossValidationResult;
    }

    /// <summary>
    /// Initializes a new instance of the PredictionModelResult class for a meta-trained model.
    /// </summary>
    /// <param name="metaLearner">The meta-learner containing the trained model and adaptation capabilities.</param>
    /// <param name="metaResult">The results from the meta-training process.</param>
    /// <param name="loraConfiguration">Optional LoRA configuration for parameter-efficient adaptation.</param>
    /// <param name="biasDetector">Optional bias detector for ethical AI evaluation.</param>
    /// <param name="fairnessEvaluator">Optional fairness evaluator for ethical AI evaluation.</param>
    /// <param name="ragRetriever">Optional retriever for RAG functionality during inference.</param>
    /// <param name="ragReranker">Optional reranker for RAG functionality during inference.</param>
    /// <param name="ragGenerator">Optional generator for RAG functionality during inference.</param>
    /// <param name="queryProcessors">Optional query processors for RAG query preprocessing.</param>
    /// <remarks>
    /// <para>
    /// This constructor is used when a model has been trained using meta-learning (e.g., MAML, Reptile, SEAL).
    /// The resulting PredictionModelResult contains the meta-trained model along with the meta-learner itself,
    /// enabling quick adaptation to new tasks with just a few examples.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a prediction result for a meta-trained model.
    ///
    /// Meta-trained models are special because:
    /// - They've learned how to learn across many different tasks
    /// - They can quickly adapt to new tasks with just a few examples (few-shot learning)
    /// - They retain the meta-learner for future adaptation
    ///
    /// After meta-training, you can:
    /// - Use Adapt() to quickly adjust the model to a new task (5-10 examples)
    /// - Use FineTune() for more extensive adaptation (100+ examples)
    /// - Save and deploy the model for rapid adaptation in production
    ///
    /// This constructor packages everything needed to use a meta-trained model:
    /// - The trained model (from the meta-learner)
    /// - The meta-learner itself (for adaptation)
    /// - Training history (loss curves, performance metrics)
    /// - Optional LoRA configuration (for efficient adaptation)
    /// </para>
    /// </remarks>
    public PredictionModelResult(
        IMetaLearner<T, TInput, TOutput> metaLearner,
        MetaTrainingResult<T> metaResult,
        ILoRAConfiguration<T>? loraConfiguration = null,
        IBiasDetector<T>? biasDetector = null,
        IFairnessEvaluator<T>? fairnessEvaluator = null,
        IRetriever<T>? ragRetriever = null,
        IReranker<T>? ragReranker = null,
        IGenerator<T>? ragGenerator = null,
        IEnumerable<IQueryProcessor>? queryProcessors = null)
    {
        Model = metaLearner.BaseModel;
        MetaLearner = metaLearner;
        MetaTrainingResult = metaResult;
        LoRAConfiguration = loraConfiguration;
        ModelMetaData = Model?.GetModelMetadata() ?? new();
        BiasDetector = biasDetector;
        FairnessEvaluator = fairnessEvaluator;
        RagRetriever = ragRetriever;
        RagReranker = ragReranker;
        RagGenerator = ragGenerator;
        QueryProcessors = queryProcessors;

        // Create placeholder OptimizationResult and NormalizationInfo for consistency
        OptimizationResult = new OptimizationResult<T, TInput, TOutput>();
        NormalizationInfo = new NormalizationInfo<T, TInput, TOutput>();
    }

    /// <summary>
    /// Initializes a new instance of the PredictionModelResult class with default values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new PredictionModelResult instance with default values for all properties. It is primarily 
    /// used for deserialization, where the properties will be populated from the deserialized data. This constructor should 
    /// not be used directly for creating models that will be used for predictions, as the Model property will be null and 
    /// the Predict method will throw an exception.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates an empty prediction model result that will be filled in later.
    /// 
    /// This default constructor:
    /// - Creates an object with default/empty values
    /// - Is primarily used during deserialization (loading from a file)
    /// - Should not be used directly when you want to make predictions
    /// 
    /// When this constructor is used:
    /// - The Model property is null
    /// - The other properties are initialized with empty objects
    /// 
    /// If you try to call Predict on an object created with this constructor without
    /// first deserializing data into it, you'll get an error because the Model is null.
    /// </para>
    /// </remarks>
    internal PredictionModelResult()
    {
    }

    /// <summary>
    /// Gets the metadata associated with the model.
    /// </summary>
    /// <returns>A ModelMetaData&lt;T&gt; object containing descriptive information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the metadata associated with the model, which is stored in the ModelMetaData property. It is
    /// implemented to satisfy the IPredictiveModel interface, which requires a method to retrieve model metadata. The
    /// metadata includes information such as the names of the input features, the name of the target variable, the date
    /// and time the model was created, the type of model, and any additional descriptive information.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns descriptive information about the model.
    ///
    /// The GetModelMetadata method:
    /// - Returns the metadata stored in the ModelMetaData property
    /// - Is required by the IPredictiveModel interface
    /// - Provides access to information about what the model does and how it works
    ///
    /// This method is useful when:
    /// - You want to display information about the model
    /// - You need to check what features the model expects
    /// - You're working with multiple models and need to identify them
    ///
    /// For example, you might call this method to get the list of feature names
    /// so you can ensure your input data has the correct columns.
    /// </para>
    /// </remarks>
    public ModelMetadata<T> GetModelMetadata()
    {
        return ModelMetaData;
    }

    /// <summary>
    /// Makes predictions using the model on the provided input data.
    /// </summary>
    /// <param name="newData">A matrix of input features, where each row represents an observation and each column represents a feature.</param>
    /// <returns>A vector of predicted values, one for each observation in the input matrix.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the Model or Normalizer is not initialized.</exception>
    /// <remarks>
    /// <para>
    /// This method makes predictions using the model on the provided input data. It first normalizes the input data using 
    /// the normalizer from the NormalizationInfo property, then passes the normalized data to the model's Predict method, 
    /// and finally denormalizes the model's outputs to obtain the final predictions. This process ensures that the input 
    /// data is preprocessed in the same way as the training data was, and that the predictions are in the same scale as 
    /// the original target variable.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes predictions on new data using the trained model.
    /// 
    /// The Predict method:
    /// - Takes a matrix of input features as its parameter
    /// - Normalizes the input data to match how the model was trained
    /// - Passes the normalized data to the model for prediction
    /// - Denormalizes the results to convert them back to the original scale
    /// - Returns a vector of predictions, one for each row in the input matrix
    /// 
    /// This method will throw an exception if:
    /// - The Model property is null (not initialized)
    /// - The Normalizer in NormalizationInfo is null (not initialized)
    /// 
    /// For example, if you have a matrix of house features (square footage, bedrooms, etc.),
    /// this method will return a vector of predicted house prices.
    /// </para>
    /// </remarks>
    public TOutput Predict(TInput newData)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        if (NormalizationInfo.Normalizer == null)
        {
            throw new InvalidOperationException("Normalizer is not initialized.");
        }

        var (normalizedNewData, _) = NormalizationInfo.Normalizer.NormalizeInput(newData);
        var normalizedPredictions = Model.Predict(normalizedNewData);

        return NormalizationInfo.Normalizer.Denormalize(normalizedPredictions, NormalizationInfo.YParams);
    }

    /// <summary>
    /// Training is not supported on PredictionModelResult. Use PredictionModelBuilder to create and train new models.
    /// </summary>
    /// <param name="input">Input training data (not used).</param>
    /// <param name="expectedOutput">Expected output values (not used).</param>
    /// <exception cref="InvalidOperationException">Always thrown - PredictionModelResult represents an already-trained model and cannot be retrained.</exception>
    /// <remarks>
    /// PredictionModelResult is a snapshot of a trained model with its optimization results and metadata.
    /// Retraining would invalidate the OptimizationResult and metadata.
    /// To train a new model or retrain with different data, use PredictionModelBuilder.Build() instead.
    /// </remarks>
    public void Train(TInput input, TOutput expectedOutput)
    {
        throw new InvalidOperationException(
            "PredictionModelResult represents an already-trained model and cannot be retrained. " +
            "The OptimizationResult and metadata reflect the original training process. " +
            "To train a new model, use PredictionModelBuilder.Build() instead.");
    }

    /// <summary>
    /// Quickly adapts the model to a new task using a few examples (few-shot learning).
    /// </summary>
    /// <param name="supportX">Input features for adaptation (typically 1-50 examples per class).</param>
    /// <param name="supportY">Target outputs for adaptation.</param>
    /// <param name="steps">Number of adaptation steps (default 10 for meta-learning, 50 for supervised).</param>
    /// <param name="learningRate">Learning rate for adaptation (default 0.01). Set to -1 to use config defaults.</param>
    /// <remarks>
    /// <para>
    /// This method performs lightweight adaptation to quickly adjust the model to a new task or domain.
    /// It is designed for scenarios where you have limited labeled data and need fast adaptation.
    /// </para>
    /// <para><b>For Beginners:</b> Adapt() is like giving your model a quick "refresher course" on a new task.
    ///
    /// <b>When to use Adapt():</b>
    /// - You have 1-50 examples of a new task
    /// - You need quick adjustment (seconds to minutes)
    /// - You want to preserve the model's general capabilities
    ///
    /// <b>How it works:</b>
    /// - <b>Meta-learning models:</b> Uses the meta-learner's fast adaptation (inner loop)
    /// - <b>Supervised models:</b> Performs quick fine-tuning with small learning rate
    /// - <b>With LoRA:</b> Only adapts small adapter layers (very efficient!)
    /// - <b>Without LoRA:</b> Updates all parameters (slower but more flexible)
    ///
    /// <b>Examples:</b>
    /// - Adapt a general image classifier to recognize a new category (5-10 images)
    /// - Adjust a language model to a specific writing style (10-20 examples)
    /// - Fine-tune a recommendation model to a new user (5-15 interactions)
    ///
    /// <b>Performance:</b>
    /// - Meta-learning + LoRA: Fastest (milliseconds to seconds)
    /// - Meta-learning alone: Fast (seconds)
    /// - Supervised + LoRA: Moderate (seconds to minutes)
    /// - Supervised alone: Slower (minutes)
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when Model is not initialized.</exception>
    /// <exception cref="ArgumentNullException">Thrown when supportX or supportY is null.</exception>
    public void Adapt(TInput supportX, TOutput supportY, int steps = -1, double learningRate = -1)
    {
        if (Model == null)
            throw new InvalidOperationException("Model is not initialized - cannot adapt");
        if (supportX == null)
            throw new ArgumentNullException(nameof(supportX), "Support set features cannot be null");
        if (supportY == null)
            throw new ArgumentNullException(nameof(supportY), "Support set targets cannot be null");

        // Determine default steps based on model type
        if (steps == -1)
        {
            steps = MetaLearner != null ? 10 : 50;
        }

        // Determine default learning rate based on model type
        // Meta-learning models use smaller LR (already meta-trained)
        // Supervised models use standard fine-tuning LR
        if (learningRate < 0)
        {
            learningRate = MetaLearner != null ? 0.001 : 0.01;
        }

        // Meta-learning path: Use fast adaptation
        if (MetaLearner != null)
        {
            // Create task from support data
            var task = new MetaLearningTask<T, TInput, TOutput>
            {
                SupportSetX = supportX,
                SupportSetY = supportY,
                QuerySetX = supportX,  // Use support as query for adaptation
                QuerySetY = supportY
            };

            // Perform fast adaptation using meta-learner
            var adaptResult = MetaLearner.AdaptAndEvaluate(task);

            // Apply adapted parameters to this model
            // The meta-learner adapts its BaseModel, so we need to copy those parameters
            var adaptedParameters = MetaLearner.BaseModel.GetParameters();
            Model.SetParameters(adaptedParameters);
        }
        else
        {
            // Supervised learning path: Quick fine-tuning
            // LoRA Integration Note: Parameter-efficient fine-tuning with LoRA requires:
            // 1. For neural networks: Access to model layers to apply LoRAConfiguration.ApplyLoRA()
            //    - Requires architectural refactoring to expose layers through IFullModel
            //    - Or adding LoRA-aware training methods to model interface
            // 2. For non-neural models (regression, polynomial): Different approach needed
            //    - Apply low-rank decomposition at parameter level
            //    - Research territory for non-layer-based models
            // For now, perform standard full-parameter adaptation
            //
            // When LoRA is properly integrated (future PR), this will become:
            // if (LoRAConfiguration != null && Model is ILayeredModel layeredModel)
            // {
            //     var loraLayers = layeredModel.GetLayers().Select(l => LoRAConfiguration.ApplyLoRA(l));
            //     // Train with LoRA-adapted layers
            // }

            // Perform gradient descent steps
            for (int step = 0; step < steps; step++)
            {
                // Use Model.Train() which performs one gradient step
                Model.Train(supportX, supportY);
            }
        }
    }

    /// <summary>
    /// Performs comprehensive fine-tuning on a dataset to optimize for a specific task.
    /// </summary>
    /// <param name="trainX">Training input features (typically 100-10,000+ examples).</param>
    /// <param name="trainY">Training target outputs.</param>
    /// <param name="epochs">Number of training epochs (default 100 for supervised, 50 for meta-learning).</param>
    /// <param name="validationX">Optional validation features for monitoring overfitting.</param>
    /// <param name="validationY">Optional validation targets.</param>
    /// <param name="learningRate">Learning rate for fine-tuning (default 0.001). Set to -1 to use config defaults.</param>
    /// <remarks>
    /// <para>
    /// This method performs extensive fine-tuning to optimize the model for a specific task or domain.
    /// It is designed for scenarios where you have substantial labeled data and computational resources.
    /// </para>
    /// <para><b>For Beginners:</b> FineTune() is like giving your model a complete "training course" on a new task.
    ///
    /// <b>When to use FineTune():</b>
    /// - You have 100+ labeled examples
    /// - You can afford longer training time (minutes to hours)
    /// - You want to maximize performance on a specific task
    /// - You're okay with the model specializing (losing some generality)
    ///
    /// <b>How it works:</b>
    /// - <b>Meta-learning models:</b> Extended adaptation with more steps
    /// - <b>Supervised models:</b> Standard fine-tuning with full optimization
    /// - <b>With LoRA:</b> Only trains adapter layers (efficient, preserves base model)
    /// - <b>Without LoRA:</b> Updates all parameters (full fine-tuning)
    ///
    /// <b>Difference from Adapt():</b>
    /// - Adapt(): 1-50 examples, 10-50 steps, preserves generality
    /// - FineTune(): 100+ examples, 100-1000+ steps, optimizes for specific task
    ///
    /// <b>Examples:</b>
    /// - Fine-tune a general classifier on domain-specific data (1000+ images)
    /// - Adapt a language model to a specific industry (10,000+ documents)
    /// - Specialize a recommender for a specific user segment (5,000+ interactions)
    ///
    /// <b>Performance:</b>
    /// - With LoRA: Faster, less memory, preserves base model
    /// - Without LoRA: Slower, more memory, fully specialized
    /// - With validation: Better generalization, prevents overfitting
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when Model is not initialized.</exception>
    /// <exception cref="ArgumentNullException">Thrown when trainX or trainY is null.</exception>
    public void FineTune(
        TInput trainX,
        TOutput trainY,
        int epochs = -1,
        TInput? validationX = default,
        TOutput? validationY = default,
        double learningRate = -1)
    {
        if (Model == null)
            throw new InvalidOperationException("Model is not initialized - cannot fine-tune");
        if (trainX == null)
            throw new ArgumentNullException(nameof(trainX), "Training features cannot be null");
        if (trainY == null)
            throw new ArgumentNullException(nameof(trainY), "Training targets cannot be null");

        // Determine default epochs based on model type
        if (epochs == -1)
        {
            epochs = MetaLearner != null ? 50 : 100;
        }

        // Determine default learning rate based on model type
        // Meta-learning models use smaller LR for fine-tuning (preserve meta-trained features)
        // Supervised models use standard fine-tuning LR
        if (learningRate < 0)
        {
            learningRate = MetaLearner != null ? 0.0001 : 0.001;
        }

        // Meta-learning path: Extended adaptation
        if (MetaLearner != null)
        {
            // For meta-learning, fine-tuning is just extended adaptation
            // Use more steps for thorough optimization
            Adapt(trainX, trainY, steps: epochs, learningRate: learningRate);
        }
        else
        {
            // Supervised learning path: Full fine-tuning
            // LoRA Integration Note: Same architectural considerations as Adapt() method
            // Parameter-efficient fine-tuning requires model architecture refactoring to expose layers
            // For now, perform standard full-parameter fine-tuning

            // Perform training epochs
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Train on full dataset
                Model.Train(trainX, trainY);

                // Validation monitoring: Future enhancement for early stopping
                // Would require loss calculation on validation set and stopping criteria
            }
        }
    }

    /// <summary>
    /// Gets the parameters of the underlying model.
    /// </summary>
    /// <returns>A vector containing the model parameters.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the Model is not initialized.</exception>
    public Vector<T> GetParameters()
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        return Model.GetParameters();
    }

    /// <summary>
    /// Setting parameters is not supported on PredictionModelResult.
    /// </summary>
    /// <param name="parameters">The parameter vector (not used).</param>
    /// <exception cref="InvalidOperationException">Always thrown - PredictionModelResult parameters cannot be modified.</exception>
    /// <remarks>
    /// Modifying parameters would invalidate the OptimizationResult which reflects the optimized parameter values.
    /// To create a model with different parameters, use PredictionModelBuilder with custom initial parameters.
    /// </remarks>
    public void SetParameters(Vector<T> parameters)
    {
        throw new InvalidOperationException(
            "PredictionModelResult parameters cannot be modified. " +
            "The current parameters reflect the optimized solution from the training process. " +
            "To create a model with different parameters, use PredictionModelBuilder.");
    }

    /// <summary>
    /// Gets the number of parameters in the underlying model.
    /// </summary>
    public int ParameterCount
    {
        get
        {
            if (Model == null)
            {
                return 0;
            }

            return Model.ParameterCount;
        }
    }

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameter vector to use.</param>
    /// <returns>A new PredictionModelResult with updated parameters.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the Model is not initialized.</exception>
    public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        var newModel = Model.WithParameters(parameters);

        // Deep-copy OptimizationResult and update its BestSolution to reference newModel
        // This ensures metadata consistency - BestSolution should always point to the current model
        var updatedOptimizationResult = OptimizationResult.DeepCopy();
        updatedOptimizationResult.BestSolution = newModel;

        // Create new result with updated optimization result
        // Use constructor that preserves BiasDetector, FairnessEvaluator, and RAG components
        return new PredictionModelResult<T, TInput, TOutput>(
            updatedOptimizationResult,
            NormalizationInfo,
            BiasDetector,
            FairnessEvaluator,
            RagRetriever,
            RagReranker,
            RagGenerator,
            QueryProcessors);
    }

    /// <summary>
    /// Gets the indices of features that are actively used by the underlying model.
    /// </summary>
    /// <returns>An enumerable of active feature indices.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the Model is not initialized.</exception>
    public IEnumerable<int> GetActiveFeatureIndices()
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        return Model.GetActiveFeatureIndices();
    }

    /// <summary>
    /// Setting active feature indices is not supported on PredictionModelResult.
    /// </summary>
    /// <param name="featureIndices">The feature indices (not used).</param>
    /// <exception cref="InvalidOperationException">Always thrown - PredictionModelResult feature configuration cannot be modified.</exception>
    /// <remarks>
    /// Changing active features would invalidate the trained model and optimization results.
    /// To train a model with different features, use PredictionModelBuilder with the desired feature configuration.
    /// </remarks>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        throw new InvalidOperationException(
            "PredictionModelResult active features cannot be modified. " +
            "The model was trained with a specific feature set. " +
            "To use different features, train a new model using PredictionModelBuilder.");
    }

    /// <summary>
    /// Checks if a specific feature is used by the underlying model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is used, false otherwise.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the Model is not initialized.</exception>
    public bool IsFeatureUsed(int featureIndex)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        return Model.IsFeatureUsed(featureIndex);
    }

    /// <summary>
    /// Gets the feature importance scores from the underlying model.
    /// </summary>
    /// <returns>A dictionary mapping feature names to importance scores.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the Model is not initialized.</exception>
    public Dictionary<string, T> GetFeatureImportance()
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        return Model.GetFeatureImportance();
    }

    /// <summary>
    /// Creates a deep copy of this PredictionModelResult.
    /// </summary>
    /// <returns>A new PredictionModelResult instance that is a deep copy of this one.</returns>
    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Cannot deep copy PredictionModelResult with null Model.");
        }

        var clonedModel = Model.DeepCopy();
        var clonedOptimizationResult = OptimizationResult.DeepCopy();

        // Update OptimizationResult.BestSolution to reference the cloned model
        // This ensures metadata consistency across the deep copy
        clonedOptimizationResult.BestSolution = clonedModel;

        var clonedNormalizationInfo = NormalizationInfo.DeepCopy();

        // Use constructor that preserves BiasDetector, FairnessEvaluator, and RAG components
        return new PredictionModelResult<T, TInput, TOutput>(
            clonedOptimizationResult,
            clonedNormalizationInfo,
            BiasDetector,
            FairnessEvaluator,
            RagRetriever,
            RagReranker,
            RagGenerator,
            QueryProcessors);
    }

    /// <summary>
    /// Creates a shallow copy of this PredictionModelResult.
    /// </summary>
    /// <returns>A new PredictionModelResult instance that is a shallow copy of this one.</returns>
    /// <remarks>
    /// This method delegates to WithParameters to ensure consistency in how OptimizationResult is handled.
    /// The cloned instance will have a new model with the same parameters and updated OptimizationResult metadata.
    /// </remarks>
    public IFullModel<T, TInput, TOutput> Clone()
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Cannot clone PredictionModelResult with null Model.");
        }

        return WithParameters(Model.GetParameters());
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the entire PredictionModelResult object, including the model, optimization results, 
    /// normalization information, and metadata. The model is serialized using its own Serialize() method, 
    /// ensuring that model-specific serialization logic is properly applied. The other components are 
    /// serialized using JSON. This approach ensures that each component of the PredictionModelResult is 
    /// serialized in the most appropriate way.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the model into a format that can be stored or transmitted.
    /// 
    /// The Serialize method:
    /// - Uses the model's own serialization method to properly handle model-specific details
    /// - Serializes other components (optimization results, normalization info, metadata) to JSON
    /// - Combines everything into a single byte array that can be saved to a file or database
    /// 
    /// This is important because:
    /// - Different model types may need to be serialized differently
    /// - It ensures all the model's internal details are properly preserved
    /// - It allows for more efficient and robust storage of the complete prediction model package
    /// </para>
    /// </remarks>
    public byte[] Serialize()
    {
        try
        {
            // Create JSON settings with custom converters for our types
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                Formatting = Formatting.Indented
            };

            // Serialize the object
            var jsonString = JsonConvert.SerializeObject(this, settings);
            return Encoding.UTF8.GetBytes(jsonString);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to serialize the model: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Deserializes a model from a byte array.
    /// </summary>
    /// <param name="data">A byte array containing the serialized model.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs a PredictionModelResult object from a serialized byte array. It reads 
    /// the serialized data of each component (model, optimization results, normalization information, 
    /// and metadata) and deserializes them using the appropriate methods. The model is deserialized 
    /// using its model-specific deserialization method, while the other components are deserialized 
    /// from JSON.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a model from a previously serialized byte array.
    /// 
    /// The Deserialize method:
    /// - Takes a byte array containing a serialized model
    /// - Extracts each component (model, optimization results, etc.)
    /// - Uses the appropriate deserialization method for each component
    /// - Reconstructs the complete PredictionModelResult object
    /// 
    /// This approach ensures:
    /// - Each model type is deserialized correctly using its own specific logic
    /// - All model parameters and settings are properly restored
    /// - The complete prediction pipeline (normalization, prediction, denormalization) is reconstructed
    /// 
    /// This method will throw an exception if the deserialization process fails for any component.
    /// </para>
    /// </remarks>
    public void Deserialize(byte[] data)
    {
        try
        {
            var jsonString = Encoding.UTF8.GetString(data);

            // Create JSON settings with custom converters for our types
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All
            };

            // Deserialize the object
            var deserializedObject = JsonConvert.DeserializeObject<PredictionModelResult<T, TInput, TOutput>>(jsonString, settings);

            if (deserializedObject != null)
            {
                Model = deserializedObject.Model;
                OptimizationResult = deserializedObject.OptimizationResult;
                NormalizationInfo = deserializedObject.NormalizationInfo;
                ModelMetaData = deserializedObject.ModelMetaData;
                BiasDetector = deserializedObject.BiasDetector;
                FairnessEvaluator = deserializedObject.FairnessEvaluator;
            }
            else
            {
                throw new InvalidOperationException("Deserialization resulted in a null object.");
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to deserialize the model: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model will be saved.</param>
    /// <remarks>
    /// <para>
    /// This method saves the serialized model to a file at the specified path. It first serializes the model to a byte array
    /// using the Serialize method, then writes the byte array to the specified file. If the file already exists, it will be
    /// overwritten. This method provides a convenient way to persist the model for later use.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the model to a file on disk.
    ///
    /// The SaveModel method:
    /// - Takes a file path where the model should be saved
    /// - Serializes the model to a byte array
    /// - Writes the byte array to the specified file
    ///
    /// This method is useful when:
    /// - You want to save a trained model for later use
    /// - You need to share a model with others
    /// - You want to deploy a model to a production environment
    ///
    /// For example, after training a model, you might save it with:
    /// `myModel.SaveModel("C:\\Models\\house_price_predictor.model");`
    ///
    /// If the file already exists, it will be overwritten.
    /// </para>
    /// </remarks>
    public void SaveModel(string filePath)
    {
        File.WriteAllBytes(filePath, Serialize());
    }

    /// <summary>
    /// Loads the model from a file.
    /// </summary>
    /// <param name="filePath">The path to the file containing the saved model.</param>
    /// <remarks>
    /// <para>
    /// This method loads a serialized model from a file at the specified path. It reads the byte array from the file
    /// and then deserializes it using the Deserialize method. This method provides a convenient way to load a previously
    /// saved model.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a model from a file on disk.
    ///
    /// The LoadFromFile method:
    /// - Takes a file path where the model is stored
    /// - Reads the byte array from the file
    /// - Deserializes the byte array to restore the model
    ///
    /// This method is useful when:
    /// - You want to load a previously trained and saved model
    /// - You need to use a model that was shared with you
    /// - You want to deploy a pre-trained model in a production environment
    ///
    /// For example, to load a model, you might use:
    /// `myModel.LoadFromFile("C:\\Models\\house_price_predictor.model");`
    ///
    /// <b>Note:</b> This method is distinct from the static LoadModel overload which requires a model factory.
    /// </para>
    /// </remarks>
    public void LoadFromFile(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Model file not found at path: {filePath}", filePath);
        }

        var data = File.ReadAllBytes(filePath);
        Deserialize(data);
    }

    /// <summary>
    /// Explicit implementation of IModelSerializer.LoadModel to avoid confusion with static LoadModel method.
    /// </summary>
    /// <param name="filePath">The path to the file containing the saved model.</param>
    void IModelSerializer.LoadModel(string filePath)
    {
        LoadFromFile(filePath);
    }

    /// <summary>
    /// Loads a model from a file.
    /// </summary>
    /// <param name="filePath">The path of the file containing the serialized model.</param>
    /// <param name="modelFactory">A factory function that creates the appropriate model type based on metadata.</param>
    /// <returns>A new PredictionModelResult&lt;T&gt; instance loaded from the file.</returns>
    /// <remarks>
    /// <para>
    /// This static method loads a serialized model from a file at the specified path. It requires a model factory function
    /// that can create the appropriate model type based on metadata. This ensures that the correct model type is instantiated
    /// before deserialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a file.
    /// 
    /// The LoadModel method:
    /// - Takes a file path where the model is stored
    /// - Uses the model factory to create the right type of model based on metadata
    /// - Reads the file and deserializes the data into a new PredictionModelResult object
    /// - Returns the fully loaded model ready for making predictions
    /// 
    /// The model factory is important because:
    /// - Different types of models (linear regression, neural networks, etc.) need different deserialization logic
    /// - The factory knows how to create the right type of model based on information in the saved file
    /// 
    /// For example, you might load a model with:
    /// `var model = PredictionModelResult<double, Matrix<double>, Vector<double>>.LoadModel(
    ///     "C:\\Models\\house_price_predictor.model", 
    ///     metadata => new LinearRegressionModel<double>());`
    /// </para>
    /// </remarks>
    public static PredictionModelResult<T, TInput, TOutput> LoadModel(
        string filePath,
        Func<ModelMetadata<T>, IFullModel<T, TInput, TOutput>> modelFactory)
    {
        // First, we need to read the file
        byte[] data = File.ReadAllBytes(filePath);

        // Extract metadata to determine model type
        var metadata = ExtractMetadataFromSerializedData(data);

        // Create a new model instance of the appropriate type
        var model = modelFactory(metadata);

        // Create a new PredictionModelResult with the model
        var result = new PredictionModelResult<T, TInput, TOutput>
        {
            Model = model
        };

        // Deserialize the data
        result.Deserialize(data);

        return result;
    }

    private static ModelMetadata<T> ExtractMetadataFromSerializedData(byte[] data)
    {
        var jsonString = Encoding.UTF8.GetString(data);
        var settings = new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.All
        };
        var deserializedObject = JsonConvert.DeserializeObject<PredictionModelResult<T, TInput, TOutput>>(jsonString, settings);
        return deserializedObject?.ModelMetaData ?? new();
    }

    /// <summary>
    /// Generates a grounded answer using the configured RAG pipeline during inference.
    /// </summary>
    /// <param name="query">The question to answer.</param>
    /// <param name="topK">Number of documents to retrieve (optional).</param>
    /// <param name="topKAfterRerank">Number of documents after reranking (optional).</param>
    /// <param name="metadataFilters">Optional filters for document selection.</param>
    /// <returns>A grounded answer with source citations.</returns>
    /// <exception cref="InvalidOperationException">Thrown when RAG components are not configured.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> Use this during inference to get AI-generated answers backed by your documents.
    /// The system will search your document collection, find the most relevant sources,
    /// and generate an answer with citations.
    /// 
    /// RAG must be configured via PredictionModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.
    /// </remarks>
    public AiDotNet.RetrievalAugmentedGeneration.Models.GroundedAnswer<T> GenerateAnswer(
        string query,
        int? topK = null,
        int? topKAfterRerank = null,
        Dictionary<string, object>? metadataFilters = null)
    {
        if (RagRetriever == null || RagReranker == null || RagGenerator == null)
        {
            throw new InvalidOperationException(
                "RAG pipeline not configured. Configure RAG components using PredictionModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
        }

        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty", nameof(query));

        var processedQuery = ProcessQueryWithProcessors(query);

        var filters = metadataFilters ?? new Dictionary<string, object>();
        var effectiveTopK = topK ?? RagRetriever.DefaultTopK;
        var retrievedDocs = RagRetriever.Retrieve(processedQuery, effectiveTopK, filters);

        var retrievedList = retrievedDocs.ToList();

        if (retrievedList.Count == 0)
        {
            return new AiDotNet.RetrievalAugmentedGeneration.Models.GroundedAnswer<T>
            {
                Query = query,
                Answer = "I couldn't find any relevant information to answer this question.",
                SourceDocuments = new List<AiDotNet.RetrievalAugmentedGeneration.Models.Document<T>>(),
                Citations = new List<string>(),
                ConfidenceScore = 0.0
            };
        }

        var rerankedDocs = RagReranker.Rerank(processedQuery, retrievedList);
        
        if (topKAfterRerank.HasValue)
        {
            rerankedDocs = rerankedDocs.Take(topKAfterRerank.Value);
        }

        var contextDocs = rerankedDocs.ToList();
        return RagGenerator.GenerateGrounded(processedQuery, contextDocs);
    }

    /// <summary>
    /// Retrieves relevant documents without generating an answer during inference.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="topK">Number of documents to retrieve (optional).</param>
    /// <param name="applyReranking">Whether to rerank results (default: true).</param>
    /// <param name="metadataFilters">Optional filters for document selection.</param>
    /// <returns>Retrieved and optionally reranked documents.</returns>
    /// <exception cref="InvalidOperationException">Thrown when RAG components are not configured.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> Use this during inference to search your document collection without generating an answer.
    /// Good for exploring what documents are available or debugging retrieval quality.
    /// 
    /// RAG must be configured via PredictionModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.
    /// </remarks>
    public IEnumerable<AiDotNet.RetrievalAugmentedGeneration.Models.Document<T>> RetrieveDocuments(
        string query,
        int? topK = null,
        bool applyReranking = true,
        Dictionary<string, object>? metadataFilters = null)
    {
        if (RagRetriever == null)
        {
            throw new InvalidOperationException(
                "RAG retriever not configured. Configure RAG components using PredictionModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
        }

        if (applyReranking && RagReranker == null)
        {
            throw new InvalidOperationException(
                "RAG reranker not configured. Either configure a reranker or call RetrieveDocuments with applyReranking = false.");
        }

        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty", nameof(query));

        var processedQuery = ProcessQueryWithProcessors(query);

        var filters = metadataFilters ?? new Dictionary<string, object>();
        var effectiveTopK = topK ?? RagRetriever.DefaultTopK;
        var docs = RagRetriever.Retrieve(processedQuery, effectiveTopK, filters);

        if (applyReranking && RagReranker != null)
        {
            docs = RagReranker.Rerank(processedQuery, docs);
        }

        return docs.ToList();
    }

    /// <summary>
    /// Processes a query through all configured query processors in sequence.
    /// </summary>
    /// <param name="query">The original query to process.</param>
    /// <returns>The processed query after applying all processors, or the original if no processors configured.</returns>
    private string ProcessQueryWithProcessors(string query)
    {
        if (QueryProcessors == null)
            return query;

        var processedQuery = query;
        foreach (var processor in QueryProcessors)
        {
            processedQuery = processor.ProcessQuery(processedQuery);
        }
        return processedQuery;
    }
}
