global using Newtonsoft.Json;
global using Formatting = Newtonsoft.Json.Formatting;
using AiDotNet.Data.Abstractions;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using AiDotNet.Interpretability;
using AiDotNet.Serialization;
using AiDotNet.Agents;
using AiDotNet.Models;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Deployment.Export;
using AiDotNet.Deployment.Export.Onnx;
using AiDotNet.Deployment.TensorRT;
using AiDotNet.Deployment.Mobile.CoreML;
using AiDotNet.Deployment.Mobile.TensorFlowLite;
using AiDotNet.Deployment.Runtime;
using AiDotNet.Reasoning;
using AiDotNet.Reasoning.Models;
using AiDotNet.LanguageModels;
using AiDotNet.Enums;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Configuration;
using AiDotNet.Tokenization.Models;

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
    /// Gets or sets the tokenizer used for text processing.
    /// </summary>
    /// <value>An implementation of ITokenizer for encoding/decoding text, or null if not configured.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The tokenizer converts text into tokens (numbers) that the model can process.
    ///
    /// When working with text-based models:
    /// - Text must be tokenized before being fed into the model
    /// - The tokenizer stores the vocabulary mapping between tokens and IDs
    /// - Use the Tokenize() method to convert text to tokens for inference
    ///
    /// Example usage:
    /// <code>
    /// var result = modelResult.Tokenize("Hello world");
    /// // result contains token IDs, attention masks, etc.
    /// </code>
    /// </para>
    /// </remarks>
    internal ITokenizer? Tokenizer { get; private set; }

    /// <summary>
    /// Gets or sets the tokenization configuration.
    /// </summary>
    internal TokenizationConfig? TokenizationConfig { get; private set; }

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
    /// Gets or sets the knowledge graph for graph-enhanced retrieval.
    /// </summary>
    /// <value>A knowledge graph containing entities and relationships, or null if Graph RAG is not configured.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The knowledge graph stores entities (like people, places, concepts) and their
    /// relationships. When you query the model, it can traverse these relationships to find related context
    /// that pure vector similarity might miss.
    /// </para>
    /// <para>
    /// This property is excluded from JSON serialization because it contains runtime infrastructure
    /// (graph store, file handles) that should be reconfigured when the model is loaded.
    /// </para>
    /// </remarks>
    [JsonIgnore]
    internal KnowledgeGraph<T>? KnowledgeGraph { get; private set; }

    /// <summary>
    /// Gets or sets the graph store backend for persistent graph storage.
    /// </summary>
    /// <value>The graph storage backend, or null if Graph RAG is not configured.</value>
    /// <remarks>
    /// <para>
    /// This property is excluded from JSON serialization because it represents runtime storage
    /// infrastructure (file handles, WAL) that must be reconfigured when the model is loaded.
    /// </para>
    /// </remarks>
    [JsonIgnore]
    internal IGraphStore<T>? GraphStore { get; private set; }

    /// <summary>
    /// Gets or sets the hybrid graph retriever for combined vector + graph retrieval.
    /// </summary>
    /// <value>A hybrid retriever combining vector similarity with graph traversal, or null if not configured.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The hybrid retriever first finds similar documents using vector search,
    /// then expands the context by traversing the knowledge graph to find related entities. This provides
    /// richer context than pure vector search alone.
    /// </para>
    /// <para>
    /// This property is excluded from JSON serialization because it contains references to
    /// runtime infrastructure (knowledge graph, document store) that must be reconfigured when loaded.
    /// </para>
    /// </remarks>
    [JsonIgnore]
    internal HybridGraphRetriever<T>? HybridGraphRetriever { get; private set; }

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
    /// Gets the agent configuration used during model building.
    /// </summary>
    /// <value>Agent configuration containing API keys and settings, or null if agent assistance wasn't used.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you enabled agent assistance during model building with ConfigureAgentAssistance(),
    /// this property stores the configuration. The API key is stored here so you can use AskAsync() on the trained
    /// model without providing the key again.
    ///
    /// Note: API keys are NOT serialized when saving the model to disk for security reasons.
    /// </para>
    /// </remarks>
    [JsonIgnore]
    internal AgentConfiguration<T>? AgentConfig { get; private set; }

    /// <summary>
    /// Gets the agent's recommendations made during model building.
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
    internal AgentRecommendation<T, TInput, TOutput>? AgentRecommendation { get; private set; }

    /// <summary>
    /// Gets the deployment configuration for model export, caching, versioning, A/B testing, and telemetry.
    /// </summary>
    /// <value>Deployment configuration aggregating all deployment-related settings, or null if not configured.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This contains all deployment-related settings configured during model building,
    /// including:
    /// - Quantization: Model compression settings (Float16/Int8)
    /// - Caching: Model caching and eviction policies
    /// - Versioning: Model version management
    /// - A/B Testing: Traffic splitting between model versions
    /// - Telemetry: Performance monitoring and metrics
    /// - Export: Platform-specific export settings
    ///
    /// These settings enable advanced deployment features like exporting models for mobile devices,
    /// managing multiple model versions, and monitoring production performance.
    ///
    /// If null, deployment features were not configured and will use defaults when needed.
    /// </para>
    /// </remarks>
    internal DeploymentConfiguration? DeploymentConfiguration { get; private set; }

    /// <summary>
    /// Gets the JIT-compiled prediction function for accelerated inference.
    /// </summary>
    /// <value>A compiled function for fast predictions, or null if JIT compilation was not enabled or not supported.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is an optimized, pre-compiled version of your model's prediction logic.
    ///
    /// When JIT compilation is enabled and the model supports it:
    /// - The model's computation graph is compiled to fast native code during building
    /// - This compiled function is stored here
    /// - Predict() automatically uses it for 5-10x faster predictions
    ///
    /// If this is null:
    /// - JIT was not enabled during model building, OR
    /// - The model doesn't support JIT compilation (e.g., layer-based neural networks)
    /// - Predictions use the normal execution path (still works, just not JIT-accelerated)
    ///
    /// The JIT-compiled function takes an array of Tensor&lt;T&gt; inputs and returns an array of Tensor&lt;T&gt; outputs,
    /// matching the model's computation graph structure.
    /// </para>
    /// </remarks>
    [JsonIgnore]  // Don't serialize - will need to be recompiled after deserialization
    private Func<Tensor<T>[], Tensor<T>[]>? JitCompiledFunction { get; set; }
    private AiDotNet.Configuration.InferenceOptimizationConfig? InferenceOptimizationConfig { get; set; }

    /// <summary>
    /// Gets the reasoning configuration for advanced Chain-of-Thought, Tree-of-Thoughts, and Self-Consistency reasoning.
    /// </summary>
    /// <value>Reasoning configuration for advanced reasoning capabilities, or null if not configured.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This configuration enables advanced reasoning capabilities that make AI
    /// models "think step by step" instead of giving quick answers that might be wrong.
    ///
    /// When reasoning is configured:
    /// - Use ReasonAsync() to solve complex problems with step-by-step reasoning
    /// - Use QuickReasonAsync() for fast answers to simple problems
    /// - Use DeepReasonAsync() for thorough analysis of complex problems
    ///
    /// The reasoning configuration controls:
    /// - How many reasoning steps to take
    /// - Whether to explore multiple solution paths (Tree-of-Thoughts)
    /// - Whether to verify answers with multiple attempts (Self-Consistency)
    /// - Step verification and refinement options
    ///
    /// If null, reasoning features were not configured. You can still use reasoning by providing
    /// configuration directly to ReasonAsync(), but you must have agent configuration set up.
    /// </para>
    /// </remarks>
    internal ReasoningConfig? ReasoningConfig { get; private set; }

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
    /// <param name="agentConfig">Optional agent configuration used during model building.</param>
    /// <param name="agentRecommendation">Optional agent recommendations from model building.</param>
    /// <param name="deploymentConfiguration">Optional deployment configuration for export, caching, versioning, A/B testing, and telemetry.</param>
    /// <param name="knowledgeGraph">Optional knowledge graph for graph-enhanced retrieval.</param>
    /// <param name="graphStore">Optional graph store backend for persistent storage.</param>
    /// <param name="hybridGraphRetriever">Optional hybrid retriever for combined vector + graph search.</param>
    public PredictionModelResult(OptimizationResult<T, TInput, TOutput> optimizationResult,
        NormalizationInfo<T, TInput, TOutput> normalizationInfo,
        IBiasDetector<T>? biasDetector = null,
        IFairnessEvaluator<T>? fairnessEvaluator = null,
        IRetriever<T>? ragRetriever = null,
        IReranker<T>? ragReranker = null,
        IGenerator<T>? ragGenerator = null,
        IEnumerable<IQueryProcessor>? queryProcessors = null,
        ILoRAConfiguration<T>? loraConfiguration = null,
        CrossValidationResult<T, TInput, TOutput>? crossValidationResult = null,
        AgentConfiguration<T>? agentConfig = null,
        AgentRecommendation<T, TInput, TOutput>? agentRecommendation = null,
        DeploymentConfiguration? deploymentConfiguration = null,
        Func<Tensor<T>[], Tensor<T>[]>? jitCompiledFunction = null,
        AiDotNet.Configuration.InferenceOptimizationConfig? inferenceOptimizationConfig = null,
        ReasoningConfig? reasoningConfig = null,
        KnowledgeGraph<T>? knowledgeGraph = null,
        IGraphStore<T>? graphStore = null,
        HybridGraphRetriever<T>? hybridGraphRetriever = null)
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
        AgentConfig = agentConfig;
        AgentRecommendation = agentRecommendation;
        DeploymentConfiguration = deploymentConfiguration;
        JitCompiledFunction = jitCompiledFunction;
        InferenceOptimizationConfig = inferenceOptimizationConfig;
        ReasoningConfig = reasoningConfig;
        KnowledgeGraph = knowledgeGraph;
        GraphStore = graphStore;
        HybridGraphRetriever = hybridGraphRetriever;
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
    /// <param name="agentConfig">Optional agent configuration for AI assistance during inference.</param>
    /// <param name="deploymentConfiguration">Optional deployment configuration for export, caching, versioning, A/B testing, and telemetry.</param>
    /// <param name="knowledgeGraph">Optional knowledge graph for graph-enhanced retrieval.</param>
    /// <param name="graphStore">Optional graph store backend for persistent storage.</param>
    /// <param name="hybridGraphRetriever">Optional hybrid retriever for combined vector + graph search.</param>
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
    /// - Optional agent configuration (for AI assistance)
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
        IEnumerable<IQueryProcessor>? queryProcessors = null,
        AgentConfiguration<T>? agentConfig = null,
        DeploymentConfiguration? deploymentConfiguration = null,
        ReasoningConfig? reasoningConfig = null,
        KnowledgeGraph<T>? knowledgeGraph = null,
        IGraphStore<T>? graphStore = null,
        HybridGraphRetriever<T>? hybridGraphRetriever = null)
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
        AgentConfig = agentConfig;
        DeploymentConfiguration = deploymentConfiguration;
        ReasoningConfig = reasoningConfig;
        KnowledgeGraph = knowledgeGraph;
        GraphStore = graphStore;
        HybridGraphRetriever = hybridGraphRetriever;

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

        // Use JIT-compiled function if available for 5-10x faster predictions
        TOutput normalizedPredictions;
        if (JitCompiledFunction != null && normalizedNewData is Tensor<T> inputTensor)
        {
            // JIT PATH: Use compiled function for accelerated inference
            var jitResult = JitCompiledFunction(new[] { inputTensor });
            if (jitResult != null && jitResult.Length > 0 && jitResult[0] is TOutput output)
            {
                normalizedPredictions = output;
            }
            else
            {
                // Fallback to model if JIT result is unexpected
                normalizedPredictions = Model.Predict(normalizedNewData);
            }
        }
        else
        {
            // NORMAL PATH: Use model's standard prediction
            normalizedPredictions = Model.Predict(normalizedNewData);
        }

        return NormalizationInfo.Normalizer.Denormalize(normalizedPredictions, NormalizationInfo.YParams);
    }

    /// <summary>
    /// Gets the default loss function used by this model for gradient computation.
    /// </summary>
    /// <exception cref="InvalidOperationException">If Model is not initialized.</exception>
    public ILossFunction<T> DefaultLossFunction
    {
        get
        {
            if (Model == null)
            {
                throw new InvalidOperationException("Model is not initialized.");
            }
            return Model.DefaultLossFunction;
        }
    }

    /// <summary>
    /// Computes gradients of the loss function with respect to model parameters WITHOUT updating parameters.
    /// </summary>
    /// <param name="input">The input data (will be normalized automatically).</param>
    /// <param name="target">The target/expected output (will be normalized automatically).</param>
    /// <param name="lossFunction">The loss function to use. If null, uses the model's default loss function.</param>
    /// <returns>A vector containing gradients with respect to all model parameters.</returns>
    /// <exception cref="InvalidOperationException">If Model or Normalizer is not initialized.</exception>
    /// <remarks>
    /// <para>
    /// This method normalizes input and target before computing gradients, maintaining consistency
    /// with the Predict method. Gradients are computed on normalized data and returned as-is
    /// (gradients are with respect to parameters, not outputs, so no denormalization is needed).
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This calculates which direction to adjust the model's parameters to reduce error,
    /// without actually changing them. Input and target are automatically normalized before
    /// gradient computation, just like Predict normalizes input automatically.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        if (NormalizationInfo.Normalizer == null)
        {
            throw new InvalidOperationException("Normalizer is not initialized.");
        }

        // Normalize input and target to maintain API consistency with Predict
        var (normalizedInput, _) = NormalizationInfo.Normalizer.NormalizeInput(input);
        var (normalizedTarget, _) = NormalizationInfo.Normalizer.NormalizeOutput(target);

        // Compute gradients on normalized data (gradients are wrt parameters, no denormalization needed)
        return Model.ComputeGradients(normalizedInput, normalizedTarget, lossFunction);
    }

    /// <summary>
    /// Applies pre-computed gradients to update the model parameters.
    /// </summary>
    /// <param name="gradients">The gradient vector to apply.</param>
    /// <param name="learningRate">The learning rate for the update.</param>
    /// <exception cref="InvalidOperationException">If Model is not initialized.</exception>
    /// <remarks>
    /// <para>
    /// This method delegates to the underlying model's ApplyGradients implementation.
    /// Updates parameters using: θ = θ - learningRate * gradients
    /// </para>
    /// <para><b>For Beginners:</b>
    /// After computing gradients, this method actually updates the model's parameters
    /// by moving them in the direction that reduces error. It delegates to the wrapped model.
    /// </para>
    /// </remarks>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        Model.ApplyGradients(gradients, learningRate);
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
        // Preserve all configuration properties to ensure deployment behavior, model adaptation,
        // training history, and Graph RAG configuration are maintained across parameter updates
        return new PredictionModelResult<T, TInput, TOutput>(
            updatedOptimizationResult,
            NormalizationInfo,
            BiasDetector,
            FairnessEvaluator,
            RagRetriever,
            RagReranker,
            RagGenerator,
            QueryProcessors,
            loraConfiguration: LoRAConfiguration,
            crossValidationResult: CrossValidationResult,
            agentConfig: AgentConfig,
            agentRecommendation: AgentRecommendation,
            deploymentConfiguration: DeploymentConfiguration,
            jitCompiledFunction: null, // JIT compilation is parameter-specific, don't copy
            inferenceOptimizationConfig: InferenceOptimizationConfig,
            knowledgeGraph: KnowledgeGraph,
            graphStore: GraphStore,
            hybridGraphRetriever: HybridGraphRetriever);
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

        // Preserve all configuration properties to ensure deployment behavior, model adaptation,
        // training history, and Graph RAG configuration are maintained across deep copy
        return new PredictionModelResult<T, TInput, TOutput>(
            clonedOptimizationResult,
            clonedNormalizationInfo,
            BiasDetector,
            FairnessEvaluator,
            RagRetriever,
            RagReranker,
            RagGenerator,
            QueryProcessors,
            loraConfiguration: LoRAConfiguration,
            crossValidationResult: CrossValidationResult,
            agentConfig: AgentConfig,
            agentRecommendation: AgentRecommendation,
            deploymentConfiguration: DeploymentConfiguration,
            jitCompiledFunction: null, // JIT compilation is model-specific, don't copy
            inferenceOptimizationConfig: InferenceOptimizationConfig,
            knowledgeGraph: KnowledgeGraph,
            graphStore: GraphStore,
            hybridGraphRetriever: HybridGraphRetriever);
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

                // Preserve RAG components and all configuration properties
                RagRetriever = deserializedObject.RagRetriever;
                RagReranker = deserializedObject.RagReranker;
                RagGenerator = deserializedObject.RagGenerator;
                QueryProcessors = deserializedObject.QueryProcessors;
                LoRAConfiguration = deserializedObject.LoRAConfiguration;
                CrossValidationResult = deserializedObject.CrossValidationResult;
                AgentConfig = deserializedObject.AgentConfig;
                AgentRecommendation = deserializedObject.AgentRecommendation;
                DeploymentConfiguration = deserializedObject.DeploymentConfiguration;
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

    /// <summary>
    /// Queries the knowledge graph to find related nodes by entity name or label.
    /// </summary>
    /// <param name="query">The search query (entity name or partial match).</param>
    /// <param name="topK">Maximum number of results to return.</param>
    /// <returns>Collection of matching graph nodes.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Graph RAG is not configured.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method searches the knowledge graph for entities matching your query.
    /// Unlike vector search which finds similar documents, this finds entities by name or label.
    ///
    /// Example:
    /// <code>
    /// var nodes = result.QueryKnowledgeGraph("Einstein", topK: 5);
    /// foreach (var node in nodes)
    /// {
    ///     Console.WriteLine($"{node.Label}: {node.Id}");
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public IEnumerable<GraphNode<T>> QueryKnowledgeGraph(string query, int topK = 10)
    {
        if (KnowledgeGraph == null)
        {
            throw new InvalidOperationException(
                "Knowledge graph not configured. Configure Graph RAG using PredictionModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
        }

        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty", nameof(query));

        return KnowledgeGraph.FindRelatedNodes(query, topK);
    }

    /// <summary>
    /// Retrieves results using hybrid vector + graph search for enhanced context retrieval.
    /// </summary>
    /// <param name="queryEmbedding">The query embedding vector.</param>
    /// <param name="topK">Number of initial candidates from vector search.</param>
    /// <param name="expansionDepth">How many hops to traverse in the graph (0 = no expansion).</param>
    /// <param name="maxResults">Maximum total results to return.</param>
    /// <returns>List of retrieval results with scores and source information.</returns>
    /// <exception cref="InvalidOperationException">Thrown when hybrid retriever is not configured.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method combines the best of both worlds:
    /// 1. First, it finds similar documents using vector similarity (like traditional RAG)
    /// 2. Then, it expands the context by traversing the knowledge graph to find related entities
    ///
    /// For example, searching for "photosynthesis" might:
    /// - Find documents about photosynthesis via vector search
    /// - Then traverse the graph to also include chlorophyll, plants, carbon dioxide
    ///
    /// This provides richer, more complete context than vector search alone.
    /// </para>
    /// </remarks>
    public List<RetrievalResult<T>> HybridRetrieve(
        Vector<T> queryEmbedding,
        int topK = 5,
        int expansionDepth = 1,
        int maxResults = 10)
    {
        if (HybridGraphRetriever == null)
        {
            throw new InvalidOperationException(
                "Hybrid graph retriever not configured. Configure Graph RAG with a document store using PredictionModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
        }

        if (queryEmbedding == null || queryEmbedding.Length == 0)
            throw new ArgumentException("Query embedding cannot be null or empty", nameof(queryEmbedding));

        return HybridGraphRetriever.Retrieve(queryEmbedding, topK, expansionDepth, maxResults);
    }

    /// <summary>
    /// Traverses the knowledge graph starting from a node using breadth-first search.
    /// </summary>
    /// <param name="startNodeId">The ID of the starting node.</param>
    /// <param name="maxDepth">Maximum traversal depth.</param>
    /// <returns>Collection of nodes reachable from the starting node in BFS order.</returns>
    /// <exception cref="InvalidOperationException">Thrown when knowledge graph is not configured.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method explores the graph starting from a specific entity,
    /// discovering all connected entities up to a specified depth.
    ///
    /// Example: Starting from "Paris", depth=2 might find:
    /// - Depth 1: France, Eiffel Tower, Seine River
    /// - Depth 2: Europe, Iron, Water
    ///
    /// This is useful for understanding the context around a specific entity.
    /// </para>
    /// </remarks>
    public IEnumerable<GraphNode<T>> TraverseGraph(string startNodeId, int maxDepth = 2)
    {
        if (KnowledgeGraph == null)
        {
            throw new InvalidOperationException(
                "Knowledge graph not configured. Configure Graph RAG using PredictionModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
        }

        if (string.IsNullOrWhiteSpace(startNodeId))
            throw new ArgumentException("Start node ID cannot be null or empty", nameof(startNodeId));

        return KnowledgeGraph.BreadthFirstTraversal(startNodeId, maxDepth);
    }

    /// <summary>
    /// Finds the shortest path between two nodes in the knowledge graph.
    /// </summary>
    /// <param name="startNodeId">The ID of the starting node.</param>
    /// <param name="endNodeId">The ID of the target node.</param>
    /// <returns>List of node IDs representing the path, or empty list if no path exists.</returns>
    /// <exception cref="InvalidOperationException">Thrown when knowledge graph is not configured.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method finds how two entities are connected.
    ///
    /// Example: Finding the path between "Einstein" and "Princeton University" might return:
    /// ["einstein", "worked_at_princeton", "princeton_university"]
    ///
    /// This is useful for understanding the relationships between concepts.
    /// </para>
    /// </remarks>
    public List<string> FindPathInGraph(string startNodeId, string endNodeId)
    {
        if (KnowledgeGraph == null)
        {
            throw new InvalidOperationException(
                "Knowledge graph not configured. Configure Graph RAG using PredictionModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
        }

        if (string.IsNullOrWhiteSpace(startNodeId))
            throw new ArgumentException("Start node ID cannot be null or empty", nameof(startNodeId));
        if (string.IsNullOrWhiteSpace(endNodeId))
            throw new ArgumentException("End node ID cannot be null or empty", nameof(endNodeId));

        return KnowledgeGraph.FindShortestPath(startNodeId, endNodeId);
    }

    /// <summary>
    /// Gets all edges (relationships) connected to a node in the knowledge graph.
    /// </summary>
    /// <param name="nodeId">The ID of the node to query.</param>
    /// <param name="direction">The direction of edges to retrieve.</param>
    /// <returns>Collection of edges connected to the node.</returns>
    /// <exception cref="InvalidOperationException">Thrown when knowledge graph is not configured.</exception>
    /// <exception cref="ArgumentException">Thrown when nodeId is null or empty.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method finds all relationships connected to an entity.
    ///
    /// Example: Getting edges for "Einstein" might return:
    /// - Outgoing: STUDIED→Physics, WORKED_AT→Princeton, BORN_IN→Germany
    /// - Incoming: INFLUENCED_BY→Newton
    /// </para>
    /// </remarks>
    public IEnumerable<GraphEdge<T>> GetNodeRelationships(string nodeId, EdgeDirection direction = EdgeDirection.Both)
    {
        if (KnowledgeGraph == null)
        {
            throw new InvalidOperationException(
                "Knowledge graph not configured. Configure Graph RAG using PredictionModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
        }

        if (string.IsNullOrWhiteSpace(nodeId))
            throw new ArgumentException("Node ID cannot be null or empty", nameof(nodeId));

        var result = new List<GraphEdge<T>>();

        if (direction == EdgeDirection.Outgoing || direction == EdgeDirection.Both)
        {
            result.AddRange(KnowledgeGraph.GetOutgoingEdges(nodeId));
        }

        if (direction == EdgeDirection.Incoming || direction == EdgeDirection.Both)
        {
            result.AddRange(KnowledgeGraph.GetIncomingEdges(nodeId));
        }

        return result;
    }

    /// <summary>
    /// Attaches Graph RAG components to a PredictionModelResult instance.
    /// </summary>
    /// <param name="knowledgeGraph">The knowledge graph to attach.</param>
    /// <param name="graphStore">The graph store backend to attach.</param>
    /// <param name="hybridGraphRetriever">The hybrid retriever to attach.</param>
    /// <remarks>
    /// This method is internal and used by PredictionModelBuilder when loading/deserializing models.
    /// Graph RAG components cannot be serialized (they contain file handles, WAL references, etc.),
    /// so the builder automatically reattaches them when loading a model that was configured with Graph RAG.
    /// Users should use PredictionModelBuilder.LoadModel() which handles this automatically.
    /// </remarks>
    internal void AttachGraphComponents(
        KnowledgeGraph<T>? knowledgeGraph = null,
        IGraphStore<T>? graphStore = null,
        HybridGraphRetriever<T>? hybridGraphRetriever = null)
    {
        KnowledgeGraph = knowledgeGraph;
        GraphStore = graphStore;
        HybridGraphRetriever = hybridGraphRetriever;
    }

    /// <summary>
    /// Attaches tokenization components to the model result.
    /// </summary>
    /// <param name="tokenizer">The tokenizer to attach.</param>
    /// <param name="config">Optional tokenization configuration.</param>
    /// <remarks>
    /// This method is internal and used by PredictionModelBuilder during model construction.
    /// </remarks>
    internal void AttachTokenizer(
        ITokenizer? tokenizer,
        TokenizationConfig? config = null)
    {
        Tokenizer = tokenizer;
        TokenizationConfig = config;
    }

    /// <summary>
    /// Tokenizes text using the configured tokenizer.
    /// </summary>
    /// <param name="text">The text to tokenize.</param>
    /// <returns>The tokenization result containing token IDs, attention mask, etc.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no tokenizer is configured.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts your text into the format the model needs.
    ///
    /// Example:
    /// <code>
    /// var result = modelResult.Tokenize("Hello, how are you?");
    /// // result.TokenIds contains [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]
    /// // result.AttentionMask contains [1, 1, 1, 1, 1, 1, 1, 1]
    /// </code>
    /// </para>
    /// </remarks>
    public TokenizationResult Tokenize(string text)
    {
        if (Tokenizer == null)
            throw new InvalidOperationException("No tokenizer configured. Use ConfigureTokenizer() in PredictionModelBuilder.");

        var options = TokenizationConfig?.ToEncodingOptions();
        return Tokenizer.Encode(text, options);
    }

    /// <summary>
    /// Tokenizes multiple texts in a batch.
    /// </summary>
    /// <param name="texts">The texts to tokenize.</param>
    /// <returns>A list of tokenization results.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no tokenizer is configured.</exception>
    public List<TokenizationResult> TokenizeBatch(List<string> texts)
    {
        if (Tokenizer == null)
            throw new InvalidOperationException("No tokenizer configured. Use ConfigureTokenizer() in PredictionModelBuilder.");

        var options = TokenizationConfig?.ToEncodingOptions();
        return Tokenizer.EncodeBatch(texts, options);
    }

    /// <summary>
    /// Decodes token IDs back into text.
    /// </summary>
    /// <param name="tokenIds">The token IDs to decode.</param>
    /// <param name="skipSpecialTokens">Whether to skip special tokens in the output.</param>
    /// <returns>The decoded text.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no tokenizer is configured.</exception>
    public string Detokenize(List<int> tokenIds, bool skipSpecialTokens = true)
    {
        if (Tokenizer == null)
            throw new InvalidOperationException("No tokenizer configured. Use ConfigureTokenizer() in PredictionModelBuilder.");

        return Tokenizer.Decode(tokenIds, skipSpecialTokens);
    }

    /// <summary>
    /// Gets whether a tokenizer is configured for this model.
    /// </summary>
    public bool HasTokenizer => Tokenizer != null;

    /// <summary>
    /// Saves the prediction model result's current state to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the model state to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the entire PredictionModelResult, including the underlying model,
    /// optimization results, normalization information, and metadata. It uses the existing
    /// Serialize method and writes the data to the provided stream.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a snapshot of your complete trained model package.
    ///
    /// When you call SaveState:
    /// - The trained model and all its parameters are written to the stream
    /// - Training results and metrics are saved
    /// - Normalization settings are preserved
    /// - All metadata is included
    ///
    /// This is particularly useful for:
    /// - Checkpointing during long optimization runs
    /// - Saving the best model found during training
    /// - Knowledge distillation workflows
    /// - Creating model backups before deployment
    ///
    /// You can later use LoadState to restore the complete model package.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="IOException">Thrown when there's an error writing to the stream.</exception>
    public virtual void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(stream));

        try
        {
            var data = this.Serialize();
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to save prediction model result state to stream: {ex.Message}", ex);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Unexpected error while saving prediction model result state: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Loads the prediction model result's state from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the model state from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes a complete PredictionModelResult that was previously saved with SaveState,
    /// restoring the model, optimization results, normalization information, and all metadata.
    /// It uses the existing Deserialize method after reading data from the stream.
    /// </para>
    /// <para><b>For Beginners:</b> This is like loading a saved snapshot of your complete trained model package.
    ///
    /// When you call LoadState:
    /// - The trained model and all its parameters are read from the stream
    /// - Training results and metrics are restored
    /// - Normalization settings are reapplied
    /// - All metadata is recovered
    ///
    /// After loading, the model package can:
    /// - Make predictions using the restored model
    /// - Access training history and metrics
    /// - Apply the same normalization as during training
    /// - Be deployed to production
    ///
    /// This is essential for:
    /// - Resuming interrupted optimization
    /// - Loading the best model after training
    /// - Deploying trained models to production
    /// - Knowledge distillation workflows
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="IOException">Thrown when there's an error reading from the stream.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the stream contains invalid or incompatible data.</exception>
    public virtual void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(stream));

        try
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            var data = ms.ToArray();

            if (data.Length == 0)
                throw new InvalidOperationException("Stream contains no data.");

            this.Deserialize(data);
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to read prediction model result state from stream: {ex.Message}", ex);
        }
        catch (InvalidOperationException)
        {
            // Re-throw InvalidOperationException from Deserialize
            throw;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to deserialize prediction model result state. The stream may contain corrupted or incompatible data: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Exports the model to ONNX format for cross-platform deployment.
    /// </summary>
    /// <param name="outputPath">The file path where the ONNX model will be saved.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX (Open Neural Network Exchange) is a universal format for AI models
    /// that works across different frameworks and platforms. Use this for:
    /// - Cross-platform deployment (Windows, Linux, macOS)
    /// - Cloud deployment
    /// - General-purpose production serving
    ///
    /// The exported model will use the export configuration specified during model building,
    /// or sensible defaults if no configuration was provided.
    ///
    /// Example:
    /// <code>
    /// var model = await new PredictionModelBuilder&lt;double&gt;()
    ///     .ConfigureExport(new ExportConfig { TargetPlatform = TargetPlatform.CPU })
    ///     .BuildAsync(x, y);
    /// model.ExportToOnnx("model.onnx");
    /// </code>
    /// </para>
    /// </remarks>
    public void ExportToOnnx(string outputPath)
    {
        if (Model == null)
            throw new InvalidOperationException("Cannot export: Model is null");

        var exportConfig = DeploymentConfiguration?.Export ?? new ExportConfig();

        var onnxConfig = new ExportConfiguration
        {
            ModelName = exportConfig.ModelName,
            TargetPlatform = exportConfig.TargetPlatform,
            OptimizeModel = exportConfig.OptimizeModel,
            BatchSize = exportConfig.BatchSize
        };

        var exporter = new OnnxModelExporter<T, TInput, TOutput>();
        exporter.Export(Model, outputPath, onnxConfig);
    }

    /// <summary>
    /// Exports the model to TensorRT format for high-performance inference on NVIDIA GPUs.
    /// </summary>
    /// <param name="outputPath">The file path where the TensorRT model will be saved.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> TensorRT is NVIDIA's high-performance inference engine.
    /// Use this when:
    /// - Deploying to servers with NVIDIA GPUs
    /// - Maximum inference speed is required
    /// - You need GPU-optimized inference
    ///
    /// TensorRT provides 2-4x faster inference than ONNX on NVIDIA hardware.
    /// Requires NVIDIA GPU to run.
    ///
    /// Example:
    /// <code>
    /// var model = await new PredictionModelBuilder&lt;double&gt;()
    ///     .ConfigureExport(new ExportConfig { TargetPlatform = TargetPlatform.TensorRT, Quantization = QuantizationMode.Float16 })
    ///     .BuildAsync(x, y);
    /// model.ExportToTensorRT("model.trt");
    /// </code>
    /// </para>
    /// </remarks>
    public void ExportToTensorRT(string outputPath)
    {
        if (Model == null)
            throw new InvalidOperationException("Cannot export: Model is null");

        var exportConfig = DeploymentConfiguration?.Export ?? new ExportConfig { TargetPlatform = TargetPlatform.TensorRT };

        var tensorRTConfig = new TensorRTConfiguration
        {
            MaxBatchSize = exportConfig.BatchSize,
            UseFp16 = exportConfig.Quantization == QuantizationMode.Float16,
            UseInt8 = exportConfig.Quantization == QuantizationMode.Int8
        };

        var converter = new TensorRTConverter<T, TInput, TOutput>();
        converter.ConvertToTensorRT(Model, outputPath, tensorRTConfig);
    }

    /// <summary>
    /// Exports the model to CoreML format for deployment on Apple devices (iOS, macOS).
    /// </summary>
    /// <param name="outputPath">The file path where the CoreML model will be saved.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> CoreML is Apple's machine learning framework.
    /// Use this when deploying to:
    /// - iPhone/iPad apps
    /// - macOS applications
    /// - Apple Watch apps
    ///
    /// CoreML models are optimized for Apple Silicon and Neural Engine,
    /// providing excellent performance on Apple devices.
    ///
    /// Example:
    /// <code>
    /// var model = await new PredictionModelBuilder&lt;double&gt;()
    ///     .ConfigureExport(new ExportConfig { TargetPlatform = TargetPlatform.CoreML, Quantization = QuantizationMode.Float16 })
    ///     .BuildAsync(x, y);
    /// model.ExportToCoreML("model.mlmodel");
    /// </code>
    /// </para>
    /// </remarks>
    public void ExportToCoreML(string outputPath)
    {
        if (Model == null)
            throw new InvalidOperationException("Cannot export: Model is null");

        var exportConfig = DeploymentConfiguration?.Export ?? new ExportConfig { TargetPlatform = TargetPlatform.CoreML };

        var coreMLConfig = new ExportConfiguration
        {
            ModelName = exportConfig.ModelName,
            TargetPlatform = exportConfig.TargetPlatform,
            OptimizeModel = exportConfig.OptimizeModel,
            BatchSize = exportConfig.BatchSize
        };

        var exporter = new CoreMLExporter<T, TInput, TOutput>();
        exporter.Export(Model, outputPath, coreMLConfig);
    }

    /// <summary>
    /// Exports the model to TensorFlow Lite format for mobile and edge deployment.
    /// </summary>
    /// <param name="outputPath">The file path where the TFLite model will be saved.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> TensorFlow Lite is designed for mobile and edge devices.
    /// Use this when deploying to:
    /// - Android apps
    /// - Raspberry Pi and edge devices
    /// - Embedded systems
    /// - IoT devices
    ///
    /// TFLite models are highly optimized for size and speed on resource-constrained devices.
    ///
    /// Example:
    /// <code>
    /// var model = await new PredictionModelBuilder&lt;double&gt;()
    ///     .ConfigureExport(new ExportConfig { TargetPlatform = TargetPlatform.TFLite, Quantization = QuantizationMode.Int8 })
    ///     .BuildAsync(x, y);
    /// model.ExportToTFLite("model.tflite");
    /// </code>
    /// </para>
    /// </remarks>
    public void ExportToTFLite(string outputPath)
    {
        if (Model == null)
            throw new InvalidOperationException("Cannot export: Model is null");

        var exportConfig = DeploymentConfiguration?.Export ?? new ExportConfig { TargetPlatform = TargetPlatform.TFLite };

        var tfliteConfig = new ExportConfiguration
        {
            ModelName = exportConfig.ModelName,
            TargetPlatform = exportConfig.TargetPlatform,
            OptimizeModel = exportConfig.OptimizeModel,
            BatchSize = exportConfig.BatchSize
        };

        var exporter = new TFLiteExporter<T, TInput, TOutput>();
        exporter.Export(Model, outputPath, tfliteConfig);
    }

    /// <summary>
    /// Creates a deployment runtime for production features like versioning, A/B testing, caching, and telemetry.
    /// </summary>
    /// <param name="modelPath">The path to the exported ONNX model file.</param>
    /// <param name="modelName">The name of the model (e.g., "HousePricePredictor").</param>
    /// <param name="version">The version identifier (e.g., "1.0.0").</param>
    /// <returns>A deployment runtime instance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The deployment runtime provides production features:
    /// - **Model Versioning**: Manage multiple model versions and roll back if needed
    /// - **A/B Testing**: Split traffic between different model versions
    /// - **Telemetry**: Track latency, throughput, errors, and metrics
    /// - **Caching**: Keep frequently-used models in memory for faster inference
    ///
    /// Before using this, you must first export your model to ONNX format.
    ///
    /// Example:
    /// <code>
    /// // Export model to ONNX
    /// model.ExportToOnnx("model.onnx");
    ///
    /// // Create runtime with deployed model
    /// var runtime = model.CreateDeploymentRuntime("model.onnx", "MyModel", "1.0.0");
    ///
    /// // Use runtime for inference with production features
    /// var prediction = await runtime.InferAsync("MyModel", "1.0.0", inputData);
    /// var stats = runtime.GetModelStatistics("MyModel");
    /// </code>
    /// </para>
    /// </remarks>
    public DeploymentRuntime<T> CreateDeploymentRuntime(string modelPath, string modelName, string version)
    {
        var runtimeConfig = new RuntimeConfiguration
        {
            EnableCaching = DeploymentConfiguration?.Caching?.Enabled ?? true,
            EnableTelemetry = DeploymentConfiguration?.Telemetry?.Enabled ?? true,
            EnableGpuAcceleration = DeploymentConfiguration?.Export?.TargetPlatform == TargetPlatform.GPU
                                    || DeploymentConfiguration?.Export?.TargetPlatform == TargetPlatform.TensorRT
        };

        var runtime = new DeploymentRuntime<T>(runtimeConfig);
        runtime.RegisterModel(modelName, version, modelPath);

        return runtime;
    }

    #region Reasoning Methods

    /// <summary>
    /// Solves a problem using advanced reasoning strategies like Chain-of-Thought, Tree-of-Thoughts, or Self-Consistency.
    /// </summary>
    /// <param name="problem">The problem or question to solve.</param>
    /// <param name="mode">The reasoning mode to use (default: Auto selects based on problem complexity).</param>
    /// <param name="config">Optional reasoning configuration (uses pre-configured settings if null).</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A complete reasoning result with answer, steps, and metrics.</returns>
    /// <exception cref="InvalidOperationException">Thrown when agent configuration is not set up.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method uses advanced AI reasoning to solve problems step by step,
    /// just like how a human would "show their work" on a complex problem.
    ///
    /// **What it does:**
    /// - Breaks down the problem into logical steps
    /// - Can explore multiple solution paths (Tree-of-Thoughts)
    /// - Can verify answers with multiple attempts (Self-Consistency)
    /// - Returns detailed reasoning chain for transparency
    ///
    /// **Reasoning modes:**
    /// - Auto: Automatically selects the best strategy
    /// - ChainOfThought: Step-by-step linear reasoning
    /// - TreeOfThoughts: Explores multiple paths, backtracks if needed
    /// - SelfConsistency: Solves multiple times, uses majority voting
    ///
    /// **Example:**
    /// <code>
    /// var result = await modelResult.ReasonAsync(
    ///     "If a train travels 60 mph for 2.5 hours, how far does it go?",
    ///     ReasoningMode.ChainOfThought
    /// );
    /// Console.WriteLine(result.FinalAnswer);  // "150 miles"
    /// Console.WriteLine(result.ReasoningChain);  // Shows step-by-step work
    /// </code>
    ///
    /// **Requirements:**
    /// - Agent configuration must be set (ConfigureAgentAssistance during building)
    /// - API key for LLM provider (OpenAI, Anthropic, etc.)
    /// </para>
    /// </remarks>
    public async Task<ReasoningResult<T>> ReasonAsync(
        string problem,
        ReasoningMode mode = ReasoningMode.Auto,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        if (AgentConfig == null || !AgentConfig.IsEnabled)
        {
            throw new InvalidOperationException(
                "Reasoning requires agent configuration. Use ConfigureAgentAssistance() during model building " +
                "to set up the LLM provider and API key required for reasoning capabilities.");
        }

        // Use stored config if none provided
        var effectiveConfig = config ?? ReasoningConfig ?? new ReasoningConfig();

        // Create chat model from agent config
        var chatModel = CreateChatModelFromAgentConfig();

        // Create internal Reasoner and delegate
        var reasoner = new Reasoner<T>(chatModel);
        return await reasoner.SolveAsync(problem, mode, effectiveConfig, cancellationToken);
    }

    /// <summary>
    /// Quickly solves a problem with minimal reasoning overhead for fast answers.
    /// </summary>
    /// <param name="problem">The problem or question to solve.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>The final answer as a string.</returns>
    /// <exception cref="InvalidOperationException">Thrown when agent configuration is not set up.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for simple problems where you need a fast answer
    /// without detailed reasoning steps. It's optimized for speed over thoroughness.
    ///
    /// **When to use:**
    /// - Simple math problems
    /// - Quick factual questions
    /// - When speed matters more than detailed explanation
    ///
    /// **Example:**
    /// <code>
    /// string answer = await modelResult.QuickReasonAsync("What is 15% of 240?");
    /// Console.WriteLine(answer);  // "36"
    /// </code>
    /// </para>
    /// </remarks>
    public async Task<string> QuickReasonAsync(
        string problem,
        CancellationToken cancellationToken = default)
    {
        if (AgentConfig == null || !AgentConfig.IsEnabled)
        {
            throw new InvalidOperationException(
                "Reasoning requires agent configuration. Use ConfigureAgentAssistance() during model building.");
        }

        var chatModel = CreateChatModelFromAgentConfig();
        var reasoner = new Reasoner<T>(chatModel);
        return await reasoner.QuickSolveAsync(problem, cancellationToken);
    }

    /// <summary>
    /// Performs deep, thorough reasoning on a complex problem using extensive exploration and verification.
    /// </summary>
    /// <param name="problem">The complex problem to analyze.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A comprehensive reasoning result with extensive exploration.</returns>
    /// <exception cref="InvalidOperationException">Thrown when agent configuration is not set up.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for complex problems that need careful analysis.
    /// It explores multiple approaches, verifies reasoning, and provides high-confidence answers.
    ///
    /// **What it does:**
    /// - Uses Tree-of-Thoughts to explore multiple solution paths
    /// - Applies verification at each step
    /// - Uses self-refinement to improve answers
    /// - Takes longer but produces more reliable results
    ///
    /// **When to use:**
    /// - Complex multi-step problems
    /// - Important decisions requiring high confidence
    /// - Problems with multiple valid approaches
    /// - When you need to understand all possibilities
    ///
    /// **Example:**
    /// <code>
    /// var result = await modelResult.DeepReasonAsync(
    ///     "Design an algorithm to find the shortest path in a weighted graph"
    /// );
    /// // Result includes multiple explored approaches and verification
    /// </code>
    /// </para>
    /// </remarks>
    public async Task<ReasoningResult<T>> DeepReasonAsync(
        string problem,
        CancellationToken cancellationToken = default)
    {
        if (AgentConfig == null || !AgentConfig.IsEnabled)
        {
            throw new InvalidOperationException(
                "Reasoning requires agent configuration. Use ConfigureAgentAssistance() during model building.");
        }

        var chatModel = CreateChatModelFromAgentConfig();
        var reasoner = new Reasoner<T>(chatModel);
        return await reasoner.DeepSolveAsync(problem, cancellationToken);
    }

    /// <summary>
    /// Solves a problem multiple times using different approaches and returns the consensus answer.
    /// </summary>
    /// <param name="problem">The problem to solve.</param>
    /// <param name="numAttempts">Number of independent solving attempts (default: 5).</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>The consensus result with voting statistics.</returns>
    /// <exception cref="InvalidOperationException">Thrown when agent configuration is not set up.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method solves the same problem multiple times independently
    /// and picks the most common answer. It's like asking 5 experts and going with the majority.
    ///
    /// **How it works:**
    /// - Solves the problem N times independently
    /// - Each attempt may use slightly different reasoning
    /// - Uses majority voting to pick the final answer
    /// - Reports confidence based on agreement level
    ///
    /// **When to use:**
    /// - Math problems where errors are common
    /// - When you need high confidence in the answer
    /// - Problems where reasoning paths can vary
    ///
    /// **Example:**
    /// <code>
    /// var result = await modelResult.ReasonWithConsensusAsync(
    ///     "What is the derivative of x^3?",
    ///     numAttempts: 5
    /// );
    /// // If 4 out of 5 attempts say "3x^2", that's the answer
    /// Console.WriteLine($"Consensus: {result.Metrics["consensus_ratio"]:P0}");
    /// </code>
    /// </para>
    /// </remarks>
    public async Task<ReasoningResult<T>> ReasonWithConsensusAsync(
        string problem,
        int numAttempts = 5,
        CancellationToken cancellationToken = default)
    {
        if (AgentConfig == null || !AgentConfig.IsEnabled)
        {
            throw new InvalidOperationException(
                "Reasoning requires agent configuration. Use ConfigureAgentAssistance() during model building.");
        }

        var chatModel = CreateChatModelFromAgentConfig();
        var reasoner = new Reasoner<T>(chatModel);
        return await reasoner.SolveWithConsensusAsync(problem, numAttempts, cancellationToken);
    }

    /// <summary>
    /// Creates a chat model from the agent configuration.
    /// </summary>
    private IChatModel<T> CreateChatModelFromAgentConfig()
    {
        if (AgentConfig == null)
            throw new InvalidOperationException("Agent configuration is required.");

        var apiKey = AgentKeyResolver.ResolveApiKey(
            AgentConfig.ApiKey,
            AgentConfig,
            AgentConfig.Provider);

        return AgentConfig.Provider switch
        {
            LLMProvider.OpenAI => new OpenAIChatModel<T>(apiKey),
            LLMProvider.Anthropic => new AnthropicChatModel<T>(apiKey),
            LLMProvider.AzureOpenAI => new AzureOpenAIChatModel<T>(
                AgentConfig.AzureEndpoint ?? throw new InvalidOperationException("Azure endpoint required"),
                apiKey,
                AgentConfig.AzureDeployment ?? "gpt-4"),
            _ => throw new ArgumentException($"Unknown provider: {AgentConfig.Provider}")
        };
    }

    #endregion

    #region IJitCompilable Implementation

    /// <summary>
    /// Gets whether the underlying model currently supports JIT compilation.
    /// </summary>
    /// <value>Returns true if the wrapped model implements IJitCompilable and supports JIT, false otherwise.</value>
    /// <remarks>
    /// <para>
    /// This property delegates to the wrapped model's SupportsJitCompilation property if the model
    /// implements IJitCompilable. If the model does not implement this interface or does not support
    /// JIT compilation, this returns false.
    /// </para>
    /// <para><b>For Beginners:</b> Whether you can use JIT compilation depends on the type of model you trained.
    ///
    /// Models that support JIT compilation (SupportsJitCompilation = true):
    /// - Linear regression models
    /// - Polynomial regression models
    /// - Ridge/Lasso regression models
    /// - Models using differentiable operations
    ///
    /// Models that do NOT support JIT (SupportsJitCompilation = false):
    /// - Decision trees
    /// - Random forests
    /// - Gradient boosted trees
    /// - Models using discrete logic
    ///
    /// If your model supports JIT:
    /// - Predictions will be 5-10x faster
    /// - The computation graph is compiled to optimized native code
    /// - You get this speedup automatically when calling Predict()
    ///
    /// If your model doesn't support JIT:
    /// - Predictions still work normally
    /// - No JIT acceleration, but still optimized for the model type
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when Model is null.</exception>
    public bool SupportsJitCompilation
    {
        get
        {
            if (Model == null)
            {
                throw new InvalidOperationException("Model is not initialized.");
            }

            // Check if the model implements IJitCompilable and supports JIT
            if (Model is IJitCompilable<T> jitModel)
            {
                return jitModel.SupportsJitCompilation;
            }

            // Model doesn't implement IJitCompilable
            return false;
        }
    }

    /// <summary>
    /// Exports the underlying model's computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the model's prediction.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Model is null.</exception>
    /// <exception cref="NotSupportedException">Thrown when the underlying model does not support JIT compilation.</exception>
    /// <remarks>
    /// <para>
    /// This method delegates to the wrapped model's ExportComputationGraph method if the model
    /// implements IJitCompilable and supports JIT compilation. If the model does not implement
    /// this interface or does not support JIT, this throws NotSupportedException.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a "recipe" of your model's calculations for JIT compilation.
    ///
    /// If your model supports JIT (SupportsJitCompilation = true):
    /// - This method creates a computation graph from your model
    /// - The graph represents all the mathematical operations your model performs
    /// - The JIT compiler uses this to create fast optimized code
    ///
    /// If your model doesn't support JIT (SupportsJitCompilation = false):
    /// - This method will throw an exception
    /// - Check SupportsJitCompilation before calling this
    /// - Decision trees, random forests, etc. cannot export computation graphs
    ///
    /// You typically don't call this method directly. It's used internally by:
    /// - PredictionModelBuilder when building models with JIT enabled
    /// - The prediction pipeline to compile models for faster inference
    ///
    /// Example of what happens inside:
    /// - Linear model: Creates graph with MatMul(X, Coefficients) + Intercept
    /// - Neural network: Creates graph with all layers and activations
    /// - Decision tree: Throws exception - cannot create computation graph
    /// </para>
    /// </remarks>
    public AiDotNet.Autodiff.ComputationNode<T> ExportComputationGraph(List<AiDotNet.Autodiff.ComputationNode<T>> inputNodes)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        // Check if the model implements IJitCompilable
        if (Model is IJitCompilable<T> jitModel)
        {
            // Check if it actually supports JIT before delegating
            if (!jitModel.SupportsJitCompilation)
            {
                throw new NotSupportedException(
                    $"The underlying model type ({Model.GetType().Name}) does not support JIT compilation. " +
                    "Check SupportsJitCompilation property before calling ExportComputationGraph.");
            }

            // Delegate to the wrapped model
            return jitModel.ExportComputationGraph(inputNodes);
        }

        // Model doesn't implement IJitCompilable at all
        throw new NotSupportedException(
            $"The underlying model type ({Model.GetType().Name}) does not implement IJitCompilable<T>. " +
            "JIT compilation is only supported for models that use differentiable computation graphs, such as " +
            "linear models, polynomial models, and neural networks. Tree-based models (decision trees, random forests, " +
            "gradient boosting) cannot be JIT compiled due to their discrete branching logic.");
    }

    #endregion
}
