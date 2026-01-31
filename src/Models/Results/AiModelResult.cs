global using Newtonsoft.Json;
global using Formatting = Newtonsoft.Json.Formatting;
using System.Linq;
using System.Runtime.CompilerServices;
using AiDotNet.AdversarialRobustness.Safety;
using AiDotNet.Agents;
using AiDotNet.Benchmarking;
using AiDotNet.Benchmarking.Models;
using AiDotNet.CheckpointManagement;
using AiDotNet.Configuration;
using AiDotNet.Data.Structures;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Deployment.Export;
using AiDotNet.Deployment.Export.Onnx;
using AiDotNet.Deployment.Mobile.CoreML;
using AiDotNet.Deployment.Mobile.TensorFlowLite;
using AiDotNet.Deployment.Runtime;
using AiDotNet.Deployment.TensorRT;
using AiDotNet.Diagnostics;
using AiDotNet.Enums;
using AiDotNet.ExperimentTracking;
using AiDotNet.Helpers;
using AiDotNet.Inference;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.Interpretability.Explainers;
using AiDotNet.Interpretability.Interfaces;
using AiDotNet.LanguageModels;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.PromptEngineering;
using AiDotNet.PromptEngineering.Analysis;
using AiDotNet.PromptEngineering.Compression;
using AiDotNet.Reasoning;
using AiDotNet.Reasoning.Models;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using AiDotNet.Serialization;
using AiDotNet.Tokenization.Configuration;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using AiDotNet.TrainingMonitoring;

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
public partial class AiModelResult<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
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
    [JsonIgnore]
    internal IFullModel<T, TInput, TOutput>? Model { get; private set; }

    /// <summary>
    /// Gets the options used to create this model result.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This stores the full configuration options used when building the model,
    /// including TTA configuration, preprocessing settings, and other optional features.
    /// May be null when deserialized from storage (legacy models without options).
    /// </para>
    /// </remarks>
    [JsonIgnore]
    internal AiModelResultOptions<T, TInput, TOutput>? Options { get; private set; }

    /// <summary>
    /// Gets the serialized model payload for the facade-hidden <see cref="Model"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is stored separately from the <see cref="Model"/> instance so we can persist models using
    /// their own binary serializer (<see cref="IModelSerializer"/>), while keeping the public-facing
    /// surface area of the result object stable and IP-conscious.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the saved "snapshot" of the model itself. When loading, we recreate
    /// the model object and then restore it from these bytes.
    /// </para>
    /// </remarks>
    [JsonProperty]
    internal byte[] SerializedModelData { get; private set; } = [];

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
    [JsonProperty]
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
    [JsonProperty]
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
    [JsonProperty]
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
    /// Gets or sets the interpretability options for model explanation methods.
    /// </summary>
    /// <value>The configured interpretability options, or null if not configured.</value>
    internal InterpretabilityOptions? InterpretabilityOptions { get; private set; }

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
    /// Gets the AutoML summary for this model, if AutoML was used during building.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property contains a redacted summary of the AutoML search (trial outcomes and scores),
    /// and intentionally excludes hyperparameter values and other proprietary details.
    /// </para>
    /// <para><b>For Beginners:</b> If you enabled AutoML, this tells you how the automatic search went.</para>
    /// </remarks>
    public AutoMLRunSummary? AutoMLSummary { get; internal set; }

    /// <summary>
    /// Gets the most recent benchmark report produced for this model, if available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is populated when benchmarking is enabled through the facade (for example,
    /// <c>AiModelBuilder.ConfigureBenchmarking(...)</c>) or when
    /// <see cref="EvaluateBenchmarksAsync"/> is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "report card" from running standardized benchmark suites.</para>
    /// </remarks>
    public BenchmarkReport? BenchmarkReport { get; internal set; }

    /// <summary>
    /// Gets the profiling report captured during training and/or inference, if profiling was enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This helps you see where time was spent during model build and prediction.</para>
    /// </remarks>
    public ProfileReport? ProfilingReport { get; internal set; }

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

    [JsonProperty]
    private AiDotNet.Configuration.InferenceOptimizationConfig? InferenceOptimizationConfig { get; set; }

    [JsonIgnore]
    private readonly object _inferenceOptimizationLock = new();

    [JsonIgnore]
    private InferenceOptimizer<T>? _inferenceOptimizer;

    [JsonIgnore]
    private NeuralNetworkBase<T>? _inferenceOptimizedNeuralModel;

    [JsonIgnore]
    private bool _inferenceOptimizationsInitialized;

    // Serving assembly uses InternalsVisibleTo; keep this internal to avoid expanding user-facing API surface.
    internal AiDotNet.Configuration.InferenceOptimizationConfig? GetInferenceOptimizationConfigForServing()
        => InferenceOptimizationConfig;

    internal ISafetyFilter<T>? SafetyFilter { get; private set; }

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

    #region Prompt Engineering Properties

    /// <summary>
    /// Gets or sets the prompt template used for generating prompts during inference.
    /// </summary>
    /// <value>An implementation of IPromptTemplate for structured prompt generation, or null if not configured.</value>
    /// <remarks>
    /// <para>
    /// The prompt template defines the structure and format of prompts sent to language models.
    /// It supports variable interpolation, allowing dynamic content to be inserted into a predefined template.
    /// </para>
    /// <para><b>For Beginners:</b> A prompt template is like a form letter with blanks that get filled in.
    ///
    /// Instead of writing a complete prompt every time, you create a template:
    /// <code>
    /// "Translate {text} from {source_language} to {target_language}"
    /// </code>
    ///
    /// Then fill in the blanks at runtime:
    /// <code>
    /// var result = model.FormatPrompt(new {
    ///     text = "Hello",
    ///     source_language = "English",
    ///     target_language = "Spanish"
    /// });
    /// // Result: "Translate Hello from English to Spanish"
    /// </code>
    ///
    /// Benefits:
    /// - Consistent prompt structure
    /// - Easy to update prompts without changing code
    /// - Supports complex multi-part prompts (system, user, assistant)
    /// </para>
    /// </remarks>
    internal IPromptTemplate? PromptTemplate { get; private set; }

    /// <summary>
    /// Gets or sets the prompt chain used for multi-step inference workflows.
    /// </summary>
    /// <value>An implementation of IPromptChain for sequential prompt processing, or null if not configured.</value>
    /// <remarks>
    /// <para>
    /// Prompt chains enable complex multi-step workflows where the output of one prompt
    /// becomes the input to the next. This supports patterns like:
    /// - Sequential processing (translate → summarize → format)
    /// - Conditional branching based on intermediate results
    /// - Parallel execution of independent steps
    /// - Map-reduce patterns for processing multiple items
    /// </para>
    /// <para><b>For Beginners:</b> A prompt chain is like an assembly line where each step does one thing.
    ///
    /// Example workflow:
    /// <code>
    /// // Chain: Translate → Summarize → Format
    /// Step 1: Translate document from Spanish to English
    /// Step 2: Summarize the translated document
    /// Step 3: Format the summary as bullet points
    /// </code>
    ///
    /// Each step takes the previous step's output as input, making complex tasks manageable.
    /// Chains can also:
    /// - Run steps in parallel when they don't depend on each other
    /// - Branch based on conditions (if sentiment is negative, escalate)
    /// - Loop over collections (summarize each chapter)
    /// </para>
    /// </remarks>
    internal IChain<string, string>? PromptChain { get; private set; }

    /// <summary>
    /// Gets or sets the prompt optimizer used for automatic prompt improvement.
    /// </summary>
    /// <value>An implementation of IPromptOptimizer&lt;T&gt; for prompt optimization, or null if not configured.</value>
    /// <remarks>
    /// <para>
    /// The prompt optimizer automatically improves prompts to achieve better results.
    /// Different optimization strategies include:
    /// - Gradient-based optimization (GRIPS, APE)
    /// - Evolutionary algorithms (genetic algorithms, particle swarm)
    /// - Bayesian optimization for efficient hyperparameter tuning
    /// - OPRO (Optimization by PRompting) using LLMs to improve prompts
    /// </para>
    /// <para><b>For Beginners:</b> A prompt optimizer automatically improves your prompts.
    ///
    /// Instead of manually tweaking prompts through trial and error, the optimizer:
    /// <code>
    /// // Before optimization:
    /// "Summarize this text"
    /// // Accuracy: 65%
    ///
    /// // After optimization:
    /// "Provide a concise summary of the main points in the following text,
    /// focusing on key facts and conclusions. Be specific and avoid generalities."
    /// // Accuracy: 89%
    /// </code>
    ///
    /// The optimizer uses various strategies to find the best prompt wording,
    /// structure, and examples for your specific task.
    /// </para>
    /// </remarks>
    internal IPromptOptimizer<T>? PromptOptimizer { get; private set; }

    /// <summary>
    /// Gets or sets the few-shot example selector for dynamic example selection.
    /// </summary>
    /// <value>An implementation of IFewShotExampleSelector for selecting examples, or null if not configured.</value>
    /// <remarks>
    /// <para>
    /// The few-shot example selector dynamically chooses the most relevant examples
    /// to include in prompts based on the current input. Selection strategies include:
    /// - Semantic similarity (embedding-based matching)
    /// - MMR (Maximal Marginal Relevance) for diverse examples
    /// - N-gram overlap for lexical similarity
    /// - Random selection for baseline comparison
    /// </para>
    /// <para><b>For Beginners:</b> Few-shot learning shows the AI examples of what you want.
    ///
    /// Instead of explaining in words, you show examples:
    /// <code>
    /// // Without examples:
    /// "Classify the sentiment of this review"
    ///
    /// // With few-shot examples:
    /// "Classify sentiment:
    ///  Review: 'Loved it!' → Positive
    ///  Review: 'Terrible waste of money' → Negative
    ///  Review: 'It was okay, nothing special' → Neutral
    ///  Review: 'Best purchase ever!'"
    /// </code>
    ///
    /// The example selector automatically picks the best examples for each input,
    /// choosing examples that are similar or relevant to the current query.
    /// </para>
    /// </remarks>
    internal IFewShotExampleSelector<T>? FewShotExampleSelector { get; private set; }

    /// <summary>
    /// Gets or sets the prompt analyzer for measuring prompt metrics.
    /// </summary>
    /// <value>An implementation of IPromptAnalyzer for prompt analysis, or null if not configured.</value>
    /// <remarks>
    /// <para>
    /// The prompt analyzer provides metrics and validation for prompts before
    /// they are sent to the model. Analysis includes:
    /// - Token counting for cost estimation
    /// - Complexity scoring for prompt difficulty
    /// - Pattern detection (instruction, question, few-shot, etc.)
    /// - Validation for potential issues (injection, length, etc.)
    /// </para>
    /// <para><b>For Beginners:</b> The analyzer is like a spell-checker for prompts.
    ///
    /// Before sending a prompt to the AI (which costs money), the analyzer tells you:
    /// <code>
    /// var metrics = model.AnalyzePrompt("Your prompt here...");
    ///
    /// Console.WriteLine($"Tokens: {metrics.TokenCount}");      // How many tokens
    /// Console.WriteLine($"Cost: ${metrics.EstimatedCost}");    // Estimated API cost
    /// Console.WriteLine($"Complexity: {metrics.ComplexityScore}"); // How complex (0-1)
    /// Console.WriteLine($"Patterns: {string.Join(", ", metrics.DetectedPatterns)}");
    /// </code>
    ///
    /// This helps you:
    /// - Avoid exceeding token limits
    /// - Estimate and control costs
    /// - Catch potential issues before API calls
    /// </para>
    /// </remarks>
    internal IPromptAnalyzer? PromptAnalyzer { get; private set; }

    /// <summary>
    /// Gets or sets the prompt compressor for reducing prompt length.
    /// </summary>
    /// <value>An implementation of IPromptCompressor for prompt compression, or null if not configured.</value>
    /// <remarks>
    /// <para>
    /// The prompt compressor reduces prompt length while preserving semantic meaning.
    /// Compression strategies include:
    /// - Redundancy removal (eliminate repetitive phrases)
    /// - Summarization (condense verbose content)
    /// - Selective context (keep most relevant information)
    /// - Token optimization (use shorter synonyms)
    /// </para>
    /// <para><b>For Beginners:</b> The compressor makes prompts shorter to save money.
    ///
    /// Shorter prompts = fewer tokens = lower costs:
    /// <code>
    /// // Before compression (80 tokens):
    /// "Please analyze the following document and provide a summary.
    ///  The document that you need to analyze is provided below.
    ///  When you analyze the document, focus on the main points."
    ///
    /// // After compression (25 tokens):
    /// "Summarize this document, focusing on main points:"
    /// </code>
    ///
    /// The compressor automatically:
    /// - Removes redundant phrases
    /// - Shortens verbose instructions
    /// - Preserves essential meaning
    /// - Reports compression metrics (tokens saved, cost savings)
    /// </para>
    /// </remarks>
    internal IPromptCompressor? PromptCompressor { get; private set; }

    #endregion

    #region Training Infrastructure Properties

    /// <summary>
    /// Gets or sets the experiment run associated with this model.
    /// </summary>
    /// <value>The experiment run that produced this model, or null if experiment tracking was not used.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides direct access to the training run for additional logging.
    ///
    /// You can use this to:
    /// - Log additional metrics after training (e.g., production performance)
    /// - Add notes about the model's behavior in production
    /// - Record artifacts like deployment logs or user feedback
    ///
    /// Example:
    /// <code>
    /// if (result.ExperimentRun != null)
    /// {
    ///     result.ExperimentRun.LogMetric("production_accuracy", 0.92);
    ///     result.ExperimentRun.AddNote("Deployed to production on 2024-01-15");
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    [JsonIgnore]
    internal IExperimentRun<T>? ExperimentRun { get; private set; }

    /// <summary>
    /// Gets or sets the experiment tracker used during training.
    /// </summary>
    /// <value>The experiment tracker for accessing other runs and experiments, or null if not configured.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This gives you access to the experiment tracking system.
    ///
    /// You can use this to:
    /// - Compare this model with other training runs
    /// - Find the best-performing model from an experiment
    /// - Start new training runs based on this one
    ///
    /// Example:
    /// <code>
    /// if (result.ExperimentTracker != null)
    /// {
    ///     var allRuns = result.ExperimentTracker.ListRuns(experimentId);
    ///     var bestRun = allRuns.OrderByDescending(r => r.GetLatestMetric("accuracy")).First();
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    [JsonIgnore]
    internal IExperimentTracker<T>? ExperimentTracker { get; private set; }

    /// <summary>
    /// Gets or sets the checkpoint manager for model persistence operations.
    /// </summary>
    /// <value>The checkpoint manager for saving and loading model checkpoints, or null if not configured.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This manages saved copies of your model.
    ///
    /// You can use this to:
    /// - Save the model after making changes (like fine-tuning)
    /// - List all saved checkpoints
    /// - Load different versions of your model
    /// - Clean up old checkpoints to save disk space
    ///
    /// Example:
    /// <code>
    /// if (result.CheckpointManager != null)
    /// {
    ///     // Save current state
    ///     result.CheckpointManager.SaveCheckpoint("after_finetuning", model);
    ///
    ///     // List available checkpoints
    ///     var checkpoints = result.CheckpointManager.ListCheckpoints();
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    [JsonIgnore]
    internal ICheckpointManager<T, TInput, TOutput>? CheckpointManager { get; private set; }

    /// <summary>
    /// Gets or sets the model registry for version and lifecycle management.
    /// </summary>
    /// <value>The model registry for managing model versions and stages, or null if not configured.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like a version control system for your models.
    ///
    /// You can use this to:
    /// - Promote this model from "Staging" to "Production"
    /// - Register fine-tuned versions as new model versions
    /// - Archive old models that are no longer needed
    /// - Compare performance across model versions
    ///
    /// Example:
    /// <code>
    /// if (result.ModelRegistry != null)
    /// {
    ///     // Promote to production
    ///     result.ModelRegistry.TransitionModelStage("my-model", 1, ModelStage.Production);
    ///
    ///     // Get current production model
    ///     var prodModel = result.ModelRegistry.GetProductionModel("my-model");
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    [JsonIgnore]
    internal IModelRegistry<T, TInput, TOutput>? ModelRegistry { get; private set; }

    /// <summary>
    /// Gets or sets the training monitor for accessing training diagnostics.
    /// </summary>
    /// <value>The training monitor containing training history and diagnostics, or null if not configured.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This gives you insights into how training went.
    ///
    /// You can use this to:
    /// - View learning curves (loss over time)
    /// - Check for signs of overfitting
    /// - Analyze gradient flow during training
    /// - Export training charts and reports
    ///
    /// Example:
    /// <code>
    /// if (result.TrainingMonitor != null)
    /// {
    ///     var history = result.TrainingMonitor.GetMetricsHistory();
    ///     var finalLoss = history["loss"].Last();
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    [JsonIgnore]
    internal ITrainingMonitor<T>? TrainingMonitor { get; private set; }

    /// <summary>
    /// Gets or sets the hyperparameter optimization result.
    /// </summary>
    /// <value>Complete hyperparameter optimization results including all trials, or null if optimization was not used.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If an optimizer searched for the best settings,
    /// this contains all the configurations it tried and how well each performed.
    ///
    /// You can use this to:
    /// - See which hyperparameters were most important
    /// - Find patterns in what made training successful
    /// - Continue optimization from where it left off
    ///
    /// Example:
    /// <code>
    /// if (result.HyperparameterOptimizationResult != null)
    /// {
    ///     var bestParams = result.HyperparameterOptimizationResult.BestParameters;
    ///     Console.WriteLine($"Best learning rate: {bestParams["learning_rate"]}");
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    internal HyperparameterOptimizationResult<T>? HyperparameterOptimizationResult { get; private set; }

    /// <summary>
    /// Gets or sets the experiment run ID from experiment tracking.
    /// </summary>
    /// <value>The unique identifier for the training run, or null if experiment tracking was not used.</value>
    internal string? ExperimentRunId { get; private set; }

    /// <summary>
    /// Gets or sets the experiment ID that this run belongs to.
    /// </summary>
    /// <value>The experiment ID grouping related training runs, or null if experiment tracking was not used.</value>
    internal string? ExperimentId { get; private set; }

    /// <summary>
    /// Gets or sets the model version from the model registry.
    /// </summary>
    /// <value>The version number assigned to this model, or null if registry was not used.</value>
    internal int? ModelVersion { get; private set; }

    /// <summary>
    /// Gets or sets the registered model name in the model registry.
    /// </summary>
    /// <value>The name under which this model is registered, or null if registry was not used.</value>
    internal string? RegisteredModelName { get; private set; }

    /// <summary>
    /// Gets or sets the checkpoint path where the model was saved during training.
    /// </summary>
    /// <value>The path to the best or latest checkpoint, or null if checkpointing was not used.</value>
    internal string? CheckpointPath { get; private set; }

    /// <summary>
    /// Gets or sets the data version hash for the training data.
    /// </summary>
    /// <value>A hash uniquely identifying the training data, or null if data versioning was not used.</value>
    internal string? DataVersionHash { get; private set; }

    /// <summary>
    /// Gets or sets the hyperparameter optimization trial ID.
    /// </summary>
    /// <value>The trial ID that produced this model, or null if optimization was not used.</value>
    internal int? HyperparameterTrialId { get; private set; }

    /// <summary>
    /// Gets or sets the hyperparameters used for training.
    /// </summary>
    /// <value>A dictionary of hyperparameter names to values, or null if not tracked.</value>
    internal Dictionary<string, object>? Hyperparameters { get; private set; }

    /// <summary>
    /// Gets or sets the training metrics history.
    /// </summary>
    /// <value>A history of metrics recorded during training, or null if not tracked.</value>
    internal Dictionary<string, List<double>>? TrainingMetricsHistory { get; private set; }

    #endregion

    /// <summary>
    /// Initializes a new instance of the AiModelResult class using an options object for clean configuration.
    /// </summary>
    /// <param name="options">The configuration options containing all settings for the prediction model result.</param>
    /// <remarks>
    /// <para>
    /// This constructor provides a cleaner API by accepting a single options object instead of many parameters.
    /// It supports two initialization paths:
    /// </para>
    /// <list type="bullet">
    ///   <item><description><b>Standard path:</b> Set OptimizationResult and NormalizationInfo for regular trained models</description></item>
    ///   <item><description><b>Meta-learning path:</b> Set MetaLearner and MetaTrainingResult for meta-trained models</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> This is the only way to create a AiModelResult.
    ///
    /// For a standard trained model:
    /// <code>
    /// var options = new AiModelResultOptions&lt;double, double[], double&gt;
    /// {
    ///     OptimizationResult = optimizationResult,
    ///     NormalizationInfo = normInfo,
    ///     BiasDetector = myBiasDetector
    /// };
    /// var result = new AiModelResult&lt;double, double[], double&gt;(options);
    /// </code>
    ///
    /// For a meta-trained model:
    /// <code>
    /// var options = new AiModelResultOptions&lt;double, double[], double&gt;
    /// {
    ///     MetaLearner = metaLearner,
    ///     MetaTrainingResult = metaResult,
    ///     LoRAConfiguration = loraConfig
    /// };
    /// var result = new AiModelResult&lt;double, double[], double&gt;(options);
    /// </code>
    ///
    /// Benefits:
    /// - Only set the options you need
    /// - Clear property names make code self-documenting
    /// - Easy to add new options without breaking existing code
    /// - IDE auto-completion helps discover available options
    /// </para>
    /// </remarks>
    internal AiModelResult(AiModelResultOptions<T, TInput, TOutput> options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        // Store the options for use by partial classes (e.g., TTA augmentation)
        Options = options;

        // Determine initialization path: meta-learning or standard
        var isMetaLearningPath = options.MetaLearner is not null;

        if (isMetaLearningPath)
        {
            // Meta-learning path: MetaLearner and MetaTrainingResult are required
            if (options.MetaLearner is null)
            {
                throw new ArgumentNullException(nameof(options), "MetaLearner cannot be null for meta-learning path");
            }

            if (options.MetaTrainingResult is null)
            {
                throw new ArgumentNullException(nameof(options), "MetaTrainingResult cannot be null for meta-learning path");
            }

            // Get model from meta-learner
            Model = options.MetaLearner.BaseModel;
            MetaLearner = options.MetaLearner;
            MetaTrainingResult = options.MetaTrainingResult;

            // Create default OptimizationResult and NormalizationInfo for consistency
            OptimizationResult = options.OptimizationResult ?? new OptimizationResult<T, TInput, TOutput>();
            NormalizationInfo = options.NormalizationInfo ?? new NormalizationInfo<T, TInput, TOutput>();
        }
        else
        {
            // Standard path: OptimizationResult and NormalizationInfo are required
            if (options.OptimizationResult is null)
            {
                throw new ArgumentNullException(nameof(options), "OptimizationResult cannot be null");
            }

            if (options.NormalizationInfo is null)
            {
                throw new ArgumentNullException(nameof(options), "NormalizationInfo cannot be null");
            }

            Model = options.OptimizationResult.BestSolution;
            OptimizationResult = options.OptimizationResult;
            NormalizationInfo = options.NormalizationInfo;
            MetaLearner = options.MetaLearner;
            MetaTrainingResult = options.MetaTrainingResult;
        }

        ModelMetaData = Model?.GetModelMetadata() ?? new();

        // Ethical AI and fairness
        BiasDetector = options.BiasDetector;
        FairnessEvaluator = options.FairnessEvaluator;
        InterpretabilityOptions = options.InterpretabilityOptions;

        // Tokenization
        Tokenizer = options.Tokenizer;
        TokenizationConfig = options.TokenizationConfig;

        // Program Synthesis (optional)
        ProgramSynthesisModel = options.ProgramSynthesisModel;
        ProgramSynthesisServingClientOptions = options.ProgramSynthesisServingClientOptions;
        ProgramSynthesisServingClient =
            options.ProgramSynthesisServingClient ??
            (options.ProgramSynthesisServingClientOptions is not null
                ? new AiDotNet.ProgramSynthesis.Serving.ProgramSynthesisServingClient(options.ProgramSynthesisServingClientOptions)
                : null);

        // RAG (Retrieval Augmented Generation)
        RagRetriever = options.RagRetriever;
        RagReranker = options.RagReranker;
        RagGenerator = options.RagGenerator;
        QueryProcessors = options.QueryProcessors;

        // Graph RAG
        KnowledgeGraph = options.KnowledgeGraph;
        GraphStore = options.GraphStore;
        HybridGraphRetriever = options.HybridGraphRetriever;

        // Cross-validation
        CrossValidationResult = options.CrossValidationResult;

        // AutoML (redacted summary)
        AutoMLSummary = options.AutoMLSummary;

        // Fine-tuning and adaptation
        LoRAConfiguration = options.LoRAConfiguration;

        // Agent assistance
        AgentConfig = options.AgentConfig;
        AgentRecommendation = options.AgentRecommendation;

        // Deployment
        DeploymentConfiguration = options.DeploymentConfiguration;
        JitCompiledFunction = options.JitCompiledFunction;
        InferenceOptimizationConfig = options.InferenceOptimizationConfig;

        // Safety & Robustness (enabled by default; opt-out via options)
        var safetyConfig = options.SafetyFilterConfiguration;
        SafetyFilter = safetyConfig?.Enabled == false
            ? null
            : safetyConfig?.Filter ?? new SafetyFilter<T>(safetyConfig?.Options ?? new SafetyFilterOptions<T>());

        // Reasoning
        ReasoningConfig = options.ReasoningConfig;

        // Prompt Engineering
        PromptTemplate = options.PromptTemplate;
        PromptChain = options.PromptChain;
        PromptOptimizer = options.PromptOptimizer;
        FewShotExampleSelector = options.FewShotExampleSelector;
        PromptAnalyzer = options.PromptAnalyzer;
        PromptCompressor = options.PromptCompressor;

        // Diagnostics / benchmarking
        ProfilingReport = options.ProfilingReport;
        BenchmarkReport = options.BenchmarkReport;

        // Training Infrastructure
        ExperimentRun = options.ExperimentRun;
        ExperimentTracker = options.ExperimentTracker;
        CheckpointManager = options.CheckpointManager;
        ModelRegistry = options.ModelRegistry;
        TrainingMonitor = options.TrainingMonitor;
        HyperparameterOptimizationResult = options.HyperparameterOptimizationResult;
        ExperimentRunId = options.ExperimentRunId;
        ExperimentId = options.ExperimentId;
        ModelVersion = options.ModelVersion;
        RegisteredModelName = options.RegisteredModelName;
        CheckpointPath = options.CheckpointPath;
        DataVersionHash = options.DataVersionHash;
        HyperparameterTrialId = options.HyperparameterTrialId;
        Hyperparameters = options.Hyperparameters;
        TrainingMetricsHistory = options.TrainingMetricsHistory;
    }

    /// <summary>
    /// Initializes a new instance of the AiModelResult class with default values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new AiModelResult instance with default values for all properties. It is primarily 
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
    internal AiModelResult()
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
    /// Gets federated learning training metadata if this model was produced via federated learning.
    /// </summary>
    /// <returns>The federated learning metadata, or null if not available.</returns>
    public FederatedLearningMetadata? GetFederatedLearningMetadata()
    {
        var metadata = GetModelMetadata();
        if (metadata.Properties != null &&
            metadata.Properties.TryGetValue(FederatedLearningMetadata.MetadataKey, out var value))
        {
            return value as FederatedLearningMetadata;
        }

        return null;
    }

    /// <summary>
    /// Runs benchmark suites against this model using the unified benchmark runner.
    /// </summary>
    /// <param name="options">Benchmarking options (suites, sample size, failure policy).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A structured benchmark report.</returns>
    /// <remarks>
    /// <para>
    /// This method is facade-first: users select benchmark suites via enums and receive a structured report.
    /// It avoids requiring users to manually wire up benchmark implementations.
    /// </para>
    /// </remarks>
    public async Task<BenchmarkReport> EvaluateBenchmarksAsync(
        BenchmarkingOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var effectiveOptions = options ?? new BenchmarkingOptions();
        var report = await BenchmarkRunner.RunAsync(this, effectiveOptions, cancellationToken).ConfigureAwait(false);

        if (effectiveOptions.AttachReportToResult)
        {
            BenchmarkReport = report;
        }

        return report;
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

        var dataForPrediction = newData;
        if (SafetyFilter != null && newData is Vector<T> vectorInput && typeof(TInput) == typeof(Vector<T>))
        {
            var validation = SafetyFilter.ValidateInput(vectorInput);
            if (!validation.IsValid)
            {
                var issues = validation.Issues.Count > 0
                    ? string.Join("; ", validation.Issues.Select(i => $"{i.Type}:{i.Severity}"))
                    : "Unknown safety validation failure.";
                throw new InvalidOperationException($"Safety validation failed: {issues}");
            }

            if (validation.SanitizedInput != null)
            {
                var sanitized = validation.SanitizedInput;
                dataForPrediction = Unsafe.As<Vector<T>, TInput>(ref sanitized);
            }
        }
        else if (SafetyFilter != null && newData is Matrix<T> matrixInput && typeof(TInput) == typeof(Matrix<T>))
        {
            var sanitizedMatrix = ValidateAndSanitizeMatrix(matrixInput);
            dataForPrediction = Unsafe.As<Matrix<T>, TInput>(ref sanitizedMatrix);
        }

        var (normalizedNewData, _) = NormalizationInfo.Normalizer.NormalizeInput(dataForPrediction);

        // Use JIT-compiled function if available for 5-10x faster predictions
        TOutput normalizedPredictions;

        // INFERENCE OPTIMIZATION PATH: apply configured inference optimizations for neural network models
        if (InferenceOptimizationConfig != null &&
            Model is NeuralNetworkBase<T> neuralModel &&
            normalizedNewData is Tensor<T> inputTensor)
        {
            var optimizedNeuralModel = EnsureStatelessInferenceOptimizationsInitialized(neuralModel);
            if (optimizedNeuralModel != null)
            {
                var optimizedOutput = optimizedNeuralModel.Predict(inputTensor);
                if (optimizedOutput is TOutput output)
                {
                    normalizedPredictions = output;
                }
                else
                {
                    // Fallback to the wrapped model if type mismatch occurs
                    normalizedPredictions = Model.Predict(normalizedNewData);
                }

                return NormalizationInfo.Normalizer.Denormalize(normalizedPredictions, NormalizationInfo.YParams);
            }
        }

        if (JitCompiledFunction != null && normalizedNewData is Tensor<T> inputTensor2)
        {
            // JIT PATH: Use compiled function for accelerated inference
            var jitResult = JitCompiledFunction(new[] { inputTensor2 });
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

        var denormalized = NormalizationInfo.Normalizer.Denormalize(normalizedPredictions, NormalizationInfo.YParams);

        if (SafetyFilter != null && denormalized is Vector<T> vectorOutput && typeof(TOutput) == typeof(Vector<T>))
        {
            var filtered = SafetyFilter.FilterOutput(vectorOutput);
            if (filtered.WasModified || !filtered.IsSafe)
            {
                var filteredOutput = filtered.FilteredOutput;
                return Unsafe.As<Vector<T>, TOutput>(ref filteredOutput);
            }
        }
        else if (SafetyFilter != null && denormalized is Matrix<T> matrixOutput && typeof(TOutput) == typeof(Matrix<T>))
        {
            var filteredMatrix = FilterMatrixOutput(matrixOutput);
            return Unsafe.As<Matrix<T>, TOutput>(ref filteredMatrix);
        }

        return denormalized;
    }

    private Matrix<T> ValidateAndSanitizeMatrix(Matrix<T> input)
    {
        if (SafetyFilter == null)
        {
            return input;
        }

        if (input.Rows == 0 || input.Columns == 0)
        {
            return input;
        }

        bool anySanitized = false;
        var sanitizedRows = new Vector<T>?[input.Rows];

        for (int i = 0; i < input.Rows; i++)
        {
            var row = input.GetRow(i);
            var validation = SafetyFilter.ValidateInput(row);
            if (!validation.IsValid)
            {
                var issues = validation.Issues.Count > 0
                    ? string.Join("; ", validation.Issues.Select(issue => $"{issue.Type}:{issue.Severity}"))
                    : "Unknown safety validation failure.";
                throw new InvalidOperationException($"Safety validation failed (row {i}): {issues}");
            }

            sanitizedRows[i] = validation.SanitizedInput;
            anySanitized |= validation.SanitizedInput != null;
        }

        if (!anySanitized)
        {
            return input;
        }

        var output = new Matrix<T>(input.Rows, input.Columns);
        for (int i = 0; i < input.Rows; i++)
        {
            var sanitizedRow = sanitizedRows[i] ?? input.GetRow(i);
            if (sanitizedRow.Length != input.Columns)
            {
                throw new InvalidOperationException($"Safety filter produced an incompatible sanitized row at index {i}.");
            }

            output.SetRow(i, sanitizedRow);
        }

        return output;
    }

    private Matrix<T> FilterMatrixOutput(Matrix<T> output)
    {
        if (SafetyFilter == null)
        {
            return output;
        }

        if (output.Rows == 0 || output.Columns == 0)
        {
            return output;
        }

        var result = new Matrix<T>(output.Rows, output.Columns);
        for (int i = 0; i < output.Rows; i++)
        {
            var row = output.GetRow(i);
            var filtered = SafetyFilter.FilterOutput(row);

            if (filtered.FilteredOutput.Length != output.Columns)
            {
                throw new InvalidOperationException($"Safety filter produced an incompatible filtered row at index {i}.");
            }

            result.SetRow(i, filtered.FilteredOutput);
        }

        return result;
    }

    /// <summary>
    /// Begins an inference session for stateful inference features (e.g., KV-cache).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Sessions are intended for serving-style workloads where you run many sequential inference steps.
    /// A session can create multiple independent sequences, each maintaining its own state (like KV-cache).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use a session when you are doing "token-by-token" inference.
    ///
    /// - Use <see cref="Predict(TInput)"/> for one-off, stateless predictions.
    /// - Use <see cref="BeginInferenceSession"/> when you need the model to remember prior calls in the same sequence.
    /// </para>
    /// </remarks>
    public InferenceSession BeginInferenceSession()
    {
        return new InferenceSession(this, InferenceOptimizationConfig);
    }

    private NeuralNetworkBase<T>? EnsureStatelessInferenceOptimizationsInitialized(NeuralNetworkBase<T> model)
    {
        if (_inferenceOptimizationsInitialized)
        {
            return _inferenceOptimizedNeuralModel;
        }

        lock (_inferenceOptimizationLock)
        {
            if (_inferenceOptimizationsInitialized)
            {
                return _inferenceOptimizedNeuralModel;
            }

            try
            {
                if (InferenceOptimizationConfig != null)
                {
                    // Stateless-only optimizations for plain Predict(): avoid stateful features that can leak across calls.
                    var statelessConfig = CreateStatelessInferenceConfig(InferenceOptimizationConfig);
                    var optimizer = new InferenceOptimizer<T>(statelessConfig);
                    var (optimizedModel, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: true);

                    _inferenceOptimizer = optimizer;
                    _inferenceOptimizedNeuralModel = anyApplied ? optimizedModel : null;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: inference optimizations failed: {ex.Message}");
                _inferenceOptimizer = null;
                _inferenceOptimizedNeuralModel = null;
            }
            finally
            {
                _inferenceOptimizationsInitialized = true;
            }

            return _inferenceOptimizedNeuralModel;
        }
    }

    private static AiDotNet.Configuration.InferenceOptimizationConfig CreateStatelessInferenceConfig(
        AiDotNet.Configuration.InferenceOptimizationConfig config)
    {
        return new AiDotNet.Configuration.InferenceOptimizationConfig
        {
            EnableFlashAttention = config.EnableFlashAttention,
            AttentionMasking = config.AttentionMasking,

            // Disable stateful/session-centric features for plain Predict().
            EnableKVCache = false,
            EnablePagedKVCache = false,
            EnableBatching = false,
            EnableSpeculativeDecoding = false
        };
    }

    /// <summary>
    /// Facade-friendly inference session that owns stateful inference internals.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This type intentionally keeps inference internals behind the facade. Users create sequences via
    /// <see cref="CreateSequence"/> and run inference via <see cref="InferenceSequence.Predict(TInput)"/>.
    /// </para>
    /// </remarks>
    public sealed class InferenceSession : IDisposable
    {
        private readonly AiModelResult<T, TInput, TOutput> _result;
        private readonly AiDotNet.Configuration.InferenceOptimizationConfig? _config;
        private bool _disposed;

        internal InferenceSession(
            AiModelResult<T, TInput, TOutput> result,
            AiDotNet.Configuration.InferenceOptimizationConfig? config)
        {
            _result = result ?? throw new ArgumentNullException(nameof(result));
            _config = config;
        }

        /// <summary>
        /// Creates an independent sequence within this session.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Each sequence represents an independent stream (e.g., one chat) and owns its own state.
        /// </para>
        /// </remarks>
        public InferenceSequence CreateSequence()
        {
            ThrowIfDisposed();
            return new InferenceSequence(_result, _config, multiLoRATask: null);
        }

        // Internal (serving/tests): allow selecting a Multi-LoRA task per sequence without expanding public API surface.
        internal InferenceSequence CreateSequence(string? multiLoRATask)
        {
            ThrowIfDisposed();
            return new InferenceSequence(_result, _config, multiLoRATask);
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            _disposed = true;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(InferenceSession));
            }
        }
    }

    /// <summary>
    /// Represents one independent, stateful inference sequence (e.g., one chat/generation stream).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A sequence may keep internal state across calls when inference optimizations are enabled (e.g., KV-cache).
    /// Call <see cref="Reset"/> to start a new logical sequence on the same object.
    /// </para>
    /// </remarks>
    public sealed class InferenceSequence : IDisposable
    {
        private readonly AiModelResult<T, TInput, TOutput> _result;
        private readonly AiDotNet.Configuration.InferenceOptimizationConfig? _config;
        private bool _disposed;

        // Session-local inference state (populated lazily when used).
        private InferenceOptimizer<T>? _sequenceOptimizer;
        private NeuralNetworkBase<T>? _sequenceOptimizedNeuralModel;
        private bool _sequenceInitialized;
        private readonly object _sequenceLock = new();

        internal InferenceSequence(
            AiModelResult<T, TInput, TOutput> result,
            AiDotNet.Configuration.InferenceOptimizationConfig? config,
            string? multiLoRATask)
        {
            _result = result ?? throw new ArgumentNullException(nameof(result));
            _config = config;
            _multiLoRATask = multiLoRATask;
        }

        private string? _multiLoRATask;

        /// <summary>
        /// Runs a prediction for the given input within this sequence.
        /// </summary>
        /// <param name="newData">The input to predict on.</param>
        /// <returns>The predicted output.</returns>
        /// <remarks>
        /// <para>
        /// When inference optimizations are configured, this method may keep and reuse sequence-local state
        /// (such as a KV-cache) across calls for improved throughput and latency.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> This is like predicting with "memory". Each call can reuse what was computed
        /// previously for the same sequence so the next call can be faster.
        /// </para>
        /// </remarks>
        public TOutput Predict(TInput newData)
        {
            ThrowIfDisposed();

            if (_result.Model == null)
            {
                throw new InvalidOperationException("Model is not initialized.");
            }

            if (_result.NormalizationInfo.Normalizer == null)
            {
                throw new InvalidOperationException("Normalizer is not initialized.");
            }

            var (normalizedNewData, _) = _result.NormalizationInfo.Normalizer.NormalizeInput(newData);

            // Session inference: use configured inference optimizations, including stateful ones, if applicable.
            if (_config != null &&
                _result.Model is NeuralNetworkBase<T> neuralModel &&
                normalizedNewData is Tensor<T> inputTensor)
            {
                var optimized = EnsureSequenceOptimizationsInitialized(neuralModel);
                if (optimized != null)
                {
                    var optimizedOutput = optimized.Predict(inputTensor);
                    if (optimizedOutput is TOutput output)
                    {
                        return _result.NormalizationInfo.Normalizer.Denormalize(output, _result.NormalizationInfo.YParams);
                    }
                }
            }

            // Fallback: normal predict path (no JIT inside a session to keep behavior consistent).
            var normalizedPredictions = _result.Model.Predict(normalizedNewData);
            return _result.NormalizationInfo.Normalizer.Denormalize(normalizedPredictions, _result.NormalizationInfo.YParams);
        }

        /// <summary>
        /// Resets sequence-local inference state.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This clears any cached state for the current sequence so the next prediction starts fresh.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> Call this when you want to start a new conversation/stream using the same sequence object.
        /// </para>
        /// </remarks>
        public void Reset()
        {
            ThrowIfDisposed();
            lock (_sequenceLock)
            {
                _sequenceOptimizer?.ClearCache();
            }
        }

        // Internal: switch Multi-LoRA task for this sequence, resetting state to avoid cache leakage.
        internal void SetMultiLoRATask(string? taskName)
        {
            ThrowIfDisposed();
            lock (_sequenceLock)
            {
                if (string.Equals(_multiLoRATask, taskName, StringComparison.Ordinal))
                    return;

                _multiLoRATask = taskName;

                try
                {
                    _sequenceOptimizer?.ClearCache();
                }
                catch
                {
                    // Best-effort.
                }

                _sequenceOptimizer = null;
                _sequenceOptimizedNeuralModel = null;
                _sequenceInitialized = false;
            }
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            try
            {
                _sequenceOptimizer?.ClearCache();
            }
            catch
            {
                // Best-effort cleanup; disposal must not throw.
            }

            _disposed = true;
        }

        // Exposed to AiDotNetTests via InternalsVisibleTo for integration verification without expanding the public API surface.
        internal Dictionary<string, object> GetInferenceStatistics()
        {
            ThrowIfDisposed();
            lock (_sequenceLock)
            {
                return _sequenceOptimizer?.GetStatistics() ?? new Dictionary<string, object>();
            }
        }

        private NeuralNetworkBase<T>? EnsureSequenceOptimizationsInitialized(NeuralNetworkBase<T> model)
        {
            if (_sequenceInitialized)
            {
                return _sequenceOptimizedNeuralModel;
            }

            lock (_sequenceLock)
            {
                if (_sequenceInitialized)
                {
                    return _sequenceOptimizedNeuralModel;
                }

                try
                {
                    if (_config != null)
                    {
                        // If Multi-LoRA is in use, isolate per-sequence task selection by cloning and selecting task
                        // before applying any further inference optimizations.
                        NeuralNetworkBase<T> modelForSequence = model;
                        bool hasMultiLoRATask = !string.IsNullOrWhiteSpace(_multiLoRATask);
                        if (hasMultiLoRATask)
                        {
                            try
                            {
                                modelForSequence = (NeuralNetworkBase<T>)model.Clone();

                                int appliedCount = 0;
                                foreach (var multi in modelForSequence.Layers.OfType<AiDotNet.LoRA.Adapters.MultiLoRAAdapter<T>>())
                                {
                                    multi.SetCurrentTask(_multiLoRATask!);
                                    appliedCount++;
                                }

                                InferenceDiagnostics.RecordDecision(
                                    area: "InferenceSession",
                                    feature: "MultiLoRA",
                                    enabled: appliedCount > 0,
                                    reason: appliedCount > 0 ? $"Task={_multiLoRATask}" : $"NoMultiLoRAAdapters(Task={_multiLoRATask})");
                            }
                            catch (Exception ex)
                            {
                                InferenceDiagnostics.RecordException("InferenceSession", "MultiLoRA", ex, $"Task={_multiLoRATask};FallbackToBaseModel");
                                modelForSequence = model;
                            }
                        }

                        // In a session, prefer causal masking defaults when user left it as Auto.
                        var sessionConfig = _config.AttentionMasking == AiDotNet.Configuration.AttentionMaskingMode.Auto
                            ? new AiDotNet.Configuration.InferenceOptimizationConfig
                            {
                                EnableFlashAttention = _config.EnableFlashAttention,
                                EnableKVCache = _config.EnableKVCache,
                                EnablePagedKVCache = _config.EnablePagedKVCache,
                                PagedKVCacheBlockSize = _config.PagedKVCacheBlockSize,
                                MaxBatchSize = _config.MaxBatchSize,
                                KVCacheMaxSizeMB = _config.KVCacheMaxSizeMB,
                                KVCachePrecision = _config.KVCachePrecision,
                                KVCacheQuantization = _config.KVCacheQuantization,
                                UseSlidingWindowKVCache = _config.UseSlidingWindowKVCache,
                                KVCacheWindowSize = _config.KVCacheWindowSize,
                                EnableBatching = _config.EnableBatching,
                                EnableSpeculativeDecoding = _config.EnableSpeculativeDecoding,
                                SpeculationPolicy = _config.SpeculationPolicy,
                                SpeculativeMethod = _config.SpeculativeMethod,
                                DraftModelType = _config.DraftModelType,
                                SpeculationDepth = _config.SpeculationDepth,
                                UseTreeSpeculation = _config.UseTreeSpeculation,
                                EnableWeightOnlyQuantization = _config.EnableWeightOnlyQuantization,
                                AttentionMasking = AiDotNet.Configuration.AttentionMaskingMode.Causal
                            }
                            : _config;

                        var optimizer = new InferenceOptimizer<T>(sessionConfig);
                        var (optimizedModel, anyApplied) = optimizer.OptimizeForInference(modelForSequence, cloneModel: ReferenceEquals(modelForSequence, model));

                        _sequenceOptimizer = optimizer;
                        // If Multi-LoRA was requested, keep the per-sequence model even when no other optimizations apply.
                        _sequenceOptimizedNeuralModel = anyApplied || !ReferenceEquals(modelForSequence, model) ? optimizedModel : null;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: inference session optimizations failed: {ex.Message}");
                    _sequenceOptimizer = null;
                    _sequenceOptimizedNeuralModel = null;
                }
                finally
                {
                    _sequenceInitialized = true;
                }

                return _sequenceOptimizedNeuralModel;
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(InferenceSequence));
            }
        }
    }

    /// <summary>
    /// Gets the default loss function used by this model for gradient computation.
    /// </summary>
    /// <exception cref="InvalidOperationException">If Model is not initialized.</exception>
    [JsonIgnore]
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
    /// Training is not supported on AiModelResult. Use AiModelBuilder to create and train new models.
    /// </summary>
    /// <param name="input">Input training data (not used).</param>
    /// <param name="expectedOutput">Expected output values (not used).</param>
    /// <exception cref="InvalidOperationException">Always thrown - AiModelResult represents an already-trained model and cannot be retrained.</exception>
    /// <remarks>
    /// AiModelResult is a snapshot of a trained model with its optimization results and metadata.
    /// Retraining would invalidate the OptimizationResult and metadata.
    /// To train a new model or retrain with different data, use AiModelBuilder.Build() instead.
    /// </remarks>
    public void Train(TInput input, TOutput expectedOutput)
    {
        throw new InvalidOperationException(
            "AiModelResult represents an already-trained model and cannot be retrained. " +
            "The OptimizationResult and metadata reflect the original training process. " +
            "To train a new model, use AiModelBuilder.Build() instead.");
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
    /// Setting parameters is not supported on AiModelResult.
    /// </summary>
    /// <param name="parameters">The parameter vector (not used).</param>
    /// <exception cref="InvalidOperationException">Always thrown - AiModelResult parameters cannot be modified.</exception>
    /// <remarks>
    /// Modifying parameters would invalidate the OptimizationResult which reflects the optimized parameter values.
    /// To create a model with different parameters, use AiModelBuilder with custom initial parameters.
    /// </remarks>
    public void SetParameters(Vector<T> parameters)
    {
        throw new InvalidOperationException(
            "AiModelResult parameters cannot be modified. " +
            "The current parameters reflect the optimized solution from the training process. " +
            "To create a model with different parameters, use AiModelBuilder.");
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
    /// <returns>A new AiModelResult with updated parameters.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the Model is not initialized.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a new model with updated parameters. The OptimizationResult is deep-copied
    /// and updated to reference the new model. NormalizationInfo is shared (shallow-copied) since
    /// normalization parameters don't change when model parameters change.
    /// </para>
    /// <para>
    /// All configuration components (prompt engineering, RAG, agents, etc.) are shallow-copied,
    /// meaning they are shared between the original and new instance. See <see cref="DeepCopy"/>
    /// for detailed documentation on which components are deep vs shallow copied.
    /// </para>
    /// </remarks>
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
        var options = new AiModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = updatedOptimizationResult,
            NormalizationInfo = NormalizationInfo,
            BiasDetector = BiasDetector,
            FairnessEvaluator = FairnessEvaluator,
            RagRetriever = RagRetriever,
            RagReranker = RagReranker,
            RagGenerator = RagGenerator,
            QueryProcessors = QueryProcessors,
            LoRAConfiguration = LoRAConfiguration,
            CrossValidationResult = CrossValidationResult,
            AutoMLSummary = AutoMLSummary,
            AgentConfig = AgentConfig,
            AgentRecommendation = AgentRecommendation,
            DeploymentConfiguration = DeploymentConfiguration,
            // JIT compilation is parameter-specific, don't copy
            InferenceOptimizationConfig = InferenceOptimizationConfig,
            ReasoningConfig = ReasoningConfig,
            KnowledgeGraph = KnowledgeGraph,
            GraphStore = GraphStore,
            HybridGraphRetriever = HybridGraphRetriever,
            MetaLearner = MetaLearner,
            MetaTrainingResult = MetaTrainingResult,
            Tokenizer = Tokenizer,
            TokenizationConfig = TokenizationConfig,
            PromptTemplate = PromptTemplate,
            PromptChain = PromptChain,
            PromptOptimizer = PromptOptimizer,
            FewShotExampleSelector = FewShotExampleSelector,
            PromptAnalyzer = PromptAnalyzer,
            PromptCompressor = PromptCompressor,
            // Training Infrastructure - shallow copy (shared references)
            ExperimentRun = ExperimentRun,
            ExperimentTracker = ExperimentTracker,
            CheckpointManager = CheckpointManager,
            ModelRegistry = ModelRegistry,
            TrainingMonitor = TrainingMonitor,
            HyperparameterOptimizationResult = HyperparameterOptimizationResult,
            ExperimentRunId = ExperimentRunId,
            ExperimentId = ExperimentId,
            ModelVersion = ModelVersion,
            RegisteredModelName = RegisteredModelName,
            CheckpointPath = CheckpointPath,
            DataVersionHash = DataVersionHash,
            HyperparameterTrialId = HyperparameterTrialId,
            Hyperparameters = Hyperparameters,
            TrainingMetricsHistory = TrainingMetricsHistory
        };

        return new AiModelResult<T, TInput, TOutput>(options);
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
    /// Setting active feature indices is not supported on AiModelResult.
    /// </summary>
    /// <param name="featureIndices">The feature indices (not used).</param>
    /// <exception cref="InvalidOperationException">Always thrown - AiModelResult feature configuration cannot be modified.</exception>
    /// <remarks>
    /// Changing active features would invalidate the trained model and optimization results.
    /// To train a model with different features, use AiModelBuilder with the desired feature configuration.
    /// </remarks>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        throw new InvalidOperationException(
            "AiModelResult active features cannot be modified. " +
            "The model was trained with a specific feature set. " +
            "To use different features, train a new model using AiModelBuilder.");
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

    #region Model Interpretability Methods

    /// <summary>
    /// Explains a single prediction using SHAP (SHapley Additive exPlanations) values.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="backgroundData">Representative background data for computing expected values.</param>
    /// <returns>A SHAP explanation showing how each feature contributed to this prediction.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model is not initialized or SHAP is not enabled.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> SHAP values tell you exactly how much each feature contributed
    /// to this specific prediction compared to an "average" prediction.
    ///
    /// Example interpretation:
    /// - Baseline: $300,000 (average house price)
    /// - SHAP for Bedrooms: +$50,000 (having 4 bedrooms adds value)
    /// - SHAP for Location: +$100,000 (good neighborhood)
    /// - SHAP for Age: -$30,000 (older house reduces value)
    /// - Prediction: $420,000
    ///
    /// The sum of SHAP values plus baseline always equals the prediction.
    /// </para>
    /// </remarks>
    public SHAPExplanation<T> ExplainWithSHAP(Vector<T> instance, Matrix<T> backgroundData)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        if (!options.EnableSHAP)
            throw new InvalidOperationException("SHAP is not enabled. Configure interpretability with EnableSHAP = true.");

        // Create a prediction function from the model
        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction();

        var explainer = new SHAPExplainer<T>(
            predictFunc,
            backgroundData,
            nSamples: options.SHAPSampleCount,
            featureNames: options.FeatureNames,
            randomState: options.RandomSeed);

        return explainer.Explain(instance);
    }

    /// <summary>
    /// Explains multiple predictions using SHAP values and returns global feature importance.
    /// </summary>
    /// <param name="data">The input instances to explain.</param>
    /// <param name="backgroundData">Representative background data for computing expected values.</param>
    /// <returns>A global SHAP explanation with aggregated feature importance across all instances.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model is not initialized or SHAP is not enabled.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This aggregates SHAP values across many predictions to show
    /// which features are most important overall. Use GetFeatureImportance() on the result
    /// to see a ranked list of features by their average absolute SHAP value.
    /// </para>
    /// </remarks>
    public GlobalSHAPExplanation<T> ExplainGlobalWithSHAP(Matrix<T> data, Matrix<T> backgroundData)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        if (!options.EnableSHAP)
            throw new InvalidOperationException("SHAP is not enabled. Configure interpretability with EnableSHAP = true.");

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction();

        var explainer = new SHAPExplainer<T>(
            predictFunc,
            backgroundData,
            nSamples: options.SHAPSampleCount,
            featureNames: options.FeatureNames,
            randomState: options.RandomSeed);

        return explainer.ExplainGlobal(data);
    }

    /// <summary>
    /// Explains a single prediction using LIME (Local Interpretable Model-agnostic Explanations).
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>A LIME explanation with local feature weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model is not initialized or LIME is not enabled.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> LIME explains predictions by fitting a simple linear model
    /// around the specific instance. It creates many slightly modified versions of your input,
    /// sees how predictions change, and fits a simple model to understand local behavior.
    ///
    /// The coefficients tell you the direction and magnitude of each feature's effect:
    /// - Positive coefficient: Feature value pushed prediction higher
    /// - Negative coefficient: Feature value pushed prediction lower
    /// - LocalR2 tells you how well the simple model approximates the complex one locally
    /// </para>
    /// </remarks>
    public LIMEExplanationResult<T> ExplainWithLIME(Vector<T> instance)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        if (!options.EnableLIME)
            throw new InvalidOperationException("LIME is not enabled. Configure interpretability with EnableLIME = true.");

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction();

        var explainer = new LIMEExplainer<T>(
            predictFunc,
            instance.Length,
            nSamples: options.LIMESampleCount,
            kernelWidth: options.LIMEKernelWidth ?? 0.75,
            featureNames: options.FeatureNames,
            randomState: options.RandomSeed);

        return explainer.Explain(instance);
    }

    /// <summary>
    /// Computes permutation feature importance for the model.
    /// </summary>
    /// <param name="X">The feature matrix to evaluate on.</param>
    /// <param name="y">The target values for evaluation.</param>
    /// <param name="scoreFunction">
    /// A function that computes a score given (actual, predicted). Higher scores = better.
    /// If null, uses R² for regression-like outputs.
    /// </param>
    /// <returns>Feature importance scores showing which features matter most.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model is not initialized or permutation importance is not enabled.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Permutation importance measures how much the model's score drops
    /// when each feature is randomly shuffled. If shuffling a feature hurts performance a lot,
    /// that feature is important.
    ///
    /// The ImportanceStds values show how consistent the importance estimates are across
    /// multiple shuffles. High standard deviation means the importance estimate is uncertain.
    /// </para>
    /// </remarks>
    public FeatureImportanceResult<T> GetPermutationFeatureImportance(Matrix<T> X, Vector<T> y, Func<Vector<T>, Vector<T>, T>? scoreFunction = null)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        if (!options.EnablePermutationImportance)
            throw new InvalidOperationException("Permutation importance is not enabled. Configure interpretability with EnablePermutationImportance = true.");

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction();

        // Default score function: R²
        scoreFunction ??= ComputeR2Score;

        var calculator = new PermutationFeatureImportance<T>(
            predictFunc,
            scoreFunction,
            nRepeats: options.PermutationRepeatCount,
            featureNames: options.FeatureNames,
            randomState: options.RandomSeed);

        return calculator.Calculate(X, y);
    }

    /// <summary>
    /// Trains a global surrogate model to approximate the complex model's behavior.
    /// </summary>
    /// <param name="X">The feature matrix to train the surrogate on.</param>
    /// <returns>A surrogate explanation with linear coefficients and fidelity score.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model is not initialized or global surrogate is not enabled.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> A global surrogate is a simple linear model that tries to mimic
    /// your complex model. If the surrogate has high fidelity (R² close to 1), you can use it
    /// to understand the complex model's overall behavior.
    ///
    /// The coefficients show:
    /// - Which features the complex model relies on most
    /// - Whether each feature has a positive or negative effect
    /// - The relative importance of each feature
    ///
    /// Low fidelity means the complex model is too nonlinear for a linear surrogate.
    /// </para>
    /// </remarks>
    public SurrogateExplanation<T> GetGlobalSurrogateExplanation(Matrix<T> X)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        if (!options.EnableGlobalSurrogate)
            throw new InvalidOperationException("Global surrogate is not enabled. Configure interpretability with EnableGlobalSurrogate = true.");

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction();

        var explainer = new GlobalSurrogateExplainer<T>(
            predictFunc,
            X.Columns,
            options.FeatureNames);

        return explainer.ExplainGlobal(X);
    }

    /// <summary>
    /// Computes Accumulated Local Effects (ALE) for all features in the dataset.
    /// </summary>
    /// <param name="data">The data matrix to use for computing ALE.</param>
    /// <param name="numIntervals">Number of intervals to divide each feature range into.</param>
    /// <returns>The ALE result containing effects for all features.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> ALE plots show how each feature affects predictions while properly
    /// handling correlated features. Unlike PDP, ALE doesn't make unrealistic assumptions about
    /// feature independence.
    ///
    /// Example:
    /// <code>
    /// var aleResult = result.GetAccumulatedLocalEffects(trainingData, 20);
    /// // aleResult.FeatureEffects[0] shows how feature 0 affects predictions
    /// // Positive values mean the feature increases predictions, negative decreases
    /// </code>
    /// </para>
    /// </remarks>
    public ALEResult<T> GetAccumulatedLocalEffects(Matrix<T> data, int numIntervals = 20)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction();
        var explainer = new AccumulatedLocalEffectsExplainer<T>(predictFunc, data, numIntervals, options.FeatureNames);
        return explainer.ExplainGlobal(data);
    }

    /// <summary>
    /// Computes Accumulated Local Effects for a specific feature.
    /// </summary>
    /// <param name="data">The data matrix to use for computing ALE.</param>
    /// <param name="featureIndex">Index of the feature to analyze.</param>
    /// <param name="numIntervals">Number of intervals to divide the feature range into.</param>
    /// <returns>The ALE result for the specific feature.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This computes ALE for a single feature, showing how
    /// changing that feature affects predictions while accounting for correlations.
    ///
    /// Example:
    /// <code>
    /// var aleResult = result.GetFeatureALE(trainingData, 0, 20);
    /// // aleResult.FeatureEffects[0] shows how feature 0 affects predictions
    /// </code>
    /// </para>
    /// </remarks>
    public ALEResult<T> GetFeatureALE(Matrix<T> data, int featureIndex, int numIntervals = 20)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction();
        var explainer = new AccumulatedLocalEffectsExplainer<T>(predictFunc, data, numIntervals, options.FeatureNames);
        return explainer.ExplainGlobal(data);
    }

    /// <summary>
    /// Detects and quantifies feature interactions using Friedman's H-statistic.
    /// </summary>
    /// <param name="data">The data matrix to analyze.</param>
    /// <param name="numSamples">Number of samples to use (for performance).</param>
    /// <returns>Feature interaction results including pairwise H-statistics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> H-statistic measures how much two features interact in the model.
    /// - H = 0 means no interaction (features contribute independently)
    /// - H = 1 means pure interaction (combined effect differs completely from individual effects)
    ///
    /// Example:
    /// <code>
    /// var interactions = result.GetFeatureInteractions(trainingData, 500);
    /// var topPairs = interactions.GetTopInteractions(5);
    /// // Shows the 5 most interacting feature pairs
    /// </code>
    /// </para>
    /// </remarks>
    public FeatureInteractionResult<T> GetFeatureInteractions(Matrix<T> data, int gridSize = 20)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction();
        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        var explainer = new FeatureInteractionExplainer<T>(predictFunc, data, gridSize, options.FeatureNames);
        return explainer.ExplainGlobal(data);
    }

    /// <summary>
    /// Explains a prediction using Integrated Gradients attribution.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="baseline">Optional baseline (default is zero vector).</param>
    /// <param name="numSteps">Number of interpolation steps (higher = more accurate).</param>
    /// <returns>Attribution scores for each feature.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Integrated Gradients computes how much each feature contributed to
    /// the prediction by integrating gradients along a path from a baseline to the input.
    /// It satisfies the completeness axiom: attributions sum to the difference between
    /// the prediction and the baseline prediction.
    ///
    /// Example:
    /// <code>
    /// var igResult = result.ExplainWithIntegratedGradients(instance, baseline: null, numSteps: 50);
    /// // igResult.Attributions[i] shows feature i's contribution
    /// // igResult.ConvergenceDelta should be close to 0 if converged
    /// </code>
    /// </para>
    /// </remarks>
    public IntegratedGradientsExplanation<T> ExplainWithIntegratedGradients(Vector<T> instance, Vector<T>? baseline = null, int numSteps = 50)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        Func<Vector<T>, Vector<T>> vectorPredictFunc = CreateVectorPredictionFunction();
        Func<Vector<T>, int, Vector<T>>? gradientFunc = TryCreateIndexedGradientFunction();

        var explainer = new IntegratedGradientsExplainer<T>(
            vectorPredictFunc,
            gradientFunc,
            instance.Length,
            numSteps,
            baseline,
            options.FeatureNames);
        return explainer.Explain(instance);
    }

    /// <summary>
    /// Generates a Grad-CAM heatmap for CNN model predictions.
    /// </summary>
    /// <param name="instance">The input instance (typically image data).</param>
    /// <param name="inputShape">Shape of the input (e.g., [1, 3, 224, 224] for batch, channels, height, width).</param>
    /// <param name="featureMapShape">Shape of the feature maps from the last conv layer.</param>
    /// <param name="useGradCAMPlusPlus">Whether to use Grad-CAM++ (better for multiple objects).</param>
    /// <returns>Grad-CAM explanation with heatmap and class activation maps.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Grad-CAM creates visual explanations showing which parts of an
    /// image the CNN model focused on when making its prediction. Brighter areas in the heatmap
    /// indicate regions more important for the prediction.
    ///
    /// Example:
    /// <code>
    /// var gradCamResult = result.ExplainWithGradCAM(imageData,
    ///     inputShape: new[] { 1, 3, 224, 224 },
    ///     featureMapShape: new[] { 1, 512, 14, 14 });
    /// // gradCamResult.Heatmap shows importance per spatial location
    /// </code>
    /// </para>
    /// </remarks>
    public GradCAMExplanation<T> ExplainWithGradCAM(Vector<T> instance, int[] inputShape, int[] featureMapShape, bool useGradCAMPlusPlus = false)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        Func<Tensor<T>, Tensor<T>> tensorPredictFunc = CreateTensorPredictionFunction();
        Func<Tensor<T>, int, Tensor<T>>? featureMapFunc = TryCreateFeatureMapFunction();
        Func<Tensor<T>, int, int, Tensor<T>>? gradientFunc = TryCreateTensorGradientFunction();

        var explainer = new GradCAMExplainer<T>(
            tensorPredictFunc,
            featureMapFunc,
            gradientFunc,
            inputShape,
            featureMapShape,
            useGradCAMPlusPlus);
        return explainer.Explain(instance);
    }

    /// <summary>
    /// Visualizes attention patterns for transformer-based models.
    /// </summary>
    /// <param name="instance">The input instance (e.g., token embeddings).</param>
    /// <param name="numLayers">Number of transformer layers in the model.</param>
    /// <param name="numHeads">Number of attention heads per layer.</param>
    /// <param name="sequenceLength">Length of the input sequence.</param>
    /// <param name="tokenLabels">Optional labels for each token/position.</param>
    /// <returns>Attention visualization with rollout and token importance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Attention visualization shows how transformer models attend to
    /// different parts of the input when making predictions. Attention rollout aggregates attention
    /// across all layers to show overall importance.
    ///
    /// Example:
    /// <code>
    /// var attentionResult = result.ExplainWithAttentionVisualization(
    ///     embeddings, numLayers: 12, numHeads: 8, sequenceLength: 128,
    ///     tokenLabels: new[] { "[CLS]", "Hello", "world", "[SEP]" });
    /// // attentionResult.AttentionRollout shows aggregated attention patterns
    /// // attentionResult.TokenImportance shows importance of each position
    /// </code>
    /// </para>
    /// </remarks>
    public AttentionExplanation<T> ExplainWithAttentionVisualization(Vector<T> instance, int numLayers, int numHeads, int sequenceLength, string[]? tokenLabels = null)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        Func<Tensor<T>, Tensor<T>> tensorPredictFunc = CreateTensorPredictionFunction();
        Func<Tensor<T>, int, Tensor<T>>? attentionFunc = TryCreateAttentionWeightsFunction();

        var explainer = new AttentionVisualizationExplainer<T>(
            tensorPredictFunc,
            attentionFunc,
            numLayers,
            numHeads,
            sequenceLength,
            tokenLabels);
        return explainer.Explain(instance);
    }

    /// <summary>
    /// Explains a prediction using DeepLIFT attribution.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="reference">Reference input (baseline) for comparison.</param>
    /// <param name="rule">The propagation rule to use (Rescale or RevealCancel).</param>
    /// <returns>DeepLIFT attributions showing feature contributions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> DeepLIFT explains predictions by comparing neuron activations
    /// to a reference input. It propagates the difference in output back through the network
    /// to assign importance to each input feature.
    ///
    /// - Rescale rule: Distributes importance proportionally to activation differences
    /// - RevealCancel rule: Separates positive and negative contributions
    ///
    /// Example:
    /// <code>
    /// var deepLiftResult = result.ExplainWithDeepLIFT(instance, referenceInput);
    /// // deepLiftResult.Attributions[i] shows feature i's contribution
    /// </code>
    /// </para>
    /// </remarks>
    public DeepLIFTExplanation<T> ExplainWithDeepLIFT(Vector<T> instance, Vector<T>? baseline = null, DeepLIFTRule rule = DeepLIFTRule.Rescale)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        Func<Vector<T>, Vector<T>> vectorPredictFunc = CreateVectorPredictionFunction();
        Func<Vector<T>, Vector<T>>? getActivations = TryCreateActivationsFunction();
        Func<Vector<T>, Vector<T>, Vector<T>>? computeMultipliers = TryCreateMultipliersFunction();

        var explainer = new DeepLIFTExplainer<T>(
            vectorPredictFunc,
            getActivations,
            computeMultipliers,
            instance.Length,
            baseline,
            options.FeatureNames,
            rule);
        return explainer.Explain(instance);
    }

    /// <summary>
    /// Explains a prediction using GradientSHAP (combines Integrated Gradients with SHAP).
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="backgroundData">Background dataset for sampling baselines.</param>
    /// <param name="numSamples">Number of baseline samples to use.</param>
    /// <returns>GradientSHAP attributions with variance estimates.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> GradientSHAP combines the efficiency of gradient-based methods
    /// with SHAP's game-theoretic foundation. It samples random baselines from background data
    /// and averages Integrated Gradients over these baselines.
    ///
    /// The variance estimates help you understand how stable the attributions are across
    /// different baseline choices.
    ///
    /// Example:
    /// <code>
    /// var gradShapResult = result.ExplainWithGradientSHAP(instance, trainingData, numSamples: 200);
    /// // gradShapResult.ShapValues[i] is feature i's SHAP value
    /// // gradShapResult.Variance[i] shows attribution uncertainty
    /// </code>
    /// </para>
    /// </remarks>
    public GradientSHAPExplanation<T> ExplainWithGradientSHAP(Vector<T> instance, Matrix<T> backgroundData, int numSamples = 200, int numSteps = 50)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        Func<Vector<T>, Vector<T>> vectorPredictFunc = CreateVectorPredictionFunction();
        Func<Vector<T>, int, Vector<T>>? gradientFunc = TryCreateIndexedGradientFunction();

        var explainer = new GradientSHAPExplainer<T>(
            vectorPredictFunc,
            gradientFunc,
            backgroundData,
            numSamples,
            numSteps,
            true, // addNoise
            0.09, // noiseStdDev
            options.FeatureNames);
        return explainer.Explain(instance);
    }

    /// <summary>
    /// Explains a prediction using Layer-wise Relevance Propagation (LRP).
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="rule">The LRP propagation rule to use.</param>
    /// <returns>LRP explanation with relevance scores per feature.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> LRP explains neural network predictions by backpropagating
    /// "relevance" from the output through each layer to the input. The relevance is conserved
    /// at each layer, ensuring attributions sum to the output.
    ///
    /// Different rules handle different layer types:
    /// - Basic: Simple proportional distribution
    /// - Epsilon: Adds stability for near-zero activations
    /// - Gamma: Emphasizes positive contributions
    /// - AlphaBeta: Separates positive and negative evidence
    ///
    /// Example:
    /// <code>
    /// var lrpResult = result.ExplainWithLRP(instance, LRPRule.Epsilon);
    /// // lrpResult.InputRelevance[i] shows feature i's relevance
    /// </code>
    /// </para>
    /// </remarks>
    public LRPExplanation<T> ExplainWithLRP(Vector<T> instance, LRPRule rule = LRPRule.Epsilon, double epsilon = 1e-4)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        Func<Vector<T>, Vector<T>> vectorPredictFunc = CreateVectorPredictionFunction();
        Func<Vector<T>, Vector<T>[]>? getLayerActivations = TryCreateLayerActivationsFunction();
        Func<int, Matrix<T>>? getLayerWeights = TryCreateLayerWeightsFunction();

        var explainer = new LayerwiseRelevancePropagationExplainer<T>(
            vectorPredictFunc,
            getLayerActivations,
            getLayerWeights,
            instance.Length,
            null, // layerSizes - will be inferred
            options.FeatureNames,
            rule,
            epsilon);
        return explainer.Explain(instance);
    }

    /// <summary>
    /// Generates a saliency map showing input gradients.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="method">The saliency method to use.</param>
    /// <param name="numSamples">Number of samples for SmoothGrad methods.</param>
    /// <returns>Saliency map showing feature importance via gradients.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saliency maps show which input features the model is most
    /// sensitive to. Areas with high gradient magnitude are features that, if changed slightly,
    /// would most affect the prediction.
    ///
    /// - VanillaGradient: Raw gradients (can be noisy)
    /// - GradientTimesInput: Gradients weighted by input values
    /// - SmoothGrad: Averages gradients with noise for smoother results
    /// - SmoothGradSquared: Like SmoothGrad but squares gradients first
    ///
    /// Example:
    /// <code>
    /// var saliency = result.GetSaliencyMap(instance, SaliencyMethod.SmoothGrad);
    /// // saliency.Saliency[i] shows feature i's sensitivity
    /// </code>
    /// </para>
    /// </remarks>
    public SaliencyMapExplanation<T> GetSaliencyMap(Vector<T> instance, SaliencyMethod method = SaliencyMethod.SmoothGrad, int smoothGradSamples = 50, double smoothGradNoise = 0.1)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        Func<Vector<T>, Vector<T>> vectorPredictFunc = CreateVectorPredictionFunction();
        Func<Vector<T>, int, Vector<T>>? gradientFunc = TryCreateIndexedGradientFunction();

        var explainer = new SaliencyMapExplainer<T>(
            vectorPredictFunc,
            gradientFunc,
            instance.Length,
            method,
            smoothGradSamples,
            smoothGradNoise,
            options.FeatureNames);
        return explainer.Explain(instance);
    }

    /// <summary>
    /// Explains a prediction using similar examples from a prototype set.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="prototypes">The set of prototype examples.</param>
    /// <param name="prototypeLabels">Optional labels for each prototype.</param>
    /// <param name="numPrototypes">Number of similar prototypes to return.</param>
    /// <param name="metric">Distance metric to use.</param>
    /// <returns>Prototype explanation with similar and contrasting examples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prototype explanations work by saying "this prediction is like
    /// these examples from the training set." This is intuitive because humans naturally explain
    /// by comparison.
    ///
    /// The explanation includes:
    /// - Similar prototypes: Examples close to the input with the same prediction
    /// - Contrast prototypes: Examples with different predictions
    /// - Distinguishing features: What makes this input different from contrasts
    ///
    /// Example:
    /// <code>
    /// var protoResult = result.ExplainWithPrototypes(instance, trainingData, labels, 5);
    /// // protoResult.SimilarPrototypes shows similar examples
    /// // protoResult.DistinguishingFeatures shows key differences
    /// </code>
    /// </para>
    /// </remarks>
    public PrototypeExplanation<T> ExplainWithPrototypes(Vector<T> instance, Matrix<T> prototypes, Vector<T>? prototypeLabels = null, int numNeighbors = 5, AiDotNet.Interpretability.Explainers.DistanceMetric metric = AiDotNet.Interpretability.Explainers.DistanceMetric.Euclidean)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction();

        var explainer = new PrototypeExplainer<T>(
            predictFunc,
            prototypes,
            prototypeLabels,
            numNeighbors,
            metric,
            options.FeatureNames);
        return explainer.Explain(instance);
    }

    /// <summary>
    /// Generates a contrastive explanation answering "Why X and not Y?"
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="factClass">The actual predicted class (the "fact").</param>
    /// <param name="foilClass">The alternative class to contrast against (the "foil").</param>
    /// <returns>Contrastive explanation with pertinent positives and negatives.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Contrastive explanations answer questions like "Why did you
    /// predict cat instead of dog?" This is how humans naturally explain decisions.
    ///
    /// The explanation includes:
    /// - Pertinent Positives: Features that support the fact (why it IS a cat)
    /// - Pertinent Negatives: Features that, if changed, would flip to the foil (why it's NOT a dog)
    /// - Feature contributions: How each feature affects the fact vs foil decision
    ///
    /// Example:
    /// <code>
    /// var contrast = result.GetContrastiveExplanation(instance, factClass: 3, foilClass: 7);
    /// // contrast.PertinentPositives: features supporting class 3
    /// // contrast.PertinentNegatives: features that would flip to class 7
    /// </code>
    /// </para>
    /// </remarks>
    public ContrastiveExplanation<T> GetContrastiveExplanation(Vector<T> instance, int factClass, int foilClass)
    {
        if (Model is null)
            throw new InvalidOperationException("Model is not initialized.");

        var options = InterpretabilityOptions ?? new InterpretabilityOptions();
        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction();

        var explainer = new ContrastiveExplainer<T>(
            predictFunc,
            instance.Length,
            options.FeatureNames);
        return explainer.Explain(instance, factClass, foilClass);
    }

    /// <summary>
    /// Creates a prediction function that takes a Matrix and returns a Vector of predictions.
    /// </summary>
    private Func<Matrix<T>, Vector<T>> CreatePredictionFunction()
    {
        return (Matrix<T> inputMatrix) =>
        {
            var predictions = new T[inputMatrix.Rows];
            for (int i = 0; i < inputMatrix.Rows; i++)
            {
                var row = inputMatrix.GetRow(i);
                var input = ConvertVectorToInput(row);
                var output = Model!.Predict(input);
                predictions[i] = ConvertOutputToScalar(output);
            }
            return new Vector<T>(predictions);
        };
    }

    /// <summary>
    /// Converts a Vector to the model's TInput type.
    /// </summary>
    private TInput ConvertVectorToInput(Vector<T> vector)
    {
        object result;

        if (typeof(TInput) == typeof(Vector<T>))
        {
            result = vector;
        }
        else if (typeof(TInput) == typeof(Matrix<T>))
        {
            // Create a single-row matrix
            var matrix = new Matrix<T>(1, vector.Length);
            for (int j = 0; j < vector.Length; j++)
                matrix[0, j] = vector[j];
            result = matrix;
        }
        else if (typeof(TInput) == typeof(Tensor<T>))
        {
            result = Tensor<T>.FromVector(vector, new[] { 1, vector.Length });
        }
        else
        {
            throw new NotSupportedException($"Cannot convert Vector<T> to {typeof(TInput).Name} for interpretability methods.");
        }

        return (TInput)result;
    }

    /// <summary>
    /// Converts the model's TOutput to a scalar value.
    /// </summary>
    /// <remarks>
    /// For multi-output models (Vector/Matrix/Tensor with more than one element),
    /// only the first element is used. This is intentional for univariate interpretability methods.
    /// </remarks>
    private T ConvertOutputToScalar(TOutput output)
    {
        if (output is T scalar)
            return scalar;

        if (output is Vector<T> vector && vector.Length > 0)
        {
            // Note: For multi-output, we use only the first element
            return vector[0];
        }

        if (output is Matrix<T> matrix && matrix.Rows > 0 && matrix.Columns > 0)
        {
            // Note: For multi-output, we use only the first element
            return matrix[0, 0];
        }

        if (output is Tensor<T> tensor && tensor.Length > 0)
        {
            // Note: For multi-output, we use only the first element
            return tensor.ToVector()[0];
        }

        throw new NotSupportedException($"Cannot convert {typeof(TOutput).Name} to scalar for interpretability methods.");
    }

    /// <summary>
    /// Computes R² score for regression evaluation.
    /// </summary>
    private static T ComputeR2Score(Vector<T> actual, Vector<T> predicted)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (actual.Length != predicted.Length || actual.Length == 0)
            return numOps.Zero;

        double meanActual = 0;
        for (int i = 0; i < actual.Length; i++)
            meanActual += numOps.ToDouble(actual[i]);
        meanActual /= actual.Length;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double a = numOps.ToDouble(actual[i]);
            double p = numOps.ToDouble(predicted[i]);
            ssRes += (a - p) * (a - p);
            ssTot += (a - meanActual) * (a - meanActual);
        }

        if (ssTot < 1e-10)
            return numOps.Zero;

        return numOps.FromDouble(1 - ssRes / ssTot);
    }

    /// <summary>
    /// Creates a scalar prediction function that takes a Vector and returns a scalar value.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This helper converts the model's predict method into a simple
    /// function that gradient-based explainers can use. The function takes feature values
    /// as input and returns a single prediction value.</para>
    /// </remarks>
    private Func<Vector<T>, T> CreateScalarPredictionFunction()
    {
        return (Vector<T> input) =>
        {
            var modelInput = ConvertVectorToInput(input);
            var output = Model!.Predict(modelInput);
            return ConvertOutputToScalar(output);
        };
    }

    /// <summary>
    /// Attempts to create a gradient function from the model if it supports gradients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gradients tell us how the output changes with respect to
    /// small changes in each input feature. If the model doesn't support analytical gradients,
    /// this returns null and the explainer will fall back to numerical approximation.</para>
    /// </remarks>
    private Func<Vector<T>, Vector<T>>? TryCreateGradientFunction()
    {
        // Check if the model supports gradient computation for interpretability
        if (Model is IInterpretableNeuralNetwork<T, TInput, TOutput> interpretableNet)
        {
            return (Vector<T> input) =>
            {
                var tensor = Tensor<T>.FromVector(input, new[] { 1, input.Length });
                var gradTensor = interpretableNet.ComputeGradient(tensor);
                return gradTensor.ToVector();
            };
        }

        // For other model types, return null to use numerical approximation
        return null;
    }

    /// <summary>
    /// Creates a prediction function for tensor input/output (used by CNN explainers).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> CNNs work with tensors (multi-dimensional arrays) rather than
    /// simple vectors. This creates a function that can handle tensor data for Grad-CAM and
    /// similar visual explanation methods.</para>
    /// </remarks>
    private Func<Tensor<T>, Tensor<T>> CreateTensorPredictionFunction()
    {
        return (Tensor<T> input) =>
        {
            if (typeof(TInput) == typeof(Tensor<T>))
            {
                var output = Model!.Predict((TInput)(object)input);
                if (output is Tensor<T> tensorOut)
                    return tensorOut;
                if (output is Vector<T> vecOut)
                    return Tensor<T>.FromVector(vecOut, new[] { 1, vecOut.Length });
                throw new InvalidOperationException("Model output cannot be converted to Tensor.");
            }

            // Convert tensor to appropriate input type
            var vector = input.ToVector();
            var modelInput = ConvertVectorToInput(vector);
            var modelOutput = Model!.Predict(modelInput);

            if (modelOutput is Tensor<T> outTensor)
                return outTensor;
            if (modelOutput is Vector<T> outVec)
                return Tensor<T>.FromVector(outVec, new[] { 1, outVec.Length });

            var scalar = ConvertOutputToScalar(modelOutput);
            return Tensor<T>.FromVector(new Vector<T>(new[] { scalar }), new[] { 1, 1 });
        };
    }

    /// <summary>
    /// Attempts to create a function that extracts feature maps and their gradients from a CNN.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Grad-CAM needs access to intermediate feature maps in a CNN
    /// and their gradients. If the model doesn't expose these, this returns null and the
    /// explainer will fall back to occlusion-based methods.</para>
    /// </remarks>
    private Func<Tensor<T>, int, (Tensor<T> featureMaps, Tensor<T> gradients)>? TryCreateFeatureGradientFunction()
    {
        // Check if model is a CNN that can provide feature map gradients for Grad-CAM
        if (Model is IConvolutionalNetwork<T, TInput, TOutput> cnnNet)
        {
            return (Tensor<T> input, int targetClass) =>
            {
                return cnnNet.GetFeatureMapsAndGradients(input, targetClass);
            };
        }

        // Return null to use occlusion-based fallback
        return null;
    }

    /// <summary>
    /// Creates a function that extracts attention matrices from transformer layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Transformer models use attention to determine which parts
    /// of the input to focus on. This function extracts those attention patterns for
    /// visualization. If the model isn't a transformer, it returns empty attention.</para>
    /// </remarks>
    private Func<Tensor<T>, List<Tensor<T>>> TryCreateAttentionExtractor()
    {
        return (Tensor<T> input) =>
        {
            // Check if model is a transformer that can provide attention weights
            if (Model is ITransformerNetwork<T, TInput, TOutput> transformerNet)
            {
                return transformerNet.GetAttentionWeights(input);
            }

            // Return empty list if model doesn't support attention extraction
            return new List<Tensor<T>>();
        };
    }

    /// <summary>
    /// Creates a function that returns both output and layer activations for DeepLIFT.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DeepLIFT needs to see what happens at each layer of the
    /// network, not just the final output. This function captures those intermediate values
    /// called "activations" so we can trace how the network processed the input.</para>
    /// </remarks>
    private Func<Vector<T>, (T output, Vector<T>[] layerActivations)> CreateActivationFunction()
    {
        return (Vector<T> input) =>
        {
            // Check if model supports activation extraction for DeepLIFT
            if (Model is IInterpretableNeuralNetwork<T, TInput, TOutput> interpretableNet)
            {
                var tensor = Tensor<T>.FromVector(input, new[] { 1, input.Length });
                var (output, activations) = interpretableNet.ForwardWithActivations(tensor);
                var scalar = output.ToVector()[0];
                var layerActivs = activations.Select(a => a.ToVector()).ToArray();
                return (scalar, layerActivs);
            }

            // Fallback: just return output with input as only "activation"
            var modelInput = ConvertVectorToInput(input);
            var modelOutput = Model!.Predict(modelInput);
            var outputScalar = ConvertOutputToScalar(modelOutput);
            return (outputScalar, new[] { input });
        };
    }

    /// <summary>
    /// Creates a function that returns output, activations, and weights for LRP.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> LRP (Layer-wise Relevance Propagation) needs detailed
    /// information about the network: what values flow through each layer (activations)
    /// and how layers are connected (weights). This enables tracing importance backwards.</para>
    /// </remarks>
    private Func<Vector<T>, (T output, Vector<T>[] activations, Matrix<T>[] weights)> CreateNetworkInfoFunction()
    {
        return (Vector<T> input) =>
        {
            // Check if model supports detailed network information extraction for LRP
            if (Model is IInterpretableNeuralNetwork<T, TInput, TOutput> interpretableNet)
            {
                var tensor = Tensor<T>.FromVector(input, new[] { 1, input.Length });
                var (output, activations, weights) = interpretableNet.ForwardWithNetworkInfo(tensor);
                var scalar = output.ToVector()[0];
                var layerActivs = activations.Select(a => a.ToVector()).ToArray();
                return (scalar, layerActivs, weights);
            }

            // Fallback: return minimal info
            var modelInput = ConvertVectorToInput(input);
            var modelOutput = Model!.Predict(modelInput);
            var outputScalar = ConvertOutputToScalar(modelOutput);

            // Return input as activation and identity as weight
            var identity = Matrix<T>.CreateIdentity(input.Length);
            return (outputScalar, new[] { input }, new[] { identity });
        };
    }

    /// <summary>
    /// Creates a function that returns the predicted class index.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For classification models, we often need to know which
    /// class was predicted (as an index like 0, 1, 2...) rather than the raw prediction
    /// values. This function finds the class with the highest probability.</para>
    /// </remarks>
    private Func<Vector<T>, int> CreateClassificationFunction()
    {
        return (Vector<T> input) =>
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            var modelInput = ConvertVectorToInput(input);
            var output = Model!.Predict(modelInput);

            // If output is a vector, return index of max
            if (output is Vector<T> probs)
            {
                int maxIdx = 0;
                T maxVal = probs[0];
                for (int i = 1; i < probs.Length; i++)
                {
                    if (numOps.GreaterThan(probs[i], maxVal))
                    {
                        maxVal = probs[i];
                        maxIdx = i;
                    }
                }
                return maxIdx;
            }

            if (output is Tensor<T> tensor)
            {
                var vec = tensor.ToVector();
                int maxIdx = 0;
                T maxVal = vec[0];
                for (int i = 1; i < vec.Length; i++)
                {
                    if (numOps.GreaterThan(vec[i], maxVal))
                    {
                        maxVal = vec[i];
                        maxIdx = i;
                    }
                }
                return maxIdx;
            }

            // Binary classification: return 0 or 1 based on threshold
            var scalar = ConvertOutputToScalar(output);
            return numOps.GreaterThan(scalar, numOps.FromDouble(0.5)) ? 1 : 0;
        };
    }

    /// <summary>
    /// Creates a function that returns class probabilities.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For classification, we often need the full probability
    /// distribution across all classes, not just the final prediction. This enables
    /// contrastive explanations ("why class A and not class B?").</para>
    /// </remarks>
    private Func<Vector<T>, Vector<T>> CreateClassProbabilityFunction()
    {
        return (Vector<T> input) =>
        {
            var modelInput = ConvertVectorToInput(input);
            var output = Model!.Predict(modelInput);

            if (output is Vector<T> probs)
                return probs;

            if (output is Tensor<T> tensor)
                return tensor.ToVector();

            // Single output: treat as binary with [1-p, p]
            var numOps = MathHelper.GetNumericOperations<T>();
            var scalar = ConvertOutputToScalar(output);
            return new Vector<T>(new[] { numOps.Subtract(numOps.One, scalar), scalar });
        };
    }

    /// <summary>
    /// Creates a vector prediction function (Vector in, Vector out).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Many explainers need a function that takes feature values
    /// and returns a vector of predictions (e.g., class probabilities). This wraps the model
    /// to provide that interface.</para>
    /// </remarks>
    private Func<Vector<T>, Vector<T>> CreateVectorPredictionFunction()
    {
        return (Vector<T> input) =>
        {
            var modelInput = ConvertVectorToInput(input);
            var output = Model!.Predict(modelInput);

            if (output is Vector<T> vec)
                return vec;

            if (output is Tensor<T> tensor)
                return tensor.ToVector();

            // Scalar output: return as single-element vector
            var scalar = ConvertOutputToScalar(output);
            return new Vector<T>(new[] { scalar });
        };
    }

    /// <summary>
    /// Creates a gradient function that takes a vector and output index, returns gradients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some explainers need gradients with respect to a specific
    /// output (e.g., a specific class probability). This function computes gradients for
    /// the specified output index.</para>
    /// </remarks>
    private Func<Vector<T>, int, Vector<T>>? TryCreateIndexedGradientFunction()
    {
        if (Model is IInterpretableNeuralNetwork<T, TInput, TOutput> interpretableNet)
        {
            return (Vector<T> input, int outputIndex) =>
            {
                var tensor = Tensor<T>.FromVector(input, new[] { 1, input.Length });
                var gradTensor = interpretableNet.ComputeGradient(tensor);
                return gradTensor.ToVector();
            };
        }

        // Return null to use numerical approximation
        return null;
    }

    /// <summary>
    /// Creates a function to get feature maps at a specific layer for a given class.
    /// </summary>
    private Func<Tensor<T>, int, Tensor<T>>? TryCreateFeatureMapFunction()
    {
        if (Model is IConvolutionalNetwork<T, TInput, TOutput> cnnNet)
        {
            return (Tensor<T> input, int targetClass) =>
            {
                var (featureMaps, _) = cnnNet.GetFeatureMapsAndGradients(input, targetClass);
                return featureMaps;
            };
        }

        return null;
    }

    /// <summary>
    /// Creates a function to get gradients for tensor input at specific output and layer.
    /// </summary>
    private Func<Tensor<T>, int, int, Tensor<T>>? TryCreateTensorGradientFunction()
    {
        if (Model is IConvolutionalNetwork<T, TInput, TOutput> cnnNet)
        {
            return (Tensor<T> input, int targetClass, int layerIndex) =>
            {
                var (_, gradients) = cnnNet.GetFeatureMapsAndGradients(input, targetClass);
                return gradients;
            };
        }

        return null;
    }

    /// <summary>
    /// Creates a function to get attention weights at a specific layer.
    /// </summary>
    private Func<Tensor<T>, int, Tensor<T>>? TryCreateAttentionWeightsFunction()
    {
        if (Model is ITransformerNetwork<T, TInput, TOutput> transformerNet)
        {
            return (Tensor<T> input, int layerIndex) =>
            {
                var attentionWeights = transformerNet.GetAttentionWeights(input);
                if (layerIndex < attentionWeights.Count)
                    return attentionWeights[layerIndex];

                // Return zeros if layer doesn't exist
                var numOps = MathHelper.GetNumericOperations<T>();
                return new Tensor<T>(new[] { 1, 1 });
            };
        }

        return null;
    }

    /// <summary>
    /// Creates a function to get layer activations for DeepLIFT.
    /// </summary>
    private Func<Vector<T>, Vector<T>>? TryCreateActivationsFunction()
    {
        if (Model is IInterpretableNeuralNetwork<T, TInput, TOutput> interpretableNet)
        {
            return (Vector<T> input) =>
            {
                var tensor = Tensor<T>.FromVector(input, new[] { 1, input.Length });
                var (_, activations) = interpretableNet.ForwardWithActivations(tensor);
                // Return concatenated activations
                if (activations.Length > 0)
                    return activations[activations.Length - 1].ToVector();
                return input;
            };
        }

        return null;
    }

    /// <summary>
    /// Creates a function to compute DeepLIFT multipliers.
    /// </summary>
    private Func<Vector<T>, Vector<T>, Vector<T>>? TryCreateMultipliersFunction()
    {
        // Multiplier computation is typically handled within DeepLIFT itself
        // Return null to use the explainer's default implementation
        return null;
    }

    /// <summary>
    /// Creates a function to get all layer activations for LRP.
    /// </summary>
    private Func<Vector<T>, Vector<T>[]>? TryCreateLayerActivationsFunction()
    {
        if (Model is IInterpretableNeuralNetwork<T, TInput, TOutput> interpretableNet)
        {
            return (Vector<T> input) =>
            {
                var tensor = Tensor<T>.FromVector(input, new[] { 1, input.Length });
                var (_, activations) = interpretableNet.ForwardWithActivations(tensor);
                return activations.Select(a => a.ToVector()).ToArray();
            };
        }

        return null;
    }

    /// <summary>
    /// Creates a function to get layer weights for LRP.
    /// </summary>
    private Func<int, Matrix<T>>? TryCreateLayerWeightsFunction()
    {
        if (Model is IInterpretableNeuralNetwork<T, TInput, TOutput> interpretableNet)
        {
            return (int layerIndex) =>
            {
                // Get weights by doing a forward pass and extracting weights
                var numOps = MathHelper.GetNumericOperations<T>();
                var dummyInput = new Vector<T>(1);
                var tensor = Tensor<T>.FromVector(dummyInput, new[] { 1, 1 });
                var (_, _, weights) = interpretableNet.ForwardWithNetworkInfo(tensor);

                if (layerIndex < weights.Length)
                    return weights[layerIndex];

                // Return identity if layer doesn't exist
                return Matrix<T>.CreateIdentity(1);
            };
        }

        return null;
    }

    #endregion

    #region Training Infrastructure Public Accessors

    /// <summary>
    /// Gets the experiment run associated with this model, if experiment tracking was configured.
    /// </summary>
    /// <returns>The experiment run, or null if experiment tracking was not used.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides access to the training run for post-training logging.
    ///
    /// Example:
    /// <code>
    /// var run = result.GetExperimentRun();
    /// if (run != null)
    /// {
    ///     run.LogMetric("production_accuracy", 0.92);
    ///     run.AddNote("Deployed to production");
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public IExperimentRun<T>? GetExperimentRun() => ExperimentRun;

    /// <summary>
    /// Gets the experiment tracker used during training, if configured.
    /// </summary>
    /// <returns>The experiment tracker, or null if not configured.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to compare training runs or start new experiments.
    ///
    /// Example:
    /// <code>
    /// var tracker = result.GetExperimentTracker();
    /// if (tracker != null)
    /// {
    ///     var allRuns = tracker.ListRuns(experimentId);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public IExperimentTracker<T>? GetExperimentTracker() => ExperimentTracker;

    /// <summary>
    /// Gets the checkpoint manager for model persistence operations.
    /// </summary>
    /// <returns>The checkpoint manager, or null if not configured.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to save model states or load previous checkpoints.
    ///
    /// Example:
    /// <code>
    /// var manager = result.GetCheckpointManager();
    /// if (manager != null)
    /// {
    ///     manager.SaveCheckpoint("after_finetuning", model, metrics);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public ICheckpointManager<T, TInput, TOutput>? GetCheckpointManager() => CheckpointManager;

    /// <summary>
    /// Gets the model registry for version and lifecycle management.
    /// </summary>
    /// <returns>The model registry, or null if not configured.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to manage model versions and stage transitions.
    ///
    /// Example:
    /// <code>
    /// var registry = result.GetModelRegistry();
    /// if (registry != null)
    /// {
    ///     registry.TransitionModelStage("my-model", 1, ModelStage.Production);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public IModelRegistry<T, TInput, TOutput>? GetModelRegistry() => ModelRegistry;

    /// <summary>
    /// Gets the training monitor for accessing training diagnostics.
    /// </summary>
    /// <returns>The training monitor, or null if not configured.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to analyze training history and diagnostics.
    ///
    /// Example:
    /// <code>
    /// var monitor = result.GetTrainingMonitor();
    /// if (monitor != null)
    /// {
    ///     var history = monitor.GetMetricsHistory();
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public ITrainingMonitor<T>? GetTrainingMonitor() => TrainingMonitor;

    /// <summary>
    /// Gets the hyperparameter optimization result, if optimization was used.
    /// </summary>
    /// <returns>The optimization result containing all trials, or null if optimization was not used.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to analyze which hyperparameters worked best.
    ///
    /// Example:
    /// <code>
    /// var hpoResult = result.GetHyperparameterOptimizationResult();
    /// if (hpoResult != null)
    /// {
    ///     Console.WriteLine($"Best params: {hpoResult.BestParameters}");
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public HyperparameterOptimizationResult<T>? GetHyperparameterOptimizationResult() => HyperparameterOptimizationResult;

    /// <summary>
    /// Gets training infrastructure metadata as a dictionary.
    /// </summary>
    /// <returns>A dictionary containing all training infrastructure metadata.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides a convenient way to access all training metadata at once.
    ///
    /// Includes:
    /// - ExperimentRunId, ExperimentId - Experiment tracking IDs
    /// - ModelVersion, RegisteredModelName - Model registry info
    /// - CheckpointPath - Where the model was checkpointed
    /// - DataVersionHash - Training data version
    /// - HyperparameterTrialId - Which optimization trial produced this model
    ///
    /// Example:
    /// <code>
    /// var metadata = result.GetTrainingInfrastructureMetadata();
    /// Console.WriteLine($"Run ID: {metadata["ExperimentRunId"]}");
    /// Console.WriteLine($"Model Version: {metadata["ModelVersion"]}");
    /// </code>
    /// </para>
    /// </remarks>
    public Dictionary<string, object?> GetTrainingInfrastructureMetadata()
    {
        return new Dictionary<string, object?>
        {
            ["ExperimentRunId"] = ExperimentRunId,
            ["ExperimentId"] = ExperimentId,
            ["ModelVersion"] = ModelVersion,
            ["RegisteredModelName"] = RegisteredModelName,
            ["CheckpointPath"] = CheckpointPath,
            ["DataVersionHash"] = DataVersionHash,
            ["HyperparameterTrialId"] = HyperparameterTrialId
        };
    }

    /// <summary>
    /// Gets the hyperparameters used for training.
    /// </summary>
    /// <returns>A dictionary of hyperparameter names to values, or null if not tracked.</returns>
    public Dictionary<string, object>? GetHyperparameters() => Hyperparameters;

    /// <summary>
    /// Gets the training metrics history.
    /// </summary>
    /// <returns>A dictionary mapping metric names to their values over time, or null if not tracked.</returns>
    public Dictionary<string, List<double>>? GetTrainingMetricsHistory() => TrainingMetricsHistory;

    /// <summary>
    /// Gets experiment tracking information as a structured object.
    /// </summary>
    /// <returns>An ExperimentInfo object containing experiment tracking data, or null if experiment tracking was not used.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides type-safe access to experiment tracking data.
    ///
    /// Example:
    /// <code>
    /// var expInfo = result.GetExperimentInfo();
    /// if (expInfo != null)
    /// {
    ///     Console.WriteLine($"Experiment: {expInfo.ExperimentId}");
    ///     Console.WriteLine($"Run: {expInfo.RunId}");
    ///
    ///     // Log additional metrics post-training
    ///     if (expInfo.ExperimentRun != null)
    ///     {
    ///         expInfo.ExperimentRun.LogMetric("production_accuracy", 0.92);
    ///     }
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public ExperimentInfo<T>? GetExperimentInfo()
    {
        // Return null if no experiment tracking was used
        if (ExperimentRunId == null && ExperimentId == null && ExperimentRun == null && ExperimentTracker == null)
        {
            return null;
        }

        return new ExperimentInfo<T>(
            ExperimentId,
            ExperimentRunId,
            ExperimentRun,
            ExperimentTracker,
            TrainingMetricsHistory,
            Hyperparameters,
            HyperparameterTrialId,
            DataVersionHash
        );
    }

    /// <summary>
    /// Gets model registry information as a structured object.
    /// </summary>
    /// <returns>A ModelRegistryInfo object containing registry data, or null if model registry was not used.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides type-safe access to model versioning and registry data.
    ///
    /// Example:
    /// <code>
    /// var registryInfo = result.GetModelRegistryInfo();
    /// if (registryInfo != null)
    /// {
    ///     Console.WriteLine($"Model: {registryInfo.RegisteredName} v{registryInfo.Version}");
    ///
    ///     // Promote to production
    ///     if (registryInfo.Registry != null)
    ///     {
    ///         registryInfo.Registry.TransitionModelStage(
    ///             registryInfo.RegisteredName,
    ///             registryInfo.Version ?? 1,
    ///             ModelStage.Production);
    ///     }
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public ModelRegistryInfo<T, TInput, TOutput>? GetModelRegistryInfo()
    {
        // Return null if no model registry was used
        if (ModelVersion == null && RegisteredModelName == null && ModelRegistry == null)
        {
            return null;
        }

        return new ModelRegistryInfo<T, TInput, TOutput>(
            RegisteredModelName,
            ModelVersion,
            ModelRegistry,
            CheckpointPath,
            CheckpointManager
        );
    }

    #endregion

    /// <summary>
    /// Creates a copy of this AiModelResult with deep-copied core model components.
    /// </summary>
    /// <returns>A new AiModelResult instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>Deep-copied components</b> (independent copies, mutations don't affect original):
    /// <list type="bullet">
    ///   <item><description>Model - The underlying predictive model</description></item>
    ///   <item><description>OptimizationResult - Training results and metrics</description></item>
    ///   <item><description>NormalizationInfo - Data normalization parameters</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Shallow-copied components</b> (shared references, mutations affect both copies):
    /// <list type="bullet">
    ///   <item><description>BiasDetector, FairnessEvaluator - Ethical AI components</description></item>
    ///   <item><description>RagRetriever, RagReranker, RagGenerator, QueryProcessors - RAG components</description></item>
    ///   <item><description>KnowledgeGraph, GraphStore, HybridGraphRetriever - Graph RAG components</description></item>
    ///   <item><description>MetaLearner, MetaTrainingResult - Meta-learning components</description></item>
    ///   <item><description>PromptTemplate, PromptChain, PromptOptimizer - Prompt engineering components</description></item>
    ///   <item><description>FewShotExampleSelector, PromptAnalyzer, PromptCompressor - Prompt engineering components</description></item>
    ///   <item><description>Tokenizer, TokenizationConfig - Tokenization components</description></item>
    ///   <item><description>AgentConfig, AgentRecommendation, ReasoningConfig - Agent/reasoning config</description></item>
    ///   <item><description>DeploymentConfiguration, InferenceOptimizationConfig - Deployment config</description></item>
    ///   <item><description>LoRAConfiguration, CrossValidationResult - Training config</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The shallow-copied components are typically stateless configuration objects or services
    /// that can be safely shared. If you need independent copies of these components, you should
    /// create new instances manually before calling DeepCopy.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new model that can be modified independently
    /// from the original for its core prediction behavior (model weights, normalization).
    /// However, configuration objects like prompt templates are shared between the original
    /// and the copy - if you modify them, both copies will see the change.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Cannot deep copy AiModelResult with null Model.");
        }

        var clonedModel = Model.DeepCopy();
        var clonedOptimizationResult = OptimizationResult.DeepCopy();

        // Update OptimizationResult.BestSolution to reference the cloned model
        // This ensures metadata consistency across the deep copy
        clonedOptimizationResult.BestSolution = clonedModel;

        var clonedNormalizationInfo = NormalizationInfo.DeepCopy();

        // Preserve all configuration properties to ensure deployment behavior, model adaptation,
        // training history, and Graph RAG configuration are maintained across deep copy
        var options = new AiModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = clonedOptimizationResult,
            NormalizationInfo = clonedNormalizationInfo,
            BiasDetector = BiasDetector,
            FairnessEvaluator = FairnessEvaluator,
            RagRetriever = RagRetriever,
            RagReranker = RagReranker,
            RagGenerator = RagGenerator,
            QueryProcessors = QueryProcessors,
            LoRAConfiguration = LoRAConfiguration,
            CrossValidationResult = CrossValidationResult,
            AutoMLSummary = AutoMLSummary,
            AgentConfig = AgentConfig,
            AgentRecommendation = AgentRecommendation,
            DeploymentConfiguration = DeploymentConfiguration,
            // JIT compilation is model-specific, don't copy
            InferenceOptimizationConfig = InferenceOptimizationConfig,
            ReasoningConfig = ReasoningConfig,
            KnowledgeGraph = KnowledgeGraph,
            GraphStore = GraphStore,
            HybridGraphRetriever = HybridGraphRetriever,
            MetaLearner = MetaLearner,
            MetaTrainingResult = MetaTrainingResult,
            Tokenizer = Tokenizer,
            TokenizationConfig = TokenizationConfig,
            PromptTemplate = PromptTemplate,
            PromptChain = PromptChain,
            PromptOptimizer = PromptOptimizer,
            FewShotExampleSelector = FewShotExampleSelector,
            PromptAnalyzer = PromptAnalyzer,
            PromptCompressor = PromptCompressor,
            // Training Infrastructure - shallow copy (shared references)
            ExperimentRun = ExperimentRun,
            ExperimentTracker = ExperimentTracker,
            CheckpointManager = CheckpointManager,
            ModelRegistry = ModelRegistry,
            TrainingMonitor = TrainingMonitor,
            HyperparameterOptimizationResult = HyperparameterOptimizationResult,
            ExperimentRunId = ExperimentRunId,
            ExperimentId = ExperimentId,
            ModelVersion = ModelVersion,
            RegisteredModelName = RegisteredModelName,
            CheckpointPath = CheckpointPath,
            DataVersionHash = DataVersionHash,
            HyperparameterTrialId = HyperparameterTrialId,
            Hyperparameters = Hyperparameters,
            TrainingMetricsHistory = TrainingMetricsHistory
        };

        return new AiModelResult<T, TInput, TOutput>(options);
    }

    /// <summary>
    /// Creates a shallow copy of this AiModelResult.
    /// </summary>
    /// <returns>A new AiModelResult instance that is a shallow copy of this one.</returns>
    /// <remarks>
    /// This method delegates to WithParameters to ensure consistency in how OptimizationResult is handled.
    /// The cloned instance will have a new model with the same parameters and updated OptimizationResult metadata.
    /// </remarks>
    public IFullModel<T, TInput, TOutput> Clone()
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Cannot clone AiModelResult with null Model.");
        }

        return WithParameters(Model.GetParameters());
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the entire AiModelResult object, including the model, optimization results, 
    /// normalization information, and metadata. The model is serialized using its own Serialize() method, 
    /// ensuring that model-specific serialization logic is properly applied. The other components are 
    /// serialized using JSON. This approach ensures that each component of the AiModelResult is 
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
            var modelToSerialize = Model ?? OptimizationResult?.BestSolution;
            if (modelToSerialize != null)
            {
                // Persist a model-owned snapshot so deserialization can restore state without relying on JSON for model internals.
                SerializedModelData = modelToSerialize.Serialize();

                // Refresh metadata for consistency and to keep ModelMetaData aligned with the persisted snapshot.
                ModelMetaData = modelToSerialize.GetModelMetadata();
            }

            // Create JSON settings with custom converters and safe type binding
            // Use TypeNameHandling.Auto instead of All to minimize type info exposure
            // Auto only emits type info when actual type differs from declared type
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.Auto,
                SerializationBinder = new SafeSerializationBinder(),
                Converters = JsonConverterRegistry.GetAllConverters(),
                Formatting = Formatting.Indented,
                ContractResolver = new AiModelResultContractResolver()
            };

            // Serialize the object to JSON bytes
            var jsonString = JsonConvert.SerializeObject(this, settings);
            var jsonBytes = Encoding.UTF8.GetBytes(jsonString);

            // Apply compression if configured
            var compressionConfig = DeploymentConfiguration?.Compression;
            if (compressionConfig != null && compressionConfig.Mode != ModelCompressionMode.None)
            {
                return CompressionHelper.Compress(jsonBytes, compressionConfig);
            }

            return jsonBytes;
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
    /// This method reconstructs a AiModelResult object from a serialized byte array. It reads 
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
    /// - Reconstructs the complete AiModelResult object
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
            // Decompress if needed (CompressionHelper automatically detects compressed data)
            var decompressedData = CompressionHelper.DecompressIfNeeded(data);
            var jsonString = Encoding.UTF8.GetString(decompressedData);

            // Create JSON settings with custom converters and safe type binding
            // Use TypeNameHandling.Auto to match serialization and minimize type info exposure
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.Auto,
                SerializationBinder = new SafeSerializationBinder(),
                Converters = JsonConverterRegistry.GetAllConverters(),
                ContractResolver = new AiModelResultContractResolver()
            };

            // Deserialize the object
            var deserializedObject = JsonConvert.DeserializeObject<AiModelResult<T, TInput, TOutput>>(jsonString, settings);

            if (deserializedObject != null)
            {
                OptimizationResult = deserializedObject.OptimizationResult;
                NormalizationInfo = deserializedObject.NormalizationInfo;
                ModelMetaData = deserializedObject.ModelMetaData;
                BiasDetector = deserializedObject.BiasDetector;
                FairnessEvaluator = deserializedObject.FairnessEvaluator;
                InferenceOptimizationConfig = deserializedObject.InferenceOptimizationConfig;
                SerializedModelData = deserializedObject.SerializedModelData;

                // Model is intentionally facade-hidden and is not serialized directly.
                // Prefer preserving the existing instance (e.g., builder-supplied), otherwise use the deserialized skeleton.
                Model ??= deserializedObject.Model ?? deserializedObject.OptimizationResult?.BestSolution;

                // Preserve RAG components and all configuration properties
                RagRetriever = deserializedObject.RagRetriever;
                RagReranker = deserializedObject.RagReranker;
                RagGenerator = deserializedObject.RagGenerator;
                QueryProcessors = deserializedObject.QueryProcessors;
                LoRAConfiguration = deserializedObject.LoRAConfiguration;
                CrossValidationResult = deserializedObject.CrossValidationResult;
                AutoMLSummary = deserializedObject.AutoMLSummary;
                AgentConfig = deserializedObject.AgentConfig;
                AgentRecommendation = deserializedObject.AgentRecommendation;
                DeploymentConfiguration = deserializedObject.DeploymentConfiguration;

                // Reset transient runtime state (will be reinitialized lazily)
                JitCompiledFunction = null;
                _inferenceOptimizer = null;
                _inferenceOptimizedNeuralModel = null;
                _inferenceOptimizationsInitialized = false;

                // Restore the model's internal state from the model-owned serialized payload when available.
                // Fall back to metadata.ModelData for older payloads that stored the snapshot there.
                var modelBytes = SerializedModelData;
                if (modelBytes == null || modelBytes.Length == 0)
                {
                    modelBytes = ModelMetaData?.ModelData ?? [];
                }

                if (Model != null && modelBytes.Length > 0)
                {
                    Model.Deserialize(modelBytes);
                    OptimizationResult.BestSolution = Model;
                }
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
    /// <returns>A new AiModelResult&lt;T&gt; instance loaded from the file.</returns>
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
    /// - Reads the file and deserializes the data into a new AiModelResult object
    /// - Returns the fully loaded model ready for making predictions
    /// 
    /// The model factory is important because:
    /// - Different types of models (linear regression, neural networks, etc.) need different deserialization logic
    /// - The factory knows how to create the right type of model based on information in the saved file
    /// 
    /// For example, you might load a model with:
    /// `var model = AiModelResult<double, Matrix<double>, Vector<double>>.LoadModel(
    ///     "C:\\Models\\house_price_predictor.model", 
    ///     metadata => new LinearRegressionModel<double>());`
    /// </para>
    /// </remarks>
    public static AiModelResult<T, TInput, TOutput> LoadModel(
        string filePath,
        Func<ModelMetadata<T>, IFullModel<T, TInput, TOutput>> modelFactory)
    {
        // First, we need to read the file
        byte[] data = File.ReadAllBytes(filePath);

        // Extract metadata to determine model type
        var metadata = ExtractMetadataFromSerializedData(data);

        // Create a new model instance of the appropriate type
        var model = modelFactory(metadata);

        // Create a new AiModelResult with the model
        var result = new AiModelResult<T, TInput, TOutput>
        {
            Model = model
        };

        // Deserialize the data
        result.Deserialize(data);

        return result;
    }

    private static ModelMetadata<T> ExtractMetadataFromSerializedData(byte[] data)
    {
        // Decompress if needed (CompressionHelper automatically detects compressed data)
        var decompressedData = CompressionHelper.DecompressIfNeeded(data);
        var jsonString = Encoding.UTF8.GetString(decompressedData);
        // Use TypeNameHandling.Auto to match serialization and minimize type info exposure
        var settings = new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.Auto,
            SerializationBinder = new SafeSerializationBinder(),
            Converters = JsonConverterRegistry.GetAllConverters(),
            ConstructorHandling = ConstructorHandling.AllowNonPublicDefaultConstructor
        };
        var deserializedObject = JsonConvert.DeserializeObject<AiModelResult<T, TInput, TOutput>>(jsonString, settings);
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
    /// RAG must be configured via AiModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.
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
                "RAG pipeline not configured. Configure RAG components using AiModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
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
    /// RAG must be configured via AiModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.
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
                "RAG retriever not configured. Configure RAG components using AiModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
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
                "Knowledge graph not configured. Configure Graph RAG using AiModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
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
                "Hybrid graph retriever not configured. Configure Graph RAG with a document store using AiModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
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
                "Knowledge graph not configured. Configure Graph RAG using AiModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
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
                "Knowledge graph not configured. Configure Graph RAG using AiModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
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
                "Knowledge graph not configured. Configure Graph RAG using AiModelBuilder.ConfigureRetrievalAugmentedGeneration() before building the model.");
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
    /// Attaches Graph RAG components to a AiModelResult instance.
    /// </summary>
    /// <param name="knowledgeGraph">The knowledge graph to attach.</param>
    /// <param name="graphStore">The graph store backend to attach.</param>
    /// <param name="hybridGraphRetriever">The hybrid retriever to attach.</param>
    /// <remarks>
    /// This method is internal and used by AiModelBuilder when loading/deserializing models.
    /// Graph RAG components cannot be serialized (they contain file handles, WAL references, etc.),
    /// so the builder automatically reattaches them when loading a model that was configured with Graph RAG.
    /// Users should use AiModelBuilder.LoadModel() which handles this automatically.
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
    /// This method is internal and used by AiModelBuilder during model construction.
    /// </remarks>
    internal void AttachTokenizer(
        ITokenizer? tokenizer,
        TokenizationConfig? config = null)
    {
        Tokenizer = tokenizer;
        TokenizationConfig = config;
    }

    /// <summary>
    /// Attaches prompt engineering components to this result.
    /// </summary>
    /// <param name="promptTemplate">The prompt template for formatting prompts.</param>
    /// <param name="promptChain">The chain for multi-step prompt execution.</param>
    /// <param name="promptOptimizer">The optimizer for improving prompt quality.</param>
    /// <param name="fewShotExampleSelector">The selector for choosing relevant few-shot examples.</param>
    /// <param name="promptAnalyzer">The analyzer for prompt metrics and validation.</param>
    /// <param name="promptCompressor">The compressor for reducing prompt token counts.</param>
    /// <remarks>
    /// This method is internal and used by AiModelBuilder during model construction.
    /// </remarks>
    internal void AttachPromptEngineering(
        IPromptTemplate? promptTemplate,
        IChain<string, string>? promptChain,
        IPromptOptimizer<T>? promptOptimizer,
        IFewShotExampleSelector<T>? fewShotExampleSelector,
        IPromptAnalyzer? promptAnalyzer,
        IPromptCompressor? promptCompressor)
    {
        PromptTemplate = promptTemplate;
        PromptChain = promptChain;
        PromptOptimizer = promptOptimizer;
        FewShotExampleSelector = fewShotExampleSelector;
        PromptAnalyzer = promptAnalyzer;
        PromptCompressor = promptCompressor;
    }

    #region Prompt Engineering Inference Methods

    /// <summary>
    /// Formats a prompt using the configured prompt template.
    /// </summary>
    /// <param name="variables">A dictionary of variable names and their values to substitute into the template.</param>
    /// <returns>The formatted prompt string with all variables substituted.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no prompt template is configured.</exception>
    /// <remarks>
    /// <para>
    /// This method uses the configured IPromptTemplate to render a prompt with the provided
    /// variable values. The template defines the structure of the prompt, and this method
    /// fills in the placeholders with actual values.
    /// </para>
    /// <para><b>For Beginners:</b> A prompt template is like a fill-in-the-blank form for AI prompts.
    ///
    /// Instead of writing different prompts for each use case, you define a template once
    /// with placeholders like {topic} or {language}, then fill them in as needed.
    ///
    /// Example:
    /// <code>
    /// // Template: "Translate the following text from {source} to {target}: {text}"
    /// var variables = new Dictionary&lt;string, string&gt;
    /// {
    ///     ["source"] = "English",
    ///     ["target"] = "Spanish",
    ///     ["text"] = "Hello, how are you?"
    /// };
    ///
    /// string prompt = modelResult.FormatPrompt(variables);
    /// // Result: "Translate the following text from English to Spanish: Hello, how are you?"
    /// </code>
    ///
    /// Benefits:
    /// - Consistent prompt structure across your application
    /// - Easy to modify prompts without changing code
    /// - Reusable templates for common tasks
    /// </para>
    /// </remarks>
    public string FormatPrompt(Dictionary<string, string> variables)
    {
        if (PromptTemplate == null)
            throw new InvalidOperationException(
                "No prompt template configured. Use ConfigurePromptTemplate() in AiModelBuilder.");

        return PromptTemplate.Format(variables);
    }

    /// <summary>
    /// Analyzes a prompt and returns detailed metrics about its structure and characteristics.
    /// </summary>
    /// <param name="prompt">The prompt text to analyze.</param>
    /// <returns>A PromptMetrics object containing token count, cost estimate, complexity score, and detected patterns.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no prompt analyzer is configured.</exception>
    /// <remarks>
    /// <para>
    /// This method uses the configured IPromptAnalyzer to examine a prompt and extract
    /// metrics such as token count, estimated cost, complexity score, variable count,
    /// example count, and detected prompt patterns.
    /// </para>
    /// <para><b>For Beginners:</b> Prompt analysis helps you understand and optimize your prompts.
    ///
    /// When you analyze a prompt, you learn:
    /// - Token count: How many "words" the AI sees (affects cost and limits)
    /// - Estimated cost: How much this prompt will cost in API fees
    /// - Complexity score: How complicated the prompt is (0 = simple, 1 = complex)
    /// - Variable count: How many {placeholders} need to be filled
    /// - Detected patterns: What type of prompt this is (question, instruction, etc.)
    ///
    /// Example:
    /// <code>
    /// string prompt = "You are a helpful assistant. Translate {text} from English to Spanish.";
    /// var metrics = modelResult.AnalyzePrompt(prompt);
    ///
    /// Console.WriteLine($"Tokens: {metrics.TokenCount}");           // e.g., 15
    /// Console.WriteLine($"Estimated cost: ${metrics.EstimatedCost}"); // e.g., $0.0001
    /// Console.WriteLine($"Complexity: {metrics.ComplexityScore}");   // e.g., 0.3
    /// Console.WriteLine($"Variables: {metrics.VariableCount}");      // e.g., 1
    /// Console.WriteLine($"Patterns: {string.Join(", ", metrics.DetectedPatterns)}");
    /// // e.g., "translation, instruction"
    /// </code>
    ///
    /// Use this to:
    /// - Estimate costs before sending to AI
    /// - Identify overly complex prompts that might confuse the model
    /// - Validate that all required variables are present
    /// </para>
    /// </remarks>
    public PromptMetrics AnalyzePrompt(string prompt)
    {
        if (PromptAnalyzer == null)
            throw new InvalidOperationException(
                "No prompt analyzer configured. Use ConfigurePromptAnalyzer() in AiModelBuilder.");

        return PromptAnalyzer.Analyze(prompt);
    }

    /// <summary>
    /// Validates a prompt and returns any detected issues or warnings.
    /// </summary>
    /// <param name="prompt">The prompt text to validate.</param>
    /// <param name="options">Optional validation options to customize the validation behavior.</param>
    /// <returns>A list of PromptIssue objects describing any problems found.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no prompt analyzer is configured.</exception>
    /// <remarks>
    /// <para>
    /// This method uses the configured IPromptAnalyzer to check a prompt for common issues
    /// such as unclosed variables, excessive length, potential injection vulnerabilities,
    /// and other problems that could affect prompt quality or safety.
    /// </para>
    /// <para><b>For Beginners:</b> Validation catches problems with your prompts before you send them.
    ///
    /// Common issues that validation detects:
    /// - Unclosed variable placeholders: "{text" instead of "{text}"
    /// - Prompts that are too long for the model's context window
    /// - Potential prompt injection attempts in user input
    /// - Missing required sections or unclear instructions
    ///
    /// Example:
    /// <code>
    /// string prompt = "Translate {text from English to Spanish."; // Note: unclosed {
    /// var issues = modelResult.ValidatePrompt(prompt);
    ///
    /// foreach (var issue in issues)
    /// {
    ///     Console.WriteLine($"[{issue.Severity}] {issue.Message}");
    ///     // Output: "[Error] Unclosed variable placeholder at position 10"
    /// }
    ///
    /// if (issues.Any(i => i.Severity == IssueSeverity.Error))
    /// {
    ///     // Don't send the prompt - fix the errors first
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public IReadOnlyList<PromptIssue> ValidatePrompt(string prompt, ValidationOptions? options = null)
    {
        if (PromptAnalyzer == null)
            throw new InvalidOperationException(
                "No prompt analyzer configured. Use ConfigurePromptAnalyzer() in AiModelBuilder.");

        return PromptAnalyzer.ValidatePrompt(prompt, options).ToList();
    }

    /// <summary>
    /// Compresses a prompt to reduce its token count while preserving essential meaning.
    /// </summary>
    /// <param name="prompt">The prompt text to compress.</param>
    /// <param name="options">Optional compression options to control the compression behavior.</param>
    /// <returns>A CompressionResult containing the compressed prompt and compression metrics.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no prompt compressor is configured.</exception>
    /// <remarks>
    /// <para>
    /// This method uses the configured IPromptCompressor to reduce the token count of a prompt
    /// while attempting to preserve its essential meaning and effectiveness. Compression can
    /// help reduce API costs and fit longer content into limited context windows.
    /// </para>
    /// <para><b>For Beginners:</b> Prompt compression makes your prompts shorter to save money and fit limits.
    ///
    /// AI APIs charge per token, and models have maximum context sizes. Compression helps by:
    /// - Removing redundant words and phrases
    /// - Shortening verbose explanations
    /// - Eliminating unnecessary whitespace
    /// - Using more concise phrasing
    ///
    /// Example:
    /// <code>
    /// string longPrompt = @"
    ///     Please take the following text and translate it from the English language
    ///     to the Spanish language. Make sure to preserve the original meaning and
    ///     tone of the text as much as possible. Here is the text: {text}";
    ///
    /// var result = modelResult.CompressPrompt(longPrompt);
    ///
    /// Console.WriteLine($"Original tokens: {result.OriginalTokenCount}");    // e.g., 50
    /// Console.WriteLine($"Compressed tokens: {result.CompressedTokenCount}"); // e.g., 20
    /// Console.WriteLine($"Saved: {result.TokensSaved} tokens ({result.CompressionRatio:P0})");
    /// Console.WriteLine($"Compressed: {result.CompressedPrompt}");
    /// // e.g., "Translate English to Spanish, preserving meaning and tone: {text}"
    /// </code>
    ///
    /// Use compression options to control behavior:
    /// <code>
    /// var options = new CompressionOptions
    /// {
    ///     TargetReduction = 0.3,      // Try to reduce by 30%
    ///     PreserveVariables = true,   // Don't remove {placeholders}
    ///     PreserveCodeBlocks = true   // Don't modify code examples
    /// };
    /// var result = modelResult.CompressPrompt(longPrompt, options);
    /// </code>
    /// </para>
    /// </remarks>
    public CompressionResult CompressPrompt(string prompt, CompressionOptions? options = null)
    {
        if (PromptCompressor == null)
            throw new InvalidOperationException(
                "No prompt compressor configured. Use ConfigurePromptCompressor() in AiModelBuilder.");

        return PromptCompressor.CompressWithMetrics(prompt, options ?? CompressionOptions.Default);
    }

    /// <summary>
    /// Executes a prompt chain synchronously with the given input.
    /// </summary>
    /// <param name="input">The initial input to the chain.</param>
    /// <returns>The output string from the chain execution.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no prompt chain is configured.</exception>
    /// <remarks>
    /// <para>
    /// This method executes a multi-step prompt workflow where each step's output becomes
    /// the next step's input. Chains enable complex workflows like translation followed
    /// by summarization, or analysis followed by formatting.
    /// </para>
    /// <para><b>For Beginners:</b> A prompt chain runs multiple AI steps in sequence.
    ///
    /// Instead of doing everything in one big prompt, chains break tasks into steps:
    /// <code>
    /// // Chain example: Translate → Summarize → Extract Keywords
    /// // Step 1: Translate document from Spanish to English
    /// // Step 2: Summarize the translated document
    /// // Step 3: Extract key points as bullet points
    /// </code>
    ///
    /// Each step takes the previous step's output as input.
    ///
    /// Benefits of chains:
    /// - Simpler prompts (each does one thing well)
    /// - Better quality (specialized prompts perform better)
    /// - Easier debugging (inspect intermediate results)
    /// - Flexible workflows (add/remove/modify steps)
    ///
    /// Example:
    /// <code>
    /// string spanishDocument = "Documento en español...";
    /// string result = modelResult.RunChain(spanishDocument);
    ///
    /// Console.WriteLine($"Result: {result}");
    /// </code>
    /// </para>
    /// </remarks>
    public string RunChain(string input)
    {
        if (PromptChain == null)
            throw new InvalidOperationException(
                "No prompt chain configured. Use ConfigurePromptChain() in AiModelBuilder.");

        return PromptChain.Run(input);
    }

    /// <summary>
    /// Executes a prompt chain asynchronously with the given input.
    /// </summary>
    /// <param name="input">The initial input to the chain.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A task that resolves to the output string from the chain execution.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no prompt chain is configured.</exception>
    /// <remarks>
    /// <para>
    /// Async version of RunChain for non-blocking execution. Essential for chains
    /// that make API calls to language models or perform other I/O operations.
    /// </para>
    /// <para><b>For Beginners:</b> Same as RunChain but doesn't block your program.
    ///
    /// Use this version when:
    /// - Running in a web application (keeps server responsive)
    /// - Processing many documents in parallel
    /// - Making actual API calls to language models
    ///
    /// Example:
    /// <code>
    /// string result = await modelResult.RunChainAsync("Input text...");
    /// Console.WriteLine(result);
    /// </code>
    ///
    /// For parallel processing:
    /// <code>
    /// var documents = new[] { "Doc 1", "Doc 2", "Doc 3" };
    /// var tasks = documents.Select(doc => modelResult.RunChainAsync(doc));
    /// var results = await Task.WhenAll(tasks);
    /// </code>
    /// </para>
    /// </remarks>
    public Task<string> RunChainAsync(string input, CancellationToken cancellationToken = default)
    {
        if (PromptChain == null)
            throw new InvalidOperationException(
                "No prompt chain configured. Use ConfigurePromptChain() in AiModelBuilder.");

        return PromptChain.RunAsync(input, cancellationToken);
    }

    /// <summary>
    /// Evaluates a reasoning benchmark using the configured facade (prompt chain or agent reasoning).
    /// </summary>
    /// <typeparam name="TScore">The numeric score type used by the benchmark (for example, double).</typeparam>
    /// <param name="benchmark">The benchmark to evaluate.</param>
    /// <param name="sampleSize">Optional number of problems to evaluate (null for all).</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>The benchmark evaluation result.</returns>
    /// <remarks>
    /// <para>
    /// This method hides the benchmark wiring so users don't have to manually provide a
    /// <c>Func&lt;string, Task&lt;string&gt;&gt;</c>. The default evaluation path is:
    /// </para>
    /// <list type="bullet">
    /// <item><description>Use <see cref="PromptChain"/> (via <see cref="RunChainAsync"/>) when configured.</description></item>
    /// <item><description>Otherwise, use agent reasoning (via <see cref="QuickReasonAsync"/>) when configured.</description></item>
    /// </list>
    /// </remarks>
    public Task<AiDotNet.Reasoning.Benchmarks.Models.BenchmarkResult<TScore>> EvaluateBenchmarkAsync<TScore>(
        IBenchmark<TScore> benchmark,
        int? sampleSize = null,
        CancellationToken cancellationToken = default)
    {
        if (benchmark is null)
        {
            throw new ArgumentNullException(nameof(benchmark));
        }

        Func<string, Task<string>> evaluateFunction;

        if (PromptChain != null)
        {
            evaluateFunction = problem => RunChainAsync(problem, cancellationToken);
        }
        else if (AgentConfig != null && AgentConfig.IsEnabled)
        {
            evaluateFunction = problem => QuickReasonAsync(problem, cancellationToken);
        }
        else
        {
            throw new InvalidOperationException(
                "Benchmark evaluation requires either a prompt chain (ConfigurePromptChain) or agent assistance (ConfigureAgentAssistance).");
        }

        return benchmark.EvaluateAsync(evaluateFunction, sampleSize, cancellationToken);
    }

    /// <summary>
    /// Selects relevant few-shot examples for a given query or context.
    /// </summary>
    /// <param name="query">The query or context to find relevant examples for.</param>
    /// <param name="maxExamples">The maximum number of examples to return.</param>
    /// <returns>A list of FewShotExample objects most relevant to the query.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no few-shot example selector is configured.</exception>
    /// <remarks>
    /// <para>
    /// This method uses the configured IFewShotExampleSelector to find the most relevant
    /// examples from a pool of available examples. The selection can be based on similarity,
    /// diversity, or other strategies depending on the selector implementation.
    /// </para>
    /// <para><b>For Beginners:</b> Few-shot examples teach the AI what you want by showing examples.
    ///
    /// Instead of just describing what you want, you show the AI examples:
    /// <code>
    /// // Without few-shot: "Translate English to Spanish"
    /// // With few-shot: "Translate English to Spanish. Examples:
    /// //   'Hello' -> 'Hola'
    /// //   'Goodbye' -> 'Adiós'
    /// //   Now translate: 'Good morning'"
    /// </code>
    ///
    /// The challenge is choosing which examples to include. This method automatically
    /// selects the most relevant examples for your specific input.
    ///
    /// Example:
    /// <code>
    /// // You have a pool of 100 translation examples
    /// // For the input "How are you?", select the 3 most relevant
    /// var examples = modelResult.SelectFewShotExamples("How are you?", maxExamples: 3);
    ///
    /// // Build your prompt with the selected examples
    /// var prompt = "Translate English to Spanish. Examples:\n";
    /// foreach (var ex in examples)
    /// {
    ///     prompt += $"  '{ex.Input}' -> '{ex.Output}'\n";
    /// }
    /// prompt += "Now translate: How are you?";
    /// </code>
    ///
    /// Selection strategies include:
    /// - Similarity-based: Choose examples most similar to the query
    /// - Diversity-based: Choose examples covering different cases
    /// - Hybrid: Balance similarity and diversity
    /// </para>
    /// </remarks>
    public IReadOnlyList<FewShotExample> SelectFewShotExamples(string query, int maxExamples = 5)
    {
        if (FewShotExampleSelector == null)
            throw new InvalidOperationException(
                "No few-shot example selector configured. Use ConfigureFewShotExampleSelector() in AiModelBuilder.");

        return FewShotExampleSelector.SelectExamples(query, maxExamples);
    }

    /// <summary>
    /// Optimizes a prompt to improve its effectiveness using an evaluation function.
    /// </summary>
    /// <param name="initialPrompt">The initial prompt to optimize.</param>
    /// <param name="evaluationFunction">A function that scores prompt performance (higher scores are better).</param>
    /// <param name="maxIterations">Maximum number of optimization iterations.</param>
    /// <returns>An optimized IPromptTemplate.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no prompt optimizer is configured.</exception>
    /// <remarks>
    /// <para>
    /// This method uses the configured IPromptOptimizer to iteratively improve a prompt.
    /// The optimizer generates variations, evaluates them using your scoring function,
    /// and selects better-performing candidates over multiple iterations.
    /// </para>
    /// <para><b>For Beginners:</b> Prompt optimization automatically improves your prompts through testing.
    ///
    /// How it works:
    /// 1. Start with your initial prompt
    /// 2. Generate variations (different wordings, structures)
    /// 3. Test each variation using your evaluation function
    /// 4. Keep the best-performing versions
    /// 5. Repeat until maxIterations is reached
    ///
    /// You provide the evaluation function that scores how well a prompt works:
    /// <code>
    /// // Evaluation function that tests accuracy
    /// Func&lt;string, double&gt; evaluate = (prompt) =>
    /// {
    ///     double correctCount = 0;
    ///     foreach (var testCase in testSet)
    ///     {
    ///         var result = model.Generate(prompt + testCase.Input);
    ///         if (result == testCase.ExpectedOutput)
    ///             correctCount++;
    ///     }
    ///     return correctCount / testSet.Count; // Returns accuracy 0.0 to 1.0
    /// };
    ///
    /// var optimized = modelResult.OptimizePrompt(
    ///     "Classify sentiment:",
    ///     evaluate,
    ///     maxIterations: 50);
    ///
    /// // Use the optimized template
    /// string finalPrompt = optimized.Format(new Dictionary&lt;string, string&gt; { ["input"] = text });
    /// </code>
    /// </para>
    /// </remarks>
    public IPromptTemplate OptimizePrompt(string initialPrompt, Func<string, T> evaluationFunction, int maxIterations = 100)
    {
        if (PromptOptimizer == null)
            throw new InvalidOperationException(
                "No prompt optimizer configured. Use ConfigurePromptOptimizer() in AiModelBuilder.");

        return PromptOptimizer.Optimize(initialPrompt, evaluationFunction, maxIterations);
    }

    /// <summary>
    /// Optimizes a prompt asynchronously using an async evaluation function.
    /// </summary>
    /// <param name="initialPrompt">The initial prompt to optimize.</param>
    /// <param name="evaluationFunction">An async function that scores prompt performance (higher scores are better).</param>
    /// <param name="maxIterations">Maximum number of optimization iterations.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A task that resolves to an optimized IPromptTemplate.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no prompt optimizer is configured.</exception>
    /// <remarks>
    /// <para>
    /// Async version of OptimizePrompt for when your evaluation function involves
    /// asynchronous operations like API calls or I/O operations.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when your scoring function calls APIs.
    ///
    /// Example with async evaluation:
    /// <code>
    /// // Async evaluation function that calls an API
    /// Func&lt;string, Task&lt;double&gt;&gt; evaluateAsync = async (prompt) =>
    /// {
    ///     var results = await TestWithApiAsync(prompt);
    ///     return CalculateAccuracy(results);
    /// };
    ///
    /// var optimized = await modelResult.OptimizePromptAsync(
    ///     "Classify sentiment:",
    ///     evaluateAsync,
    ///     maxIterations: 50);
    /// </code>
    ///
    /// Benefits:
    /// - Doesn't block your program during optimization
    /// - Can be cancelled if needed
    /// - Handles async API calls efficiently
    /// </para>
    /// </remarks>
    public Task<IPromptTemplate> OptimizePromptAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations = 100,
        CancellationToken cancellationToken = default)
    {
        if (PromptOptimizer == null)
            throw new InvalidOperationException(
                "No prompt optimizer configured. Use ConfigurePromptOptimizer() in AiModelBuilder.");

        return PromptOptimizer.OptimizeAsync(initialPrompt, evaluationFunction, maxIterations, cancellationToken);
    }

    /// <summary>
    /// Checks whether a prompt template is configured and available for use.
    /// </summary>
    /// <returns>True if a prompt template is configured; otherwise, false.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to check if you can call FormatPrompt().
    ///
    /// Example:
    /// <code>
    /// if (modelResult.HasPromptTemplate)
    /// {
    ///     var prompt = modelResult.FormatPrompt(variables);
    /// }
    /// else
    /// {
    ///     // Use a default prompt or throw an error
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public bool HasPromptTemplate => PromptTemplate != null;

    /// <summary>
    /// Checks whether a prompt analyzer is configured and available for use.
    /// </summary>
    /// <returns>True if a prompt analyzer is configured; otherwise, false.</returns>
    public bool HasPromptAnalyzer => PromptAnalyzer != null;

    /// <summary>
    /// Checks whether a prompt compressor is configured and available for use.
    /// </summary>
    /// <returns>True if a prompt compressor is configured; otherwise, false.</returns>
    public bool HasPromptCompressor => PromptCompressor != null;

    /// <summary>
    /// Checks whether a prompt chain is configured and available for use.
    /// </summary>
    /// <returns>True if a prompt chain is configured; otherwise, false.</returns>
    public bool HasPromptChain => PromptChain != null;

    /// <summary>
    /// Checks whether a few-shot example selector is configured and available for use.
    /// </summary>
    /// <returns>True if a few-shot example selector is configured; otherwise, false.</returns>
    public bool HasFewShotExampleSelector => FewShotExampleSelector != null;

    /// <summary>
    /// Checks whether a prompt optimizer is configured and available for use.
    /// </summary>
    /// <returns>True if a prompt optimizer is configured; otherwise, false.</returns>
    public bool HasPromptOptimizer => PromptOptimizer != null;

    #endregion

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
            throw new InvalidOperationException("No tokenizer configured. Use ConfigureTokenizer() in AiModelBuilder.");

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
            throw new InvalidOperationException("No tokenizer configured. Use ConfigureTokenizer() in AiModelBuilder.");

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
            throw new InvalidOperationException("No tokenizer configured. Use ConfigureTokenizer() in AiModelBuilder.");

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
    /// This method serializes the entire AiModelResult, including the underlying model,
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
    /// This method deserializes a complete AiModelResult that was previously saved with SaveState,
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
    /// var model = await new AiModelBuilder&lt;double&gt;()
    ///     .ConfigureExport(new ExportConfig { TargetPlatform = TargetPlatform.CPU })
    ///     .BuildAsync();
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
    /// var model = await new AiModelBuilder&lt;double&gt;()
    ///     .ConfigureExport(new ExportConfig { TargetPlatform = TargetPlatform.TensorRT, Quantization = QuantizationMode.Float16 })
    ///     .BuildAsync();
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
            Precision = exportConfig.Quantization switch
            {
                QuantizationMode.Float16 => TensorRTPrecision.FP16,
                QuantizationMode.Int8 => TensorRTPrecision.INT8,
                _ => TensorRTPrecision.FP32
            }
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
    /// var model = await new AiModelBuilder&lt;double&gt;()
    ///     .ConfigureExport(new ExportConfig { TargetPlatform = TargetPlatform.CoreML, Quantization = QuantizationMode.Float16 })
    ///     .BuildAsync();
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
    /// var model = await new AiModelBuilder&lt;double&gt;()
    ///     .ConfigureExport(new ExportConfig { TargetPlatform = TargetPlatform.TFLite, Quantization = QuantizationMode.Int8 })
    ///     .BuildAsync();
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
    /// - AiModelBuilder when building models with JIT enabled
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

        if (Model is not IJitCompilable<T> jitModel)
        {
            throw new NotSupportedException(
                $"The underlying model type ({Model.GetType().Name}) does not implement IJitCompilable<T>. " +
                "JIT compilation is only supported for models that use differentiable computation graphs, such as " +
                "linear models, polynomial models, and neural networks. Tree-based models (decision trees, random forests, " +
                "gradient boosting) cannot be JIT compiled due to their discrete branching logic.");
        }

        if (!jitModel.SupportsJitCompilation)
        {
            throw new NotSupportedException(
                $"The underlying model type ({Model.GetType().Name}) does not support JIT compilation. " +
                "Check SupportsJitCompilation property before calling ExportComputationGraph.");
        }

        return jitModel.ExportComputationGraph(inputNodes);
    }

    #endregion
}
