using AiDotNet.Augmentation;
using AiDotNet.Benchmarking.Models;
using AiDotNet.CheckpointManagement;
using AiDotNet.Configuration;
using AiDotNet.Data.Structures;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Diagnostics;
using AiDotNet.ExperimentTracking;
using AiDotNet.HyperparameterOptimization;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.Models.Results;
using AiDotNet.Preprocessing;
using AiDotNet.ProgramSynthesis.Serving;
using AiDotNet.PromptEngineering.Analysis;
using AiDotNet.PromptEngineering.Compression;
using AiDotNet.Reasoning;
using AiDotNet.Reasoning.Models;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Configuration;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.TrainingMonitoring;
using AiDotNet.TrainingMonitoring.ExperimentTracking;

namespace AiDotNet.Models.Options;

/// <summary>
/// Represents the configuration options for creating a AiModelResult.
/// </summary>
/// <remarks>
/// <para>
/// This class consolidates all the configuration parameters needed to construct a AiModelResult
/// into a single, organized object. Instead of passing 20+ parameters to constructors, this options
/// class groups related settings together for better readability and maintainability.
/// </para>
/// <para>
/// The options are organized into logical categories:
/// <list type="bullet">
///   <item><description>Core Model: The trained model and its optimization/normalization data</description></item>
///   <item><description>Ethical AI: Bias detection and fairness evaluation components</description></item>
///   <item><description>RAG: Retrieval-Augmented Generation components for document retrieval</description></item>
///   <item><description>Graph RAG: Knowledge graph components for enhanced retrieval</description></item>
///   <item><description>Prompt Engineering: Templates, chains, optimizers, and analysis tools</description></item>
///   <item><description>Fine-tuning: LoRA and meta-learning configurations</description></item>
///   <item><description>Agent &amp; Reasoning: AI agent and advanced reasoning configurations</description></item>
///   <item><description>Deployment: Export, caching, versioning, and telemetry settings</description></item>
///   <item><description>Inference: JIT compilation and optimization configurations</description></item>
///   <item><description>Tokenization: Text tokenizer and configuration</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> This class is like a settings container for creating a trained model.
///
/// Instead of writing code like this (hard to read):
/// <code>
/// var result = new AiModelResult(opt, norm, bias, fair, ret, rerank, gen, ...20 more);
/// </code>
///
/// You can write this (easy to read):
/// <code>
/// var options = new AiModelResultOptions&lt;double, Matrix, Vector&gt;
/// {
///     OptimizationResult = opt,
///     NormalizationInfo = norm,
///     BiasDetector = bias,
///     PromptTemplate = myTemplate,
///     // ... only set what you need
/// };
/// var result = new AiModelResult(options);
/// </code>
///
/// Benefits:
/// - You only set the options you need (everything else has sensible defaults)
/// - Named properties make it clear what each setting does
/// - Easy to see all available settings via IntelliSense
/// - Adding new options doesn't break existing code
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt;, Vector&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The type of output predictions (e.g., Vector&lt;T&gt;).</typeparam>
public class AiModelResultOptions<T, TInput, TOutput> : ModelOptions
{
    // ============================================================================
    // Core Model Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the trained model used for making predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the actual model that will perform predictions. If not set explicitly,
    /// it will be derived from OptimizationResult.BestSolution during construction.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "brain" of your prediction system.
    /// Usually you don't need to set this directly - it comes from the optimization result.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput>? Model { get; set; }

    /// <summary>
    /// Gets or sets the fitted text vectorizer used to turn raw text into model features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Set when the model was trained on text through <c>ConfigureTextVectorizer(...)</c>. The result keeps the
    /// fitted vectorizer so callers can predict directly from strings via <c>PredictText(...)</c>.
    /// </para>
    /// <para><b>For Beginners:</b> A text vectorizer turns sentences into numbers the model can learn from. Keeping
    /// the same fitted vectorizer means new text is converted exactly the same way at prediction time.
    /// </para>
    /// </remarks>
    public ITextVectorizer<T>? TextVectorizer { get; set; }

    /// <summary>
    /// Gets or sets the results of the optimization process that created the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains detailed information about how the model was trained, including performance
    /// metrics on training, validation, and test datasets, as well as the training history.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how well your model was trained.
    /// It includes scores like R-squared, RMSE, and tells you if the model might be overfitting.
    /// </para>
    /// </remarks>
    public OptimizationResult<T, TInput, TOutput>? OptimizationResult { get; set; }

    /// <summary>
    /// Gets or sets the preprocessing pipeline information for data transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stores the fitted preprocessing pipeline used during training so that new input data
    /// can be transformed the same way. This replaces the legacy <see cref="NormalizationInfo"/>
    /// with a more flexible, composable pipeline approach supporting scalers, encoders, imputers,
    /// and feature generators.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers all the data transformations applied during training:
    /// - Scaling (StandardScaler, MinMaxScaler, etc.)
    /// - Encoding (OneHotEncoder, LabelEncoder, etc.)
    /// - Missing value handling (SimpleImputer, KNNImputer, etc.)
    /// - Feature generation (PolynomialFeatures, SplineTransformer, etc.)
    ///
    /// When you make predictions on new data, it applies the same transformations so the model
    /// understands the input correctly.
    /// </para>
    /// </remarks>
    public PreprocessingInfo<T, TInput, TOutput>? PreprocessingInfo { get; set; }

    /// <summary>
    /// Gets or sets the postprocessing pipeline configured via
    /// <see cref="AiModelBuilder{T,TInput,TOutput}.ConfigurePostprocessing(AiDotNet.Postprocessing.PostprocessingPipeline{T,TOutput,TOutput})"/>.
    /// </summary>
    /// <value>
    /// A <see cref="AiDotNet.Postprocessing.PostprocessingPipeline{T,TOutput,TOutput}"/>
    /// applied to the model's raw output during inference, or null if no
    /// postprocessing was configured.
    /// </value>
    /// <remarks>
    /// <para>
    /// Applied inside <see cref="AiModelResult{T,TInput,TOutput}.Predict"/>
    /// after the model produces its raw output (and after any
    /// <c>PreprocessingInfo</c> target-inverse transform). Without this
    /// wiring the configured postprocessing pipeline was stored on the
    /// builder but never invoked on predictions — a stored-but-not-
    /// consumed regression of the same shape as PR #1357 (#1361 family).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Postprocessing is the "format the answer"
    /// step that runs AFTER the model predicts. Common examples are
    /// applying softmax to convert raw logits into probabilities,
    /// decoding class indices back into label strings, or applying
    /// thresholds to classification scores. Setting this property here
    /// is how the builder hands the pipeline to the runtime so each
    /// <c>Predict</c> call applies it automatically.
    /// </para>
    /// </remarks>
    public AiDotNet.Postprocessing.PostprocessingPipeline<T, TOutput, TOutput>? PostprocessingPipeline { get; set; }

    /// <summary>
    /// Optional sample of model-output predictions (NOT training targets)
    /// handed in alongside an unfitted <see cref="PostprocessingPipeline"/>
    /// so the <see cref="AiModelResult{T,TInput,TOutput}"/> ctor can
    /// lazy-fit the pipeline at construction time instead of throwing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Important:</b> the postprocessing pipeline transforms model
    /// PREDICTIONS (e.g. raw logits → softmax probabilities, raw scores →
    /// thresholded labels), so its <c>Fit</c> needs the distribution of
    /// model outputs, not the training-target distribution. If you
    /// supply training targets here the pipeline will be fit on the
    /// wrong distribution and silently transform predictions
    /// incorrectly at inference time (review #1368 C8efy). The
    /// <see cref="AiModelBuilder{T,TInput,TOutput}"/> supervised path
    /// produces this sample correctly by calling
    /// <c>bestSolution.Predict(XTrain)</c> internally; only direct
    /// <c>AiModelResultOptions</c> construction callers need to
    /// materialise it themselves.
    /// </para>
    /// </remarks>
    /// <remarks>
    /// <para>
    /// When <see cref="PostprocessingPipeline"/> is non-null and not yet
    /// fitted, the ctor will call <c>PostprocessingPipeline.Fit(this)</c>
    /// if this value is non-null; if it is null the ctor throws the
    /// fail-fast diagnostic (review #1368 C6WMS). AiModelBuilder's
    /// supervised path normally fits the pipeline itself before
    /// constructing the result; this slot exists for direct
    /// <c>AiModelResultOptions</c> construction paths (federated /
    /// meta-learning / distributed) that own the trained model and
    /// training data but haven't called Fit yet.
    /// </para>
    /// <para>
    /// <b>Important — fit-sample shape:</b> <typeparamref name="TOutput"/>
    /// must be a <i>batched</i> representation (e.g. <c>Tensor&lt;T&gt;</c>
    /// with shape <c>[batch, features]</c> or <c>Matrix&lt;T&gt;</c>) when
    /// the pipeline contains data-distribution-learning transformers
    /// (StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler,
    /// PowerTransformer, LabelEncoder, etc.) — a single-row sample will
    /// fit those transformers to degenerate statistics
    /// (zero variance, max == min). For these cases supply enough rows for
    /// the statistic to stabilise (typically ≥ 256 rows; for power
    /// transformers ≥ 4096), or pre-fit the pipeline yourself via
    /// <c>PostprocessingPipeline.Fit(...)</c> on representative model
    /// outputs and leave this slot null (review #1368 C7mlj).
    /// </para>
    /// </remarks>
    public TOutput? PostprocessingFitSample { get; set; }

    /// <summary>
    /// Gets or sets the knowledge-distillation options configured via
    /// <see cref="AiModelBuilder{T,TInput,TOutput}.ConfigureKnowledgeDistillation"/>.
    /// </summary>
    /// <value>
    /// A <see cref="KnowledgeDistillationOptions{T,TInput,TOutput}"/>
    /// instance carrying the configured teacher / temperature / alpha
    /// settings, or null if no distillation was configured.
    /// </value>
    /// <remarks>
    /// <para>
    /// Carried through to <see cref="AiModelResult{T,TInput,TOutput}"/>
    /// so consumers can drive distillation manually post-build via a
    /// teacher-aware loss function. Without this slot the options were
    /// stored on the builder but silently dropped before the result
    /// surface — discovered by AiDotNet#1345 Bucket9
    /// ConfigureKnowledgeDistillation test.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Knowledge distillation is a technique where
    /// a smaller "student" model learns to mimic a larger "teacher"
    /// model — the temperature setting controls how soft the teacher's
    /// predictions become, and alpha balances the student's own labels
    /// against the teacher's soft targets. This property surfaces those
    /// configured settings on the result so downstream tooling can run
    /// the distillation loop.
    /// </para>
    /// </remarks>
    public AiDotNet.Models.Options.KnowledgeDistillationOptions<T, TInput, TOutput>? KnowledgeDistillationOptions { get; set; }

    /// <summary>
    /// Gets or sets an optional AutoML run summary for this trained model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is intended for facade outputs and should not contain hyperparameter values, weights, or other
    /// sensitive implementation details. If AutoML was not used, this can be null.
    /// </para>
    /// <para><b>For Beginners:</b> If you used AutoML, this stores a short history of the AutoML search
    /// (how many trials ran and what the scores looked like), without exposing internal tuning details.</para>
    /// </remarks>
    public AutoMLRunSummary? AutoMLSummary { get; set; }

    // ============================================================================
    // Ethical AI Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the bias detector for identifying potential biases in model predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When set, enables bias detection capabilities on the resulting model.
    /// Use this to check if your model treats different demographic groups fairly.
    /// </para>
    /// <para><b>For Beginners:</b> This helps you check if your model is unfair to certain groups.
    /// For example, it can detect if a loan approval model treats men and women differently.
    /// </para>
    /// </remarks>
    public IBiasDetector<T>? BiasDetector { get; set; }

    /// <summary>
    /// Gets or sets the fairness evaluator for computing fairness metrics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When set, enables fairness evaluation capabilities including demographic parity,
    /// equalized odds, and other fairness metrics.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates specific numbers that tell you how fair your model is.
    /// Higher fairness scores mean your model treats different groups more equally.
    /// </para>
    /// </remarks>
    public IFairnessEvaluator<T>? FairnessEvaluator { get; set; }

    /// <summary>
    /// Gets or sets the interpretability options for model explanation methods.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Configures which interpretability methods (SHAP, LIME, Permutation Importance, Global Surrogate)
    /// are available and their parameters for explaining model predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how your model explains its predictions.
    /// When configured, you can ask the model "why did you make this prediction?" and get
    /// answers like "Age increased the prediction by +5000, Income decreased it by -2000."
    /// </para>
    /// </remarks>
    public InterpretabilityOptions? InterpretabilityOptions { get; set; }

    // ============================================================================
    // RAG (Retrieval-Augmented Generation) Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the retriever for finding relevant documents during inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The retriever searches for documents or passages that are relevant to the input query.
    /// It typically uses vector similarity search to find the most similar content.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a smart search engine that finds relevant information
    /// from your document collection to help the model answer questions better.
    /// </para>
    /// </remarks>
    public IRetriever<T>? RagRetriever { get; set; }

    /// <summary>
    /// Gets or sets the reranker for improving document relevance ordering.
    /// </summary>
    /// <remarks>
    /// <para>
    /// After initial retrieval, the reranker reorders documents by more carefully
    /// evaluating their relevance to the query, typically improving result quality.
    /// </para>
    /// <para><b>For Beginners:</b> After finding documents, this component sorts them again
    /// to put the most relevant ones first. It's like a second opinion on what's most useful.
    /// </para>
    /// </remarks>
    public IReranker<T>? RagReranker { get; set; }

    /// <summary>
    /// Gets or sets the generator for creating answers from retrieved context.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The generator takes the retrieved documents and the original query to produce
    /// a natural language response that incorporates the retrieved information.
    /// </para>
    /// <para><b>For Beginners:</b> This takes the documents found by the retriever and
    /// uses them to write a helpful answer to your question.
    /// </para>
    /// </remarks>
    public IGenerator<T>? RagGenerator { get; set; }

    /// <summary>
    /// Gets or sets the query processors for preprocessing search queries.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Query processors can expand, rewrite, or enhance queries before they are
    /// sent to the retriever, improving search quality.
    /// </para>
    /// <para><b>For Beginners:</b> These improve your search queries before searching.
    /// For example, they might add synonyms or fix spelling mistakes.
    /// </para>
    /// </remarks>
    public IEnumerable<IQueryProcessor>? QueryProcessors { get; set; }

    // ============================================================================
    // Graph RAG Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the knowledge graph for entity-relationship-based retrieval.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The knowledge graph stores entities (people, places, concepts) and their relationships,
    /// enabling retrieval that follows semantic connections beyond simple text similarity.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a web of connected facts. When you ask a question,
    /// it can follow connections like "Einstein worked at Princeton" to find related information.
    /// </para>
    /// </remarks>
    public KnowledgeGraph<T>? KnowledgeGraph { get; set; }

    /// <summary>
    /// Gets or sets the graph store backend for persistent graph storage.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Provides the underlying storage mechanism for the knowledge graph,
    /// supporting persistence, transactions, and efficient graph traversal.
    /// </para>
    /// <para><b>For Beginners:</b> This is the database that stores your knowledge graph.
    /// It keeps the graph saved even when your program restarts.
    /// </para>
    /// </remarks>
    public IGraphStore<T>? GraphStore { get; set; }

    /// <summary>
    /// Gets or sets the hybrid retriever combining vector search with graph traversal.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Combines traditional vector similarity search with knowledge graph traversal,
    /// providing richer context by following entity relationships.
    /// </para>
    /// <para><b>For Beginners:</b> This combines regular search with following connections in
    /// the knowledge graph. It finds documents that are similar AND documents about related topics.
    /// </para>
    /// </remarks>
    public HybridGraphRetriever<T>? HybridGraphRetriever { get; set; }

    // ============================================================================
    // Prompt Engineering Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the prompt template for formatting model inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A prompt template provides a reusable structure for creating prompts with variable
    /// placeholders that can be filled in at runtime.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a fill-in-the-blank form for creating prompts.
    ///
    /// Example:
    /// <code>
    /// var template = new SimplePromptTemplate("Translate {text} from {source} to {target}");
    /// var prompt = template.Format(new Dictionary&lt;string, string&gt;
    /// {
    ///     ["text"] = "Hello",
    ///     ["source"] = "English",
    ///     ["target"] = "Spanish"
    /// });
    /// // Result: "Translate Hello from English to Spanish"
    /// </code>
    /// </para>
    /// </remarks>
    public IPromptTemplate? PromptTemplate { get; set; }


    /// <summary>
    /// Gets or sets the prompt optimizer for automatically improving prompts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A prompt optimizer automatically refines prompts to achieve better performance
    /// on a specific task. Strategies include discrete search, gradient-based methods,
    /// evolutionary algorithms, and Bayesian optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This automatically improves your prompts by trying variations
    /// and keeping the ones that work best. It's like A/B testing for prompts.
    /// </para>
    /// </remarks>
    public IPromptOptimizer<T>? PromptOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the few-shot example selector for choosing examples to include in prompts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A few-shot example selector chooses the most relevant examples to include in prompts
    /// based on the current query. Strategies include random, fixed, semantic similarity,
    /// diversity-based, and clustering-based selection.
    /// </para>
    /// <para><b>For Beginners:</b> When you want to show the AI examples of what you want,
    /// this picks the best examples to show based on your current question.
    ///
    /// For example, if you're translating a technical document, it picks technical examples
    /// rather than casual conversation examples.
    /// </para>
    /// </remarks>
    public IFewShotExampleSelector<T>? FewShotExampleSelector { get; set; }

    /// <summary>
    /// Gets or sets the prompt analyzer for computing prompt metrics and validation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The prompt analyzer computes metrics like token count, estimated cost, complexity score,
    /// and validates prompts for potential issues before sending them to an LLM.
    /// </para>
    /// <para><b>For Beginners:</b> This checks your prompts before you use them.
    /// It tells you how many tokens they use (affecting cost), how complex they are,
    /// and warns you about potential problems.
    /// </para>
    /// </remarks>
    public IPromptAnalyzer? PromptAnalyzer { get; set; }

    /// <summary>
    /// Gets or sets the prompt compressor for reducing token counts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The prompt compressor reduces the token count of prompts while preserving their
    /// semantic meaning. This can significantly reduce API costs for long prompts.
    /// </para>
    /// <para><b>For Beginners:</b> This makes your prompts shorter without losing meaning.
    /// Shorter prompts cost less money when using paid AI services like GPT-4 or Claude.
    /// </para>
    /// </remarks>
    public IPromptCompressor? PromptCompressor { get; set; }

    // ============================================================================
    // Fine-tuning & Adaptation Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the LoRA configuration for parameter-efficient fine-tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// LoRA (Low-Rank Adaptation) enables efficient fine-tuning by adding small adapter
    /// layers instead of modifying all model parameters. This makes adaptation faster
    /// and requires less memory.
    /// </para>
    /// <para><b>For Beginners:</b> This allows you to customize the model for your specific use
    /// without retraining the whole thing. It's much faster and cheaper than full fine-tuning.
    /// </para>
    /// </remarks>
    public ILoRAConfiguration<T>? LoRAConfiguration { get; set; }

    /// <summary>
    /// Gets or sets the meta-learner for few-shot adaptation capabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If the model was trained using meta-learning (MAML, Reptile, SEAL), this contains
    /// the meta-learner that can quickly adapt the model to new tasks with just a few examples.
    /// </para>
    /// <para><b>For Beginners:</b> This is for models trained to "learn how to learn."
    /// With just 5-10 examples, it can adapt the model to a completely new task.
    /// </para>
    /// </remarks>
    public IMetaLearner<T, TInput, TOutput>? MetaLearner { get; set; }

    /// <summary>
    /// Gets or sets the results from meta-learning training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains performance history and statistics from the meta-training process,
    /// including loss curves, accuracy across different tasks, and convergence information.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how well the meta-learning training went.
    /// It includes charts of how the model improved during training.
    /// </para>
    /// </remarks>
    public MetaTrainingResult<T>? MetaTrainingResult { get; set; }

    // ============================================================================
    // Cross-validation Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the results from cross-validation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains fold-by-fold performance metrics and aggregated statistics from
    /// cross-validation, helping assess model consistency and generalization.
    /// </para>
    /// <para><b>For Beginners:</b> Cross-validation tests your model multiple times on different
    /// parts of the data. This stores all those test results so you can see how consistently
    /// your model performs.
    /// </para>
    /// </remarks>
    public CrossValidationResult<T, TInput, TOutput>? CrossValidationResult { get; set; }

    // ============================================================================
    // Agent & Reasoning Properties
    // ============================================================================


    /// <summary>
    /// Gets or sets the reasoning configuration for advanced reasoning capabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Configures advanced reasoning strategies like Chain-of-Thought, Tree-of-Thoughts,
    /// and Self-Consistency for complex problem-solving.
    /// </para>
    /// <para><b>For Beginners:</b> This enables the model to "think step by step" when solving
    /// complex problems, similar to how humans work through difficult questions.
    /// </para>
    /// </remarks>
    public ReasoningConfig? ReasoningConfig { get; set; }

    // ============================================================================
    // Deployment Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the deployment configuration for model export and production use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Aggregates all deployment-related settings including quantization, caching,
    /// versioning, A/B testing, telemetry, and export configurations.
    /// </para>
    /// <para><b>For Beginners:</b> This contains all the settings for using your model in production:
    /// - Quantization: Make the model smaller/faster
    /// - Caching: Remember previous predictions
    /// - Versioning: Track different model versions
    /// - A/B Testing: Compare model versions
    /// - Telemetry: Monitor performance
    /// </para>
    /// </remarks>
    public DeploymentConfiguration? DeploymentConfiguration { get; set; }

    /// <summary>
    /// Gets or sets information about model quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains quantization details such as strategy (GPTQ, AWQ, SmoothQuant), bit width,
    /// compression ratio, and per-layer quantization statistics.
    /// </para>
    /// <para><b>For Beginners:</b> Quantization makes your model smaller and faster by using
    /// fewer bits per weight. This property tells you what quantization was applied.
    /// </para>
    /// </remarks>
    public QuantizationInfo? QuantizationInfo { get; set; }

    // ============================================================================
    // Inference Optimization Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the JIT-compiled prediction function for accelerated inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When JIT compilation is enabled and the model supports it, this contains
    /// a pre-compiled, optimized version of the prediction function for faster inference.
    /// </para>
    /// <para><b>For Beginners:</b> This is a speed-optimized version of your model's prediction code.
    /// When available, it can make predictions 5-10x faster than the normal code path.
    /// </para>
    /// </remarks>
    public Func<Tensor<T>[], Tensor<T>[]>? JitCompiledFunction { get; set; }

    /// <summary>
    /// Gets or sets the inference optimization configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains settings for runtime inference optimizations including batch processing,
    /// memory management, and computational optimizations.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how predictions are made at runtime.
    /// You can tune settings like batch size and memory usage for your specific hardware.
    /// </para>
    /// </remarks>
    public InferenceOptimizationConfig? InferenceOptimizationConfig { get; set; }

    /// <summary>
    /// JIT compilation configuration applied on every Predict call.
    /// </summary>
    public AiDotNet.Configuration.JitCompilationConfig? JitCompilationConfig { get; set; }

    /// <summary>
    /// Determinism policy. When false (default), Predict re-asserts deterministic
    /// mode on the calling thread. When true, allows nondeterministic kernels.
    /// </summary>
    public bool AllowNondeterminism { get; set; }

    // ============================================================================
    // Augmentation Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the unified augmentation configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This configuration covers both training-time augmentation and Test-Time
    /// Augmentation (TTA) for improved inference accuracy. It includes:
    /// - Core settings (enabled, probability, seed)
    /// - TTA settings (enabled by default, aggregation method)
    /// - Modality-specific settings (image, tabular, audio, text, video)
    /// </para>
    /// <para><b>For Beginners:</b> Augmentation creates variations of your data:
    /// - During training: Helps the model learn to recognize objects regardless of orientation, lighting, etc.
    /// - During inference (TTA): Makes predictions on multiple variations and combines them for better accuracy.
    ///
    /// Example use cases:
    /// - Image classification: Train on flipped, rotated versions; predict on multiple views
    /// - Tabular data: Add noise, apply MixUp for regularization
    /// - Audio: Apply time stretch, pitch shift for robustness
    /// </para>
    /// </remarks>
    public AugmentationConfig? AugmentationConfig { get; set; }

    // ============================================================================
    // Safety & Robustness Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the safety filter configuration used to validate inputs and filter outputs during inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When null, safety filtering defaults to enabled with standard options.
    /// Set <see cref="SafetyFilterConfiguration{T}.Enabled"/> to false to opt out.
    /// </para>
    /// </remarks>
    public SafetyFilterConfiguration<T>? SafetyFilterConfiguration { get; set; }

    // ============================================================================
    // Tokenization Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the tokenizer for text encoding and decoding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The tokenizer converts text into tokens (numerical IDs) that the model can process,
    /// and converts model outputs back into readable text.
    /// </para>
    /// <para><b>For Beginners:</b> This converts text into numbers that the model understands,
    /// and converts the model's number outputs back into text you can read.
    /// </para>
    /// </remarks>
    public ITokenizer? Tokenizer { get; set; }

    /// <summary>
    /// Gets or sets the tokenization configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains settings for the tokenizer including vocabulary size, special tokens,
    /// padding behavior, and truncation settings.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how text is converted to tokens:
    /// - Maximum length (how much text to process)
    /// - Padding (how to handle short texts)
    /// - Special tokens (like [START] and [END] markers)
    /// </para>
    /// </remarks>
    public TokenizationConfig? TokenizationConfig { get; set; }

    // ============================================================================
    // Program Synthesis Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the Program Synthesis model used for code tasks (optional).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Program Synthesis is independent of the main prediction model. This allows a single <c>AiModelResult</c>
    /// to expose code-task capabilities even when the primary model uses non-tensor data types (for example,
    /// <c>Vector&lt;T&gt;</c> or <c>Matrix&lt;T&gt;</c>).
    /// </para>
    /// <para><b>For Beginners:</b> This is the specialized model used for code-related tasks such as code generation or repair.
    /// You typically configure this via <c>AiModelBuilder.ConfigureProgramSynthesis(...)</c>.
    /// </para>
    /// </remarks>
    public IFullModel<T, Tensor<T>, Tensor<T>>? ProgramSynthesisModel { get; set; }

    /// <summary>
    /// Gets or sets the Program Synthesis Serving client (optional).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When provided, Program Synthesis inference (code tasks, sandboxed execution, evaluation) can be routed through
    /// <c>AiDotNet.Serving</c> by default to isolate untrusted code and keep proprietary logic on the server side.
    /// </para>
    /// <para><b>For Beginners:</b> This lets your app call a secure server to run code tasks safely.
    ///
    /// Instead of running code on your machine (which can be unsafe), you can point AiDotNet to a Serving instance that
    /// runs everything in a sandbox.
    /// </para>
    /// </remarks>
    public IProgramSynthesisServingClient? ProgramSynthesisServingClient { get; set; }

    /// <summary>
    /// Gets or sets the options used to create a default <see cref="ProgramSynthesisServingClient"/> when no explicit client is provided.
    /// </summary>
    public ProgramSynthesisServingClientOptions? ProgramSynthesisServingClientOptions { get; set; }

    // ============================================================================
    // Diagnostics Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the profiling report from training and inference operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains comprehensive profiling data collected during model training including
    /// operation timing, memory allocations, and performance statistics with percentiles.
    /// </para>
    /// <para><b>For Beginners:</b> This shows you how long different parts of training took.
    /// It includes:
    /// - How long each operation took (e.g., forward pass, backward pass)
    /// - Memory usage during training
    /// - Performance percentiles (P50, P95, P99) for statistical analysis
    /// - Identification of performance bottlenecks
    /// </para>
    /// </remarks>
    public ProfileReport? ProfilingReport { get; set; }

    /// <summary>
    /// Weight-streaming activity report from the underlying
    /// <c>WeightRegistry</c> if streaming engaged during the build
    /// (issue #1222 task #186). Null when streaming stayed off (the
    /// common case for models that fit in RAM).
    /// </summary>
    /// <value>
    /// A <see cref="AiDotNet.Deployment.Configuration.WeightStreamingReport"/>
    /// snapshot of the streaming pool state at result-construction time
    /// (disk read count, eviction count, prefetch hit/miss/issue counts,
    /// resident-bytes high-water, compression ratio, the effective
    /// threshold that drove the engagement decision, and a flag
    /// distinguishing auto-detected from user-forced engagement). Null
    /// when streaming wasn't engaged for this build — most callers can
    /// treat null as "model fit in RAM, no streaming overhead paid".
    /// </value>
    /// <remarks>
    /// <para>Forwarded to <c>AiModelResult.WeightStreamingReport</c> via
    /// the options ctor so callers can inspect streaming behaviour:
    /// disk-read counts, eviction counts, prefetch hit/miss ratio.</para>
    /// <para><b>For Beginners:</b> Big language and vision models can
    /// have so many parameters (think tens of billions) that they don't
    /// fit in your computer's RAM all at once. AiDotNet's "weight
    /// streaming" feature solves this by keeping the model on disk and
    /// loading only the parts it needs for each layer's computation,
    /// then evicting them when memory gets full. This property tells
    /// you whether streaming was used and how busy it was during the
    /// build:</para>
    /// <list type="bullet">
    /// <item>If null: the model is small enough to live in RAM, no
    /// streaming was needed (zero overhead paid).</item>
    /// <item>If non-null: streaming engaged. Look at the report's
    /// counters to understand cost — high disk-read counts mean
    /// frequent paging from disk; high prefetch-miss counts mean the
    /// pool's window-size was too small for the model's access pattern;
    /// high eviction count is normal for models well above the
    /// threshold and just means the pool is doing its job.</item>
    /// </list>
    /// <para>You usually don't set this directly — the
    /// <c>AiModelBuilder</c> populates it automatically when streaming
    /// engages (auto-detect) or when you explicitly opt in via
    /// <c>ConfigureWeightStreaming(new WeightStreamingConfig { Enabled = true })</c>.</para>
    /// </remarks>
    public AiDotNet.Deployment.Configuration.WeightStreamingReport? WeightStreamingReport { get; set; }

    // ============================================================================
    // Training Infrastructure Properties
    // ============================================================================

    /// <summary>
    /// Gets or sets the memory management configuration for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Memory management includes gradient checkpointing, activation pooling,
    /// and model sharding to enable training of larger models.
    /// </para>
    /// <para><b>For Beginners:</b> This configuration helps you train larger models
    /// by optimizing how memory is used during training. It can reduce memory usage
    /// by 40-50% at the cost of ~30% extra compute time.
    /// </para>
    /// </remarks>
    public Training.Memory.TrainingMemoryConfig? MemoryConfig { get; set; }

    /// <summary>
    /// Gets or sets the experiment run ID from experiment tracking.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When experiment tracking is configured, this contains the unique identifier
    /// for the training run, enabling reproducibility and comparison with other runs.
    /// </para>
    /// <para><b>For Beginners:</b> This is a unique ID for your training session.
    /// You can use it to find and compare this training run with others later.
    /// </para>
    /// </remarks>
    public string? ExperimentRunId { get; set; }

    /// <summary>
    /// Gets or sets the experiment ID that this run belongs to.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The experiment ID groups related training runs together for organization
    /// and comparison purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This groups training runs together.
    /// For example, all runs testing different learning rates might belong to the same experiment.
    /// </para>
    /// </remarks>
    public string? ExperimentId { get; set; }

    /// <summary>
    /// Gets or sets the model version from the model registry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When model registry is configured, this contains the version number
    /// assigned to this model, enabling version tracking and rollback.
    /// </para>
    /// <para><b>For Beginners:</b> This is the version number of your model.
    /// Like software versions (v1.0, v2.0), this helps track model improvements over time.
    /// </para>
    /// </remarks>
    public int? ModelVersion { get; set; }

    /// <summary>
    /// Gets or sets the registered model name in the model registry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The name under which this model is registered in the model registry,
    /// used for retrieval and deployment.
    /// </para>
    /// <para><b>For Beginners:</b> This is the name of your model in the registry.
    /// You can use this name to load the model later or deploy it to production.
    /// </para>
    /// </remarks>
    public string? RegisteredModelName { get; set; }

    /// <summary>
    /// Gets or sets the checkpoint path where the model was saved during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When checkpoint management is configured, this contains the path to the
    /// best or latest checkpoint, enabling training resumption or model loading.
    /// </para>
    /// <para><b>For Beginners:</b> This is where your model was saved during training.
    /// If training is interrupted, you can resume from this checkpoint.
    /// </para>
    /// </remarks>
    public string? CheckpointPath { get; set; }

    /// <summary>
    /// Gets or sets the data version hash for the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When data version control is configured, this contains a hash that uniquely
    /// identifies the training data used, enabling reproducibility.
    /// </para>
    /// <para><b>For Beginners:</b> This is a fingerprint of your training data.
    /// It ensures you can always know exactly what data was used to train this model.
    /// </para>
    /// </remarks>
    public string? DataVersionHash { get; set; }

    /// <summary>
    /// Gets or sets the hyperparameter optimization trial ID.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When hyperparameter optimization is used, this identifies which trial
    /// produced this model, linking back to the optimization history.
    /// </para>
    /// <para><b>For Beginners:</b> If an optimizer searched for the best settings,
    /// this tells you which attempt (trial) produced this model.
    /// </para>
    /// </remarks>
    public int? HyperparameterTrialId { get; set; }

    /// <summary>
    /// Gets or sets the hyperparameters used for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A dictionary of hyperparameter names to values that were used during training,
    /// enabling reproducibility and comparison between runs.
    /// </para>
    /// <para><b>For Beginners:</b> These are the settings (like learning rate, batch size)
    /// that were used to train your model. Saving them lets you reproduce the training.
    /// </para>
    /// </remarks>
    public Dictionary<string, object>? Hyperparameters { get; set; }

    /// <summary>
    /// Gets or sets the training metrics history.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains the history of metrics (loss, accuracy, etc.) recorded during training,
    /// enabling visualization and analysis of training progress.
    /// </para>
    /// <para><b>For Beginners:</b> This is a log of how your model improved during training.
    /// You can use it to create charts showing loss going down over time.
    /// </para>
    /// </remarks>
    public Dictionary<string, List<double>>? TrainingMetricsHistory { get; set; }

    // ============================================================================
    // Training Infrastructure Components
    // ============================================================================

    /// <summary>
    /// Gets or sets the experiment run associated with this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When experiment tracking was used during training, this provides direct access
    /// to the run object for logging additional metrics, artifacts, or notes post-training.
    /// </para>
    /// <para><b>For Beginners:</b> This is the training session record. You can use it to:
    /// - Log additional metrics after training completes
    /// - Add notes about model performance in production
    /// - Record artifacts like deployment logs
    /// </para>
    /// </remarks>
    public IExperimentRun<T>? ExperimentRun { get; set; }

    /// <summary>
    /// Gets or sets the experiment tracker used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Provides access to the experiment tracking system for retrieving other runs,
    /// creating comparison reports, or starting new related experiments.
    /// </para>
    /// <para><b>For Beginners:</b> This gives you access to the experiment tracking system.
    /// You can use it to:
    /// - Compare this model with other training runs
    /// - Find the best-performing model from an experiment
    /// - Start new training runs based on this one
    /// </para>
    /// </remarks>
    public IExperimentTracker<T>? ExperimentTracker { get; set; }

    /// <summary>
    /// Gets or sets the checkpoint manager for model persistence operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Provides access to checkpoint operations including saving updated models,
    /// listing available checkpoints, and managing checkpoint lifecycle.
    /// </para>
    /// <para><b>For Beginners:</b> This manages saved copies of your model. You can use it to:
    /// - Save the model after making changes (like fine-tuning)
    /// - List all saved checkpoints
    /// - Load different versions of your model
    /// - Clean up old checkpoints to save disk space
    /// </para>
    /// </remarks>
    public ICheckpointManager<T, TInput, TOutput>? CheckpointManager { get; set; }

    /// <summary>
    /// Gets or sets the model registry for version and lifecycle management.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Provides access to the model registry for transitioning model stages,
    /// registering new versions, and managing model lifecycle.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a version control system for your models.
    /// You can use it to:
    /// - Promote this model from "Staging" to "Production"
    /// - Register fine-tuned versions as new model versions
    /// - Archive old models that are no longer needed
    /// - Compare performance across model versions
    /// </para>
    /// </remarks>
    public IModelRegistry<T, TInput, TOutput>? ModelRegistry { get; set; }

    /// <summary>
    /// Gets or sets the training monitor for accessing training diagnostics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Provides access to training diagnostics including learning curves,
    /// gradient statistics, and training progress information.
    /// </para>
    /// <para><b>For Beginners:</b> This gives you insights into how training went.
    /// You can use it to:
    /// - View learning curves (loss over time)
    /// - Check for signs of overfitting
    /// - Analyze gradient flow during training
    /// - Export training charts and reports
    /// </para>
    /// </remarks>
    public ITrainingMonitor<T>? TrainingMonitor { get; set; }

    /// <summary>
    /// Gets or sets the hyperparameter optimization result.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When hyperparameter optimization was used, this contains the complete results
    /// including all tried configurations and their scores.
    /// </para>
    /// <para><b>For Beginners:</b> If an optimizer searched for the best settings,
    /// this contains all the configurations it tried and how well each performed.
    /// You can use it to:
    /// - See which hyperparameters were most important
    /// - Find patterns in what made training successful
    /// - Continue optimization from where it left off
    /// </para>
    /// </remarks>
    public HyperparameterOptimizationResult<T>? HyperparameterOptimizationResult { get; set; }

    /// <summary>
    /// Gets or sets the most recent benchmark report produced during model build/evaluation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is populated when the user enables benchmarking through the facade (for example,
    /// <c>AiModelBuilder.ConfigureBenchmarking(...)</c>) and a benchmark report is generated.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "report card" produced when you run benchmark suites
    /// like GSM8K or MMLU. It contains summary metrics and timing.
    /// </para>
    /// </remarks>
    public BenchmarkReport? BenchmarkReport { get; set; }
}
