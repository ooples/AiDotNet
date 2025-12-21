global using AiDotNet.Agents;
global using AiDotNet.Configuration;
global using AiDotNet.DataProcessor;
global using AiDotNet.Deployment.Configuration;
global using AiDotNet.Diagnostics;
global using AiDotNet.DistributedTraining;
global using AiDotNet.Enums;
global using AiDotNet.FeatureSelectors;
global using AiDotNet.FitDetectors;
global using AiDotNet.FitnessCalculators;
global using AiDotNet.Helpers;
global using AiDotNet.KnowledgeDistillation;
global using AiDotNet.LanguageModels;
global using AiDotNet.LossFunctions;
global using AiDotNet.MetaLearning;
global using AiDotNet.MixedPrecision;
global using AiDotNet.Models;
global using AiDotNet.Models.Inputs;
global using AiDotNet.Models.Options;
global using AiDotNet.Normalizers;
global using AiDotNet.Optimizers;
global using AiDotNet.OutlierRemoval;
global using AiDotNet.PromptEngineering.Chains;
global using AiDotNet.PromptEngineering.FewShot;
global using AiDotNet.PromptEngineering.Optimization;
global using AiDotNet.PromptEngineering.Templates;
global using AiDotNet.Reasoning.Models;
global using AiDotNet.Regularization;
global using AiDotNet.RetrievalAugmentedGeneration.Graph;
global using AiDotNet.Tokenization.Configuration;
global using AiDotNet.Tokenization.HuggingFace;
global using AiDotNet.Tokenization.Interfaces;
global using AiDotNet.Tools;
global using AiDotNet.Tensors.Helpers;
global using AiDotNet.UncertaintyQuantification.Layers;
global using AiDotNet.LinearAlgebra;

using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.Policies;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Models.Options;
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
/// </remarks>
public partial class PredictionModelBuilder<T, TInput, TOutput> : IPredictionModelBuilder<T, TInput, TOutput>
{
    private IFeatureSelector<T, TInput>? _featureSelector;
    private INormalizer<T, TInput, TOutput>? _normalizer;
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

    // Tokenization configuration
    private ITokenizer? _tokenizer;
    private TokenizationConfig? _tokenizationConfig;

    // Prompt engineering configuration
    private IPromptTemplate? _promptTemplate;
    private IChain<string, string>? _promptChain;
    private IPromptOptimizer<T>? _promptOptimizer;
    private IFewShotExampleSelector<T>? _fewShotExampleSelector;
    private IPromptAnalyzer? _promptAnalyzer;
    private IPromptCompressor? _promptCompressor;

    private UncertaintyQuantificationOptions? _uncertaintyQuantificationOptions;
    private AiDotNet.Models.Inputs.UncertaintyCalibrationData<TInput, TOutput>? _uncertaintyCalibrationData;

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
        // RL TRAINING PATH - check if RL options are configured with an environment
        if (_rlOptions?.Environment is not null)
        {
            // Use episodes from options (default: 1000)
            int episodes = _rlOptions.Episodes;
            bool verbose = _rlOptions.LogFrequency > 0;
            return await BuildRLInternalAsync(episodes, verbose);
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
            return await BuildSupervisedInternalAsync(features, labels);
        }

        // META-LEARNING PATH - check if meta-learner is configured
        if (_metaLearner is not null)
        {
            return BuildMetaLearningInternalAsync();
        }

        // No training path configured
        throw new InvalidOperationException(
            "BuildAsync() requires one of the following to be configured first:\n" +
            "- ConfigureReinforcementLearning() for RL training\n" +
            "- ConfigureDataLoader() for supervised learning\n" +
            "- ConfigureMetaLearning() for meta-learning\n" +
            "For supervised learning, configure a data loader via ConfigureDataLoader() and then call BuildAsync().");
    }

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

        var deepEnsembleTemplate = _uncertaintyQuantificationOptions is { Enabled: true, Method: UncertaintyQuantificationMethod.DeepEnsemble }
            ? _model.DeepCopy()
            : null;

        OptimizationResult<T, TInput, TOutput> optimizationResult;
        var optimizationInputData = OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest);

        // Check if knowledge distillation is configured
        if (_knowledgeDistillationOptions != null)
        {
            // KNOWLEDGE DISTILLATION PATH
            optimizationResult = await PerformKnowledgeDistillationAsync(
                model,
                finalOptimizer,
                XTrain,
                yTrain,
                XVal,
                yVal,
                XTest,
                yTest);
        }
        else
        {
            // REGULAR TRAINING PATH
            // Optimize the final model on the full training set (using distributed optimizer if configured)
            optimizationResult = finalOptimizer.Optimize(optimizationInputData);
        }

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

        // Return PredictionModelResult with CV results, agent data, JIT compilation, and reasoning config
        var options = new PredictionModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = optimizationResult,
            NormalizationInfo = normInfo,
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
            ReasoningConfig = _reasoningConfig,
            KnowledgeGraph = _knowledgeGraph,
            GraphStore = _graphStore,
            HybridGraphRetriever = _hybridGraphRetriever,
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            PromptTemplate = _promptTemplate,
            PromptChain = _promptChain,
            PromptOptimizer = _promptOptimizer,
            FewShotExampleSelector = _fewShotExampleSelector,
            PromptAnalyzer = _promptAnalyzer,
            PromptCompressor = _promptCompressor,
            ProfilingReport = profilerSession?.GetReport()
        };

        var finalResult = new PredictionModelResult<T, TInput, TOutput>(options);

        finalResult.SetUncertaintyQuantificationOptions(_uncertaintyQuantificationOptions);
        TryComputeAndAttachDeepEnsembleModels(finalResult, deepEnsembleTemplate, optimizationInputData, optimizer, _uncertaintyQuantificationOptions);
        TryComputeAndAttachUncertaintyCalibrationArtifacts(finalResult);

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
            InferenceOptimizationConfig = _inferenceOptimizationConfig,
            ReasoningConfig = _reasoningConfig,
            KnowledgeGraph = _knowledgeGraph,
            GraphStore = _graphStore,
            HybridGraphRetriever = _hybridGraphRetriever,
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            PromptTemplate = _promptTemplate,
            PromptChain = _promptChain,
            PromptOptimizer = _promptOptimizer,
            FewShotExampleSelector = _fewShotExampleSelector,
            PromptAnalyzer = _promptAnalyzer,
            PromptCompressor = _promptCompressor,
            ProfilingReport = profilerSession?.GetReport()
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

        // Create normalization info (RL doesn't use normalization like supervised learning)
        var normInfo = new NormalizationInfo<T, TInput, TOutput>();

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
            PromptTemplate = _promptTemplate,
            PromptChain = _promptChain,
            PromptOptimizer = _promptOptimizer,
            FewShotExampleSelector = _fewShotExampleSelector,
            PromptAnalyzer = _promptAnalyzer,
            PromptCompressor = _promptCompressor,
            ProfilingReport = profilerSession?.GetReport()
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
    /// - <b>Auto</b>: Let ILGPU select the best device (CUDA for NVIDIA, OpenCL for AMD/Intel)
    /// - <b>CUDA</b>: Force NVIDIA CUDA backend (throws if NVIDIA GPU not available)
    /// - <b>OpenCL</b>: Force OpenCL backend (works with NVIDIA, AMD, Intel, throws if no GPU)
    /// - <b>CPU</b>: Force CPU-only execution (equivalent to UsageLevel.AlwaysCpu)
    /// </para>
    /// </remarks>
    private void ApplyGpuConfiguration()
    {
#if !NET462
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

        // Note on DeviceType (CUDA vs OpenCL):
        // The current ILGPU-based implementation auto-selects the best device type via GetPreferredDevice().
        // Explicit device type selection (CUDA vs OpenCL) would require:
        // 1. Enumerating available accelerators by type
        // 2. Filtering by CUDA vs OpenCL vs CPU
        // 3. Creating accelerator from filtered list
        // 4. Passing accelerator to GpuEngine constructor
        //
        // This is a future enhancement. For now, GpuDeviceType.Auto is implicitly used,
        // which lets ILGPU choose the best device (CUDA for NVIDIA, OpenCL for AMD/Intel).
        //
        // To add explicit device type support:
        // - Modify GpuEngine constructor to accept optional AcceleratorType filter
        // - Enumerate devices: context.Devices.Where(d => d.AcceleratorType == AcceleratorType.Cuda)
        // - Create accelerator from filtered device
        //
        // This would allow users to force CUDA or OpenCL when multiple options are available,
        // but adds complexity and is rarely needed since Auto already picks the fastest option.
        if (_gpuAccelerationConfig.DeviceType != AiDotNet.Engines.GpuDeviceType.Auto)
        {
            Console.WriteLine($"[AiDotNet] Warning: Explicit device type ({_gpuAccelerationConfig.DeviceType}) is not yet implemented.");
            Console.WriteLine("[AiDotNet] Using Auto device selection (CUDA for NVIDIA, OpenCL for AMD/Intel).");
            Console.WriteLine("[AiDotNet] This is the recommended setting and provides optimal performance.");
        }
#else
        // GPU acceleration is not supported in .NET Framework 4.6.2
        // ILGPU requires .NET Standard 2.1 or higher, which is not available in net462
        if (_gpuAccelerationConfig != null && _gpuAccelerationConfig.UsageLevel != AiDotNet.Engines.GpuUsageLevel.AlwaysCpu)
        {
            Console.WriteLine("[AiDotNet] Warning: GPU acceleration is not supported in .NET Framework 4.6.2");
            Console.WriteLine("[AiDotNet] Using CPU execution (ILGPU requires .NET Standard 2.1+)");
            Console.WriteLine("[AiDotNet] To use GPU acceleration, target net8.0 or higher");
        }
#endif
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
        var u1 = 1.0 - rng.NextDouble();
        var u2 = 1.0 - rng.NextDouble();
        var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
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
