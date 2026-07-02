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
/// Configuration (Configure* fluent methods) partial of <see cref="AiModelBuilder{T, TInput, TOutput}"/>. Split out of the
/// 9.5k-LoC main file (audit-2026-05 finding #12) for reviewability; no behaviour change.
/// </summary>
public partial class AiModelBuilder<T, TInput, TOutput>
{
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigurePreprocessing(
        Action<PreprocessingPipeline<T, TInput, TInput>>? pipelineBuilder = null)
    {
        _dataPipeline.ConfigurePreprocessing(pipelineBuilder);
        _preprocessingPipeline = _dataPipeline.PreprocessingPipeline;
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigurePreprocessing(
        IDataTransformer<T, TInput, TInput>? transformer = null)
    {
        _dataPipeline.ConfigurePreprocessing(transformer);
        _preprocessingPipeline = _dataPipeline.PreprocessingPipeline;
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigurePreprocessing(
        PreprocessingPipeline<T, TInput, TInput>? pipeline = null)
    {
        _dataPipeline.ConfigurePreprocessing(pipeline);
        _preprocessingPipeline = _dataPipeline.PreprocessingPipeline;
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigurePostprocessing(
        Action<PostprocessingPipeline<T, TOutput, TOutput>>? pipelineBuilder = null)
    {
        _dataPipeline.ConfigurePostprocessing(pipelineBuilder);
        _postprocessingPipeline = _dataPipeline.PostprocessingPipeline;
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigurePostprocessing(
        IDataTransformer<T, TOutput, TOutput>? transformer = null)
    {
        _dataPipeline.ConfigurePostprocessing(transformer);
        _postprocessingPipeline = _dataPipeline.PostprocessingPipeline;
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigurePostprocessing(
        PostprocessingPipeline<T, TOutput, TOutput>? pipeline = null)
    {
        _dataPipeline.ConfigurePostprocessing(pipeline);
        _postprocessingPipeline = _dataPipeline.PostprocessingPipeline;
        return this;
    }

    /// <summary>
    /// Caps the number of training rows that the post-train pipeline-fit
    /// step feeds into <c>bestSolution.Predict(...)</c>. Default (when not
    /// called) is no cap — the full <c>XTrain</c> tensor goes through one
    /// extra forward pass solely to materialise predictions for the
    /// pipeline's <c>Fit</c>. For users with large training sets and
    /// postprocessing transformers that stabilise on a subsample
    /// (StandardScaler, MinMaxScaler, label encoders), capping here cuts
    /// the doubled Build-time inference cost (review #1368 C7HAu).
    /// </summary>
    /// <param name="maxRows">Maximum number of leading training rows to
    /// feed into Predict for fit. Pass a non-positive value to clear the
    /// cap (revert to full-set fit).</param>
    // NOTE: deliberately not named `ConfigurePostprocessingFitMaxRows` —
    // the YAML source-generator scans `Configure*` methods and would
    // misrender a primitive `int?` parameter as a POCO YAML section. This
    // is an opt-in perf knob, not a configuration surface that belongs in
    // a YAML model recipe.
    public AiModelBuilder<T, TInput, TOutput> SetPostprocessingFitMaxRows(int? maxRows)
    {
        _dataPipeline.SetPostprocessingFitMaxRows(maxRows);
        _postprocessingFitMaxRows = _dataPipeline.PostprocessingFitMaxRows;
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigureRegularization(IRegularization<T, TInput, TOutput> regularization)
    {
        _trainingCore.ConfigureRegularization(regularization);
        _regularization = _trainingCore.Regularization;
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigureFitnessCalculator(IFitnessCalculator<T, TInput, TOutput> calculator)
    {
        _trainingCore.ConfigureFitnessCalculator(calculator);
        _fitnessCalculator = _trainingCore.FitnessCalculator;
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigureFitDetector(IFitDetector<T, TInput, TOutput> detector)
    {
        _trainingCore.ConfigureFitDetector(detector);
        _fitDetector = _trainingCore.FitDetector;
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigureModel(IFullModel<T, TInput, TOutput> model)
    {
        _trainingCore.ConfigureModel(model);
        _model = _trainingCore.Model;
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
    /// <summary>
    /// Configures TARGET (label) scaling for regression: the targets are scaled (default: z-score via
    /// <see cref="AiDotNet.Preprocessing.TargetStandardScaler{T,TOutput}"/>) before training — fit on the
    /// TRAINING split only — and <c>Predict</c> automatically inverse-transforms model outputs back to the
    /// ORIGINAL target units. Pass a custom pipeline to control the transformation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If your target is, say, a price in the hundreds while your features are
    /// scaled to ~1, gradient training struggles. This scales the target down for training and scales
    /// predictions back up for you — no manual inverse-transform needed. Regression only: never scale
    /// class labels.
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureTargetScaling(
        AiDotNet.Preprocessing.PreprocessingPipeline<T, TOutput, TOutput>? pipeline = null)
    {
        if (pipeline is null)
        {
            pipeline = new AiDotNet.Preprocessing.PreprocessingPipeline<T, TOutput, TOutput>();
            pipeline.Add(new AiDotNet.Preprocessing.TargetStandardScaler<T, TOutput>());
        }

        _targetPipeline = pipeline;
        return this;
    }

    /// <summary>
    /// Configures GROUPED training for ranking-style objectives: each inner list is a set of TRAINING row
    /// indices forming one coherent query group (e.g. one date's cross-section for a learning-to-rank
    /// model). Per epoch the model trains once per GROUP slice instead of once on the pooled set —
    /// pooled training gives pairwise/listwise ranking losses conflicting targets across groups (the same
    /// features map to different within-group ranks on different dates), collapsing the net to a constant.
    /// Neural models with Tensor inputs only; epochs come from the configured optimizer's MaxIterations.
    /// </summary>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureTrainingGroups(IReadOnlyList<IReadOnlyList<int>> groups)
    {
        if (groups is null || groups.Count == 0)
        {
            throw new ArgumentException("At least one training group is required.", nameof(groups));
        }

        _trainingGroups = groups;
        return this;
    }

    public IAiModelBuilder<T, TInput, TOutput> ConfigureOptimizer(IOptimizer<T, TInput, TOutput> optimizationAlgorithm)
    {
        _trainingCore.ConfigureOptimizer(optimizationAlgorithm);
        _optimizer = _trainingCore.Optimizer;
        return this;
    }

    /// <inheritdoc />
    public IAiModelBuilder<T, TInput, TOutput> ConfigureLicenseKey(AiDotNetLicenseKey licenseKey)
    {
        _licensing!.ConfigureLicenseKey(licenseKey);
        _licenseKey = _licensing.LicenseKey;
        _licenseValidator = _licensing.Validator;
        return this;
    }

    /// <summary>
    /// Enables federated learning training using the provided options.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the single entry point for all federated learning configurations.
    /// Set <see cref="FederatedLearningOptions.Mode"/> to choose horizontal (default) or vertical FL.
    /// All v2 subsystems (PSI, MPC, TEE, ZK verification, graph FL, unlearning, fairness, advanced
    /// compression, drift detection) are configured via properties on the options object.</para>
    ///
    /// <para>Optional injectable interface parameters allow advanced users to provide custom
    /// implementations that override the defaults derived from options.</para>
    /// </remarks>
    /// <param name="options">Federated learning configuration options (horizontal or vertical mode, privacy, compression, etc.).</param>
    /// <param name="aggregationStrategy">Optional aggregation strategy override (null uses defaults based on options).</param>
    /// <param name="clientSelectionStrategy">Optional client selection strategy override (null uses defaults based on options).</param>
    /// <param name="serverOptimizer">Optional server-side optimizer override (null uses defaults based on options).</param>
    /// <param name="heterogeneityCorrection">Optional heterogeneity correction strategy override.</param>
    /// <param name="homomorphicEncryptionProvider">Optional homomorphic encryption provider for encrypted aggregation.</param>
    /// <param name="privateSetIntersection">Optional custom PSI protocol for entity alignment in vertical FL or graph FL.</param>
    /// <param name="secureComputationProtocol">Optional custom MPC protocol for secure gradient operations.</param>
    /// <param name="teeProvider">Optional custom TEE provider for hardware-backed secure aggregation.</param>
    /// <param name="zkProofSystem">Optional custom zero-knowledge proof system for verifiable FL.</param>
    /// <param name="federatedUnlearner">Optional custom unlearning implementation for GDPR compliance.</param>
    /// <param name="driftDetector">Optional custom drift detector for concept drift adaptation.</param>
    /// <param name="contributionEvaluator">Optional custom contribution evaluator for client value assessment.</param>
    /// <param name="fairnessConstraint">Optional custom fairness constraint for equitable model performance.</param>
    /// <returns>This builder instance for method chaining.</returns>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureFederatedLearning(
        FederatedLearningOptions options,
        IAggregationStrategy<IFullModel<T, TInput, TOutput>>? aggregationStrategy = null,
        IClientSelectionStrategy? clientSelectionStrategy = null,
        IFederatedServerOptimizer<T>? serverOptimizer = null,
        IFederatedHeterogeneityCorrection<T>? heterogeneityCorrection = null,
        IHomomorphicEncryptionProvider<T>? homomorphicEncryptionProvider = null,
        FederatedLearning.PSI.IPrivateSetIntersection? privateSetIntersection = null,
        FederatedLearning.MPC.ISecureComputationProtocol<T>? secureComputationProtocol = null,
        FederatedLearning.TEE.ITeeProvider<T>? teeProvider = null,
        FederatedLearning.Verification.IZkProofSystem? zkProofSystem = null,
        FederatedLearning.Unlearning.IFederatedUnlearner<T>? federatedUnlearner = null,
        FederatedLearning.DriftDetection.IFederatedDriftDetector<T>? driftDetector = null,
        FederatedLearning.Fairness.IClientContributionEvaluator<T>? contributionEvaluator = null,
        FederatedLearning.Fairness.IFairnessConstraint<T>? fairnessConstraint = null)
    {
        Guard.NotNull(options);
        _federatedLearningOptions = options;
        _federatedAggregationStrategy = aggregationStrategy;
        _federatedClientSelectionStrategy = clientSelectionStrategy;
        _federatedServerOptimizer = serverOptimizer;
        _federatedHeterogeneityCorrection = heterogeneityCorrection;
        _federatedHomomorphicEncryptionProvider = homomorphicEncryptionProvider;
        _federatedPrivateSetIntersection = privateSetIntersection;
        _federatedSecureComputationProtocol = secureComputationProtocol;
        _federatedTeeProvider = teeProvider;
        _federatedZkProofSystem = zkProofSystem;
        _federatedUnlearner = federatedUnlearner;
        _federatedDriftDetector = driftDetector;
        _federatedContributionEvaluator = contributionEvaluator;
        _federatedFairnessConstraint = fairnessConstraint;
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
    /// var result = await new AiModelBuilder&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;()
    ///     .ConfigureModel(network)
    ///     .ConfigureOptimizer(optimizer)
    ///     .ConfigureMixedPrecision()  // Enable mixed-precision
    ///     .BuildAsync();
    ///
    /// // Or with custom configuration
    /// builder.ConfigureMixedPrecision(MixedPrecisionConfig.Conservative());
    /// </code>
    /// </example>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureMixedPrecision(MixedPrecisionConfig? config = null)
    {
        // Fail FAST at configure-time when T is not float. Mixed precision
        // is only meaningful for FP32 master + FP16/BF16 working weights,
        // and the EnableMixedPrecision path that BuildAsync invokes also
        // rejects non-float T. Validating here too means the misconfiguration
        // surfaces at the line the user typed, not at first Train() call —
        // and prevents the "configured but never consumed" footgun that
        // motivated this whole PR family (review #1362).
        if (typeof(T) != typeof(float))
        {
            throw new NotSupportedException(
                $"Mixed-precision training requires T = float; got T = {typeof(T).Name}. " +
                $"Use AiModelBuilder<float, ...> when calling ConfigureMixedPrecision.");
        }
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
    /// After building your model, use the reasoning methods on AiModelResult:
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
    /// var result = await new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureAgentAssistance(agentConfig)
    ///     .ConfigureReasoning()
    ///     .BuildAsync();
    ///
    /// // Use reasoning on the trained model
    /// var reasoningResult = await result.ReasonAsync(
    ///     "Explain why this prediction was made and what factors contributed most?",
    ///     ReasoningMode.ChainOfThought
    /// );
    /// // Result is available in the returned value
    /// </code>
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureReasoning(ReasoningConfig? config = null)
    {
        _reasoningConfig = config ?? new ReasoningConfig();
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
    /// var result = await new AiModelBuilder&lt;double, ...&gt;()
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigureInferenceOptimizations(AiDotNet.Configuration.InferenceOptimizationConfig? config = null)
    {
        _inferenceOptimizationConfig = config ?? AiDotNet.Configuration.InferenceOptimizationConfig.Default;
        return this;
    }

    /// <summary>
    /// Enables JIT (Just-In-Time) compilation for the built model's forward and
    /// backward passes.
    /// </summary>
    /// <param name="config">JIT compilation configuration. If <c>null</c>, uses
    /// <see cref="AiDotNet.Configuration.JitCompilationConfig.Default"/>.</param>
    /// <returns>This builder instance for fluent chaining.</returns>
    /// <remarks>
    /// <para>
    /// JIT compilation traces the model's computation graph on the first call at
    /// each input shape and replays the compiled plan on subsequent calls,
    /// eliminating virtual dispatch, per-op allocation, and bounds-checking
    /// overhead. Typical gains are 1.5-3x on CPU and up to 10x on GPU.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> think of this as turning your model from an
    /// interpreter into a compiled binary. The first prediction is a little
    /// slower (while the library studies your model). Every prediction after
    /// that is much faster. If anything goes wrong during compilation, the
    /// library silently falls back to the original execution path — so enabling
    /// this is safe.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureJitCompilation(
        AiDotNet.Configuration.JitCompilationConfig? config = null)
    {
        _jitCompilationConfig = config ?? AiDotNet.Configuration.JitCompilationConfig.Default;
        _jitCompilationConfig.Validate();
        return this;
    }

    /// <summary>
    /// Captures a snapshot of the active acceleration environment (SIMD, GPU, native BLAS)
    /// at build time, logs it through <paramref name="logger"/> if supplied, and surfaces the
    /// structured snapshot on <c>PredictionModelResult.AccelerationSnapshot</c>.
    /// </summary>
    /// <param name="logger">
    /// Optional callback that receives the formatted report string. When null, the
    /// report is written to <see cref="Console.WriteLine(string)"/>.
    /// </param>
    /// <returns>This builder for fluent chaining.</returns>
    /// <remarks>
    /// <para>
    /// Useful for production observability — shows exactly which of AVX2/AVX-512/NEON,
    /// CUDA/OpenCL/HIP, OpenBLAS/CLBlast/MKL are active on the target host. Users can
    /// assert against the returned snapshot in CI (e.g., fail build if AVX-512 isn't
    /// detected on an Intel Xeon host).
    /// </para>
    /// <para>
    /// Wraps <c>AiDotNet.Tensors.Engines.PlatformDetector</c> and
    /// <c>AiDotNet.Tensors.Engines.NativeLibraryDetector</c>.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ReportAccelerationStatus(Action<string>? logger = null)
    {
        _reportAccelerationAtBuild = true;
        _accelerationLogger = logger;
        return this;
    }

    /// <summary>
    /// Enables disk-backed caching of compiled inference plans. Plans are saved after
    /// the first compilation and loaded transparently on subsequent process starts,
    /// skipping the trace+compile cost of cold start.
    /// </summary>
    /// <param name="directory">
    /// Filesystem directory where plan files are stored. Created if missing. Plans are
    /// keyed by (concrete model type, element type, structure version, input shape)
    /// so one directory can host plans for multiple models.
    /// </param>
    /// <returns>This builder for fluent chaining.</returns>
    /// <remarks>
    /// <para>
    /// PyTorch-parity equivalent: <c>torch.jit.save</c> + <c>torch.jit.load</c>.
    /// Plans are tied to the host's hardware fingerprint (via Tensors'
    /// <c>PlanCompatibilityInfo</c>); plans compiled on one host are rejected on
    /// incompatible hardware, triggering a fresh compile.
    /// </para>
    /// <para>
    /// Caching is opt-in — without this call, plans live only for the process
    /// lifetime via the in-memory <c>CompiledModelCache</c>.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigurePlanCaching(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory))
            throw new ArgumentException("Plan cache directory must be a non-empty path.", nameof(directory));

        AiDotNet.NeuralNetworks.PlanCache.SetCurrent(new AiDotNet.NeuralNetworks.PlanCache(directory));
        return this;
    }

    /// <summary>
    /// Enables low-level per-tensor-op profiling via Tensors'
    /// <c>PerformanceProfiler.Instance</c>. After <c>BuildAsync()</c> returns, kernel
    /// timings are captured on <c>AiModelResult.TensorsOperationProfile</c>.
    /// Orthogonal to the higher-level <c>ConfigureProfiling</c> (AiDotNet workflow
    /// timings) — both can be enabled together to get a full picture.
    /// </summary>
    /// <returns>This builder for fluent chaining.</returns>
    /// <remarks>
    /// <para>
    /// The profiler wraps every engine op in a dispatchable scope — expect a small
    /// overhead (~1-3%) during the measured window. Disable in production latency-
    /// sensitive paths. PyTorch-parity equivalent: low-level
    /// <c>torch.profiler.profile</c> CUDA/CPU op breakdown.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> EnableTensorsOpProfiling()
    {
        _tensorsOpProfilingEnabled = true;
        AiDotNet.Tensors.Engines.Optimization.PerformanceProfiler.Instance.Enabled = true;
        return this;
    }

    /// <summary>
    /// Opts out of the builder's deterministic-by-default policy. Call this when
    /// you want the engine to pick the fastest available kernels even if they
    /// produce slightly different floating-point results across runs or hardware.
    /// </summary>
    /// <returns>This builder for fluent chaining.</returns>
    /// <remarks>
    /// <para>
    /// By default, <see cref="BuildAsync"/> calls
    /// <c>AiDotNetEngine.SetDeterministicMode(true)</c> so every model built by
    /// this library produces bitwise-identical results across runs on the same
    /// hardware.
    /// </para>
    /// <para>
    /// <b>Do NOT call this in production serving</b> where reproducibility
    /// matters for debugging, regression tests, or regulatory compliance.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> AllowNondeterminism()
    {
        _allowNondeterminism = true;
        return this;
    }

    // Uncertainty quantification configuration lives in AiModelBuilder.UncertaintyQuantification.cs to keep this file focused.

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
    /// var result = await new AiModelBuilder&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;()
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigureGpuAcceleration(GpuAccelerationConfig? config = null)
    {
        _gpuAccelerationConfig = config ?? new GpuAccelerationConfig();
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDataLoader(IDataLoader<T> dataLoader)
    {
        _dataPipeline.ConfigureDataLoader(dataLoader);
        _dataLoader = _dataPipeline.DataLoader;
        return this;
    }

    /// <summary>
    /// Configures data preparation operations that change row count (outlier removal, augmentation).
    /// </summary>
    /// <param name="pipelineBuilder">Action to configure the data preparation pipeline.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Data preparation handles operations that add or remove data points:
    /// - <b>Outlier removal:</b> Removes unusual data points that might confuse the model
    /// - <b>Data augmentation (SMOTE):</b> Creates synthetic samples to balance imbalanced classes
    /// </para>
    /// <para>
    /// These operations are only applied during training, not during prediction.
    /// </para>
    /// <para>
    /// <b>Example:</b>
    /// <code>
    /// builder.ConfigureDataPreparation(prep => prep
    ///     .Add(new OutlierRemovalOperation&lt;double&gt;(new IsolationForest&lt;double&gt;()))
    ///     .Add(new AugmentationOperation&lt;double&gt;(new SmoteAugmenter&lt;double&gt;())));
    /// </code>
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDataPreparation(
        Action<DataPreparationPipeline<T>> pipelineBuilder)
    {
        _dataPipeline.ConfigureDataPreparation(pipelineBuilder);
        _dataPreparationPipeline = _dataPipeline.DataPreparationPipeline;
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
    /// var result = await new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureDataLoader(DataLoaders.FromArrays(features, labels))
    ///     .ConfigureModel(model)
    ///     .BuildAsync();
    /// </code>
    ///
    /// Example with meta-learning:
    /// <code>
    /// var result = await new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureMetaLearning(metaLearner)
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    public Task<AiModelResult<T, TInput, TOutput>> BuildAsync() => BuildAsync(CancellationToken.None);

    /// <summary>
    /// Pushes the configured gradient-checkpointing segment size onto the
    /// current model. Called from <c>BuildAsync</c> at the top of the build
    /// flow AND after any code path that reassigns <c>_model</c> (e.g., the
    /// AutoML path that selects the best candidate).
    /// </summary>
    private void ApplyGradientCheckpointingFromMemoryConfig()
    {
        if (_model is not NeuralNetworks.NeuralNetworkBase<T> checkpointingTarget)
            return;

        int effective = 0; // default: disabled
        if (_memoryConfig is { UseGradientCheckpointing: true } memCfg)
        {
            int layerCount = checkpointingTarget.Layers.Count;
            effective = memCfg.CheckpointEveryNLayers > 0
                ? memCfg.CheckpointEveryNLayers
                : Math.Max(1, (int)Math.Sqrt(Math.Max(1, layerCount)));
        }

        checkpointingTarget.SetGradientCheckpointingSegmentSize(effective);
    }

    /// <summary>
    /// Wraps a trained model's <c>Predict</c> in a <c>CompiledModelCache</c>-backed
    /// function matching the <c>JitCompiledFunction</c> shape expected by
    /// <see cref="AiModelResult{T, TInput, TOutput}"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is how JIT compilation reaches every model whose <c>Predict</c> override
    /// bypasses the base class. The wrapper traces the model's forward pass under
    /// GraphMode on the first call at each input shape, compiles it into a flat
    /// replay plan, and re-executes the plan for subsequent calls.
    /// </para>
    /// <para>
    /// Nested-GraphMode defense: during tracing the wrapper temporarily sets
    /// <c>TensorCodecOptions.EnableCompilation = false</c> so any call into
    /// <c>NeuralNetworkBase{T}.PredictCompiled</c> from within the model's own
    /// <c>Predict</c> falls through to <c>PredictEager</c> instead of opening a
    /// second, conflicting GraphMode scope.
    /// </para>
    /// <para>
    /// Scope: only wraps <see cref="NeuralNetworks.NeuralNetworkBase{T}"/>
    /// descendants. Diffusion models (extend <c>DiffusionModelBase</c>, not
    /// <c>NeuralNetworkBase</c>) and non-neural models (regression, trees) return
    /// <c>null</c> so the result's Predict path stays on its existing code path.
    /// Multi-step inference (autoregressive text, RNN unroll) that relies on
    /// per-step scalar control flow will either trace correctly if the loop is
    /// unrolled at compile time, or fall through to eager via the try/catch —
    /// if it traces correctly but produces wrong results on replay (a real risk
    /// for models with non-Engine tensor access), set
    /// <see cref="AiDotNet.Configuration.JitCompilationConfig.ThrowOnFailure"/>
    /// in tests to catch silent divergence.
    /// </para>
    /// </remarks>
    private Func<Tensor<T>[], Tensor<T>[]>? BuildCompiledPredictFunction(
        IFullModel<T, TInput, TOutput>? model)
    {
        if (model is null) return null;
        if (_jitCompilationConfig is null || !_jitCompilationConfig.Enabled) return null;

        // Only NeuralNetworkBase<T> models go through the Engine-op forward pass
        // that GraphMode can record. Regression/tree models use scalar math that
        // wouldn't benefit from graph compilation even if it were traceable.
        if (model is not NeuralNetworks.NeuralNetworkBase<T> nnModel) return null;

        // ThrowOnFailure is enforced at THIS wrapper level (the catch below).
        // We deliberately do NOT push it onto the model instance — multiple
        // results may share one model with different strict-mode policies, and
        // writing per-result config onto a shared instance would race. The
        // lower-level NeuralNetworkBase.PredictCompiled keeps its own silent-
        // fallback-with-Trace behavior; strict mode applies only to the
        // builder-wrapped path that consults ThrowOnFailure here.

        var cache = new AiDotNet.Tensors.Engines.Compilation.CompiledModelCache<T>();
        bool throwOnFailure = _jitCompilationConfig.ThrowOnFailure;
        var jitConfig = _jitCompilationConfig;

        return (inputs) =>
        {
            if (inputs is null || inputs.Length == 0)
            {
                throw new ArgumentException(
                    "JitCompiledFunction requires at least one input tensor.",
                    nameof(inputs));
            }

            var input = inputs[0];

            // Each call applies the JIT config to this thread's TensorCodecOptions
            // — request-pool workers don't inherit the thread-static state set on
            // the builder thread.
            jitConfig.ApplyToTensorCodec();

            try
            {
                var plan = cache.GetOrCompileInference(input, () =>
                {
                    // Guard against nested-GraphMode. When the trace lambda invokes
                    // nnModel.Predict, any subclass still using the base default
                    // would re-enter PredictCompiled which opens a second GraphMode
                    // scope — the inner compile would drop the outer trace's ops.
                    // Forcing EnableCompilation=false here makes PredictCompiled
                    // fall through to PredictEager, recording the ops into our
                    // outer trace instead.
                    var savedOptions = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
                    var traceOptions = new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions
                    {
                        EnableCompilation = false,
                        EnableDataflowFusion = savedOptions.EnableDataflowFusion,
                        EnableAlgebraicBackward = savedOptions.EnableAlgebraicBackward,
                        EnableSpectralDecomposition = savedOptions.EnableSpectralDecomposition,
                        SpectralErrorTolerance = savedOptions.SpectralErrorTolerance,
                        DataflowFusionMaxHidden = savedOptions.DataflowFusionMaxHidden,
                        EnableConvBnFusion = savedOptions.EnableConvBnFusion,
                        EnableAttentionFusion = savedOptions.EnableAttentionFusion,
                        EnablePointwiseFusion = savedOptions.EnablePointwiseFusion,
                        EnableConstantFolding = savedOptions.EnableConstantFolding,
                        EnableForwardCSE = savedOptions.EnableForwardCSE,
                        EnableBlasBatch = savedOptions.EnableBlasBatch,
                        EnableMixedPrecision = savedOptions.EnableMixedPrecision
                    };
                    AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(traceOptions);
                    try
                    {
                        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
                        // Tensors 0.50.1 changed GetOrCompileInference from Action to
                        // Func<Tensor<T>> — the tracer now binds the plan output to
                        // whatever the lambda returns, rather than inferring it from
                        // the last recorded op. Return the Predict result explicitly.
                        return nnModel.Predict(input);
                    }
                    finally
                    {
                        AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(savedOptions);
                    }
                });

                return new[] { plan.Execute() };
            }
            catch (Exception ex) when (!throwOnFailure)
            {
                // Compilation blew up OR replay couldn't rebind inputs cleanly.
                // Log via Trace so the failure is observable in production telemetry
                // — silent fallback would make JIT regressions invisible until perf
                // surveys catch them. Then fall back to eager Predict so the call
                // succeeds. Wrap the fallback in NoGradScope to match the trace's
                // inference semantics (no tape recording during Predict). Next call
                // retries compilation since the cache didn't store a plan for the
                // failed shape.
                System.Diagnostics.Trace.TraceWarning(
                    $"JIT fallback for {nnModel.GetType().FullName} " +
                    $"with input shape [{string.Join(", ", input.Shape)}]: {ex.GetType().Name}: {ex.Message}");

                using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
                return new[] { nnModel.Predict(input) };
            }
        };
    }

    public async Task<AiModelResult<T, TInput, TOutput>> BuildAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // Propagate the builder's license key to ModelPersistenceGuard so it can
        // validate during serialize/deserialize operations within BuildAsync.
        using var licenseScope = Helpers.ModelPersistenceGuard.SetActiveLicenseKey(_licenseKey);

        // Apply gradient checkpointing to the model BEFORE any training runs.
        // The helper reads _memoryConfig and writes O(sqrt(N)) segment size
        // into the NeuralNetworkBase's lazy checkpointing field. Re-applied
        // after any code path that reassigns _model (e.g., AutoML).
        ApplyGradientCheckpointingFromMemoryConfig();

        // Apply weight-streaming overrides BEFORE any forward pass so the
        // model's first Predict / Train sees the user's intent (force-on,
        // force-off, or default auto-detect). Idempotent: no-op when the
        // user didn't call ConfigureWeightStreaming, and the model's own
        // ctor-time auto-detect already ran.
        ApplyWeightStreamingConfig();

        // Apply JIT compilation config so every subsequent step in BuildAsync
        // sees the configured TensorCodecOptions. CompiledModelCache engages
        // automatically when Enabled=true; Enabled=false short-circuits to eager.
        _jitCompilationConfig?.ApplyToTensorCodec();

        // Deterministic-by-default: force bitwise-reproducible kernels unless
        // the caller opted out via AllowNondeterminism().
        AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(!_allowNondeterminism);

        // Validate RAG pipeline composition if any RAG components were configured
        bool hasAnyRAG = _ragRetriever != null || _ragReranker != null || _ragGenerator != null
            || _queryProcessors != null || _graphStore != null || _knowledgeGraph != null;
        if (hasAnyRAG)
        {
            var ragValidation = RetrievalAugmentedGeneration.Configuration.PipelineValidator.ValidateRAGConfiguration(
                hasRetriever: _ragRetriever != null,
                hasReranker: _ragReranker != null,
                hasGenerator: _ragGenerator != null,
                hasQueryProcessors: _queryProcessors != null,
                hasDocumentStore: _graphStore != null,
                hasKnowledgeGraph: _knowledgeGraph != null,
                hasGraphStore: _graphStore != null);

            if (!ragValidation.IsValid)
            {
                throw new InvalidOperationException(
                    "RAG pipeline configuration is invalid:\n" +
                    string.Join("\n", ragValidation.Errors));
            }
        }

        AiModelResult<T, TInput, TOutput> result;

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

        // SELF-SUPERVISED / GENERATIVE TRAINING PATH - the model owns its objective
        // (e.g. diffusion noise prediction), so the supervised optimizer path (which fits
        // Predict(X) ≈ Y) cannot train it. Route it to a per-sample epoch loop that calls
        // the model's own Train. Detected by capability (ISelfSupervisedModel), not by type.
        if (_model is Interfaces.ISelfSupervisedModel)
        {
            result = await BuildSelfSupervisedInternalAsync(cancellationToken);
            await RunBenchmarksIfConfiguredAsync(result).ConfigureAwait(false);
            return result;
        }

        // DATA LOADER PATH - check if data loader is configured and provides input/output data
        if (_dataLoader is IInputOutputDataLoader<T, TInput, TOutput> inputOutputLoader)
        {
            // Load data if not already loaded
            if (!_dataLoader.IsLoaded)
            {
                await _dataLoader.LoadAsync(cancellationToken);
            }

            cancellationToken.ThrowIfCancellationRequested();

            // Get features and labels from the typed loader
            var features = inputOutputLoader.Features;
            var labels = inputOutputLoader.Labels;

            // Delegate to the internal supervised training method
            result = await BuildSupervisedInternalAsync(features, labels, cancellationToken);
            await RunBenchmarksIfConfiguredAsync(result).ConfigureAwait(false);
            return result;
        }

        // STREAMING DATA LOADER PATH - check if data loader is a streaming loader
        if (_dataLoader is IStreamingDataLoader<T, TInput, TOutput> streamingLoader)
        {
            // Load/prepare the streaming loader if not already loaded
            if (!_dataLoader.IsLoaded)
            {
                await _dataLoader.LoadAsync(cancellationToken);
            }

            cancellationToken.ThrowIfCancellationRequested();

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

    private AiModelResult<T, TInput, TOutput> BuildProgramSynthesisInferenceOnlyResult()
    {
        // Preprocessing requires fitted statistics - for inference-only builds, the pipeline must be pre-fitted
        if (_preprocessingPipeline is not null && !_preprocessingPipeline.IsFitted)
        {
            throw new InvalidOperationException(
                "Inference-only builds require a pre-fitted preprocessing pipeline. " +
                "Either fit the pipeline on training data first, preprocess data externally, or omit ConfigurePreprocessing().");
        }

        // Wire document transformers for inference-only builds
        ConfigureDocumentTransformers(_model);

        // Ensure inference-only builds still honor configured GPU acceleration.
        ApplyGpuConfiguration();

        var optimizationResult = new OptimizationResult<T, TInput, TOutput>
        {
            BestSolution = _model ?? throw new InvalidOperationException("Model has not been configured. Call ConfigureModel() before BuildForInference().")
        };

        // Inference-only path has no training data — pipeline must be
        // pre-fitted. The shared helper throws with a clear diagnostic if
        // postprocessing is configured but unfit (review #1368 C6WJG).
        FitPostprocessingIfNeeded(optimizationResult.BestSolution, default, nameof(BuildProgramSynthesisInferenceOnlyResult));

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

        var options = new AiModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = optimizationResult,
            PreprocessingInfo = _preprocessingPipeline is not null && _preprocessingPipeline.IsFitted
                ? new PreprocessingInfo<T, TInput, TOutput>(_preprocessingPipeline)
                : null,
            PostprocessingPipeline = _postprocessingPipeline,
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            ProgramSynthesisModel = _programSynthesisModel,
            ProgramSynthesisServingClient = _programSynthesisServingClient,
            ProgramSynthesisServingClientOptions = _programSynthesisServingClientOptions,
            InferenceOptimizationConfig = _inferenceOptimizationConfig,
            JitCompilationConfig = _jitCompilationConfig,
            JitCompiledFunction = BuildCompiledPredictFunction(optimizationResult.BestSolution),
            AllowNondeterminism = _allowNondeterminism,
            AugmentationConfig = _augmentationConfig,
            ReasoningConfig = _reasoningConfig,
            DeploymentConfiguration = deploymentConfig,
            BiasDetector = _biasDetector,
            FairnessEvaluator = _fairnessEvaluator,
            InterpretabilityOptions = _interpretabilityOptions,
            RagRetriever = _ragRetriever,
            RagReranker = _ragReranker,
            RagGenerator = _ragGenerator,
            QueryProcessors = _queryProcessors,
            KnowledgeGraph = _knowledgeGraph,
            GraphStore = _graphStore,
            HybridGraphRetriever = _hybridGraphRetriever,
            PromptTemplate = null,
            PromptOptimizer = null,
            FewShotExampleSelector = null,
            PromptAnalyzer = null,
            PromptCompressor = null,
            MemoryConfig = _memoryConfig,
            // Mirror the supervised / AutoML / RL result builders: surface
            // the streaming report when the inference-only model is itself
            // a NeuralNetworkBase that auto-detected (or the caller forced)
            // streaming. Inference-only builds still benefit from streaming
            // for very large models served read-only, so the telemetry
            // shouldn't go missing on this path.
            WeightStreamingReport = BuildWeightStreamingReport(),
        };

        var programSynthesisResult = AttachDiagnostics(new AiModelResult<T, TInput, TOutput>(options));
        ProcessKnowledgeGraphOptions(programSynthesisResult);
        AttachSafetyPipeline(programSynthesisResult);
        AttachAdversarialRobustness(programSynthesisResult);
        return programSynthesisResult;
    }

    private void AttachSafetyPipeline(AiModelResult<T, TInput, TOutput> result)
    {
        if (_safetyPipelineConfig != null)
        {
            result.SafetyPipeline = AiDotNet.Safety.SafetyPipelineFactory<T>.Create(_safetyPipelineConfig);
        }
    }

    /// <summary>
    /// Threads any <see cref="_adversarialRobustnessConfiguration"/> set via
    /// <see cref="ConfigureAdversarialRobustness"/> into the constructed
    /// <see cref="AiModelResult{T, TInput, TOutput}"/> so the runtime
    /// adversarial-robustness API (<c>PredictWithDefense</c>,
    /// <c>EvaluateRobustness</c>) actually picks up the user's settings.
    /// </summary>
    /// <remarks>
    /// Prior to this method the <see cref="ConfigureAdversarialRobustness"/>
    /// call only stored the configuration in
    /// <see cref="_adversarialRobustnessConfiguration"/>; the field was never
    /// read elsewhere, so the call had no observable effect (issue #1357 —
    /// "ConfigureAdversarialRobustness stores config, never consumes it").
    /// This method mirrors <see cref="AttachSafetyPipeline"/> and is invoked
    /// from every Build path so per-sample-train consumers also see the
    /// configuration on the returned result.
    /// </remarks>
    private void AttachAdversarialRobustness(AiModelResult<T, TInput, TOutput> result)
    {
        if (_adversarialRobustnessConfiguration is null || !_adversarialRobustnessConfiguration.Enabled)
        {
            return;
        }

        // Always surface the underlying options so EvaluateRobustness /
        // PredictWithDefense can read them at inference time even when no
        // custom defense was supplied.
        result.SetAdversarialRobustnessOptions(_adversarialRobustnessConfiguration.Options);

        if (_adversarialRobustnessConfiguration.CustomDefense is not null)
        {
            result.SetAdversarialDefense(_adversarialRobustnessConfiguration.CustomDefense);
        }
    }

    private void ProcessKnowledgeGraphOptions(AiModelResult<T, TInput, TOutput> result)
    {
        if (_knowledgeGraphOptions == null)
            return;

        if (_knowledgeGraph == null)
        {
            throw new InvalidOperationException(
                "KnowledgeGraph options were configured via ConfigureKnowledgeGraph(), " +
                "but no KnowledgeGraph instance was provided. " +
                "Call ConfigureRetrievalAugmentedGeneration() to set up the KnowledgeGraph, then ConfigureKnowledgeGraph() to enable advanced features.");
        }

        var kgResult = new Models.Results.KnowledgeGraphResult<T>();

        // Run KG construction from text if configured
        if (_knowledgeGraphOptions.ConstructionTexts != null &&
            _knowledgeGraphOptions.ConstructionTexts.Count > 0)
        {
            var constructor = new AiDotNet.RetrievalAugmentedGeneration.Graph.Construction.KGConstructor<T>();
            foreach (var text in _knowledgeGraphOptions.ConstructionTexts)
            {
                if (!string.IsNullOrWhiteSpace(text))
                {
                    constructor.ConstructFromText(text, _knowledgeGraph, _knowledgeGraphOptions.ConstructionOptions);
                }
            }
        }

        var needsLinkPrediction = _knowledgeGraphOptions.GetEffectiveEnableLinkPrediction();
        var needsTraining = _knowledgeGraphOptions.GetEffectiveTrainEmbeddings();

        if (needsLinkPrediction && !needsTraining)
        {
            throw new InvalidOperationException(
                "EnableLinkPrediction requires TrainEmbeddings to be true. " +
                "Link prediction evaluation needs a trained embedding model.");
        }

        List<GraphEdge<T>>? testEdges = null;
        KnowledgeGraph<T> trainingGraph = _knowledgeGraph;

        // Split edges for evaluation when both link prediction and training are enabled.
        if (needsLinkPrediction && needsTraining)
        {
            var allEdges = _knowledgeGraph.GetAllEdges().ToList();
            var testFraction = _knowledgeGraphOptions.GetEffectiveLinkPredictionTestFraction();
            var maxTestEdges = _knowledgeGraphOptions.GetEffectiveLinkPredictionMaxTestEdges();
            int testSize = Math.Min((int)Math.Round(allEdges.Count * testFraction), maxTestEdges);

            // Ensure at least 5 edges remain for training after the split
            const int minTrainingEdges = 5;
            if (testSize > 0 && allEdges.Count - testSize >= minTrainingEdges)
            {
                // Shuffle edges for a random split using seed from embedding options if available
                var splitRng = _knowledgeGraphOptions.EmbeddingOptions?.Seed is int seed
                    ? RandomHelper.CreateSeededRandom(seed)
                    : RandomHelper.CreateSecureRandom();
                for (int i = allEdges.Count - 1; i > 0; i--)
                {
                    int j = splitRng.Next(i + 1);
                    (allEdges[i], allEdges[j]) = (allEdges[j], allEdges[i]);
                }

                testEdges = allEdges.GetRange(allEdges.Count - testSize, testSize);
                var trainEdges = allEdges.GetRange(0, allEdges.Count - testSize);

                // Build a training-only graph (same nodes, fewer edges)
                trainingGraph = new KnowledgeGraph<T>();
                foreach (var node in _knowledgeGraph.GetAllNodes())
                    trainingGraph.AddNode(node);
                foreach (var edge in trainEdges)
                    trainingGraph.AddEdge(edge);
            }
        }

        // Train embeddings if requested
        if (_knowledgeGraphOptions.GetEffectiveTrainEmbeddings())
        {
            var embedding = CreateEmbeddingModel(_knowledgeGraphOptions.GetEffectiveEmbeddingType());
            var trainingResult = embedding.Train(trainingGraph, _knowledgeGraphOptions.EmbeddingOptions);
            kgResult.EmbeddingTrainingResult = trainingResult;
            kgResult.TrainedEmbedding = embedding;
        }

        // Set up GraphRAG options with mode from KG options
        var ragOptions = _knowledgeGraphOptions.GraphRAGOptions ?? new GraphRAGOptions();
        if (_knowledgeGraphOptions.GraphRAGMode.HasValue)
        {
            ragOptions.Mode = _knowledgeGraphOptions.GraphRAGMode.Value;
        }

        // Create EnhancedGraphRAG (uses full graph for retrieval)
        var enhancedRAG = new EnhancedGraphRAG<T>(_knowledgeGraph, ragOptions);

        // Build community index for Global/Drift modes
        var effectiveMode = ragOptions.Mode ?? Enums.GraphRAGMode.Local;
        if (effectiveMode == Enums.GraphRAGMode.Global || effectiveMode == Enums.GraphRAGMode.Drift)
        {
            enhancedRAG.BuildCommunityIndex();
            kgResult.CommunityStructure = enhancedRAG.CommunityStructure;

            // Generate community summaries
            if (kgResult.CommunityStructure != null)
            {
                var summarizer = new CommunitySummarizer<T>();
                kgResult.CommunitySummaries = summarizer.Summarize(_knowledgeGraph, kgResult.CommunityStructure);
            }
        }

        kgResult.EnhancedGraphRAG = enhancedRAG;

        // Run link prediction evaluation on held-out test edges
        if (needsLinkPrediction && kgResult.TrainedEmbedding != null && testEdges != null && testEdges.Count > 0)
        {
            var predictor = new LinkPredictor<T>(kgResult.TrainedEmbedding);
            var testTriples = testEdges.Select(e => (e.SourceId, e.RelationType, e.TargetId));

            // Evaluate against the full graph (standard filtered setting)
            kgResult.LinkPredictionEvaluation = predictor.EvaluateModel(_knowledgeGraph, testTriples);
        }

        result.KnowledgeGraphResult = kgResult;
    }

    private static IKnowledgeGraphEmbedding<T> CreateEmbeddingModel(KGEmbeddingType embeddingType)
    {
        return embeddingType switch
        {
            KGEmbeddingType.TransE => new TransEEmbedding<T>(),
            KGEmbeddingType.RotatE => new RotatEEmbedding<T>(),
            KGEmbeddingType.ComplEx => new ComplExEmbedding<T>(),
            KGEmbeddingType.DistMult => new DistMultEmbedding<T>(),
            KGEmbeddingType.TemporalTransE => new TemporalTransEEmbedding<T>(),
            _ => throw new NotSupportedException(
                $"Unknown KGEmbeddingType '{embeddingType}'. " +
                $"Supported types: {string.Join(", ", Enum.GetNames(typeof(KGEmbeddingType)))}")
        };
    }

    private Task RunBenchmarksIfConfiguredAsync(AiModelResult<T, TInput, TOutput> result)
    {
        if (_benchmarkingOptions is null || _benchmarkingOptions.Suites is null || _benchmarkingOptions.Suites.Length == 0)
        {
            return Task.CompletedTask;
        }

        return result.EvaluateBenchmarksAsync(_benchmarkingOptions);
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
    private AiModelResult<T, TInput, TOutput> BuildMetaLearningInternalAsync()
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

        // Meta-learning has no single training-X tensor (the meta-learner
        // trains over a distribution of tasks). The shared helper throws a
        // clear diagnostic when postprocessing is configured but unfit
        // here, redirecting the user to pre-fit (review #1368 C6WJG).
        FitPostprocessingIfNeeded(null, default, nameof(BuildMetaLearningInternalAsync));

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

        // Create AiModelResult with meta-learning options
        var metaOptions = new AiModelResultOptions<T, TInput, TOutput>
        {
            MetaLearner = _metaLearner,
            MetaTrainingResult = metaResult,
            PostprocessingPipeline = _postprocessingPipeline,
            JitCompilationConfig = _jitCompilationConfig,
            AllowNondeterminism = _allowNondeterminism,
            LoRAConfiguration = _loraConfiguration,
            BiasDetector = _biasDetector,
            FairnessEvaluator = _fairnessEvaluator,
            InterpretabilityOptions = _interpretabilityOptions,
            RagRetriever = _ragRetriever,
            RagReranker = _ragReranker,
            RagGenerator = _ragGenerator,
            QueryProcessors = _queryProcessors,
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
            PromptTemplate = null,
            PromptOptimizer = null,
            FewShotExampleSelector = null,
            PromptAnalyzer = null,
            PromptCompressor = null,
            ProfilingReport = profilerSession?.GetReport(),
            WeightStreamingReport = BuildWeightStreamingReport(),
            MemoryConfig = _memoryConfig
        };

        var result = AttachDiagnostics(new AiModelResult<T, TInput, TOutput>(metaOptions));
        ProcessKnowledgeGraphOptions(result);
        AttachSafetyPipeline(result);
        AttachAdversarialRobustness(result);

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
                $"RL AutoML requires AiModelBuilder<T, Vector<T>, Vector<T>>. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
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
    // ── SELF-SUPERVISED / GENERATIVE TRAINING ────────────────────────────────
    // Trains a model that owns its objective (ISelfSupervisedModel, e.g. diffusion).
    // Runs epochs over the data calling model.Train(sample, sample) per sample — the model
    // turns each sample into its own learning signal (diffusion: add noise at a random
    // timestep, learn to predict it). Epochs come from the configured optimizer's
    // MaxIterations, else a sensible default.
    private async Task<AiModelResult<T, TInput, TOutput>> BuildSelfSupervisedInternalAsync(
        CancellationToken cancellationToken)
    {
        if (_model is not IFullModel<T, TInput, TOutput> model)
        {
            throw new InvalidOperationException(
                "Self-supervised training requires a model configured via ConfigureModel(...).");
        }

        var samples = await GatherSelfSupervisedSamplesAsync(cancellationToken);
        if (samples.Count == 0)
        {
            throw new InvalidOperationException(
                "Self-supervised training has no data. Provide samples via ConfigureDataLoader(...).");
        }

        int epochs = _optimizer is not null
            ? Math.Max(1, _optimizer.GetOptions().MaxIterations)
            : 100;

        var rng = _allowNondeterminism ? new Random() : new Random(42);
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            foreach (var sample in samples.OrderBy(_ => rng.Next()))
            {
                if (sample is null) continue;
                // Self-supervised: the target is ignored — the model derives its own objective
                // from the input (e.g. diffusion adds noise and predicts it). Self-supervised
                // models use the same type for input and output (Tensor→Tensor), so the input
                // doubles as the (unused) expected-output argument.
                var target = (TOutput)(object)sample;
                model.Train(sample, target);
            }
        }

        var result = AttachDiagnostics(new AiModelResult<T, TInput, TOutput>());
        result.Model = model;
        if (_knowledgeGraph != null || _graphStore != null || _hybridGraphRetriever != null)
        {
            result.AttachGraphComponents(_knowledgeGraph, _graphStore, _hybridGraphRetriever);
        }
        if (_tokenizer != null)
        {
            result.AttachTokenizer(_tokenizer, _tokenizationConfig);
        }
        return result;
    }

    // Materialize per-sample inputs from the configured data loader for self-supervised
    // training, splitting a batched tensor along its leading (batch) dimension.
    private async Task<List<TInput>> GatherSelfSupervisedSamplesAsync(CancellationToken cancellationToken)
    {
        var samples = new List<TInput>();
        if (_dataLoader is IInputOutputDataLoader<T, TInput, TOutput> loader)
        {
            if (!_dataLoader.IsLoaded)
            {
                await _dataLoader.LoadAsync(cancellationToken);
            }
            SplitBatchedSamples(loader.Features, samples);
        }
        return samples;
    }

    // Split a batched TInput into per-sample TInputs: Tensor<T> [N, …] → N × [1, …];
    // otherwise treat the value as a single sample.
    private static void SplitBatchedSamples(TInput features, List<TInput> into)
    {
        if (features is Tensor<T> t && t.Shape.Length >= 2 && t.Shape[0] > 0)
        {
            int n = t.Shape[0];
            int stride = 1;
            for (int d = 1; d < t.Shape.Length; d++) stride *= t.Shape[d];
            var data = t.ToVector();
            var sampleShape = new int[t.Shape.Length];
            sampleShape[0] = 1;
            for (int d = 1; d < t.Shape.Length; d++) sampleShape[d] = t.Shape[d];
            for (int k = 0; k < n; k++)
            {
                var v = new Vector<T>(stride);
                for (int j = 0; j < stride; j++) v[j] = data[k * stride + j];
                if (new Tensor<T>(sampleShape, v) is TInput ti) into.Add(ti);
            }
        }
        else if (features is not null)
        {
            into.Add(features);
        }
    }

    private async Task<AiModelResult<T, TInput, TOutput>> BuildRLInternalAsync(int episodes, bool verbose)
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
                // RL AutoML reassigns _model to whichever agent was selected.
                // Re-apply weight-streaming config (mirrors the supervised
                // AutoML path above) so the user's per-instance threshold /
                // Enabled override flows to the actual training instance.
                // Closes review-comment #1271.s-NU.
                ApplyWeightStreamingConfig();
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

        // RL doesn't use preprocessing like supervised learning - set to null
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

        // RL has no single training-X tensor (the agent trains by acting in
        // an environment, not against a static dataset). The shared helper
        // throws a clear diagnostic when postprocessing is configured but
        // unfit here, redirecting the user to pre-fit (review #1368 C6WJG).
        FitPostprocessingIfNeeded(optimizationResult.BestSolution, default, nameof(BuildRLInternalAsync));

        // Return standard AiModelResult
        // Note: This Build() overload doesn't perform JIT compilation (only the main Build() does),
        // so JitCompiledFunction is not set
        var rlOptions = new AiModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = optimizationResult,
            PreprocessingInfo = preprocessingInfo,
            PostprocessingPipeline = _postprocessingPipeline,
            AutoMLSummary = autoMLSummary,
            BiasDetector = _biasDetector,
            FairnessEvaluator = _fairnessEvaluator,
            InterpretabilityOptions = _interpretabilityOptions,
            RagRetriever = _ragRetriever,
            RagReranker = _ragReranker,
            RagGenerator = _ragGenerator,
            QueryProcessors = _queryProcessors,
            LoRAConfiguration = _loraConfiguration,
            DeploymentConfiguration = deploymentConfig,
            InferenceOptimizationConfig = _inferenceOptimizationConfig,
            JitCompilationConfig = _jitCompilationConfig,
            JitCompiledFunction = BuildCompiledPredictFunction(optimizationResult.BestSolution),
            AllowNondeterminism = _allowNondeterminism,
            ReasoningConfig = _reasoningConfig,
            KnowledgeGraph = _knowledgeGraph,
            GraphStore = _graphStore,
            HybridGraphRetriever = _hybridGraphRetriever,
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            ProgramSynthesisModel = _programSynthesisModel,
            ProgramSynthesisServingClient = _programSynthesisServingClient,
            ProgramSynthesisServingClientOptions = _programSynthesisServingClientOptions,
            PromptTemplate = null,
            PromptOptimizer = null,
            FewShotExampleSelector = null,
            PromptAnalyzer = null,
            PromptCompressor = null,
            ProfilingReport = profilerSession?.GetReport(),
            WeightStreamingReport = BuildWeightStreamingReport(),
            MemoryConfig = _memoryConfig
        };

        var result = AttachDiagnostics(new AiModelResult<T, TInput, TOutput>(rlOptions));
        ProcessKnowledgeGraphOptions(result);
        AttachSafetyPipeline(result);
        AttachAdversarialRobustness(result);

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
    public TOutput Predict(TInput newData, AiModelResult<T, TInput, TOutput> modelResult)
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
    public void SaveModel(AiModelResult<T, TInput, TOutput> modelResult, string filePath)
    {
        if (modelResult is null)
        {
            throw new ArgumentNullException(nameof(modelResult));
        }

        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        if (modelResult.Model is not Interfaces.IModelSerializer serializer)
        {
            throw new InvalidOperationException(
                "The model does not implement IModelSerializer. " +
                "Cannot save as encrypted AIMF format.");
        }

        int[] inputShape = modelResult.Model is Interfaces.IModelShape shape
            ? shape.GetInputShape()
            : Array.Empty<int>();
        int[] outputShape = modelResult.Model is Interfaces.IModelShape outShape
            ? outShape.GetOutputShape()
            : Array.Empty<int>();

        // Resolve encryption key: license key > build key > trial key
        string? resolvedKey = LicenseKeyResolver.Resolve(_licenseKey);
        byte[]? decryptionToken = null;
        bool isTrialOperation = false;

        if (_licenseKey is not null && resolvedKey is null)
        {
            // A license key was configured but could not be resolved — fail closed
            throw new InvalidOperationException(
                "A license key was configured but could not be resolved. " +
                "Verify the license key is valid. Cannot save model without a resolved key.");
        }

        if (resolvedKey is not null)
        {
            // Licensed user — always validate through LicenseValidator
            // This ensures env/file keys are also validated, not just keys with a ServerUrl
            var effectiveLicenseKey = _licenseKey ?? new AiDotNetLicenseKey(resolvedKey);
            var validator = new LicenseValidator(effectiveLicenseKey);
            var validationResult = validator.Validate();

            if (validationResult.Status != LicenseKeyStatus.Active &&
                validationResult.Status != LicenseKeyStatus.ValidationPending)
            {
                throw new Exceptions.LicenseRequiredException(
                    validationResult.Status switch
                    {
                        LicenseKeyStatus.Expired => Exceptions.TrialExpirationReason.LicenseExpired,
                        LicenseKeyStatus.Revoked => Exceptions.TrialExpirationReason.LicenseInvalid,
                        LicenseKeyStatus.SeatLimitReached => Exceptions.TrialExpirationReason.SeatLimitReached,
                        _ => Exceptions.TrialExpirationReason.LicenseInvalid
                    });
            }

            decryptionToken = validationResult.DecryptionToken;
        }
        else
        {
            // No license key — this is a trial save
            isTrialOperation = true;
            // Use a deterministic trial encryption key derived from the machine fingerprint.
            resolvedKey = GenerateTrialEncryptionKey();
        }

        using (ModelPersistenceGuard.InternalOperation())
        {
            ModelLoader.SaveEncrypted(serializer, filePath, resolvedKey, inputShape, outputShape,
                decryptionToken: decryptionToken);
        }

        // Record trial operation only after save succeeded
        if (isTrialOperation)
        {
            var trialManager = new TrialStateManager();
            trialManager.RecordOperationOrThrow();
        }
    }

    /// <summary>
    /// Generates a deterministic encryption key for trial-mode saves.
    /// The key is derived from the machine fingerprint so that trial models
    /// can be loaded on the same machine during the trial period.
    /// </summary>
    private static string GenerateTrialEncryptionKey()
    {
        string machineId = MachineFingerprint.GetMachineId();
        byte[] keyMaterial = System.Text.Encoding.UTF8.GetBytes("AiDotNet.Trial.EncKey.v1:" + machineId);

        using var sha = System.Security.Cryptography.SHA256.Create();
        byte[] hash = sha.ComputeHash(keyMaterial);

        return Convert.ToBase64String(hash);
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
    /// <b>Design Note:</b> SaveModel requires the model to implement <c>IModelSerializer</c> for encrypted saves,
    /// while LoadModel auto-detects the model type from the AIMF header via <c>ModelTypeRegistry</c>.
    /// This asymmetry is intentional: the save path needs the serializer to produce bytes,
    /// while the load path uses the type name embedded in the header to reconstruct the model.
    /// </remarks>
    public AiModelResult<T, TInput, TOutput> LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);
        }

        // Resolve encryption key and validate license/trial
        string? resolvedKey = LicenseKeyResolver.Resolve(_licenseKey);
        byte[]? decryptionToken = null;
        bool isTrialLoad = false;

        if (_licenseKey is not null && resolvedKey is null)
        {
            throw new InvalidOperationException(
                "A license key was configured but could not be resolved. " +
                "Verify the license key is valid.");
        }

        if (resolvedKey is not null)
        {
            // Licensed user — always validate through LicenseValidator
            // This ensures env/file keys are also validated, not just keys with a ServerUrl
            var effectiveLicenseKey = _licenseKey ?? new AiDotNetLicenseKey(resolvedKey);
            var validator = new LicenseValidator(effectiveLicenseKey);
            var validationResult = validator.Validate();

            if (validationResult.Status != LicenseKeyStatus.Active &&
                validationResult.Status != LicenseKeyStatus.ValidationPending)
            {
                throw new Exceptions.LicenseRequiredException(
                    validationResult.Status switch
                    {
                        LicenseKeyStatus.Expired => Exceptions.TrialExpirationReason.LicenseExpired,
                        LicenseKeyStatus.Revoked => Exceptions.TrialExpirationReason.LicenseInvalid,
                        LicenseKeyStatus.SeatLimitReached => Exceptions.TrialExpirationReason.SeatLimitReached,
                        _ => Exceptions.TrialExpirationReason.LicenseInvalid
                    });
            }

            decryptionToken = validationResult.DecryptionToken;
        }
        else
        {
            // No license key — this is a trial load
            isTrialLoad = true;
            resolvedKey = GenerateTrialEncryptionKey();
        }

        // All models must be encrypted AIMF format
        if (!ModelFileHeader.HasHeader(filePath))
        {
            throw new InvalidOperationException(
                "This file is not in the encrypted AIMF format. " +
                "Only models saved with AiModelBuilder.SaveModel() can be loaded. " +
                "Re-save your model using AiModelBuilder.SaveModel() to convert it.");
        }

        byte[] data = File.ReadAllBytes(filePath);
        IModelSerializer model;
        using (ModelPersistenceGuard.InternalOperation())
        {
            try
            {
                model = ModelLoader.LoadFromBytes<T>(data, resolvedKey, decryptionToken);
            }
            catch (System.Security.Cryptography.CryptographicException) when (resolvedKey is not null)
            {
                // If decryption fails with the license key, try the trial key as fallback.
                // This handles the migration case where a user saved during trial and later upgraded.
                string trialKey = GenerateTrialEncryptionKey();
                if (trialKey != resolvedKey)
                {
                    model = ModelLoader.LoadFromBytes<T>(data, trialKey);
                }
                else
                {
                    throw;
                }
            }
        }

        if (model is Interfaces.IFullModel<T, TInput, TOutput> fullModel)
        {
            // Record trial operation only after successful load
            if (isTrialLoad)
            {
                var trialManager = new TrialStateManager();
                trialManager.RecordOperationOrThrow();
            }

            var result = AttachDiagnostics(new AiModelResult<T, TInput, TOutput>());
            result.Model = fullModel;

            // Reattach Graph RAG components if configured
            if (_knowledgeGraph != null || _graphStore != null || _hybridGraphRetriever != null)
            {
                result.AttachGraphComponents(_knowledgeGraph, _graphStore, _hybridGraphRetriever);
            }

            // Reattach tokenizer if configured
            if (_tokenizer != null)
            {
                result.AttachTokenizer(_tokenizer, _tokenizationConfig);
            }

            return result;
        }

        // Fallback: serialize through the builder's format for non-IFullModel serializers.
        // Wrap in InternalOperation to avoid double-counting (LoadModel already enforced above)
        byte[] decryptedPayload;
        using (ModelPersistenceGuard.InternalOperation())
        {
            decryptedPayload = model.Serialize();
        }

        using (ModelPersistenceGuard.InternalOperation())
        {
            return DeserializeModel(decryptedPayload);
        }
    }

    /// <summary>
    /// Validates the configured license key, reusing a cached validator to preserve
    /// in-memory state (e.g., offline grace period tracking).
    /// </summary>
    private LicenseValidationResult ValidateLicense()
    {
        if (_licenseKey is null)
        {
            throw new InvalidOperationException("No license key configured.");
        }

        _licenseValidator ??= new LicenseValidator(_licenseKey);
        return _licenseValidator.Validate();
    }
}
