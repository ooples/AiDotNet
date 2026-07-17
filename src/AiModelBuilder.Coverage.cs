using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Agentic.Tools;
using AiDotNet.Clustering.Evaluation;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.Evaluation.Metrics;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.SelfSupervisedLearning;

namespace AiDotNet;

/// <summary>
/// Extended coverage configuration extensions for AiModelBuilder.
/// Provides Configure methods for data splitting, evaluation metrics, text vectorization,
/// document storage, benchmarking, physics-informed specifications, cluster metrics,
/// curriculum scheduling, exploration strategies, self-supervised learning, stopping criteria,
/// time series decomposition, distillation strategies, model compression, and tools.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private IDataSplitter<T>? _configuredDataSplitter;
    private IClassificationMetric<T>? _configuredClassificationMetric;
    private IRegressionMetric<T>? _configuredRegressionMetric;
    private ITextVectorizer<T>? _configuredTextVectorizer;
    private IClusterMetric<T>? _configuredClusterMetric;
    private IExternalClusterMetric<T>? _configuredExternalClusterMetric;
    private ICurriculumScheduler<T>? _configuredCurriculumScheduler;
    private ReinforcementLearning.Policies.Exploration.IExplorationStrategy<T>? _configuredExplorationStrategy;
    private ReinforcementLearning.IntrinsicMotivation.IIntrinsicRewardModule<T>? _configuredIntrinsicRewardModule;
    private double _configuredIntrinsicRewardWeight = 0.5;
    private IStoppingCriterion<T>? _configuredStoppingCriterion;
    private ITimeSeriesDecomposition<T>? _configuredTimeSeriesDecomposition;
    private IDistillationStrategy<T>? _configuredDistillationStrategy;
    private IModelCompressionStrategy<T>? _configuredModelCompressionStrategy;
    private IAgentTool? _configuredTool;
    private IEnvironment<T>? _configuredEnvironment;
    private IAdversarialAttack<T, TInput, TOutput>? _configuredAdversarialAttack;
    private IAdversarialDefense<T, TInput, TOutput>? _configuredAdversarialDefense;
    private ICertifiedDefense<T, TInput, TOutput>? _configuredCertifiedDefense;
    private ActiveLearning.Interfaces.IQueryStrategy<T, TInput, TOutput>? _configuredQueryStrategy;
    private IReadOnlyList<TInput>? _queryStrategyPool;
    private int _queryStrategyBatchSize = 10;
    private double _queryStrategyDiversityWeight = 0.5;
    private IAudioEnhancer<T>? _configuredAudioEnhancer;
    private RetrievalAugmentedGeneration.VectorSearch.ISimilarityMetric<T>? _configuredSimilarityMetric;



    /// <summary>
    /// Configures a data splitting strategy for dividing datasets into train/test/validation sets.
    /// </summary>
    /// <param name="splitter">The data splitter implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Data splitting divides your dataset into separate portions for
    /// training and evaluation. Common strategies include random split, stratified split (preserving
    /// class distribution), k-fold cross-validation split, and time-based split for temporal data.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDataSplitter(IDataSplitter<T> splitter)
    {
        _configuredDataSplitter = splitter;
        return this;
    }

    /// <summary>
    /// Configures a classification metric for evaluating classifier performance.
    /// </summary>
    /// <param name="metric">The classification metric implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Classification metrics measure how well your model categorizes
    /// data. Common metrics include Accuracy, Precision, Recall, F1-Score, AUC-ROC, and
    /// Matthews Correlation Coefficient. Different metrics are better suited for different
    /// scenarios (e.g., Precision for minimizing false positives, Recall for minimizing false negatives).</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureClassificationMetric(IClassificationMetric<T> metric)
    {
        _configuredClassificationMetric = metric;
        return this;
    }

    /// <summary>
    /// Configures a regression metric for evaluating regression model performance.
    /// </summary>
    /// <param name="metric">The regression metric implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Regression metrics measure how close your model's numeric predictions
    /// are to the actual values. Common metrics include Mean Squared Error (MSE), Root Mean Squared
    /// Error (RMSE), Mean Absolute Error (MAE), R-Squared, and Mean Absolute Percentage Error (MAPE).</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureRegressionMetric(IRegressionMetric<T> metric)
    {
        _configuredRegressionMetric = metric;
        return this;
    }

    /// <summary>
    /// Configures a text vectorizer for converting text data into numeric feature vectors.
    /// </summary>
    /// <param name="vectorizer">The text vectorizer implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Text vectorizers convert words and documents into numbers that
    /// machine learning models can process. Common methods include TF-IDF (term frequency-inverse
    /// document frequency), Bag of Words, Word2Vec embeddings, and character n-grams.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureTextVectorizer(ITextVectorizer<T> vectorizer)
    {
        _configuredTextVectorizer = vectorizer;
        return this;
    }

    // ConfigureDocumentStore removed: pass the store via ConfigureRetrievalAugmentedGeneration(documentStore: ...), which consumes it (HybridGraphRetriever).

    // ConfigureBenchmark removed: benchmarking is a post-build action, not a build input. Call
    // AiModelResult.EvaluateBenchmarkAsync(...) on the trained result.

    // ConfigurePDESpecification removed: the PDE a physics-informed model must satisfy is a
    // constructor parameter of that model. Set it on the model's options — the one door.

    /// <summary>
    /// Configures an internal cluster metric for evaluating clustering quality without ground truth labels.
    /// </summary>
    /// <param name="metric">The cluster metric implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Internal cluster metrics evaluate how well data is grouped
    /// without needing correct labels. They measure properties like compactness (how tight clusters are)
    /// and separation (how far apart clusters are). Common metrics include Silhouette Score,
    /// Davies-Bouldin Index, and Calinski-Harabasz Index.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureClusterMetric(IClusterMetric<T> metric)
    {
        _configuredClusterMetric = metric;
        return this;
    }

    /// <summary>
    /// Configures an external cluster metric for evaluating clustering quality against ground truth labels.
    /// </summary>
    /// <param name="metric">The external cluster metric implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> External cluster metrics compare your clustering results against
    /// known correct groupings (ground truth). They measure how well the algorithm recovered the true
    /// structure. Common metrics include Adjusted Rand Index, Normalized Mutual Information,
    /// V-Measure, and Fowlkes-Mallows Index.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureExternalClusterMetric(IExternalClusterMetric<T> metric)
    {
        _configuredExternalClusterMetric = metric;
        return this;
    }

    /// <summary>
    /// Configures a curriculum scheduler for ordering training samples by difficulty.
    /// </summary>
    /// <param name="scheduler">The curriculum scheduler implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Curriculum learning presents training examples in a meaningful order,
    /// typically starting with easy examples and gradually introducing harder ones. This mimics how
    /// humans learn and can lead to faster convergence and better generalization. Schedulers control
    /// the pacing of difficulty progression.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureCurriculumScheduler(ICurriculumScheduler<T> scheduler)
    {
        _configuredCurriculumScheduler = scheduler;

        // Route it to where curriculum learning actually reads a scheduler from. The build passes
        // CurriculumLearningOptions.CustomScheduler to the CurriculumLearner; parking the value in a
        // private field meant this method never reached it. Curriculum learning only runs when
        // ConfigureCurriculumLearning supplied options with a dataset, so if none exist yet the
        // value is held and applied when they arrive (below), making the two calls order-independent.
        if (_curriculumLearningOptions is not null)
        {
            _curriculumLearningOptions.CustomScheduler = scheduler;
        }

        return this;
    }

    /// <summary>
    /// Configures an exploration strategy for reinforcement learning agents.
    /// </summary>
    /// <param name="strategy">The exploration strategy implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Exploration strategies determine how RL agents balance trying
    /// new actions (exploration) versus using known good actions (exploitation). Common strategies
    /// include Epsilon-Greedy (random exploration with probability epsilon), Boltzmann (softmax-based),
    /// UCB (Upper Confidence Bound), and Noisy Networks.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureExplorationStrategy(ReinforcementLearning.Policies.Exploration.IExplorationStrategy<T> strategy)
    {
        _configuredExplorationStrategy = strategy;

        // Route it to where RL training actually reads the exploration override from. If RL options
        // already exist, fill in their ExplorationStrategy (unless the options set one explicitly).
        // If they don't exist yet, BuildRLInternalAsync falls back to _configuredExplorationStrategy,
        // so the two calls are order-independent either way.
        if (_rlOptions is not null && _rlOptions.ExplorationStrategy is null)
        {
            _rlOptions.ExplorationStrategy = strategy;
        }

        return this;
    }

    /// <summary>
    /// Configures curiosity (intrinsic-motivation) exploration for reinforcement learning: a novelty
    /// bonus is added to the environment reward each step so the agent seeks unfamiliar states, which is
    /// what makes sparse-reward tasks learnable.
    /// </summary>
    /// <param name="module">
    /// The intrinsic-reward module. When <c>null</c>, the industry-leading default is used: Random Network
    /// Distillation (prediction error against a fixed random network as the novelty signal).
    /// </param>
    /// <param name="weight">Weight applied to the intrinsic reward before adding it to the extrinsic reward. Defaults to 0.5.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> If your environment only rewards the agent rarely, the agent has no
    /// signal to learn from most of the time. Curiosity gives it a bonus for discovering new situations,
    /// so it keeps exploring purposefully. The bonus fades as situations become familiar.</para>
    /// <para>
    /// Applies to the reinforcement-learning training path (with a configured environment); the mean
    /// intrinsic reward is surfaced on the result's reinforcement-learning metrics.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureCuriosity(
        ReinforcementLearning.IntrinsicMotivation.IIntrinsicRewardModule<T>? module = null, double weight = 0.5)
    {
        _configuredIntrinsicRewardModule = module
            ?? new ReinforcementLearning.IntrinsicMotivation.RandomNetworkDistillation<T>(seed: _rlOptions?.Seed);
        _configuredIntrinsicRewardWeight = weight;

        // Order-independent with ConfigureReinforcementLearning / ConfigureEnvironment: fill the options
        // if they already exist; otherwise BuildRLInternalAsync falls back to the configured field.
        if (_rlOptions is not null && _rlOptions.IntrinsicRewardModule is null)
        {
            _rlOptions.IntrinsicRewardModule = _configuredIntrinsicRewardModule;
            _rlOptions.IntrinsicRewardWeight = weight;
        }

        return this;
    }

    /// <summary>
    /// Configures a self-supervised learning method for learning representations without labeled data.
    /// </summary>
    /// <param name="method">The SSL method implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Self-supervised learning creates its own supervision signal from
    /// unlabeled data, learning useful representations without human annotations. Methods include
    /// contrastive learning (SimCLR, MoCo), masked prediction (BERT, MAE), and bootstrap methods
    /// (BYOL, SimSiam). These representations can then be fine-tuned for downstream tasks.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureSelfSupervisedLearningMethod(ISelfSupervisedLearningMethod<T> method)
    {
        if (method is null) throw new ArgumentNullException(nameof(method));

        // Write through to the SSL config so the method reaches the pretraining hook. Create the
        // config if ConfigureSelfSupervisedLearning has not run yet; that overload reuses this
        // instance, so the two compose in either call order.
        (_selfSupervisedLearningConfig ??= new SelfSupervisedLearning.SelfSupervisedLearningConfig<T>()).Method = method;
        return this;
    }

    /// <summary>
    /// Configures a stopping criterion for active learning loops.
    /// </summary>
    /// <param name="criterion">The stopping criterion implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Stopping criteria determine when an active learning loop should
    /// stop requesting more labeled data. Criteria include reaching a performance threshold,
    /// exhausting the labeling budget, detecting convergence in model performance, or reaching
    /// a maximum number of iterations.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureStoppingCriterion(IStoppingCriterion<T> criterion)
    {
        _configuredStoppingCriterion = criterion;
        return this;
    }

    /// <summary>
    /// Configures a time series decomposition method for separating time series into components.
    /// </summary>
    /// <param name="decomposition">The time series decomposition implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Time series decomposition breaks a time series into its constituent
    /// components: trend (long-term direction), seasonality (repeating patterns), and residual (noise).
    /// Methods include STL (Seasonal and Trend decomposition using Loess), classical decomposition,
    /// EMD (Empirical Mode Decomposition), and wavelet decomposition.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureTimeSeriesDecomposition(ITimeSeriesDecomposition<T> decomposition)
    {
        _configuredTimeSeriesDecomposition = decomposition;
        return this;
    }

    /// <summary>
    /// Configures a knowledge distillation strategy for transferring knowledge between models.
    /// </summary>
    /// <param name="strategy">The distillation strategy implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Knowledge distillation transfers learned knowledge from a large,
    /// complex teacher model to a smaller, faster student model. The student learns to mimic the
    /// teacher's behavior, achieving similar performance with fewer parameters. Strategies include
    /// soft label distillation, feature matching, and attention transfer.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDistillationStrategy(IDistillationStrategy<T> strategy)
    {
        _configuredDistillationStrategy = strategy;
        // The strategy IS the distillation parameter — flow it onto the KD options so it reaches
        // training (and round-trips onto AiModelResult.KnowledgeDistillationOptions). Handles the
        // case where ConfigureKnowledgeDistillation was called first; the reverse ordering is
        // reconciled in ConfigureKnowledgeDistillation.
        if (_knowledgeDistillationOptions is not null)
        {
            _knowledgeDistillationOptions.Strategy = strategy;
        }
        return this;
    }

    /// <summary>
    /// Configures a model compression strategy for reducing model size and inference cost.
    /// </summary>
    /// <param name="strategy">The model compression strategy implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Model compression reduces the size and computational cost of
    /// trained models while preserving accuracy. Techniques include pruning (removing unimportant
    /// weights), quantization (using lower precision numbers), weight sharing, and low-rank
    /// factorization. Compressed models run faster and use less memory.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureModelCompressionStrategy(IModelCompressionStrategy<T> strategy)
    {
        _configuredModelCompressionStrategy = strategy;
        return this;
    }

    /// <summary>
    /// Configures a tool for agent-based systems and function calling.
    /// </summary>
    /// <param name="tool">The tool implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tools are callable functions that AI agents can use to interact
    /// with external systems. Examples include web search, code execution, file operations,
    /// API calls, and database queries. Tools extend an agent's capabilities beyond pure
    /// text generation to real-world actions.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureTool(IAgentTool tool)
    {
        _configuredTool = tool;
        return this;
    }

    // ConfigureNoiseScheduler removed: the noise schedule is a constructor parameter of the diffusion
    // model that uses it. Set it on that model's options — the one door.

    /// <summary>
    /// Configures a reinforcement learning environment for agent training.
    /// </summary>
    /// <param name="environment">The environment implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> An RL environment defines the world an agent interacts with.
    /// It provides observations, accepts actions, and returns rewards. Environments can simulate
    /// games, robotics tasks, trading scenarios, or any sequential decision-making problem.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureEnvironment(IEnvironment<T> environment)
    {
        _configuredEnvironment = environment;

        // Route it to where RL actually reads an environment from. The build gates the RL training
        // path on RLTrainingOptions.Environment and drives that instance (Reset/Step) from there;
        // parking the value in a private field meant this method never reached it and configuring an
        // environment silently fell through to the supervised path. If no RL options exist yet,
        // configuring an environment is itself the request to train in it, so create them with the
        // standard loop defaults; otherwise fill in the environment they are missing. Holding the
        // value in _configuredEnvironment as well is what lets ConfigureReinforcementLearning carry
        // it over when options arrive later, making the two calls order-independent.
        if (_rlOptions is null)
        {
            _rlOptions = RLTrainingOptions<T>.Default(environment);
        }
        else
        {
            _rlOptions.Environment = environment;
        }

        return this;
    }

    /// <summary>
    /// Configures an adversarial attack method for evaluating model robustness.
    /// </summary>
    /// <param name="attack">The adversarial attack implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adversarial attacks generate small, carefully crafted perturbations
    /// to test whether your model can be fooled. Understanding these attacks helps build more robust
    /// models. Common methods include FGSM, PGD, C&amp;W, and DeepFool.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureAdversarialAttack(IAdversarialAttack<T, TInput, TOutput> attack)
    {
        _configuredAdversarialAttack = attack;
        return this;
    }

    /// <summary>
    /// Configures an adversarial defense method for improving model robustness.
    /// </summary>
    /// <param name="defense">The adversarial defense implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adversarial defenses protect models against adversarial attacks.
    /// They make models more robust to small input perturbations. Techniques include adversarial
    /// training, input preprocessing, gradient masking, and certified defenses.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureAdversarialDefense(IAdversarialDefense<T, TInput, TOutput> defense)
    {
        _configuredAdversarialDefense = defense;
        return this;
    }

    /// <summary>
    /// Configures a certified defense for providing formal robustness guarantees.
    /// </summary>
    /// <param name="defense">The certified defense implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Certified defenses provide mathematical guarantees about model
    /// robustness. Unlike empirical defenses, they can prove that no adversarial perturbation within
    /// a given radius can change the prediction. Methods include randomized smoothing, interval bound
    /// propagation, and convex relaxation.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureCertifiedDefense(ICertifiedDefense<T, TInput, TOutput> defense)
    {
        _configuredCertifiedDefense = defense;
        return this;
    }

    /// <summary>
    /// Configures a query strategy for active learning sample selection.
    /// </summary>
    /// <param name="strategy">The query strategy implementation to use.</param>
    /// <param name="unlabeledPool">
    /// The per-sample unlabeled pool to rank. Required for the selection to run — query strategies are
    /// generic over the input type, so (unlike the Tensor-typed ConfigureActiveLearning) the pool cannot
    /// be derived by splitting the batched training data. When null, the strategy is recorded but no
    /// selection is produced.
    /// </param>
    /// <param name="batchSize">How many samples to select for labeling. Defaults to 10.</param>
    /// <param name="diversityWeight">Redundancy penalty for batch selection (0 = pure uncertainty). Defaults to 0.5.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Query strategies determine which unlabeled samples to select for
    /// labeling in active learning. The goal is to choose the most informative samples that will
    /// improve the model most efficiently. Common strategies include uncertainty sampling,
    /// query-by-committee, expected model change, and diversity sampling.</para>
    /// <para>
    /// The strategy's per-sample scores feed the same diversity-aware batch selection as
    /// <c>ConfigureActiveLearning</c> (BADGE / facility-location style), with redundancy measured via the
    /// three-tier representation cascade. The result lands on
    /// <see cref="AiModelResult{T, TInput, TOutput}.ActiveLearningSelection"/>.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureQueryStrategy(
        ActiveLearning.Interfaces.IQueryStrategy<T, TInput, TOutput> strategy,
        IReadOnlyList<TInput>? unlabeledPool = null, int batchSize = 10, double diversityWeight = 0.5)
    {
        _configuredQueryStrategy = strategy;
        _queryStrategyPool = unlabeledPool;
        _queryStrategyBatchSize = batchSize;
        _queryStrategyDiversityWeight = diversityWeight;
        return this;
    }

    /// <summary>
    /// Configures an audio enhancer for improving audio quality.
    /// </summary>
    /// <param name="enhancer">The audio enhancer implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Audio enhancers improve the quality of audio signals through
    /// techniques like noise reduction, speech enhancement, bandwidth extension, and
    /// dereverberation. They are used in telecommunications, hearing aids, and audio production.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureAudioEnhancer(IAudioEnhancer<T> enhancer)
    {
        _configuredAudioEnhancer = enhancer;
        // Apply enhancement as a composable, fitted preprocessing step over audio-tensor inputs; its Fit
        // estimates the noise profile from the training audio so train and inference are cleaned consistently.
        _dataPipeline.AddPreprocessingStep(
            new Preprocessing.Audio.AudioEnhancementTransformer<T, TInput>(enhancer), "audio_enhancer");
        _preprocessingPipeline = _dataPipeline.PreprocessingPipeline;
        return this;
    }

    /// <summary>
    /// Configures a similarity metric for vector similarity search operations.
    /// </summary>
    /// <param name="metric">The similarity metric implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Similarity metrics measure how alike two vectors are, which is
    /// essential for nearest-neighbor search and retrieval systems. Common metrics include cosine
    /// similarity, dot product, Euclidean distance, and Jaccard similarity.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureSimilarityMetric(RetrievalAugmentedGeneration.VectorSearch.ISimilarityMetric<T> metric)
    {
        _configuredSimilarityMetric = metric;
        return this;
    }
}
