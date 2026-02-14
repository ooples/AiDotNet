using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Clustering.Evaluation;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.Evaluation.Metrics;
using AiDotNet.Interfaces;
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
    private IDataTransformer<T, TInput, TInput>? _configuredDataTransformer;
    private IDataSplitter<T>? _configuredDataSplitter;
    private IClassificationMetric<T>? _configuredClassificationMetric;
    private IRegressionMetric<T>? _configuredRegressionMetric;
    private ITextVectorizer<T>? _configuredTextVectorizer;
    private IDocumentStore<T>? _configuredDocumentStore;
    private IBenchmark<T>? _configuredBenchmark;
    private PhysicsInformed.Interfaces.IPDESpecification<T>? _configuredPDESpecification;
    private IClusterMetric<T>? _configuredClusterMetric;
    private IExternalClusterMetric<T>? _configuredExternalClusterMetric;
    private ICurriculumScheduler<T>? _configuredCurriculumScheduler;
    private ReinforcementLearning.Policies.Exploration.IExplorationStrategy<T>? _configuredExplorationStrategy;
    private ISSLMethod<T>? _configuredSSLMethod;
    private IStoppingCriterion<T>? _configuredStoppingCriterion;
    private ITimeSeriesDecomposition<T>? _configuredTimeSeriesDecomposition;
    private IDistillationStrategy<T>? _configuredDistillationStrategy;
    private IModelCompressionStrategy<T>? _configuredModelCompressionStrategy;
    private ITool? _configuredTool;

    /// <summary>
    /// Configures a data transformer for preprocessing or postprocessing data transformations.
    /// </summary>
    /// <param name="transformer">The data transformer implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Data transformers convert your data from one form to another,
    /// such as scaling numbers to a standard range, encoding text as numbers, or reducing the
    /// number of features. Available transformers include StandardScaler, MinMaxScaler,
    /// OneHotEncoder, PCA, PolynomialFeatures, SimpleImputer, and hundreds more.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDataTransformer(IDataTransformer<T, TInput, TInput> transformer)
    {
        _configuredDataTransformer = transformer;
        return this;
    }

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

    /// <summary>
    /// Configures a document store for persisting and retrieving documents with vector similarity search.
    /// </summary>
    /// <param name="store">The document store implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Document stores save and retrieve documents along with their vector
    /// embeddings, enabling similarity search. They are a core component of Retrieval-Augmented
    /// Generation (RAG) systems. Available stores include in-memory, file-based, and database-backed
    /// implementations.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDocumentStore(IDocumentStore<T> store)
    {
        _configuredDocumentStore = store;
        return this;
    }

    /// <summary>
    /// Configures a benchmark for evaluating and comparing model performance systematically.
    /// </summary>
    /// <param name="benchmark">The benchmark implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Benchmarks provide standardized tests for comparing models.
    /// They measure aspects like accuracy, speed, memory usage, and robustness across different
    /// datasets and scenarios, helping you choose the best model for your needs.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureBenchmark(IBenchmark<T> benchmark)
    {
        _configuredBenchmark = benchmark;
        return this;
    }

    /// <summary>
    /// Configures a PDE specification for physics-informed neural network training.
    /// </summary>
    /// <param name="specification">The PDE specification implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> PDE (Partial Differential Equation) specifications define the
    /// physical laws that physics-informed neural networks must respect. Examples include the
    /// heat equation, wave equation, Navier-Stokes equations, and Maxwell's equations. The neural
    /// network learns to satisfy these equations while fitting observed data.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigurePDESpecification(PhysicsInformed.Interfaces.IPDESpecification<T> specification)
    {
        _configuredPDESpecification = specification;
        return this;
    }

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
    public IAiModelBuilder<T, TInput, TOutput> ConfigureSSLMethod(ISSLMethod<T> method)
    {
        _configuredSSLMethod = method;
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
    public IAiModelBuilder<T, TInput, TOutput> ConfigureTool(ITool tool)
    {
        _configuredTool = tool;
        return this;
    }
}
