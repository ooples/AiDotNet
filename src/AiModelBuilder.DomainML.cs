using AiDotNet.Clustering.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Document.Interfaces;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.PointCloud.Interfaces;
using AiDotNet.Scoring;
using AiDotNet.Video.Interfaces;

namespace AiDotNet;

/// <summary>
/// Domain-specific machine learning configuration extensions for AiModelBuilder.
/// Provides Configure methods for audio, speech, diffusion, active/continual/online learning,
/// causal inference, drift detection, survival analysis, video, point cloud, document processing,
/// finance, radial basis functions, distance metrics, embeddings, scoring, explainability, and RL agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private IAudioEffect<T>? _configuredAudioEffect;
    private IActiveLearningStrategy<T>? _configuredActiveLearningStrategy;
    private Tensor<T>? _activeLearningPool;
    private int _activeLearningBatchSize = 10;
    private double _activeLearningDiversityWeight = 0.5;
    private bool _continualLearningEnabled;
    private ContinualLearning.Interfaces.IContinualLearningStrategy<T, TInput, TOutput>? _continualLearningStrategy;
    private ContinualLearning.Interfaces.IContinualLearnerConfig<T>? _continualLearningConfig;
    private ContinualLearning.Results.ContinualLearningResult<T>? _continualLearningTaskResult;
    private ContinualLearning.Results.ContinualEvaluationResult<T>? _continualLearningRetention;
    private AiDotNet.DriftDetection.IDriftDetector<T>? _configuredDriftDetector;
    private IDistanceMetric<T>? _configuredDistanceMetric;
    private IEmbeddingModel<T>? _configuredEmbeddingModel;
    private IModelExplainer<T>? _configuredModelExplainer;

    /// <summary>
    /// Configures an audio effect for audio signal processing pipelines.
    /// </summary>
    /// <param name="audioEffect">The audio effect implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Audio effects transform sound signals, such as applying
    /// reverb, equalization, noise reduction, or pitch shifting. These are useful for audio
    /// preprocessing and augmentation in speech and music ML pipelines.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureAudioEffect(IAudioEffect<T> audioEffect)
    {
        _configuredAudioEffect = audioEffect;
        return this;
    }

    /// <summary>
    /// Configures an active learning strategy for intelligently selecting training samples.
    /// </summary>
    /// <param name="strategy">The active learning strategy implementation to use.</param>
    /// <param name="unlabeledPool">
    /// The pool of unlabeled samples to rank. When <c>null</c>, the held-out test partition is used as the
    /// candidate pool, so a useful "which samples is the model least sure about" ranking is produced with
    /// no extra configuration.
    /// </param>
    /// <param name="batchSize">How many samples to select for labeling. Defaults to 10.</param>
    /// <param name="diversityWeight">
    /// Redundancy penalty for batch selection (0 = pure uncertainty, higher = more diversity pressure).
    /// Defaults to 0.5.
    /// </param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Active learning helps your model learn more efficiently by
    /// choosing the most informative samples to label next. Instead of labeling all your data,
    /// the model asks for labels on the examples it's most uncertain about, saving labeling effort.</para>
    /// <para>
    /// The configured strategy supplies per-sample informativeness; on top of that, the batch is chosen
    /// with a diversity/redundancy penalty (BADGE / facility-location style) so it covers the pool rather
    /// than collecting near-duplicate uncertain samples. Diversity is measured in the strongest available
    /// space — authentic BADGE gradient embeddings, then a model-exposed representation, then input
    /// features. The ranking and selected batch are surfaced on
    /// <see cref="AiModelResult{T, TInput, TOutput}.ActiveLearningSelection"/>.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureActiveLearning(
        IActiveLearningStrategy<T> strategy, Tensor<T>? unlabeledPool = null, int batchSize = 10, double diversityWeight = 0.5)
    {
        _configuredActiveLearningStrategy = strategy;
        _activeLearningPool = unlabeledPool;
        _activeLearningBatchSize = batchSize;
        _activeLearningDiversityWeight = diversityWeight;
        return this;
    }

    /// <summary>
    /// Configures continual learning so the model learns this task without forgetting earlier ones.
    /// </summary>
    /// <param name="strategy">
    /// The continual-learning strategy (EWC, LwF, GEM, MAS, SI). When <c>null</c>, the industry-standard
    /// default is used: Elastic Weight Consolidation combined with experience replay — a
    /// regularization+rehearsal hybrid that outperforms either alone.
    /// </param>
    /// <param name="config">
    /// Optional continual-learning configuration (learning rate, epochs, replay memory size). Defaults to
    /// sensible values when <c>null</c>.
    /// </param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Continual learning (lifelong learning) enables your model to
    /// learn new tasks over time without forgetting what it learned before. Traditional neural
    /// networks suffer from "catastrophic forgetting" - continual learning techniques like
    /// EWC, LwF, and GEM prevent this.</para>
    /// <para>
    /// There is one model: the strategy is applied to the model configured via <c>ConfigureModel</c>.
    /// Training this Build routes through the continual learner, which preserves prior tasks; the
    /// per-task retention report (retained accuracy, average forgetting, forward/backward transfer) is
    /// surfaced on <see cref="AiModelResult{T, TInput, TOutput}.ContinualLearningReport"/>.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureContinualLearning(
        ContinualLearning.Interfaces.IContinualLearningStrategy<T, TInput, TOutput>? strategy = null,
        ContinualLearning.Interfaces.IContinualLearnerConfig<T>? config = null)
    {
        _continualLearningEnabled = true;
        _continualLearningStrategy = strategy;
        _continualLearningConfig = config;
        return this;
    }

    /// <summary>
    /// Configures a drift detector for monitoring changes in data distribution over time.
    /// </summary>
    /// <param name="driftDetector">The drift detector implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Drift detection monitors whether the data your model sees in
    /// production is changing compared to what it was trained on. If drift is detected, it may
    /// signal that your model needs retraining. Common detectors include DDM, ADWIN, and
    /// Page-Hinkley methods.</para>
    /// <para>
    /// The configured detector drives a two-lens <see cref="DriftDetection.DriftMonitor{T}"/> calibrated
    /// on the training residuals and checked against the held-out test stream. The attributed
    /// <see cref="AiModelResult{T, TInput, TOutput}.DriftReport"/> (concept vs covariate drift) and the
    /// live <see cref="AiModelResult{T, TInput, TOutput}.DriftMonitor"/> are surfaced on the built result;
    /// production code streams live <c>(predicted, actual)</c> pairs through the same monitor.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDriftDetection(AiDotNet.DriftDetection.IDriftDetector<T> driftDetector)
    {
        _configuredDriftDetector = driftDetector;
        return this;
    }

    // ConfigureVideoModel removed: IVideoModel<T> : IFullModel; use ConfigureModel(...).
    // ConfigurePointCloudModel removed: IPointCloudModel<T> -> INeuralNetwork<T> -> IFullModel; use ConfigureModel(...).
    // ConfigureDocumentModel removed: IDocumentModel<T> : IFullModel; use ConfigureModel(...).
    // ConfigureFinancialModel removed: IFinancialModel<T> : IFullModel; use ConfigureModel(...).

    // ConfigureRadialBasisFunction removed: an RBF is a constructor parameter of the RBF network /
    // interpolator that uses it. Set it on that model's options — the one door.

    /// <summary>
    /// Configures a distance metric for measuring similarity between data points.
    /// </summary>
    /// <param name="distanceMetric">The distance metric implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Distance metrics measure how different two data points are.
    /// They are fundamental to clustering, nearest-neighbor search, and many other ML algorithms.
    /// Common metrics include Euclidean, Manhattan, Cosine, and Mahalanobis distance.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDistanceMetric(IDistanceMetric<T> distanceMetric)
    {
        _configuredDistanceMetric = distanceMetric;
        return this;
    }

    /// <summary>
    /// Configures an embedding model for learning dense vector representations.
    /// </summary>
    /// <param name="embeddingModel">The embedding model implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Embedding models convert high-dimensional or categorical data
    /// into compact, dense vectors that capture semantic meaning. Similar items end up close
    /// together in the embedding space. Used for text (Word2Vec, BERT), images, graphs, and more.</para>
    /// <para>
    /// An embedding model is a preprocessing/transform component (text → dense vector), not a trainable
    /// predictive model, so it is not routed through training. The configured embedder is surfaced on the
    /// built result as <see cref="AiModelResult{T, TInput, TOutput}.EmbeddingModel"/> so callers can embed
    /// new inputs at inference time consistently with how features were prepared.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureEmbeddingModel(IEmbeddingModel<T> embeddingModel)
    {
        _configuredEmbeddingModel = embeddingModel;
        return this;
    }

    /// <summary>
    /// Configures a scoring rule for evaluating probabilistic predictions.
    /// </summary>
    /// <param name="scorer">The scoring rule implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    // ConfigureScoringRule was removed: the scoring rule is configured on the model's options
    // (NGBoostRegressionOptions<T>.ScoringRule, a nullable IScoringRule<T>), which is the one door.
    // A builder-level setter could not reach an already-constructed model whose rule is read in its
    // constructor.

    /// <summary>
    /// Configures a model explainer for understanding model predictions.
    /// </summary>
    /// <param name="explainer">The model explainer implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Model explainers help you understand why a model makes specific
    /// predictions. This is crucial for trust, debugging, and regulatory compliance. Methods
    /// include SHAP (feature importance), LIME (local explanations), Integrated Gradients,
    /// and Attention visualization.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureModelExplainer(IModelExplainer<T> explainer)
    {
        _configuredModelExplainer = explainer;
        return this;
    }

}
