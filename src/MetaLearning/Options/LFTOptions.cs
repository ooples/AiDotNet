using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Learned Feature-Wise Transformation (LFT), Tseng, Lee, Huang and Yang,
/// "Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation" (ICLR 2020,
/// arXiv:2001.08735).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Metric-based few-shot learners "often fail to generalize to unseen domains due to large
/// discrepancy of the feature distribution across domains". LFT's answer is to stop trying to make
/// the encoder domain-invariant and instead show the metric function many DIFFERENT feature
/// distributions during training, by perturbing intermediate features with affine transforms whose
/// scale and bias are sampled from learned hyper-parameters.
/// </para>
/// <para>
/// The transformations exist only during training — they are removed before the model is used — so
/// they cost nothing at inference. What survives is a metric function that has been trained against
/// distribution shift rather than against one fixed distribution.
/// </para>
/// <para>
/// <b>For Beginners:</b> A model trained only on photographs often does badly on sketches, because
/// the numbers coming out of its feature extractor look different. Rather than trying to make those
/// numbers identical for photos and sketches, this deliberately jiggles them during training — a
/// little differently every time — so the comparison step learns not to depend on their exact
/// scale. How much to jiggle is itself learned, by checking which amount helps most on a domain
/// held out from that training step.
/// </para>
/// </remarks>
public class LFTOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    /// <summary>
    /// Gets or sets the feature encoder E and metric model M. The feature-wise transformation is
    /// applied to the features this produces.
    /// </summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    /// <summary>
    /// Gets or sets the width of the feature vector the transformation modulates, i.e. the channel
    /// count C. The hyper-parameter vectors <c>theta_gamma</c> and <c>theta_beta</c> are this long.
    /// </summary>
    /// <value>Defaults to 64.</value>
    public int FeatureDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the metric head's output width — the number of classes the few-shot task
    /// discriminates between.
    /// </summary>
    /// <value>Defaults to 5, the standard 5-way few-shot setting.</value>
    public int OutputDimension { get; set; } = 5;

    /// <summary>
    /// Gets or sets the initial value of every entry of <c>theta_gamma</c>, the hyper-parameter
    /// governing the SCALE term's spread.
    /// </summary>
    /// <remarks>
    /// The paper's pre-determined setting, used when the hyper-parameters are hand-tuned rather than
    /// learned, is 0.3. It is the starting point here even when
    /// <see cref="LearnTransformationHyperparameters"/> is on.
    /// </remarks>
    /// <value>Defaults to 0.3 (the paper's value).</value>
    public double InitialScaleHyperparameter { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the initial value of every entry of <c>theta_beta</c>, the hyper-parameter
    /// governing the BIAS term's spread.
    /// </summary>
    /// <value>Defaults to 0.5 (the paper's value).</value>
    public double InitialBiasHyperparameter { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether the transformation hyper-parameters are LEARNED by the paper's
    /// learning-to-learn split, rather than left at their initial values.
    /// </summary>
    /// <remarks>
    /// True reproduces the paper's main method; false reproduces its hand-tuned ablation, which the
    /// paper also reports and which is a legitimate cheaper configuration.
    /// </remarks>
    /// <value>True by default.</value>
    public bool LearnTransformationHyperparameters { get; set; } = true;

    /// <summary>
    /// Gets or sets the fraction of each task batch assigned to the PSEUDO-SEEN domain, the
    /// remainder becoming the pseudo-unseen domain.
    /// </summary>
    /// <remarks>
    /// The paper samples "non-overlapping pseudo-seen and pseudo-unseen domains" each iteration.
    /// The split must leave at least one task on each side or the second stage has nothing to
    /// measure against.
    /// </remarks>
    /// <value>Defaults to 0.5, an even split.</value>
    public double PseudoSeenFraction { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the step size used to update the transformation hyper-parameters in the second
    /// stage.
    /// </summary>
    /// <value>Defaults to 1e-3.</value>
    public double HyperparameterLearningRate { get; set; } = 1e-3;

    /// <inheritdoc cref="IMetaLearnerOptions{T}.InnerLearningRate"/>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <inheritdoc cref="IMetaLearnerOptions{T}.OuterLearningRate"/>
    public double OuterLearningRate { get; set; } = 0.001;

    public int AdaptationSteps { get; set; } = 5;
    public int MetaBatchSize { get; set; } = 4;
    public int NumMetaIterations { get; set; } = 1000;
    public double? GradientClipThreshold { get; set; } = 10.0;
    public int? RandomSeed { get => Seed; set => Seed = value; }
    public int EvaluationTasks { get; set; } = 100;
    public int EvaluationFrequency { get; set; } = 100;
    public bool EnableCheckpointing { get; set; } = false;
    public int CheckpointFrequency { get; set; } = 500;
    public bool UseFirstOrder { get; set; } = true;
    public ILossFunction<T>? LossFunction { get; set; }
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    public LFTOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() =>
        MetaModel != null &&
        FeatureDimension > 0 &&
        OutputDimension > 0 &&
        InitialScaleHyperparameter > 0 &&
        InitialBiasHyperparameter > 0 &&
        PseudoSeenFraction > 0.0 && PseudoSeenFraction < 1.0 &&
        HyperparameterLearningRate > 0 &&
        InnerLearningRate > 0 &&
        OuterLearningRate > 0 &&
        AdaptationSteps > 0 &&
        MetaBatchSize > 0;

    public IMetaLearnerOptions<T> Clone() => new LFTOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        FeatureDimension = FeatureDimension,
        OutputDimension = OutputDimension,
        InitialScaleHyperparameter = InitialScaleHyperparameter,
        InitialBiasHyperparameter = InitialBiasHyperparameter,
        LearnTransformationHyperparameters = LearnTransformationHyperparameters,
        PseudoSeenFraction = PseudoSeenFraction,
        HyperparameterLearningRate = HyperparameterLearningRate,
    };
}
