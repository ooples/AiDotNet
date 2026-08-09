using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Memory-based Parameter Adaptation (MbPA), Sprechmann et al., ICLR 2018
/// (arXiv:1802.10542).
/// </summary>
/// <remarks>
/// <para>
/// MbPA stores <c>(embedding, target)</c> pairs in an episodic memory. At prediction time it looks
/// up the K nearest neighbours of the query's embedding and takes a few gradient steps that fit the
/// OUTPUT network to those neighbours, regularized toward the trained parameters. The adapted
/// parameters produce that one output and are then discarded.
/// </para>
/// <para>
/// The defaults sit inside the ranges the paper reports: K and the step count both in [1, 20], and a
/// local learning rate in [0, 1].
/// </para>
/// <para>
/// <b>For Beginners:</b> A normal network has to see a lot of examples, many times, before it
/// changes its mind about anything. MbPA gives it a notebook. When a new input arrives, it flips to
/// the most similar things it has written down and briefly tunes its last layer to fit those,
/// answers, and then tears the page out. Nothing is permanently relearned, so old knowledge is not
/// overwritten — but the answer reflects the relevant memories.
/// </para>
/// </remarks>
public class MbPAOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    /// <summary>
    /// Gets or sets the embedding network f_gamma. Its output is the memory key for an input, and it
    /// is held FIXED during local adaptation — only the output network is adapted.
    /// </summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    /// <summary>
    /// Gets or sets the local adaptation learning rate alpha_M.
    /// </summary>
    /// <remarks>
    /// The paper's central claim is that this can be MUCH higher than a training learning rate,
    /// because the update is local and transient: "Much higher learning rates can be used for this
    /// local adaptation". It reports a sweep over [0, 1] and an optimum of 0.15 for language
    /// modelling.
    /// </remarks>
    /// <value>Defaults to 0.15.</value>
    public double LocalLearningRate { get; set; } = 0.15;

    /// <summary>
    /// Gets or sets beta, the strength of the pull back toward the trained parameters.
    /// </summary>
    /// <remarks>
    /// This is the MAP prior <c>log p(theta_x | theta) ~ -||theta_x - theta||^2 / (2 alpha_M)</c>.
    /// Without it the local steps would be free to walk arbitrarily far from the trained solution on
    /// the evidence of a handful of neighbours.
    /// </remarks>
    /// <value>Defaults to 0.1.</value>
    public double RegularizationBeta { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets T, the number of local gradient steps taken per prediction.
    /// </summary>
    /// <value>Defaults to 1. The paper sweeps [1, 20] and finds T = 1 optimal for language tasks.</value>
    public int LocalAdaptationSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets K, the number of neighbours retrieved from the episodic memory.
    /// </summary>
    /// <value>Defaults to 8. The paper sweeps [1, 20] and reports performance saturating near 50.</value>
    public int NumNeighbors { get; set; } = 8;

    /// <summary>
    /// Gets or sets the maximum number of <c>(embedding, target)</c> pairs held in the episodic
    /// memory. The oldest entry is evicted when full.
    /// </summary>
    /// <value>Defaults to 1024. The paper tests 100 to 5000.</value>
    public int MemorySize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets epsilon in the retrieval kernel <c>kern(h, q) = 1 / (eps + ||h - q||^2)</c>,
    /// which keeps the weight finite when a stored key coincides with the query.
    /// </summary>
    /// <value>Defaults to 1e-6.</value>
    public double KernelEpsilon { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the width of the embedding produced by <see cref="MetaModel"/>, which is both
    /// the memory key width and the output network's input width.
    /// </summary>
    /// <value>Defaults to 64.</value>
    public int FeatureDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the width of the output network's output — the number of classes for a
    /// categorical head, or the target width for a Gaussian one.
    /// </summary>
    /// <value>Defaults to 5.</value>
    public int OutputDimension { get; set; } = 5;

    /// <summary>
    /// Gets or sets the output distribution the local adaptation fits, i.e. what
    /// <c>log p(v | h, theta_x)</c> means.
    /// </summary>
    /// <value>
    /// Defaults to <see cref="MbPAOutputDistribution.Categorical"/>, which is what both of the
    /// paper's task families (image classification and language modelling) use.
    /// </value>
    public MbPAOutputDistribution OutputDistribution { get; set; } = MbPAOutputDistribution.Categorical;

    /// <summary>
    /// Gets or sets whether observed examples are written into the episodic memory during training.
    /// </summary>
    /// <value>True by default; the memory is what the method is named for.</value>
    public bool WriteMemoryDuringTraining { get; set; } = true;

    /// <inheritdoc cref="IMetaLearnerOptions{T}.InnerLearningRate"/>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <inheritdoc cref="IMetaLearnerOptions{T}.OuterLearningRate"/>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of adaptation steps used by the generic meta-learning contract.
    /// </summary>
    /// <remarks>
    /// MbPA's own step count is <see cref="LocalAdaptationSteps"/>. This exists to satisfy
    /// <see cref="IMetaLearnerOptions{T}"/> and is not used by the local adaptation.
    /// </remarks>
    public int AdaptationSteps { get; set; } = 1;

    /// <inheritdoc cref="IMetaLearnerOptions{T}.MetaBatchSize"/>
    public int MetaBatchSize { get; set; } = 4;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.NumMetaIterations"/>
    public int NumMetaIterations { get; set; } = 1000;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.GradientClipThreshold"/>
    public double? GradientClipThreshold { get; set; } = 10.0;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.RandomSeed"/>
    public int? RandomSeed { get => Seed; set => Seed = value; }
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EvaluationTasks"/>
    public int EvaluationTasks { get; set; } = 100;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EvaluationFrequency"/>
    public int EvaluationFrequency { get; set; } = 100;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EnableCheckpointing"/>
    public bool EnableCheckpointing { get; set; } = false;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.CheckpointFrequency"/>
    public int CheckpointFrequency { get; set; } = 500;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.UseFirstOrder"/>
    public bool UseFirstOrder { get; set; } = true;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.LossFunction"/>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <inheritdoc cref="IMetaLearnerOptions{T}.MetaOptimizer"/>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <inheritdoc cref="IMetaLearnerOptions{T}.InnerOptimizer"/>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <inheritdoc cref="IMetaLearnerOptions{T}.DataLoader"/>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    public MbPAOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() =>
        MetaModel != null &&
        LocalLearningRate > 0 &&
        RegularizationBeta >= 0 &&
        LocalAdaptationSteps > 0 &&
        NumNeighbors > 0 &&
        MemorySize > 0 &&
        KernelEpsilon > 0 &&
        FeatureDimension > 0 &&
        OutputDimension > 0 &&
        InnerLearningRate > 0 &&
        OuterLearningRate > 0 &&
        MetaBatchSize > 0;

    public IMetaLearnerOptions<T> Clone() => new MbPAOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        LocalLearningRate = LocalLearningRate, RegularizationBeta = RegularizationBeta,
        LocalAdaptationSteps = LocalAdaptationSteps, NumNeighbors = NumNeighbors,
        MemorySize = MemorySize, KernelEpsilon = KernelEpsilon,
        FeatureDimension = FeatureDimension, OutputDimension = OutputDimension,
        OutputDistribution = OutputDistribution,
        WriteMemoryDuringTraining = WriteMemoryDuringTraining
    };
}
