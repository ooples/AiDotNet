using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for sparse-MAML, von Oswald et al., "Learning where to learn: Gradient
/// sparsity in meta and continual learning" (NeurIPS 2021, arXiv:2110.14402).
/// </summary>
/// <remarks>
/// <para>
/// sparse-MAML meta-learns WHICH weights the inner loop may change, rather than adapting all of
/// them, so that "patterned sparsity emerges" and tasks interfere less.
/// </para>
/// <para>
/// <b>For Beginners:</b> Learning from a handful of examples by changing every weight is a good way
/// to memorize those examples and learn nothing general. These settings control a switch per weight
/// saying whether it may change at all, and how fast those switches are tuned. They all start ON, so
/// whichever ones end up off were turned off because that generalized better.
/// </para>
/// </remarks>
public class SparseMAMLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    public double InnerLearningRate { get; set; } = 0.01;
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

    /// <summary>
    /// Gets or sets the initial value of every gate logit, which sets the starting sparsity.
    /// </summary>
    /// <remarks>
    /// A positive logit starts every weight adaptable, so sparsity has to be DISCOVERED by the
    /// meta-objective rather than imposed. The paper's finding is that "patterned sparsity emerges
    /// from this process" — starting from an already-sparse mask would beg that question.
    /// </remarks>
    /// <value>Defaults to 1.0, i.e. all weights initially adaptable.</value>
    public double InitialGateLogit { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the step size for meta-updating the gate logits.
    /// </summary>
    /// <value>Defaults to 1e-2.</value>
    public double GateLearningRate { get; set; } = 1e-2;

    /// <summary>
    /// Gets or sets whether per-parameter learning rates are meta-learned alongside the mask.
    /// </summary>
    /// <remarks>
    /// The paper's "more expressive model where learning rates are meta-learned", in which it finds
    /// that "sparse learning also emerges" even when the model is free not to be sparse. With this
    /// off, the gate is a pure on/off mask over which weights the inner loop touches.
    /// </remarks>
    /// <value>False by default, the paper's sparse-MAML variant.</value>
    public bool MetaLearnPerParameterRates { get; set; } = false;

    /// <summary>
    /// Gets or sets the gate value below which a weight counts as masked OUT when sparsity is
    /// reported.
    /// </summary>
    /// <remarks>
    /// Reporting only; it does not gate the update, which uses the continuous gate so the mask stays
    /// differentiable with respect to its logits.
    /// </remarks>
    /// <value>Defaults to 0.5.</value>
    public double SparsityThreshold { get; set; } = 0.5;

    public SparseMAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() =>
        MetaModel != null &&
        InnerLearningRate > 0 &&
        OuterLearningRate > 0 &&
        AdaptationSteps > 0 &&
        MetaBatchSize > 0 &&
        GateLearningRate > 0 &&
        SparsityThreshold > 0.0 && SparsityThreshold < 1.0;

    public IMetaLearnerOptions<T> Clone() => new SparseMAMLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps,
        InitialGateLogit = InitialGateLogit,
        GateLearningRate = GateLearningRate,
        MetaLearnPerParameterRates = MetaLearnPerParameterRates,
        SparsityThreshold = SparsityThreshold, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
    };
}
