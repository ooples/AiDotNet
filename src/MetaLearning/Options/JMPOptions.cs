using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for JMP (Joint Multi-Phase meta-learning).
/// </summary>
/// <remarks>
/// JMP uses a multi-phase inner loop with separate learning rates and regularization.
/// Phase 1 (coarse) uses a higher learning rate for fast, rough adaptation.
/// Phase 2 (fine) uses a lower learning rate with stronger regularization toward the
/// Phase 1 result for careful refinement.
/// </remarks>
public class JMPOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    /// <summary>The model whose parameters the meta-learner trains to adapt quickly to new tasks.</summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    /// <inheritdoc/>
    public double InnerLearningRate { get; set; } = 0.01;
    /// <inheritdoc/>
    public double OuterLearningRate { get; set; } = 0.001;
    /// <inheritdoc/>
    public int AdaptationSteps { get; set; } = 5;
    /// <inheritdoc/>
    public int MetaBatchSize { get; set; } = 4;
    /// <inheritdoc/>
    public int NumMetaIterations { get; set; } = 1000;
    /// <inheritdoc/>
    public double? GradientClipThreshold { get; set; } = 10.0;
    /// <inheritdoc/>
    public int? RandomSeed { get => Seed; set => Seed = value; }
    /// <inheritdoc/>
    public int EvaluationTasks { get; set; } = 100;
    /// <inheritdoc/>
    public int EvaluationFrequency { get; set; } = 100;
    /// <inheritdoc/>
    public bool EnableCheckpointing { get; set; } = false;
    /// <inheritdoc/>
    public int CheckpointFrequency { get; set; } = 500;
    /// <summary>Whether to use the first-order approximation of the meta-gradient. Default: true.</summary>
    public bool UseFirstOrder { get; set; } = true;
    /// <summary>Loss function for meta-training; null uses the default multi-class classification loss.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>
    /// Optimizer for the meta-update. When set, it takes the outer step at its own learning rate (so
    /// <see cref="OuterLearningRate"/> no longer applies) and keeps its state across meta-iterations; null uses
    /// plain SGD at <see cref="OuterLearningRate"/>. It must not be the same instance as <see cref="InnerOptimizer"/>.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>
    /// Must stay null: this algorithm uses a two-phase inner learning-rate schedule, which assumes plain SGD inner steps, so a supplied
    /// optimizer is rejected when the learner is built.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Episodic data loader that supplies the meta-training tasks; null when tasks are passed to training directly.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>Fraction of adaptation steps in Phase 1 (coarse). Default: 0.4.</summary>
    public double Phase1Fraction { get; set; } = 0.4;

    /// <summary>Learning rate multiplier for Phase 1 (coarse). Default: 2.0.</summary>
    public double Phase1LRMultiplier { get; set; } = 2.0;

    /// <summary>Learning rate multiplier for Phase 2 (fine). Default: 0.5.</summary>
    public double Phase2LRMultiplier { get; set; } = 0.5;

    /// <summary>L2 regularization during Phase 2 toward Phase 1 result. Default: 0.1.</summary>
    public double PhaseRegWeight { get; set; } = 0.1;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public JMPOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() =>
        MetaModel != null &&
        InnerLearningRate > 0 &&
        OuterLearningRate > 0 &&
        AdaptationSteps >= 2 &&
        MetaBatchSize > 0 &&
        Phase1Fraction > 0 && Phase1Fraction < 1 &&
        Phase1LRMultiplier > 0 &&
        Phase2LRMultiplier > 0 &&
        PhaseRegWeight >= 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new JMPOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        Phase1Fraction = Phase1Fraction, Phase1LRMultiplier = Phase1LRMultiplier,
        Phase2LRMultiplier = Phase2LRMultiplier, PhaseRegWeight = PhaseRegWeight
    };
}
