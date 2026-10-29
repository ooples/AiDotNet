using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Meta-LoRA (Low-Rank Adaptation for Meta-Learning).
/// </summary>
/// <remarks>
/// <para>
/// Meta-LoRA applies the Low-Rank Adaptation principle to meta-learning: instead of adapting
/// all model parameters during the inner loop (as in MAML), it meta-learns a set of low-rank
/// basis vectors and only adapts a small number of coefficients per task. This drastically
/// reduces the inner-loop parameter count from d to r (where r &lt;&lt; d).
/// </para>
/// <para><b>Key Parameters:</b>
/// <list type="bullet">
/// <item><see cref="Rank"/> — number of low-rank basis vectors (controls capacity vs efficiency)</item>
/// <item><see cref="ScalingAlpha"/> — scales the LoRA update magnitude (analogous to alpha/r in standard LoRA)</item>
/// </list>
/// </para>
/// </remarks>
public class MetaLoRAOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// <inheritdoc/>
    public bool UseFirstOrder { get; set; } = true;
    /// <summary>Loss function used in the inner and outer loops; null uses the meta-model's default loss.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>Optimizer for the outer (meta) update; null uses the algorithm's default.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>Optimizer for the inner, per-task adaptation loop; null uses the algorithm's default.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Episodic data loader that supplies the meta-training tasks; null when tasks are passed to training directly.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Number of low-rank basis vectors used for adaptation. Higher rank gives more
    /// expressive task-specific updates but increases inner-loop cost. Default: 4.
    /// </summary>
    public int Rank { get; set; } = 4;

    /// <summary>
    /// Scaling factor applied to the LoRA update: adapted = base + (alpha / rank) * sum(c_i * v_i).
    /// Analogous to alpha in standard LoRA. Default: 1.0.
    /// </summary>
    public double ScalingAlpha { get; set; } = 1.0;

    /// <summary>
    /// Standard deviation for initializing the low-rank basis vectors. Default: 0.01.
    /// </summary>
    public double BasisInitStdDev { get; set; } = 0.01;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public MetaLoRAOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && Rank > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new MetaLoRAOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        Rank = Rank, ScalingAlpha = ScalingAlpha, BasisInitStdDev = BasisInitStdDev
    };
}
