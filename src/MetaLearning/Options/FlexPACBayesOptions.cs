using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Flex-PAC-Bayes: Flexible PAC-Bayesian Meta-Learning
/// with data-dependent prior construction.
/// </summary>
/// <remarks>
/// <para>
/// Flex-PAC-Bayes extends PAC-Bayesian meta-learning by constructing the prior from a
/// fraction of the support data ("prior data") and computing the PAC-Bayesian bound
/// on the remaining data ("bound data"). This data-dependent prior construction yields
/// tighter generalization bounds. The "flex" parameter (λ) interpolates between
/// standard PAC-Bayes (λ=1) and pure empirical risk minimization (λ→0).
/// </para>
/// <para><b>Key bound:</b>
/// <code>
/// L_flex = (1-f)*L_bound + f*L_prior + (λ * KL(Q || P_data)) / (2 * n_bound)
/// where f = PriorDataFraction, P_data = prior constructed from f-fraction of data,
///       Q = posterior after adaptation, λ = FlexParameter
/// </code>
/// </para>
/// </remarks>
public class FlexPACBayesOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Fraction of support data used to construct the data-dependent prior (0 &lt; f &lt; 1). Default: 0.5.
    /// </summary>
    public double PriorDataFraction { get; set; } = 0.5;

    /// <summary>
    /// KL divergence coefficient in the PAC-Bayesian bound. Default: 0.1.
    /// </summary>
    public double KLCoefficient { get; set; } = 0.1;

    /// <summary>
    /// Flex parameter λ controlling the trade-off between bound tightness and regularization.
    /// λ=1.0 gives standard PAC-Bayes; λ→0 gives pure ERM. Default: 1.0.
    /// </summary>
    public double FlexParameter { get; set; } = 1.0;

    /// <summary>
    /// Confidence parameter δ for the PAC-Bayesian bound. Default: 0.05.
    /// </summary>
    public double Delta { get; set; } = 0.05;

    /// <summary>
    /// Initial log-variance for the posterior distribution. Default: -3.0.
    /// </summary>
    public double InitialLogVariance { get; set; } = -3.0;

    public FlexPACBayesOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && PriorDataFraction > 0 && PriorDataFraction < 1 && FlexParameter >= 0 && KLCoefficient > 0;
    public IMetaLearnerOptions<T> Clone() => new FlexPACBayesOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        PriorDataFraction = PriorDataFraction, KLCoefficient = KLCoefficient,
        FlexParameter = FlexParameter, Delta = Delta, InitialLogVariance = InitialLogVariance
    };
}
