
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Types of epsilon decay schedules for Reptile meta-learning.
/// </summary>
public enum EpsilonDecaySchedule
{
    /// <summary>
    /// Linear decay from initial epsilon to final epsilon.
    /// </summary>
    Linear,

    /// <summary>
    /// Cosine annealing schedule.
    /// </summary>
    Cosine,

    /// <summary>
    /// Step decay schedule (halves at regular intervals).
    /// </summary>
    Step,

    /// <summary>
    /// Exponential decay schedule.
    /// </summary>
    Exponential
}

/// <summary>
/// Configuration for the Reptile meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Reptile is a first-order meta-learning algorithm that learns good initialization
/// parameters by repeatedly moving toward task-adapted parameters.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls how Reptile learns:
///
/// - <b>InnerLearningRate:</b> How quickly to adapt to each task (typical: 0.01)
/// - <b>MetaLearningRate (Epsilon):</b> How much to move toward adapted parameters (typical: 0.001)
/// - <b>InnerSteps:</b> How many gradient steps per task (typical: 5-10)
/// - <b>MetaBatchSize:</b> For Reptile, typically 1 (online updates)
///
/// Reptile is simpler than MAML:
/// - No second-order derivatives needed
/// - Just averages parameter updates from tasks
/// - Works well with standard optimizers
/// </para>
/// </remarks>
public class ReptileTrainerConfig<T> : IMetaLearnerConfig<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public T InnerLearningRate { get; set; } = NumOps.FromDouble(0.01);

    /// <inheritdoc/>
    public T MetaLearningRate { get; set; } = NumOps.FromDouble(0.001);

    /// <inheritdoc/>
    public int InnerSteps { get; set; } = 5;

    /// <inheritdoc/>
    public int MetaBatchSize { get; set; } = 1;

    /// <inheritdoc/>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to use momentum for meta-updates.
    /// </summary>
    /// <value>
    /// If true, applies momentum to meta-parameter updates. Default is false.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Momentum helps accelerate updates and reduces oscillations.
    ///
    /// Standard Reptile updates:
    /// θ_new = θ_old + ε * (θ_adapted - θ_old)
    ///
    /// Reptile with momentum:
    /// v_new = μ * v_old + ε * (θ_adapted - θ_old)
    /// θ_new = θ_old + v_new
    ///
    /// Where:
    /// - μ is the momentum coefficient (typically 0.9)
    /// - v is the velocity (accumulated updates)
    /// - ε is the meta-learning rate
    ///
    /// Momentum helps:
    /// - Faster convergence in consistent directions
    /// - Dampens oscillations
    /// - Escapes shallow local minima
    /// </para>
    /// </remarks>
    public bool UseMomentum { get; set; } = false;

    /// <summary>
    /// Gets or sets the momentum coefficient for meta-updates.
    /// </summary>
    /// <value>
    /// Momentum coefficient (μ) in range [0, 1). Default is 0.9.
    /// Only used if UseMomentum is true.
    /// </value>
    /// <remarks>
    /// - 0.0: No momentum (standard Reptile)
    /// - 0.5: Moderate momentum
    /// - 0.9: Strong momentum (commonly used)
    /// - 0.99: Very strong momentum (may need tuning)
    /// </remarks>
    public T MomentumCoefficient { get; set; } = NumOps.FromDouble(0.9);

    /// <summary>
    /// Gets or sets whether to use Nesterov momentum.
    /// </summary>
    /// <value>
    /// If true, uses Nesterov accelerated gradient (NAG). Default is false.
    /// Only used if UseMomentum is true.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Nesterov momentum looks ahead before applying momentum.
    ///
    /// Standard momentum:
    /// v = μ * v + ε * ∇L(θ)
    /// θ = θ - v
    ///
    /// Nesterov momentum:
    /// v = μ * v + ε * ∇L(θ - μ * v)
    /// θ = θ - v
    ///
    /// Nesterov momentum often converges faster by anticipating the momentum update.
    /// </para>
    /// </remarks>
    public bool UseNesterovMomentum { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use schedule-based epsilon decay.
    /// </summary>
    /// <value>
    /// If true, decays meta-learning rate over time. Default is false.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Learning rate decay helps stabilize training.
    ///
    /// Epsilon schedule options:
    /// - Linear: ε_t = ε_0 * (1 - t/T)
    /// - Cosine: ε_t = ε_0 * 0.5 * (1 + cos(π * t/T))
    /// - Step: ε_t = ε_0 * 0.5^(floor(t/step_size))
    ///
    /// Where t is current iteration and T is total iterations.
    /// </para>
    /// </remarks>
    public bool UseEpsilonDecay { get; set; } = false;

    /// <summary>
    /// Gets or sets the epsilon decay schedule type.
    /// </summary>
    /// <value>
    /// Type of decay schedule. Default is Linear.
    /// Only used if UseEpsilonDecay is true.
    /// </value>
    public EpsilonDecaySchedule EpsilonDecaySchedule { get; set; } = EpsilonDecaySchedule.Linear;

    /// <summary>
    /// Gets or sets the final epsilon value when using decay.
    /// </summary>
    /// <value>
    /// Final meta-learning rate after decay. Default is 0.0001.
    /// Only used if UseEpsilonDecay is true.
    /// </value>
    public T FinalEpsilon { get; set; } = NumOps.FromDouble(0.0001);

    /// <summary>
    /// Creates a default Reptile configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default values based on the original Reptile paper (Nichol et al., 2018):
    /// - Inner learning rate: 0.01 (conservative for stability)
    /// - Meta learning rate: 0.001 (10x smaller than inner rate)
    /// - Inner steps: 5 (balance between adaptation and speed)
    /// - Meta batch size: 1 (online updates, typical for Reptile)
    /// - Num meta iterations: 1000 (standard training duration)
    /// </remarks>
    public ReptileTrainerConfig()
    {
    }

    /// <summary>
    /// Creates a Reptile configuration with custom values.
    /// </summary>
    /// <param name="innerLearningRate">Learning rate for task adaptation.</param>
    /// <param name="metaLearningRate">Meta-learning rate (epsilon in Reptile).</param>
    /// <param name="innerSteps">Number of gradient steps per task.</param>
    /// <param name="metaBatchSize">Number of tasks per meta-update (typically 1 for Reptile).</param>
    /// <param name="numMetaIterations">Total number of meta-training iterations.</param>
    /// <param name="useMomentum">Whether to use momentum for meta-updates.</param>
    /// <param name="momentumCoefficient">Momentum coefficient (μ).</param>
    /// <param name="useNesterovMomentum">Whether to use Nesterov momentum.</param>
    /// <param name="useEpsilonDecay">Whether to use epsilon decay.</param>
    /// <param name="epsilonDecaySchedule">Type of epsilon decay schedule.</param>
    /// <param name="finalEpsilon">Final epsilon value after decay.</param>
    public ReptileTrainerConfig(
        double innerLearningRate,
        double metaLearningRate,
        int innerSteps,
        int metaBatchSize = 1,
        int numMetaIterations = 1000,
        bool useMomentum = false,
        double momentumCoefficient = 0.9,
        bool useNesterovMomentum = false,
        bool useEpsilonDecay = false,
        EpsilonDecaySchedule epsilonDecaySchedule = EpsilonDecaySchedule.Linear,
        double finalEpsilon = 0.0001)
    {
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        MetaLearningRate = NumOps.FromDouble(metaLearningRate);
        InnerSteps = innerSteps;
        MetaBatchSize = metaBatchSize;
        NumMetaIterations = numMetaIterations;
        UseMomentum = useMomentum;
        MomentumCoefficient = NumOps.FromDouble(momentumCoefficient);
        UseNesterovMomentum = useNesterovMomentum;
        UseEpsilonDecay = useEpsilonDecay;
        EpsilonDecaySchedule = epsilonDecaySchedule;
        FinalEpsilon = NumOps.FromDouble(finalEpsilon);
    }

    /// <inheritdoc/>
    public bool IsValid()
    {
        var innerLr = Convert.ToDouble(InnerLearningRate);
        var metaLr = Convert.ToDouble(MetaLearningRate);
        var momentumCoeff = Convert.ToDouble(MomentumCoefficient);
        var finalEpsilon = Convert.ToDouble(FinalEpsilon);

        return innerLr > 0 && innerLr <= 1.0 &&
               metaLr > 0 && metaLr <= 1.0 &&
               InnerSteps > 0 && InnerSteps <= 100 &&
               MetaBatchSize > 0 && MetaBatchSize <= 128 &&
               NumMetaIterations > 0 && NumMetaIterations <= 1000000 &&
               momentumCoeff >= 0 && momentumCoeff < 1.0 &&
               finalEpsilon > 0 && finalEpsilon < 1.0;
    }
}
