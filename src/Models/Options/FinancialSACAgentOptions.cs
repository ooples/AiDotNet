namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Financial SAC (Soft Actor-Critic) trading agent.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// The defaults are Haarnoja et al. 2018a (the ICML paper, Table 1): Adam at 3e-4 for every network, γ = 0.99,
/// a replay buffer of 10⁶, minibatches of 256, two hidden layers of 256 ReLU units, τ = 0.005, one gradient step per
/// environment step, a state-value network, and a fixed temperature (SACAlpha 0.2, the inverse of the paper's reward
/// scale of 5). <see cref="Sac2018b"/> gives the follow-up paper's setting, which learns the temperature instead.
/// </para>
/// <para>
/// <b>For Beginners:</b> SAC learns a policy that earns reward while staying as random as it can afford to, which
/// keeps it exploring. These options choose the paper version and its training settings.
/// </para>
/// </remarks>
public class FinancialSACAgentOptions<T> : TradingAgentOptions<T>
{
    /// <summary>Creates Haarnoja et al. 2018a's defaults.</summary>
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public FinancialSACAgentOptions()
    {
        // Haarnoja et al. 2018a, Table 1: two hidden layers of 256 ReLU units for every network, minibatches of
        // 256 and a replay buffer of a million transitions.
        HiddenLayers = new[] { 256, 256 };
        BatchSize = 256;
        ReplayBufferSize = 1_000_000;
        // The 2018a paper holds the temperature fixed (as a reward scale); the 2018b preset learns it.
        AutoTuneAlpha = false;
    }

    /// <summary>Copies every option, the base's included.</summary>
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public FinancialSACAgentOptions(FinancialSACAgentOptions<T> other) : base(other)
    {
        // Base properties are copied by the base copy constructor; only SAC's own are listed here.
        InitialLogAlpha = other.InitialLogAlpha;
        TargetEntropyRatio = other.TargetEntropyRatio;
        ActorHead = other.ActorHead;
        Variant = other.Variant;
        PolicyCriticAggregate = other.PolicyCriticAggregate;
        SimultaneousUpdates = other.SimultaneousUpdates;
        TemperatureParameter = other.TemperatureParameter;
    }

    /// <summary>
    /// Haarnoja et al. 2018b ("Soft Actor-Critic Algorithms and Applications"): no value network, target Q-networks,
    /// and the temperature learned against a target entropy of −dim(A), starting from α = 1.
    /// </summary>
    public static FinancialSACAgentOptions<T> Sac2018b() => new()
    {
        Variant = Enums.SacVariant.AutomaticTemperature,
        AutoTuneAlpha = true,
        InitialLogAlpha = 0.0,
    };

    /// <summary>Which SAC paper the agent follows (2018a's value network by default).</summary>
    public Enums.SacVariant Variant { get; set; } = Enums.SacVariant.ValueNetwork;

    /// <summary>
    /// The learned temperature's starting point, as log α (0, so α = 1, softlearning's initial value). Used when
    /// <see cref="TradingAgentOptions{T}.AutoTuneAlpha"/> is set; a fixed temperature is <see cref="TradingAgentOptions{T}.SACAlpha"/>.
    /// </summary>
    public double InitialLogAlpha { get; set; } = 0.0;

    /// <summary>
    /// The target entropy per action dimension for the learned temperature (−1, so −dim(A), the 2018b paper's heuristic).
    /// </summary>
    public double TargetEntropyRatio { get; set; } = -1.0;

    /// <summary>
    /// How the policy predicts its spread: from the state, as both papers' Gaussian policy does (the default), or as
    /// one learned value per action dimension.
    /// </summary>
    public Enums.SacActorHead ActorHead { get; set; } = Enums.SacActorHead.StateConditionedGaussian;

    /// <summary>
    /// How the policy update combines the two Q-functions: their minimum, as both papers specify (the default); the
    /// first alone (the 2018a code); or their mean (softlearning).
    /// </summary>
    public Enums.SacCriticAggregate PolicyCriticAggregate { get; set; } = Enums.SacCriticAggregate.Minimum;

    /// <summary>
    /// Whether every network's step is computed from the same pre-step parameters, as the 2018a code runs all its
    /// training ops in one session call, rather than one after another in the order of the papers' Algorithm 1 (the
    /// default). Simultaneous updates share one policy sample between the value target, the policy loss and the
    /// temperature loss, as that code does.
    /// </summary>
    public bool SimultaneousUpdates { get; set; }

    /// <summary>
    /// What a learned temperature takes its gradient steps on: α itself, as the 2018b paper's Algorithm 1 writes it
    /// (the default), or log α, as softlearning does.
    /// </summary>
    public Enums.SacTemperatureParameter TemperatureParameter { get; set; } = Enums.SacTemperatureParameter.Alpha;
}
