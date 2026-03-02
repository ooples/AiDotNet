namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Financial SAC (Soft Actor-Critic) trading agent.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAC is an off-policy actor-critic algorithm that maximizes
/// both reward and entropy (exploration). It is well-suited for continuous action
/// spaces like portfolio weight allocation. These options extend the base trading
/// agent options with SAC-specific parameters.
/// </para>
/// </remarks>
public class FinancialSACAgentOptions<T> : TradingAgentOptions<T>
{
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public FinancialSACAgentOptions() { }

    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public FinancialSACAgentOptions(FinancialSACAgentOptions<T> other) : this()
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        // Copy base class (TradingAgentOptions) properties
        LearningRate = other.LearningRate;
        DiscountFactor = other.DiscountFactor;
        LossFunction = other.LossFunction;
        Seed = other.Seed;
        BatchSize = other.BatchSize;
        ReplayBufferSize = other.ReplayBufferSize;
        TargetUpdateFrequency = other.TargetUpdateFrequency;
        WarmupSteps = other.WarmupSteps;
        EpsilonStart = other.EpsilonStart;
        EpsilonEnd = other.EpsilonEnd;
        EpsilonDecay = other.EpsilonDecay;
        StateSize = other.StateSize;
        ActionSize = other.ActionSize;
        ContinuousActions = other.ContinuousActions;
        HiddenLayers = other.HiddenLayers;
        InitialCapital = other.InitialCapital;
        TransactionCost = other.TransactionCost;
        MaxPositionSize = other.MaxPositionSize;
        RiskFreeRate = other.RiskFreeRate;
        AllowShortSelling = other.AllowShortSelling;
        UseRiskAdjustedReward = other.UseRiskAdjustedReward;
        VariancePenalty = other.VariancePenalty;
        RewardScale = other.RewardScale;

        // Copy SAC-specific properties
        InitialLogAlpha = other.InitialLogAlpha;
        TargetEntropyRatio = other.TargetEntropyRatio;
    }

    /// <summary>
    /// Initial log alpha value for automatic temperature tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how much the agent explores vs exploits.
    /// When auto-tuning is enabled, this is just the starting point.
    /// </para>
    /// </remarks>
    public double InitialLogAlpha { get; set; } = 0.0;

    /// <summary>
    /// Target entropy ratio relative to action dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sets the target randomness level as a fraction of
    /// the maximum possible entropy. Values closer to 1.0 encourage more exploration.
    /// </para>
    /// </remarks>
    public double TargetEntropyRatio { get; set; } = -1.0;
}
