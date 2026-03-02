namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the MarketMakingAgent.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Market making is a specific trading strategy where an
/// agent provides liquidity to the market by quoting both a buy and a sell price.
/// These options control how the agent manages its inventory and sets its spreads.
/// </para>
/// </remarks>
public class MarketMakingOptions<T> : TradingAgentOptions<T>
{
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public MarketMakingOptions() { }

    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public MarketMakingOptions(MarketMakingOptions<T> other) : this()
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

        // Copy MarketMaking-specific properties
        MaxInventory = other.MaxInventory;
        InventoryPenalty = other.InventoryPenalty;
        BaseSpread = other.BaseSpread;
    }

    /// <summary>
    /// The maximum inventory size the agent can hold.
    /// </summary>
    public int MaxInventory { get; set; } = 100;

    /// <summary>
    /// Penalty coefficient for holding inventory (encourages neutral position).
    /// </summary>
    public double InventoryPenalty { get; set; } = 0.01;

    /// <summary>
    /// The base spread around the mid-price.
    /// </summary>
    public double BaseSpread { get; set; } = 0.001;

    /// <summary>
    /// Validates the market making options.
    /// </summary>
    public override void Validate()
    {
        base.Validate();
        if (MaxInventory < 1)
            throw new ArgumentException("MaxInventory must be at least 1.", nameof(MaxInventory));
        if (InventoryPenalty < 0)
            throw new ArgumentException("InventoryPenalty must be non-negative.", nameof(InventoryPenalty));
        if (BaseSpread < 0)
            throw new ArgumentException("BaseSpread must be non-negative.", nameof(BaseSpread));
    }
}
