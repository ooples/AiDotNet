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
    /// Maximum absolute inventory, as an OVERRIDE of the environment's own limit.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Precedence.</b> <c>null</c> (the default) means "unset": the environment's own
    /// <c>maxInventory</c> constructor argument binds. When set, it REPLACES that value once the environment
    /// receives these options through <c>TradingEnvironment.ApplyAgentOverrides</c>. Exactly one limit is
    /// ever in force.
    /// </para>
    /// </remarks>
    public int? MaxInventory { get; set; }

    /// <summary>
    /// Penalty coefficient per unit of held inventory, as an OVERRIDE of the environment's own penalty.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Precedence.</b> <c>null</c> (the default) means "unset": the environment's own
    /// <c>inventoryPenalty</c> constructor argument binds. When set, it REPLACES that value — it is never
    /// added on top of it, so the penalty cannot be charged twice.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Holding a big long or short position while making markets is risky, so the
    /// reward is docked in proportion to inventory. This is how strongly that applies.
    /// </para>
    /// </remarks>
    public double? InventoryPenalty { get; set; }

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
        if (MaxInventory is int maxInventory && maxInventory < 1)
            throw new ArgumentException("MaxInventory must be at least 1 when set.", nameof(MaxInventory));
        if (InventoryPenalty is double inventoryPenalty && (inventoryPenalty < 0 || double.IsNaN(inventoryPenalty)))
            throw new ArgumentException("InventoryPenalty must be non-negative when set.", nameof(InventoryPenalty));
        if (BaseSpread < 0)
            throw new ArgumentException("BaseSpread must be non-negative.", nameof(BaseSpread));
    }
}
