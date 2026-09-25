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
    public MarketMakingOptions(MarketMakingOptions<T> other) : base(other)
    {
        // Base properties are copied by the base copy constructor; only market-making's own are listed here.
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
    /// <para><b>For Beginners:</b> A market maker ends up holding whatever other people sell it. This is the
    /// most it is allowed to hold in either direction before it must stop quoting that side — the cap that
    /// keeps one bad run from turning into an unbounded position.</para>
    /// </remarks>
    /// <value>
    /// A count of units, at least <c>1</c> when set. <c>null</c> (the default) leaves the environment's own
    /// <c>maxInventory</c> in force rather than overriding it.
    /// </value>
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
    /// <value>
    /// A reward penalty per unit of held inventory. Must be non-negative and finite when set; <c>0</c>
    /// disables the penalty and larger values make the agent flatten its position more eagerly.
    /// <c>null</c> (the default) leaves the environment's own <c>inventoryPenalty</c> in force.
    /// </value>
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
        if (InventoryPenalty is double inventoryPenalty
            && (inventoryPenalty < 0 || double.IsNaN(inventoryPenalty) || double.IsInfinity(inventoryPenalty)))
            throw new ArgumentException(
                "InventoryPenalty must be a non-negative, finite number when set.", nameof(InventoryPenalty));
        if (BaseSpread < 0 || double.IsNaN(BaseSpread) || double.IsInfinity(BaseSpread))
            throw new ArgumentException(
                "BaseSpread must be a non-negative, finite number.", nameof(BaseSpread));
    }
}
