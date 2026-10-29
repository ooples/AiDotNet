using System.Collections.Generic;

namespace AiDotNet.Finance.Options;

/// <summary>Call or put.</summary>
public enum OptionRight
{
    /// <summary>The right to buy the underlying at the strike.</summary>
    Call,

    /// <summary>The right to sell the underlying at the strike.</summary>
    Put,
}

/// <summary>Buy (long) or sell (short) a leg.</summary>
public enum OptionTradeAction
{
    /// <summary>Buy the leg (a long position).</summary>
    Buy,

    /// <summary>Sell the leg (a short position).</summary>
    Sell,
}

/// <summary>
/// Broker-INDEPENDENT risk archetype of an option strategy — what the position actually IS, regardless of
/// how any particular broker numbers its approval tiers. Brokers map these to their own level numbers
/// (and disagree with each other — e.g. cash-secured puts are L1 at some brokers, L2 at others), so the
/// platform classifies into these stable archetypes and lets a <see cref="BrokerOptionsProfile"/> map them
/// to broker levels.
/// </summary>
public enum OptionStrategyClass
{
    /// <summary>Not an options strategy (stock only / empty).</summary>
    None = 0,

    /// <summary>Covered by owned stock or fully cash-secured: covered call, protective put, cash-secured put.</summary>
    CoveredOrSecured = 1,

    /// <summary>Long options only — risk capped at premium: long call/put, long straddle/strangle.</summary>
    LongPremium = 2,

    /// <summary>Defined-risk multi-leg spread: every short leg hedged by a long (verticals, condors, butterflies).</summary>
    DefinedRiskSpread = 3,

    /// <summary>Naked/uncovered short EQUITY options — large/undefined risk.</summary>
    NakedEquity = 4,

    /// <summary>Naked/uncovered short INDEX options — the highest-risk tier (cash-settled, gap risk).</summary>
    NakedIndex = 5,
}

/// <summary>
/// Broker options-approval tiers. Most brokers gate strategies behind escalating levels — commonly FIVE,
/// not three, and numbered inconsistently across brokers. The platform enforces the AUTHORIZED level so an
/// account can never place a strategy above what it (and the operator) is cleared for.
/// </summary>
public enum OptionsApprovalLevel
{
    /// <summary>Not approved for options trading.</summary>
    None = 0,

    /// <summary>Tier 1; in the default scheme, covered and cash-secured strategies.</summary>
    Level1 = 1,

    /// <summary>Tier 2; in the default scheme, long options (risk capped at the premium).</summary>
    Level2 = 2,

    /// <summary>Tier 3; in the default scheme, defined-risk spreads.</summary>
    Level3 = 3,

    /// <summary>Tier 4; in the default scheme, naked equity options.</summary>
    Level4 = 4,

    /// <summary>Tier 5; in the default scheme, naked index options.</summary>
    Level5 = 5,
}

/// <summary>
/// A broker's mapping from the stable <see cref="OptionStrategyClass"/> archetypes to its own approval
/// <see cref="OptionsApprovalLevel"/> numbers, plus the level this account is authorized for. Brokers
/// differ, so this is configurable; <see cref="Default"/> is the common scheme (covered/secured = L1,
/// long premium = L2, defined-risk spreads = L3, naked equity = L4, naked index = L5).
/// </summary>
public sealed class BrokerOptionsProfile
{
    private readonly IReadOnlyDictionary<OptionStrategyClass, OptionsApprovalLevel> _map;

    /// <summary>The approval level this account is cleared for; strategies above it are refused.</summary>
    public OptionsApprovalLevel Authorized { get; }

    /// <summary>Creates a profile for an account authorized up to <paramref name="authorized"/>.</summary>
    /// <param name="authorized">The approval level the account is cleared for.</param>
    /// <param name="map">
    /// The broker's mapping from strategy class to approval level; null uses <see cref="DefaultMap"/>. Classes it
    /// omits require the highest level.
    /// </param>
    public BrokerOptionsProfile(OptionsApprovalLevel authorized, IReadOnlyDictionary<OptionStrategyClass, OptionsApprovalLevel>? map = null)
    {
        Authorized = authorized;
        _map = map ?? DefaultMap;
    }

    /// <summary>The common five-tier scheme. Override per broker (e.g. a broker that puts cash-secured
    /// puts at Level 2) by passing a custom map.</summary>
    public static readonly IReadOnlyDictionary<OptionStrategyClass, OptionsApprovalLevel> DefaultMap =
        new Dictionary<OptionStrategyClass, OptionsApprovalLevel>
        {
            [OptionStrategyClass.None] = OptionsApprovalLevel.None,
            [OptionStrategyClass.CoveredOrSecured] = OptionsApprovalLevel.Level1,
            [OptionStrategyClass.LongPremium] = OptionsApprovalLevel.Level2,
            [OptionStrategyClass.DefinedRiskSpread] = OptionsApprovalLevel.Level3,
            [OptionStrategyClass.NakedEquity] = OptionsApprovalLevel.Level4,
            [OptionStrategyClass.NakedIndex] = OptionsApprovalLevel.Level5,
        };

    /// <summary>A fully-authorized default-scheme profile (Level5) — convenient for tests/research.</summary>
    public static BrokerOptionsProfile Default { get; } = new(OptionsApprovalLevel.Level5);

    /// <summary>The broker approval level that <paramref name="cls"/> requires.</summary>
    /// <param name="cls">The strategy's risk archetype.</param>
    /// <returns>
    /// The required approval level. A class the map omits requires <see cref="OptionsApprovalLevel.Level5"/>, the
    /// highest, so an incomplete custom map refuses the strategy rather than permitting it or throwing.
    /// </returns>
    public OptionsApprovalLevel LevelFor(OptionStrategyClass cls) =>
        _map.TryGetValue(cls, out var level) ? level : OptionsApprovalLevel.Level5;

    /// <summary>Whether this account may place <paramref name="strategy"/>: options are authorized and the strategy's required level is at most <see cref="Authorized"/>.</summary>
    /// <param name="strategy">The strategy to check.</param>
    /// <returns>True when the strategy is permitted.</returns>
    public bool IsPermitted(OptionStrategy strategy) =>
        Authorized != OptionsApprovalLevel.None && (int)LevelFor(strategy.Class) <= (int)Authorized;

    /// <summary>Null if permitted, else a human-readable denial reason.</summary>
    public string? Deny(OptionStrategy strategy)
    {
        if (Authorized == OptionsApprovalLevel.None)
        {
            return "options trading not authorized for this account";
        }

        var required = LevelFor(strategy.Class);
        return (int)required <= (int)Authorized
            ? null
            : $"'{strategy.Name}' is {strategy.Class} ({required}) but account is authorized only to {Authorized}";
    }
}
