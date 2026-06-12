using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Finance.Options;

/// <summary>One option leg of a strategy. Quantity is in CONTRACTS (each = <see cref="ContractMultiplier"/> shares).</summary>
public sealed record OptionLeg(
    OptionRight Right,
    OptionTradeAction Action,
    double Strike,
    double ExpiryYears,
    int Contracts = 1)
{
    public const int ContractMultiplier = 100;

    /// <summary>Signed share-equivalent exposure sign: long call / short put are bullish (+), etc.</summary>
    public int SignedContracts => Action == OptionTradeAction.Buy ? Contracts : -Contracts;
}

/// <summary>An optional stock leg (covered strategies pair stock with options).</summary>
public sealed record StockLeg(OptionTradeAction Action, int Shares)
{
    public int SignedShares => Action == OptionTradeAction.Buy ? Shares : -Shares;
}

/// <summary>
/// A named option strategy: a set of <see cref="OptionLeg"/>s (plus an optional <see cref="StockLeg"/>),
/// with the broker approval level it requires, whether its risk is defined, and its expiry payoff
/// profile (max loss, max gain, breakevens) computed by sampling the payoff over a spot grid — robust
/// for ANY leg combination, not just the textbook ones. Net Greeks come from <see cref="BlackScholes{T}"/>.
///
/// <para>The <see cref="RequiredLevel"/> classifier is the gate that lets the platform enforce a broker's
/// approval tier: a strategy with an unhedged short call is <see cref="OptionsApprovalLevel.Uncovered"/>;
/// a vertical (short hedged by a long) is <see cref="OptionsApprovalLevel.Level3"/>; long-only is Level 2;
/// covered-by-stock / cash-secured is Level 1. It errs HIGH when unsure (fail-safe).</para>
/// </summary>
public sealed class OptionStrategy
{
    public string Name { get; }
    public IReadOnlyList<OptionLeg> Legs { get; }
    public StockLeg? Stock { get; }

    /// <summary>True when the underlying is a (cash-settled) index — naked index options are the
    /// highest broker tier, distinct from naked equity options.</summary>
    public bool IsIndexUnderlying { get; }

    public OptionStrategy(string name, IReadOnlyList<OptionLeg> legs, StockLeg? stock = null, bool isIndexUnderlying = false)
    {
        Name = name;
        Legs = legs ?? throw new ArgumentNullException(nameof(legs));
        Stock = stock;
        IsIndexUnderlying = isIndexUnderlying;
    }

    /// <summary>The broker-independent risk archetype (mapped to a broker level via <see cref="BrokerOptionsProfile"/>).</summary>
    public OptionStrategyClass Class => Classify();

    /// <summary>True when the worst-case loss is bounded (no naked/uncovered short exposure).</summary>
    public bool IsDefinedRisk => Class is not (OptionStrategyClass.NakedEquity or OptionStrategyClass.NakedIndex);

    private OptionStrategyClass Classify()
    {
        var shortCalls = Legs.Where(l => l is { Right: OptionRight.Call, Action: OptionTradeAction.Sell }).ToList();
        var longCalls = Legs.Where(l => l is { Right: OptionRight.Call, Action: OptionTradeAction.Buy }).ToList();
        var shortPuts = Legs.Where(l => l is { Right: OptionRight.Put, Action: OptionTradeAction.Sell }).ToList();
        var longPuts = Legs.Where(l => l is { Right: OptionRight.Put, Action: OptionTradeAction.Buy }).ToList();

        int shortCallContracts = shortCalls.Sum(l => l.Contracts);
        int longCallContracts = longCalls.Sum(l => l.Contracts);
        int shortPutContracts = shortPuts.Sum(l => l.Contracts);
        int longPutContracts = longPuts.Sum(l => l.Contracts);

        var naked = IsIndexUnderlying ? OptionStrategyClass.NakedIndex : OptionStrategyClass.NakedEquity;
        var cls = OptionStrategyClass.None;

        // Short calls: covered by stock (100 sh/contract) → covered; by long calls → defined-risk spread; else naked.
        if (shortCallContracts > 0)
        {
            int stockShares = Stock is { Action: OptionTradeAction.Buy } ? Stock.Shares : 0;
            int coveredByStock = stockShares / OptionLeg.ContractMultiplier;
            int remaining = shortCallContracts - coveredByStock;
            if (remaining <= 0)
            {
                cls = Max(cls, OptionStrategyClass.CoveredOrSecured);
            }
            else if (longCallContracts >= remaining)
            {
                cls = Max(cls, OptionStrategyClass.DefinedRiskSpread);
            }
            else
            {
                return naked; // uncovered short call — gate immediately at the naked tier
            }
        }

        // Short puts: covered by long puts → defined-risk spread; otherwise treated as cash-secured.
        if (shortPutContracts > 0)
        {
            cls = longPutContracts >= shortPutContracts
                ? Max(cls, OptionStrategyClass.DefinedRiskSpread)
                : Max(cls, OptionStrategyClass.CoveredOrSecured);
        }

        // Long-only options (no shorts handled above). A long option HEDGING an owned stock position
        // (protective put: long stock + long put) is covered/secured (Level 1); a standalone long option
        // is long-premium (Level 2).
        if (cls == OptionStrategyClass.None && (longCallContracts > 0 || longPutContracts > 0))
        {
            bool protectsOwnedStock = Stock is { Action: OptionTradeAction.Buy } && longPutContracts > 0;
            cls = protectsOwnedStock ? OptionStrategyClass.CoveredOrSecured : OptionStrategyClass.LongPremium;
        }

        return cls;
    }

    private static OptionStrategyClass Max(OptionStrategyClass a, OptionStrategyClass b) => (OptionStrategyClass)Math.Max((int)a, (int)b);

    /// <summary>
    /// Expiry payoff profile sampled over a spot grid spanning [0, gridMax×maxStrike]. Returns the net
    /// per-share P&amp;L at expiry (premium debit/credit included) at worst and best sampled spot, plus the
    /// approximate breakeven spots (sign changes). <paramref name="netPremium"/> is the net cash flow to
    /// open (negative = debit paid, positive = credit received), per share.
    /// </summary>
    public (double MaxLoss, double MaxGain, IReadOnlyList<double> Breakevens) PayoffProfile(double netPremium, int gridPoints = 400, double gridMaxMultiple = 3.0)
    {
        double maxStrike = Legs.Count > 0 ? Legs.Max(l => l.Strike) : 1.0;
        double hi = Math.Max(1.0, maxStrike * gridMaxMultiple);
        double step = hi / gridPoints;

        double maxLoss = double.PositiveInfinity, maxGain = double.NegativeInfinity;
        var breakevens = new List<double>();
        double prevPnl = double.NaN, prevSpot = 0;

        for (int i = 0; i <= gridPoints; i++)
        {
            double spot = i * step;
            double pnl = PayoffAtExpiry(spot) + netPremium;
            maxLoss = Math.Min(maxLoss, pnl);
            maxGain = Math.Max(maxGain, pnl);

            if (!double.IsNaN(prevPnl) && Math.Sign(prevPnl) != Math.Sign(pnl) && prevPnl != pnl)
            {
                // Linear interpolate the zero crossing for an approximate breakeven.
                double t = prevPnl / (prevPnl - pnl);
                breakevens.Add(prevSpot + t * (spot - prevSpot));
            }

            prevPnl = pnl;
            prevSpot = spot;
        }

        return (maxLoss, maxGain, breakevens);
    }

    /// <summary>Per-share intrinsic payoff of all legs (+ stock) at a given expiry spot, excluding premium.</summary>
    public double PayoffAtExpiry(double spot)
    {
        double pnl = 0;
        foreach (var leg in Legs)
        {
            double intrinsic = leg.Right == OptionRight.Call ? Math.Max(0, spot - leg.Strike) : Math.Max(0, leg.Strike - spot);
            pnl += leg.SignedContracts * intrinsic; // per-share basis; contracts scale the per-share payoff
        }

        if (Stock is not null)
        {
            // Stock contributes (spot - 0) linearly; an entry reference isn't needed for shape/risk relative
            // to the options, so we measure stock P&L relative to spot=strike-neutral via its signed shares.
            pnl += Stock.SignedShares / (double)OptionLeg.ContractMultiplier * spot;
        }

        return pnl;
    }

    // ── Standard strategy factories, each tagged by the level it will classify to ──

    /// <summary>Level 1: long 100 shares + short 1 call (income on owned stock).</summary>
    public static OptionStrategy CoveredCall(double callStrike, double expiryYears, int contracts = 1) =>
        new("Covered Call",
            [new OptionLeg(OptionRight.Call, OptionTradeAction.Sell, callStrike, expiryYears, contracts)],
            new StockLeg(OptionTradeAction.Buy, contracts * OptionLeg.ContractMultiplier));

    /// <summary>Level 1: short put fully cash-secured (willing to buy the stock at the strike).</summary>
    public static OptionStrategy CashSecuredPut(double putStrike, double expiryYears, int contracts = 1) =>
        new("Cash-Secured Put",
            [new OptionLeg(OptionRight.Put, OptionTradeAction.Sell, putStrike, expiryYears, contracts)]);

    /// <summary>Level 1: long 100 shares + long 1 put (downside protection).</summary>
    public static OptionStrategy ProtectivePut(double putStrike, double expiryYears, int contracts = 1) =>
        new("Protective Put",
            [new OptionLeg(OptionRight.Put, OptionTradeAction.Buy, putStrike, expiryYears, contracts)],
            new StockLeg(OptionTradeAction.Buy, contracts * OptionLeg.ContractMultiplier));

    /// <summary>Level 2: long call (defined risk = premium).</summary>
    public static OptionStrategy LongCall(double strike, double expiryYears, int contracts = 1) =>
        new("Long Call", [new OptionLeg(OptionRight.Call, OptionTradeAction.Buy, strike, expiryYears, contracts)]);

    /// <summary>Level 2: long put.</summary>
    public static OptionStrategy LongPut(double strike, double expiryYears, int contracts = 1) =>
        new("Long Put", [new OptionLeg(OptionRight.Put, OptionTradeAction.Buy, strike, expiryYears, contracts)]);

    /// <summary>Level 2: long straddle (long call + long put at the same strike) — long volatility.</summary>
    public static OptionStrategy LongStraddle(double strike, double expiryYears, int contracts = 1) =>
        new("Long Straddle",
        [
            new OptionLeg(OptionRight.Call, OptionTradeAction.Buy, strike, expiryYears, contracts),
            new OptionLeg(OptionRight.Put, OptionTradeAction.Buy, strike, expiryYears, contracts),
        ]);

    /// <summary>Level 3: bull call (debit) spread — long lower-strike call, short higher-strike call.</summary>
    public static OptionStrategy BullCallSpread(double longStrike, double shortStrike, double expiryYears, int contracts = 1) =>
        new("Bull Call Spread",
        [
            new OptionLeg(OptionRight.Call, OptionTradeAction.Buy, longStrike, expiryYears, contracts),
            new OptionLeg(OptionRight.Call, OptionTradeAction.Sell, shortStrike, expiryYears, contracts),
        ]);

    /// <summary>Level 3: bear put (debit) spread — long higher-strike put, short lower-strike put.</summary>
    public static OptionStrategy BearPutSpread(double longStrike, double shortStrike, double expiryYears, int contracts = 1) =>
        new("Bear Put Spread",
        [
            new OptionLeg(OptionRight.Put, OptionTradeAction.Buy, longStrike, expiryYears, contracts),
            new OptionLeg(OptionRight.Put, OptionTradeAction.Sell, shortStrike, expiryYears, contracts),
        ]);

    /// <summary>Level 3: iron condor — short put spread + short call spread (defined-risk, short volatility).
    /// This is the canonical "sell vol" structure the vol lane uses when forecast vol &lt; implied.</summary>
    public static OptionStrategy IronCondor(double longPut, double shortPut, double shortCall, double longCall, double expiryYears, int contracts = 1) =>
        new("Iron Condor",
        [
            new OptionLeg(OptionRight.Put, OptionTradeAction.Buy, longPut, expiryYears, contracts),
            new OptionLeg(OptionRight.Put, OptionTradeAction.Sell, shortPut, expiryYears, contracts),
            new OptionLeg(OptionRight.Call, OptionTradeAction.Sell, shortCall, expiryYears, contracts),
            new OptionLeg(OptionRight.Call, OptionTradeAction.Buy, longCall, expiryYears, contracts),
        ]);

    /// <summary>Naked short call (uncovered, large risk) — modeled so the gate can REJECT it (NakedEquity,
    /// or NakedIndex when <paramref name="isIndex"/>).</summary>
    public static OptionStrategy NakedCall(double strike, double expiryYears, int contracts = 1, bool isIndex = false) =>
        new("Naked Call", [new OptionLeg(OptionRight.Call, OptionTradeAction.Sell, strike, expiryYears, contracts)], isIndexUnderlying: isIndex);
}
