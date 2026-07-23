using System;

namespace AiDotNet.Finance.Options;

/// <summary>Whether the vol edge says to sell, buy, or sit out volatility.</summary>
public enum VolStance
{
    NoTrade,
    SellVolatility,
    BuyVolatility,
}

/// <summary>
/// The volatility edge: our FORECAST realized vol vs the option market's IMPLIED vol, and the resulting
/// stance. Edge = (implied − forecast) / forecast: when implied richly exceeds our forecast, vol is
/// overpriced (sell it); when our forecast exceeds implied, vol is cheap (buy it).
/// </summary>
public sealed record VolEdge(double ForecastVol, double ImpliedVol, double Edge, VolStance Stance);

/// <summary>
/// Turns a realized-volatility FORECAST (the one signal that is actually predictable — see the platform's
/// vol research) into an options stance and a concrete DEFINED-RISK structure, by comparing it to the
/// option market's implied vol:
/// <list type="bullet">
/// <item>forecast realized vol ≪ implied → vol is overpriced → <b>sell vol</b> via an iron condor (collect
/// premium, defined risk) with short strikes ~1 implied-σ out.</item>
/// <item>forecast realized vol ≫ implied → vol is cheap → <b>buy vol</b> via a long straddle at the money.</item>
/// </list>
/// This is the monetization bridge for the vol edge. It needs an IMPLIED vol (from an option chain) as
/// input — wire an options-chain feed to supply it. The DECISION is pure + testable here.
/// </summary>
public static class VolatilityOptionsSignal
{
    /// <summary>Compute the vol edge + stance. Thresholds are the relative gap required to act (default
    /// 15%) — a buffer so we only trade a meaningful, cost-surviving mispricing.</summary>
    public static VolEdge Evaluate(double forecastRealizedVol, double impliedVol, double sellThreshold = 0.15, double buyThreshold = 0.15)
    {
        if (forecastRealizedVol <= 0 || impliedVol <= 0)
        {
            return new VolEdge(forecastRealizedVol, impliedVol, 0, VolStance.NoTrade);
        }

        double edge = (impliedVol - forecastRealizedVol) / forecastRealizedVol;
        var stance = edge >= sellThreshold ? VolStance.SellVolatility
            : edge <= -buyThreshold ? VolStance.BuyVolatility
            : VolStance.NoTrade;
        return new VolEdge(forecastRealizedVol, impliedVol, edge, stance);
    }

    /// <summary>
    /// Recommend a concrete defined-risk structure for the edge, with strikes placed off the implied
    /// 1-σ move over the option's life (σ·spot·√T). Returns null when there's no trade, or when the
    /// account isn't authorized for the structure (the broker profile gates it — iron condors are L3).
    /// </summary>
    public static OptionStrategy? Recommend(
        VolEdge edge, double spot, double expiryYears, BrokerOptionsProfile profile,
        double shortStrikeSigma = 1.0, double wingSigma = 2.0)
    {
        if (profile is null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        if (spot <= 0 || expiryYears <= 0 || edge.Stance == VolStance.NoTrade)
        {
            return null;
        }

        // 1-σ move over the life of the option, in price terms, from the IMPLIED vol (annualized).
        double oneSigma = impliedMove(edge.ImpliedVol, spot, expiryYears);

        OptionStrategy strategy = edge.Stance switch
        {
            // Sell vol: short put + short call ~1σ out, long wings ~2σ out (defined risk).
            VolStance.SellVolatility => OptionStrategy.IronCondor(
                longPut: spot - wingSigma * oneSigma,
                shortPut: spot - shortStrikeSigma * oneSigma,
                shortCall: spot + shortStrikeSigma * oneSigma,
                longCall: spot + wingSigma * oneSigma,
                expiryYears: expiryYears),

            // Buy vol: long straddle at the money.
            VolStance.BuyVolatility => OptionStrategy.LongStraddle(strike: spot, expiryYears: expiryYears),

            _ => OptionStrategy.LongStraddle(spot, expiryYears),
        };

        // Gate by the account's authorized options level (e.g. an L2 account can't place an iron condor).
        return profile.IsPermitted(strategy) ? strategy : null;
    }

    private static double impliedMove(double impliedVol, double spot, double expiryYears)
        => impliedVol * spot * Math.Sqrt(expiryYears);
}
