using AiDotNet.Finance.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// The vol-edge → options bridge (#76/#71): forecast realized vol vs implied vol decides sell/buy/no-trade,
/// and maps to a defined-risk structure gated by the account's options level. Pins the stance thresholds,
/// the structure choice, the level gate, and that selling vol when implied is rich is the canonical case.
/// </summary>
public class VolatilityOptionsSignalTests
{
    [Fact]
    public void Implied_richly_above_forecast_says_SELL_vol()
    {
        // forecast 20% realized, implied 30% → edge +50% → sell vol.
        var edge = VolatilityOptionsSignal.Evaluate(forecastRealizedVol: 0.20, impliedVol: 0.30);
        Assert.Equal(VolStance.SellVolatility, edge.Stance);
        Assert.True(edge.Edge > 0.4);
    }

    [Fact]
    public void Forecast_above_implied_says_BUY_vol()
    {
        // forecast 35% realized, implied 20% → edge -43% → buy vol.
        var edge = VolatilityOptionsSignal.Evaluate(forecastRealizedVol: 0.35, impliedVol: 0.20);
        Assert.Equal(VolStance.BuyVolatility, edge.Stance);
    }

    [Fact]
    public void Small_gap_is_NoTrade()
    {
        var edge = VolatilityOptionsSignal.Evaluate(forecastRealizedVol: 0.20, impliedVol: 0.21); // +5% < 15%
        Assert.Equal(VolStance.NoTrade, edge.Stance);
    }

    [Fact]
    public void SellVol_recommends_a_defined_risk_iron_condor_with_strikes_around_spot()
    {
        var edge = VolatilityOptionsSignal.Evaluate(0.20, 0.40); // strong sell
        var strat = VolatilityOptionsSignal.Recommend(edge, spot: 100, expiryYears: 0.0833 /* ~1 mo */, BrokerOptionsProfile.Default);

        Assert.NotNull(strat);
        Assert.Equal("Iron Condor", strat!.Name);
        Assert.Equal(OptionStrategyClass.DefinedRiskSpread, strat.Class);
        Assert.True(strat.IsDefinedRisk);
        // 4 legs straddling spot: long put < short put < short call < long call.
        var strikes = strat.Legs.Select(l => l.Strike).OrderBy(s => s).ToList();
        Assert.Equal(4, strikes.Count);
        Assert.True(strikes[0] < strikes[1] && strikes[1] < 100 && strikes[2] > 100 && strikes[2] < strikes[3]);
    }

    [Fact]
    public void BuyVol_recommends_an_at_the_money_long_straddle()
    {
        var edge = VolatilityOptionsSignal.Evaluate(0.40, 0.20); // strong buy
        var strat = VolatilityOptionsSignal.Recommend(edge, spot: 100, expiryYears: 0.0833, BrokerOptionsProfile.Default);
        Assert.NotNull(strat);
        Assert.Equal("Long Straddle", strat!.Name);
        Assert.Equal(OptionStrategyClass.LongPremium, strat.Class);
    }

    [Fact]
    public void Account_not_authorized_for_the_structure_gets_NO_recommendation()
    {
        // A Level-1 account cannot place an iron condor (L3) — the gate returns null even on a sell signal.
        var level1 = new BrokerOptionsProfile(OptionsApprovalLevel.Level1);
        var edge = VolatilityOptionsSignal.Evaluate(0.20, 0.40);
        var strat = VolatilityOptionsSignal.Recommend(edge, spot: 100, expiryYears: 0.0833, level1);
        Assert.Null(strat);
    }

    [Fact]
    public void NoTrade_or_bad_inputs_recommend_nothing()
    {
        var flat = VolatilityOptionsSignal.Evaluate(0.20, 0.205);
        Assert.Null(VolatilityOptionsSignal.Recommend(flat, 100, 0.0833, BrokerOptionsProfile.Default));

        var sell = VolatilityOptionsSignal.Evaluate(0.20, 0.40);
        Assert.Null(VolatilityOptionsSignal.Recommend(sell, spot: 0, expiryYears: 0.0833, BrokerOptionsProfile.Default));
    }
}
