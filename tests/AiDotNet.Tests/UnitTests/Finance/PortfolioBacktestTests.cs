using System;
using System.Collections.Generic;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Finance.Trading.Evaluation;
using AiDotNet.Finance.Trading.Rewards;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Tests for the portfolio backtest evaluator + no-skill baselines — the honest-eval seam a greenfield
/// trading-agent experiment is ranked on (untouched holdout, vs baseline), never training loss.
/// </summary>
public sealed class PortfolioBacktestTests
{
    private static double[] Ramp(double start, double step, int n)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = start + i * step;
        return a;
    }

    private static PortfolioManagerEnvironment<double> Env(IReadOnlyList<double[]> prices, double lev = 1.0) =>
        new(prices, null, windowSize: 5, initialCapital: 100_000, reward: new TotalReturnReward(),
            maxLeverage: lev, transactionCost: 0.0, slippageCoefficient: 0.0, annualBorrowCost: 0.0);

    [Fact]
    [Trait("category", "unit")]
    public void Full_long_on_a_rising_market_scores_positive_return_and_sharpe()
    {
        var env = Env(new List<double[]> { Ramp(100, 1, 60) });
        var result = PortfolioBacktest.Run(env, _ => new Vector<double>(new[] { 1.0 }));

        Assert.True(result.FinalValue > 100_000, $"final {result.FinalValue}");
        Assert.True(result.TotalReturn > 0, $"total return {result.TotalReturn}");
        Assert.True(result.AnnualizedSharpe > 0, $"sharpe {result.AnnualizedSharpe}");
        Assert.True(result.MaxDrawdown < 0.02, $"a monotone rise should have ~0 drawdown, got {result.MaxDrawdown}");
        Assert.True(result.Steps > 0);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Flat_policy_holds_capital_and_takes_no_drawdown()
    {
        var env = Env(new List<double[]> { Ramp(100, 1, 60) });
        var result = PortfolioBacktest.Run(env, BaselinePolicies.Flat<double>(1));

        Assert.Equal(100_000.0, result.FinalValue, 0);
        Assert.Equal(0.0, result.TotalReturn, 6);
        Assert.Equal(0.0, result.MaxDrawdown, 6);
        Assert.Equal(0.0, result.AverageTurnover, 6);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Equal_weight_baseline_grows_on_a_rising_multi_asset_market()
    {
        var env = Env(new List<double[]> { Ramp(100, 2, 60), Ramp(50, 1, 60) });
        var result = PortfolioBacktest.Run(env, BaselinePolicies.EqualWeight<double>(2));

        Assert.True(result.TotalReturn > 0, $"equal-weight on a rising market should grow, got {result.TotalReturn}");
        Assert.Equal(60 - 5, result.Steps); // time - windowSize
    }

    [Fact]
    [Trait("category", "unit")]
    public void Momentum_baseline_longs_the_winner_and_beats_equal_weight_on_a_trend()
    {
        // Asset 0 trends up, asset 1 trends down. Momentum should hold ONLY the winner and beat equal-weight,
        // which wastes half its book on the falling asset. (2 assets, no feature columns → totalColumns == 2.)
        var prices = new List<double[]> { Ramp(100, 2, 60), Ramp(100, -1.5, 60) };

        var momentum = PortfolioBacktest.Run(Env(prices), BaselinePolicies.Momentum<double>(windowSize: 5, totalColumns: 2, tradableCount: 2));
        var equalWeight = PortfolioBacktest.Run(Env(prices), BaselinePolicies.EqualWeight<double>(2));

        Assert.True(momentum.TotalReturn > 0, $"momentum should grow on the up-trending winner, got {momentum.TotalReturn}");
        Assert.True(momentum.TotalReturn > equalWeight.TotalReturn,
            $"momentum ({momentum.TotalReturn}) should beat equal-weight ({equalWeight.TotalReturn}) by avoiding the loser");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Sortino_and_calmar_are_computed_and_sane_on_a_rising_but_choppy_market()
    {
        // A net-rising series WITH regular small dips (so downside deviation is defined). Sortino must be
        // positive and, since it penalizes only downside, at least the Sharpe; Calmar (annual return / max DD)
        // must be positive on a net gain.
        var choppy = new double[80];
        double p = 100;
        for (int i = 0; i < 80; i++)
        {
            choppy[i] = p;
            p += (i % 4 == 3) ? -0.8 : 1.2; // three up steps, one down — rises net, with real downside
        }

        var r = PortfolioBacktest.Run(Env(new List<double[]> { choppy }), _ => new Vector<double>(new[] { 1.0 }));

        Assert.True(r.AnnualizedSortino > 0, $"sortino {r.AnnualizedSortino}");
        Assert.True(r.AnnualizedSortino >= r.AnnualizedSharpe - 1e-9, "sortino only penalizes downside, so >= sharpe");
        Assert.True(r.Calmar > 0, $"calmar {r.Calmar}");
    }

    [Fact]
    [Trait("category", "unit")]
    public void RiskParity_allocates_more_to_the_calmer_asset()
    {
        // windowSize 3, 2 tradable columns. Asset 0 is steady (low vol); asset 1 is volatile (high vol).
        // Observation layout: [t0a0,t0a1, t1a0,t1a1, t2a0,t2a1].
        var state = new Vector<double>(new[] { 100.0, 100.0, 101.0, 110.0, 102.0, 95.0 });
        var policy = BaselinePolicies.RiskParity<double>(windowSize: 3, totalColumns: 2, tradableCount: 2);

        var w = policy(state);
        Assert.True(w[0] > w[1], $"risk-parity should over-weight the calmer asset: {w[0]} vs {w[1]}");
        Assert.True(Math.Abs((w[0] + w[1]) - 1.0) < 1e-9, "long-only weights should sum to 1");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Drawdown_is_captured_when_the_market_rises_then_falls()
    {
        // Rise to a peak, then fall well below it — a full-long book must record a real drawdown.
        var prices = new double[60];
        for (int i = 0; i < 30; i++) prices[i] = 100 + i * 2;      // 100 → 158
        for (int i = 30; i < 60; i++) prices[i] = 158 - (i - 29) * 3; // fall back toward ~68
        var env = Env(new List<double[]> { prices });

        var result = PortfolioBacktest.Run(env, _ => new Vector<double>(new[] { 1.0 }));
        Assert.True(result.MaxDrawdown > 0.10, $"expected a material drawdown, got {result.MaxDrawdown}");
    }
}
