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
