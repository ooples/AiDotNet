using AiDotNet.Finance.Risk;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>Verifies the Sharpe/Sortino/Calmar risk ratios against hand-computed values + properties.</summary>
public class RiskRatiosTests
{
    [Fact]
    public void Constant_returns_give_zero_ratios()
    {
        var flat = new[] { 0.01, 0.01, 0.01 };
        Assert.Equal(0.0, RiskRatios<double>.Sharpe(flat), 9);  // zero variance
        Assert.Equal(0.0, RiskRatios<double>.Sortino(flat), 9); // no downside
        Assert.Equal(0.0, RiskRatios<double>.Calmar(flat), 9);  // no drawdown
    }

    [Fact]
    public void Sortino_matches_hand_computed()
    {
        // returns [0.2, -0.1], rf=0, ppy=1: mean 0.05; downsideDev = sqrt(((-0.1)^2)/2) = sqrt(0.005)
        var r = new[] { 0.2, -0.1 };
        var expected = 0.05 / System.Math.Sqrt(0.005);
        Assert.Equal(expected, RiskRatios<double>.Sortino(r, riskFreePerPeriod: 0.0, periodsPerYear: 1), 6);
    }

    [Fact]
    public void Calmar_matches_hand_computed()
    {
        // [0.5, -0.5]: equity 1 -> 1.5 -> 0.75; maxDD = (1.5-0.75)/1.5 = 0.5.
        // ppy=2 (=count) → annualizedReturn = 0.75 - 1 = -0.25 → Calmar = -0.25/0.5 = -0.5.
        Assert.Equal(-0.5, RiskRatios<double>.Calmar([0.5, -0.5], periodsPerYear: 2), 6);
        // Monotonic up → no drawdown → 0.
        Assert.Equal(0.0, RiskRatios<double>.Calmar([0.2, 0.2], periodsPerYear: 2), 9);
    }

    [Fact]
    public void Sortino_exceeds_sharpe_when_upside_dominates_volatility()
    {
        // Big upside spikes inflate total std (hurting Sharpe) but not downside dev (Sortino ignores them).
        var r = new[] { 0.3, 0.005, 0.3, 0.005, -0.01 };
        Assert.True(RiskRatios<double>.Sortino(r) > RiskRatios<double>.Sharpe(r));
    }
}
