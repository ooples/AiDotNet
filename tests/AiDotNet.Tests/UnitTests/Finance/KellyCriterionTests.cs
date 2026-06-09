using AiDotNet.Finance.Portfolio;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>Verifies Kelly-criterion position sizing against hand-computed values.</summary>
public class KellyCriterionTests
{
    [Fact]
    public void Discrete_even_odds_60pct_edge_is_0_2()
    {
        // f* = p - (1-p)/b = 0.6 - 0.4/1 = 0.2
        Assert.Equal(0.2, KellyCriterion<double>.Discrete(0.6, 1.0), 6);
    }

    [Fact]
    public void Discrete_no_edge_clamps_to_zero()
    {
        Assert.Equal(0.0, KellyCriterion<double>.Discrete(0.5, 1.0), 6); // exactly break-even
        Assert.Equal(0.0, KellyCriterion<double>.Discrete(0.4, 1.0), 6); // negative edge → no bet
        Assert.Equal(0.0, KellyCriterion<double>.Discrete(0.6, 0.0), 6); // degenerate odds
    }

    [Fact]
    public void Continuous_is_mean_over_variance()
    {
        // f* = μ / σ² = 0.1 / 0.04 = 2.5
        Assert.Equal(2.5, KellyCriterion<double>.Continuous(0.1, 0.04), 6);
        Assert.Equal(0.0, KellyCriterion<double>.Continuous(0.1, 0.0), 6); // zero variance guard
    }

    [Fact]
    public void Fractional_scales_kelly()
    {
        Assert.Equal(1.25, KellyCriterion<double>.Fractional(2.5, 0.5), 6); // half-Kelly
    }

    [Fact]
    public void FromReturns_uses_mean_and_population_variance()
    {
        // returns [0.2, 0.0] → mean 0.1, population var ((0.1)²+(0.1)²)/2 = 0.01 → 0.1/0.01 = 10
        Assert.Equal(10.0, KellyCriterion<double>.FromReturns([0.2, 0.0]), 6);
        Assert.Equal(5.0, KellyCriterion<double>.FromReturns([0.2, 0.0], 0.5), 6); // half-Kelly
        Assert.Equal(0.0, KellyCriterion<double>.FromReturns([0.05]), 6); // <2 points → 0
    }
}
