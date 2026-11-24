using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Finance.Trading.Rewards;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Tests for the greenfield portfolio-manager RL environment + pluggable rewards: the differential Sharpe reward
/// is genuinely risk-adjusted (prefers smooth gains), the gross-leverage budget is enforced, observation-only
/// feature columns are never traded, and a long book grows on a rising market.
/// </summary>
public sealed class PortfolioManagerEnvironmentTests
{
    // ---- observation scale ------------------------------------------------------------------------

    [Fact]
    [Trait("category", "unit")]
    public void Account_state_in_the_observation_is_the_same_for_any_capital_and_equals_the_book()
    {
        // Prices 100 and 50: raw units and cash scale with capital; weights and the cash fraction do not.
        var prices = new List<double[]> { Enumerable.Repeat(100.0, 10).ToArray(), Enumerable.Repeat(50.0, 10).ToArray() };
        Vector<double> AfterOneStep(double capital, bool scaleFree)
        {
            var env = new PortfolioManagerEnvironment<double>(prices, null, windowSize: 2, initialCapital: capital,
                reward: new TotalReturnReward(), transactionCost: 0, slippageCoefficient: 0, annualBorrowCost: 0)
            { ScaleFreeAccountState = scaleFree };
            env.Reset();
            return env.Step(new Vector<double>(new[] { 0.6, -0.3 })).NextState;
        }

        var small = AfterOneStep(1.0, true);
        var large = AfterOneStep(100_000.0, true);
        Assert.Equal(small.Length, large.Length);
        for (int i = 0; i < small.Length; i++) Assert.Equal(small[i], large[i], 9);
        // Layout: 2 x 2 window prices, then the 2 weights, then cash / value (1 - 0.6 + 0.3).
        Assert.Equal(0.6, large[4], 9);
        Assert.Equal(-0.3, large[5], 9);
        Assert.Equal(0.7, large[6], 9);

        var raw = AfterOneStep(100_000.0, false);
        Assert.Equal(600.0, raw[4], 6);   // units = 0.6 x 100,000 / 100
        Assert.Equal(70_000.0, raw[6], 6);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Account_ratios_are_both_marked_at_the_observed_price_row()
    {
        // Half of 1,000 buys 5 units at 100; the observation that follows the step is at the next row, where the price
        // is 200. Marked there the book is 500 cash + 1,000 of the asset: weight 2/3, cash 1/3 (they sum to one).
        var prices = new List<double[]> { new[] { 100.0, 100.0, 100.0, 200.0, 200.0, 200.0, 200.0, 200.0 } };
        var env = new PortfolioManagerEnvironment<double>(prices, null, windowSize: 2, initialCapital: 1000.0,
            reward: new TotalReturnReward(), transactionCost: 0, slippageCoefficient: 0, annualBorrowCost: 0);
        env.Reset();
        var observation = env.Step(new Vector<double>(new[] { 0.5 })).NextState;
        // Layout: the 2-row price window, then the weight, then cash / value.
        Assert.Equal(200.0, observation[1], 9);
        Assert.Equal(2.0 / 3.0, observation[2], 9);
        Assert.Equal(1.0 / 3.0, observation[3], 9);
    }

    // ---- reward math (deterministic) ----------------------------------------------------------

    [Fact]
    [Trait("category", "unit")]
    public void DifferentialSharpe_prefers_smooth_gains_over_volatile_ones_of_equal_mean()
    {
        // Two return streams with the SAME mean (+1%/step) but different variance. A risk-adjusted objective must
        // score the smooth one higher — that is the whole point of the differential Sharpe ratio.
        var smooth = Enumerable.Repeat(0.01, 40).ToArray();
        var volatile1 = new double[40];
        for (int i = 0; i < 40; i++) volatile1[i] = (i % 2 == 0) ? 0.05 : -0.03; // mean = 0.01

        double SumReward(double[] returns)
        {
            var reward = new DifferentialSharpeReward(eta: 0.05);
            double sum = 0;
            foreach (var r in returns)
                sum += reward.Reward(new PortfolioRewardContext(r, 0, 1, 0, 0));
            return sum;
        }

        Assert.True(SumReward(smooth) > SumReward(volatile1),
            "differential Sharpe should reward the smooth stream more than the equal-mean volatile one");
    }

    [Fact]
    [Trait("category", "unit")]
    public void TotalReturnReward_applies_turnover_and_drawdown_penalties()
    {
        var reward = new TotalReturnReward(turnoverPenalty: 0.5, drawdownPenalty: 0.2);
        // return 0.02, turnover 0.1, drawdown 0.05 → 0.02 − 0.5·0.1 − 0.2·0.05 = 0.02 − 0.05 − 0.01 = -0.04
        double r = reward.Reward(new PortfolioRewardContext(0.02, 0.1, 1.0, 0.0, 0.05));
        Assert.Equal(-0.04, r, 10);
    }

    [Fact]
    [Trait("category", "unit")]
    public void DifferentialSharpe_reset_clears_running_statistics()
    {
        var reward = new DifferentialSharpeReward(eta: 0.05);
        // VARIED returns build up nonzero variance, so mid-run the reward is a differential (not the raw return).
        foreach (var r in new[] { 0.05, -0.02, 0.04, -0.01, 0.03, 0.06, -0.03, 0.02 })
            reward.Reward(new PortfolioRewardContext(r, 0, 1, 0, 0));
        double afterRun = reward.Reward(new PortfolioRewardContext(0.03, 0, 1, 0, 0));
        reward.Reset();
        double afterReset = reward.Reward(new PortfolioRewardContext(0.03, 0, 1, 0, 0));
        // Post-reset first step bootstraps to the raw return (0.03); mid-run it is a differential, not 0.03.
        Assert.Equal(0.03, afterReset, 10);
        Assert.NotEqual(0.03, afterRun, 6);
    }

    // ---- environment behavior ----------------------------------------------------------------

    private static double[] Ramp(double start, double stepUp, int n)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = start + i * stepUp;
        return a;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Action_space_is_tradable_count_and_feature_columns_are_not_traded()
    {
        var prices = new List<double[]> { Ramp(100, 1, 30), Ramp(50, 0.5, 30) };   // 2 tradable
        var features = new List<double[]> { Ramp(0.01, 0, 30), Ramp(-0.02, 0, 30) }; // 2 observation-only
        var env = new PortfolioManagerEnvironment<double>(
            prices, features, windowSize: 5, initialCapital: 100_000, reward: new TotalReturnReward());

        Assert.Equal(2, env.ActionSpaceSize);              // == tradable count, NOT 2+2
        Assert.True(env.IsContinuousActionSpace);

        env.Reset();
        // Hold everything flat (zero weights) → no trades, feature columns can never create exposure.
        var (_, _, _, info) = env.Step(new Vector<double>(new[] { 0.0, 0.0 }));
        Assert.Equal(100_000.0, Convert.ToDouble(info["portfolioValue"]), 0); // flat book stays at capital
    }

    [Fact]
    [Trait("category", "unit")]
    public void Gross_leverage_budget_is_enforced()
    {
        var prices = new List<double[]> { Ramp(100, 1, 40), Ramp(100, 1, 40), Ramp(100, 1, 40) };
        var env = new PortfolioManagerEnvironment<double>(
            prices, null, windowSize: 5, initialCapital: 100_000, reward: new TotalReturnReward(),
            maxLeverage: 1.0, transactionCost: 0.0, slippageCoefficient: 0.0, annualBorrowCost: 0.0);

        env.Reset();
        // Ask for 1.0 in every name → gross 3.0, well over the 1.0 budget. Must be scaled down.
        env.Step(new Vector<double>(new[] { 1.0, 1.0, 1.0 }));
        Assert.True(env.GrossExposure <= 1.0 + 1e-6, $"gross exposure {env.GrossExposure} exceeded the leverage budget");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Long_book_grows_on_a_rising_market()
    {
        // Two steadily-rising assets; go fully long and hold. Value must end above starting capital.
        var prices = new List<double[]> { Ramp(100, 2, 40), Ramp(80, 1.5, 40) };
        var env = new PortfolioManagerEnvironment<double>(
            prices, null, windowSize: 5, initialCapital: 100_000, reward: new TotalReturnReward(),
            maxLeverage: 1.0, transactionCost: 0.0005, slippageCoefficient: 0.0001, annualBorrowCost: 0.0);

        env.Reset();
        double value = 100_000;
        var action = new Vector<double>(new[] { 0.5, 0.5 }); // fully invested, within budget
        for (int i = 0; i < 25; i++)
        {
            var (_, _, done, info) = env.Step(action);
            value = Convert.ToDouble(info["portfolioValue"]);
            if (done) break;
        }

        Assert.True(value > 100_000, $"a long book on a rising market should grow (ended at {value:N0})");
    }
}
