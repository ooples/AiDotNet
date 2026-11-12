using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Finance.Trading.Rewards;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Reference checks for the rebalancing step of the portfolio environments: a cash-constrained buy book is
/// scaled as a whole (every buy by the same fraction), the way a target-weight rebalancer such as FinRL's
/// portfolio environment allocates the cash, rather than being filled greedily in asset order.
/// </summary>
public class PortfolioRebalanceReferenceTests
{
    [Fact]
    [Trait("category", "unit")]
    public void Portfolio_trading_environment_executes_every_scaled_buy()
    {
        // Constant prices (1, 3); 1000 cash; 0.1% cost; target 50/50. The buy book costs 1001 > 1000, so both
        // buys are scaled by 1000/1001. The old per-asset loop then dropped the second buy entirely because its
        // scaled cost exceeded the remaining cash by one ulp.
        var data = new Tensor<double>(new[] { 10, 2 });
        for (int t = 0; t < 10; t++)
        {
            data[t, 0] = 1.0;
            data[t, 1] = 3.0;
        }

        var env = new PortfolioTradingEnvironment<double>(data, windowSize: 2, initialCapital: 1000.0, transactionCost: 0.001);
        env.Reset();
        var (_, _, _, info) = env.Step(new Vector<double>(new[] { 0.5, 0.5 }));

        var positions = (Vector<double>)info["positions"];
        double scale = 1000.0 / 1001.0;
        Assert.Equal(500.0 * scale, positions[0], 6);
        Assert.Equal(500.0 / 3.0 * scale, positions[1], 6);
        double cash = Convert.ToDouble(info["cash"]);
        Assert.True(cash >= 0.0 && cash < 1e-6, $"cash after a fully invested rebalance should be ~0, was {cash}");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Portfolio_manager_environment_scales_a_cash_short_buy_book_evenly()
    {
        // Two identical assets at 100; 100k cash; target +1 / +1 (gross 2 within the 2x budget). The buys want
        // 200k of stock against 100k of cash, so each is filled to the same half: 500 units each. Filling in
        // asset order would give asset 0 all 1000 units and asset 1 nothing.
        var prices = new List<double[]>
        {
            Enumerable.Repeat(100.0, 20).ToArray(),
            Enumerable.Repeat(100.0, 20).ToArray()
        };
        var env = new PortfolioManagerEnvironment<double>(
            prices, null, windowSize: 4, initialCapital: 100_000, reward: new TotalReturnReward(),
            maxLeverage: 2.0, transactionCost: 0.0, slippageCoefficient: 0.0, annualBorrowCost: 0.0,
            annualHoldingCost: 0.0, allowShortSelling: true);

        env.Reset();
        var (_, _, _, info) = env.Step(new Vector<double>(new[] { 1.0, 1.0 }));

        var positions = (Vector<double>)info["positions"];
        Assert.Equal(500.0, positions[0], 6);
        Assert.Equal(500.0, positions[1], 6);
        Assert.Equal(0.0, Convert.ToDouble(info["cash"]), 6);
    }
}
