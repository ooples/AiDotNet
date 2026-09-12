using System;
using System.IO;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Models.Options;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// <c>TransactionCost</c>, <c>InventoryPenalty</c> and <c>MaxInventory</c> are settable on the agent options
/// and are also constructor arguments of the environments that actually enforce them. They used to be read by
/// nothing on the agent side. They now OVERRIDE the environment: unset (null) means the environment's value
/// binds, set means the agent's value REPLACES it.
/// </summary>
/// <remarks>
/// Each test pins both halves of that contract, and the "set" half is written as an equality against an
/// environment CONSTRUCTED with the override value — which is what rules out the failure mode of applying the
/// agent's value in addition to the environment's.
/// </remarks>
public sealed class TradingFrictionOverrideTests
{
    private static Tensor<double> Prices(int bars)
    {
        var data = new Tensor<double>(new[] { bars, 1 });
        double price = 100.0;
        for (int i = 0; i < bars; i++)
        {
            data[i, 0] = price;
            price *= (i % 2 == 0) ? 1.02 : 0.99;
        }

        return data;
    }

    private static StockTradingEnvironment<double> StockEnvironment(double transactionCost) =>
        new(Prices(40), windowSize: 3, initialCapital: 1_000.0, tradeSize: 1.0,
            transactionCost: transactionCost, allowShortSelling: true, randomStart: false, seed: 7);

    /// <summary>Buys for a few steps and returns the resulting portfolio value — sensitive to trading cost.</summary>
    private static double RunStockEpisode(StockTradingEnvironment<double> env)
    {
        env.Reset();
        var buy = new Vector<double>(new[] { 0.0, 0.0, 1.0 });
        double last = 0.0;
        for (int i = 0; i < 6; i++)
        {
            var (_, _, done, info) = env.Step(buy);
            last = Convert.ToDouble(info["portfolioValue"]);
            if (done) break;
        }

        return last;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Unset_TransactionCost_leaves_the_environments_own_cost_in_force()
    {
        var options = Options(Dqn, 4, 3, seed: 1);
        Assert.Null(options.TransactionCost);

        var overridden = StockEnvironment(transactionCost: 0.10);
        overridden.ApplyAgentOverrides(options);

        double withOverrideApplied = RunStockEpisode(overridden);
        double untouched = RunStockEpisode(StockEnvironment(transactionCost: 0.10));

        Assert.Equal(untouched, withOverrideApplied, 9);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Set_TransactionCost_replaces_the_environments_cost_and_is_not_added_to_it()
    {
        var options = Options(Dqn, 4, 3, seed: 1);
        options.TransactionCost = 0.02;

        var overridden = StockEnvironment(transactionCost: 0.10);
        overridden.ApplyAgentOverrides(options);

        double actual = RunStockEpisode(overridden);
        double replaced = RunStockEpisode(StockEnvironment(transactionCost: 0.02));
        double summed = RunStockEpisode(StockEnvironment(transactionCost: 0.12));
        double original = RunStockEpisode(StockEnvironment(transactionCost: 0.10));

        Assert.Equal(replaced, actual, 9);
        Assert.NotEqual(original, actual, 9);
        Assert.NotEqual(summed, actual, 9);
    }

    private static MarketMakingEnvironment<double> QuotingEnvironment(
        double inventoryPenalty, int maxInventory) =>
        new(Prices(40), windowSize: 3, initialCapital: 1_000.0, tradeSize: 1.0,
            baseSpread: 0.01, orderArrivalRate: 0.9, maxInventory: maxInventory,
            inventoryPenalty: inventoryPenalty, transactionCost: 0.0, allowShortSelling: true,
            randomStart: false, maxEpisodeLength: 0, seed: 11);

    /// <summary>Quotes tightly for several steps and returns the summed reward — sensitive to the penalty.</summary>
    private static double RunQuotingEpisode(MarketMakingEnvironment<double> env)
    {
        env.Reset();
        var quote = new Vector<double>(new[] { 0.0, 0.0 });
        double total = 0.0;
        for (int i = 0; i < 10; i++)
        {
            var (_, reward, done, _) = env.Step(quote);
            total += reward;
            if (done) break;
        }

        return total;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Unset_InventoryPenalty_leaves_the_environments_own_penalty_in_force()
    {
        var options = (MarketMakingOptions<double>)Options(MarketMaking, 4, 2, seed: 2);
        Assert.Null(options.InventoryPenalty);

        var overridden = QuotingEnvironment(inventoryPenalty: 0.05, maxInventory: 10);
        overridden.ApplyAgentOverrides(options);

        Assert.Equal(
            RunQuotingEpisode(QuotingEnvironment(inventoryPenalty: 0.05, maxInventory: 10)),
            RunQuotingEpisode(overridden),
            9);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Set_InventoryPenalty_replaces_the_environments_penalty_and_is_not_added_to_it()
    {
        var options = (MarketMakingOptions<double>)Options(MarketMaking, 4, 2, seed: 2);
        options.InventoryPenalty = 0.01;

        var overridden = QuotingEnvironment(inventoryPenalty: 0.05, maxInventory: 10);
        overridden.ApplyAgentOverrides(options);

        double actual = RunQuotingEpisode(overridden);
        double replaced = RunQuotingEpisode(QuotingEnvironment(inventoryPenalty: 0.01, maxInventory: 10));
        double summed = RunQuotingEpisode(QuotingEnvironment(inventoryPenalty: 0.06, maxInventory: 10));

        Assert.Equal(replaced, actual, 9);
        Assert.NotEqual(summed, actual, 9);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Unset_MaxInventory_leaves_the_environments_own_limit_in_force()
    {
        var options = (MarketMakingOptions<double>)Options(MarketMaking, 4, 2, seed: 2);
        Assert.Null(options.MaxInventory);

        var overridden = QuotingEnvironment(inventoryPenalty: 0.0, maxInventory: 6);
        overridden.ApplyAgentOverrides(options);

        Assert.Equal(
            RunQuotingEpisode(QuotingEnvironment(inventoryPenalty: 0.0, maxInventory: 6)),
            RunQuotingEpisode(overridden),
            9);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Set_MaxInventory_replaces_the_environments_limit()
    {
        var options = (MarketMakingOptions<double>)Options(MarketMaking, 4, 2, seed: 2);
        options.MaxInventory = 1;

        var overridden = QuotingEnvironment(inventoryPenalty: 0.0, maxInventory: 6);
        overridden.ApplyAgentOverrides(options);

        double actual = RunQuotingEpisode(overridden);
        double replaced = RunQuotingEpisode(QuotingEnvironment(inventoryPenalty: 0.0, maxInventory: 1));
        double loose = RunQuotingEpisode(QuotingEnvironment(inventoryPenalty: 0.0, maxInventory: 6));

        Assert.Equal(replaced, actual, 9);
        Assert.NotEqual(loose, actual, 9);
    }

    [Fact]
    [Trait("category", "unit")]
    public void An_incompatible_market_making_checkpoint_is_rejected_with_an_explanation()
    {
        // A checkpoint whose parameter layout does not match the agent loading it — the position every
        // pre-critic market-making checkpoint is now in, since the agent gained a critic and two target
        // networks. Produced here by saving an agent of a different state width rather than by
        // hand-assembling bytes, so the test does not depend on the serializer's private layout.
        var savedOptions = (MarketMakingOptions<double>)Options(MarketMaking, 6, 2, seed: 3);
        using var savedAgent = (MarketMakingAgent<double>)Create(MarketMaking, savedOptions);
        byte[] checkpoint = savedAgent.Serialize();

        var loadOptions = (MarketMakingOptions<double>)Options(MarketMaking, 4, 2, seed: 3);
        using var loadingAgent = (MarketMakingAgent<double>)Create(MarketMaking, loadOptions);
        Assert.NotEqual(savedAgent.ParameterCount, loadingAgent.ParameterCount);

        var mismatch = Assert.Throws<InvalidDataException>(() => loadingAgent.Deserialize(checkpoint));

        Assert.Contains("cannot load this checkpoint", mismatch.Message);
        Assert.Contains("critic", mismatch.Message);
        Assert.Contains("never read the reward", mismatch.Message);
        Assert.Contains("Retrain", mismatch.Message);
    }

    [Fact]
    [Trait("category", "unit")]
    public void A_market_making_checkpoint_of_the_right_shape_still_round_trips()
    {
        // The guard must reject only genuinely incompatible checkpoints, not every load.
        var options = (MarketMakingOptions<double>)Options(MarketMaking, 4, 2, seed: 4);
        using var saved = (MarketMakingAgent<double>)Create(MarketMaking, options);
        byte[] checkpoint = saved.Serialize();

        using var restored = (MarketMakingAgent<double>)Create(
            MarketMaking, (MarketMakingOptions<double>)Options(MarketMaking, 4, 2, seed: 4));
        restored.Deserialize(checkpoint);

        var before = saved.GetParameters();
        var after = restored.GetParameters();
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
        {
            Assert.Equal(before[i], after[i], 12);
        }
    }

}
