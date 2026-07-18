using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Finance.Trading.Evaluation;
using AiDotNet.Finance.Trading.Rewards;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Tests for the feed-forward (memoryless) policy agent — the controlled counterpart to
/// <see cref="RecurrentPolicyAgent{T}"/>. It must be MEMORYLESS (same current input → same action regardless of
/// history — the opposite of the recurrent agent), and train end-to-end through the harness. Together the two
/// agents make recurrent-vs-MLP a clean experiment where only memory differs.
/// </summary>
public sealed class FeedforwardPolicyAgentTests
{
    private static Vector<double> State(params double[] v) => new(v);

    [Fact]
    [Trait("category", "unit")]
    public void Policy_is_memoryless_same_input_same_action_regardless_of_history()
    {
        var a = new FeedforwardPolicyAgent<double>(stateDim: 3, actionDim: 2, hidden: 8, seed: 1);
        var b = new FeedforwardPolicyAgent<double>(stateDim: 3, actionDim: 2, hidden: 8, seed: 1);

        // Different histories fed to identical-weight agents.
        a.SelectAction(State(1, 0, 0), explore: false);
        a.SelectAction(State(1, 1, 0), explore: false);
        b.SelectAction(State(-1, 0, 1), explore: false);

        // Same current observation → the memoryless policy must give the SAME action (history is ignored).
        var actA = a.SelectAction(State(0.5, 0.5, 0.5), explore: false);
        var actB = b.SelectAction(State(0.5, 0.5, 0.5), explore: false);

        Assert.Equal(actA[0], actB[0], 12);
        Assert.Equal(actA[1], actB[1], 12);
    }

    [Fact(Timeout = 120000)]
    [Trait("category", "unit")]
    public async Task Trains_end_to_end_through_the_harness_with_finite_holdout_metrics()
    {
        await Task.Yield();
        var prices = new List<double[]>
        {
            Enumerable.Range(0, 40).Select(i => 100.0 + i).ToArray(),
            Enumerable.Range(0, 40).Select(i => 100.0 + 0.5 * i).ToArray(),
        };
        var trainEnv = new PortfolioManagerEnvironment<double>(
            prices, null, windowSize: 5, initialCapital: 100_000, reward: new DifferentialSharpeReward());

        var agent = new FeedforwardPolicyAgent<double>(
            trainEnv.ObservationSpaceDimension, trainEnv.ActionSpaceSize, hidden: 16, learningRate: 1e-3, seed: 3);

        double meanReturn = PortfolioAgentTrainer.Train(agent, trainEnv, episodes: 3);
        Assert.True(double.IsFinite(meanReturn), $"training return {meanReturn} not finite");

        var evalEnv = new PortfolioManagerEnvironment<double>(
            prices, null, windowSize: 5, initialCapital: 100_000, reward: new DifferentialSharpeReward());
        var result = PortfolioBacktest.Run(evalEnv, s => agent.SelectAction(s, explore: false));

        Assert.True(double.IsFinite(result.FinalValue) && result.FinalValue > 0, $"final value {result.FinalValue}");
        Assert.True(result.Steps > 0);
    }
}
