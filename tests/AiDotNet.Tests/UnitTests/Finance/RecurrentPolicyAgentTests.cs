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
/// Tests for the recurrent (LSTM) portfolio policy agent: it is genuinely HISTORY-dependent (the defining
/// property of a recurrent policy — same current input yields different actions depending on the past), its
/// recurrent state resets per episode, and it trains end-to-end through the harness on the portfolio environment
/// without error and yields finite holdout metrics. RL convergence itself is a research result (ranked via the
/// experiment runner), not a unit assertion.
/// </summary>
public sealed class RecurrentPolicyAgentTests
{
    private static Vector<double> State(params double[] v) => new(v);

    [Fact]
    [Trait("category", "unit")]
    public void Policy_is_history_dependent_same_input_different_past_gives_different_action()
    {
        // Two agents with identical weights (same seed). Feed them the SAME final observation but DIFFERENT
        // history first. A memoryless policy would return the same action; a recurrent one must differ.
        var a = new RecurrentPolicyAgent<double>(stateDim: 3, actionDim: 2, hidden: 8, seed: 1);
        var b = new RecurrentPolicyAgent<double>(stateDim: 3, actionDim: 2, hidden: 8, seed: 1);

        // Different histories.
        a.SelectAction(State(1, 0, 0), explore: false);
        a.SelectAction(State(1, 1, 0), explore: false);
        b.SelectAction(State(-1, 0, 1), explore: false);
        b.SelectAction(State(0, -1, 1), explore: false);

        // Same current observation.
        var actA = a.SelectAction(State(0.5, 0.5, 0.5), explore: false);
        var actB = b.SelectAction(State(0.5, 0.5, 0.5), explore: false);

        double diff = Math.Abs(actA[0] - actB[0]) + Math.Abs(actA[1] - actB[1]);
        Assert.True(diff > 1e-9, "recurrent policy should condition on history — actions were identical");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Reset_episode_restores_the_agent_to_a_fresh_state()
    {
        var agent = new RecurrentPolicyAgent<double>(stateDim: 3, actionDim: 2, hidden: 8, seed: 2);
        // A fresh, identical-weight agent (same seed) that only ever sees the probe — the reference for "clean state".
        var fresh = new RecurrentPolicyAgent<double>(stateDim: 3, actionDim: 2, hidden: 8, seed: 2);

        // Establish a history on `agent`, then read the action for a probe observation.
        agent.SelectAction(State(1, 1, 1), explore: false);
        agent.SelectAction(State(1, 1, 1), explore: false);
        var afterHistory = agent.SelectAction(State(0.2, 0.2, 0.2), explore: false);

        // After reset, the SAME probe from a clean state must (a) differ from the mid-history read, and
        // (b) EXACTLY match the fresh agent's action for the probe — i.e. reset fully restores the hidden state,
        // not merely perturbs it.
        agent.ResetEpisode();
        var afterReset = agent.SelectAction(State(0.2, 0.2, 0.2), explore: false);
        var freshAction = fresh.SelectAction(State(0.2, 0.2, 0.2), explore: false);

        double changed = Math.Abs(afterHistory[0] - afterReset[0]) + Math.Abs(afterHistory[1] - afterReset[1]);
        Assert.True(changed > 1e-9, "hidden state should reset per episode (differs from the mid-history read)");
        Assert.Equal(freshAction[0], afterReset[0], 12);
        Assert.Equal(freshAction[1], afterReset[1], 12);
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

        var agent = new RecurrentPolicyAgent<double>(
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
