using System;
using System.Reflection;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Evidence that <see cref="MarketMakingAgent{T}"/> learns from the reward.
/// </summary>
/// <remarks>
/// <para>
/// Before the fix the agent had no critic at all, and <c>Train()</c> looped over the replay batch doing
/// <c>_policyNetwork.Train(exp.State, exp.Action)</c> — regressing the quoting policy onto the quotes it had
/// itself produced. The stored reward was read by nothing. Whatever the agent happened to quote became its
/// own target, so a quote that lost money was reinforced exactly as hard as one that made money.
/// </para>
/// </remarks>
public sealed class MarketMakingLearnsFromRewardTests
{
    private const int StateSize = 4;
    private const int ActionSize = 2;

    private static Vector<double> Quote(double value)
    {
        var v = new Vector<double>(ActionSize);
        for (int i = 0; i < ActionSize; i++) v[i] = value;
        return v;
    }

    /// <summary>
    /// A one-state quoting bandit: the wide quote (+1) pays +1, the tight quote (-1) pays -1. Terminal
    /// transitions, so the value target is exactly the reward.
    /// </summary>
    private static MarketMakingAgent<double> CreateAgent(int seed, out Vector<double> state)
    {
        var options = (MarketMakingOptions<double>)Options(MarketMaking, StateSize, ActionSize, seed);
        options.BatchSize = 8;
        options.WarmupSteps = 0;
        options.LearningRate = 0.01;
        options.Tau = 0.05;
        state = State(StateSize, salt: 1);
        return (MarketMakingAgent<double>)Create(MarketMaking, options);
    }

    private static void StoreQuotingExperience(MarketMakingAgent<double> agent, Vector<double> state, int pairs)
    {
        for (int i = 0; i < pairs; i++)
        {
            agent.StoreExperience(state, Quote(1.0), 1.0, state, done: true);
            agent.StoreExperience(state, Quote(-1.0), -1.0, state, done: true);
        }
    }

    private static INeuralNetwork<double> Network(MarketMakingAgent<double> agent, string fieldName)
    {
        var field = typeof(MarketMakingAgent<double>).GetField(
            fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (field is null)
        {
            throw new InvalidOperationException(
                $"MarketMakingAgent has no field '{fieldName}': the agent has no critic, so the reward "
                + "cannot enter any update.");
        }

        if (field.GetValue(agent) is not INeuralNetwork<double> network)
        {
            throw new InvalidOperationException($"Field '{fieldName}' does not hold an INeuralNetwork<double>.");
        }

        return network;
    }

    private static double EvaluateQ(INeuralNetwork<double> critic, Vector<double> state, Vector<double> action)
    {
        var stateAction = new Vector<double>(state.Length + action.Length);
        for (int i = 0; i < state.Length; i++) stateAction[i] = state[i];
        for (int i = 0; i < action.Length; i++) stateAction[state.Length + i] = action[i];
        return critic.Predict(Tensor<double>.FromVector(stateAction)).ToVector()[0];
    }

    private static double MaxAbsoluteDifference(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length) return double.PositiveInfinity;
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Market_making_policy_follows_the_reward_not_its_own_past_quotes()
    {
        // Two identically-seeded agents trained on MIRRORED rewards. Both quote +1 and -1 equally often;
        // only which one pays differs. Pre-fix the policy is regressed onto BOTH stored quotes with equal
        // weight, so it converges to their mean (0) either way and the two runs cannot separate.
        double Learn(double rewardedQuote, int seed)
        {
            using var agent = CreateAgent(seed, out var state);
            for (int i = 0; i < 32; i++)
            {
                agent.StoreExperience(state, Quote(rewardedQuote), 1.0, state, done: true);
                agent.StoreExperience(state, Quote(-rewardedQuote), -1.0, state, done: true);
            }

            for (int i = 0; i < 200; i++)
            {
                agent.Train();
            }

            return agent.SelectAction(state, training: false)[0];
        }

        double rewardingWide = Learn(1.0, seed: 41);
        double rewardingTight = Learn(-1.0, seed: 41);

        Assert.True(rewardingWide > rewardingTight + 0.2,
            $"the quoting policy did not follow the reward: rewarding the wide quote produced {rewardingWide:F4}, "
            + $"rewarding the tight quote produced {rewardingTight:F4} (identical data, mirrored reward).");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Market_making_critic_ranks_the_profitable_quote_higher()
    {
        using var agent = CreateAgent(seed: 42, out var state);
        var critic = Network(agent, "_critic");

        var good = Quote(1.0);
        var bad = Quote(-1.0);

        var before = critic.GetParameters().Clone();
        StoreQuotingExperience(agent, state, pairs: 32);
        for (int i = 0; i < 200; i++)
        {
            agent.Train();
        }

        Assert.True(MaxAbsoluteDifference(before, critic.GetParameters()) > 1e-12,
            "the critic was never updated: the reward did not enter the value function.");

        double qGood = EvaluateQ(critic, state, good);
        double qBad = EvaluateQ(critic, state, bad);

        Assert.True(qGood > qBad,
            $"critic ranks the losing quote at least as high as the profitable one (Q(good)={qGood:F4}, Q(bad)={qBad:F4}).");
        Assert.True(qGood > 0.0, $"Q(good)={qGood:F4} did not become positive despite a reward of +1.");
        Assert.True(qBad < 0.0, $"Q(bad)={qBad:F4} did not become negative despite a reward of -1.");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Market_making_target_networks_are_independent_and_lag()
    {
        using var agent = CreateAgent(seed: 43, out var state);
        var critic = Network(agent, "_critic");
        var targetCritic = Network(agent, "_targetCritic");

        Assert.False(ReferenceEquals(critic, targetCritic), "the target critic IS the online critic.");
        Assert.True(MaxAbsoluteDifference(critic.GetParameters(), targetCritic.GetParameters()) < 1e-12,
            "target critic did not start as a copy of its online critic.");

        var targetBefore = targetCritic.GetParameters().Clone();
        StoreQuotingExperience(agent, state, pairs: 16);
        for (int i = 0; i < 20; i++)
        {
            agent.Train();
        }

        Assert.True(MaxAbsoluteDifference(targetBefore, targetCritic.GetParameters()) > 1e-12,
            "the target critic never moved: the soft update does nothing.");
        Assert.True(MaxAbsoluteDifference(critic.GetParameters(), targetCritic.GetParameters()) > 1e-12,
            "the target critic is identical to the online critic; tau is being ignored.");
    }
}
