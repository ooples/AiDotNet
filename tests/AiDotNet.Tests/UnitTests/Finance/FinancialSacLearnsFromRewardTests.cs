using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Evidence that <see cref="FinancialSACAgent{T}"/> actually implements Soft Actor-Critic.
/// </summary>
/// <remarks>
/// <para>
/// Before the fix the agent was SAC in name only: <c>Train()</c> called <c>_actor.Train(states, actions)</c>
/// — regressing the actor onto the very actions it had itself taken — and did nothing else. The twin critics
/// were never trained, <c>UpdateTargetNetworks</c> was an empty method body, and the REWARD was never read by
/// any update. An agent like that cannot distinguish a profitable action from a ruinous one, and nothing in
/// the training curve says so.
/// </para>
/// <para>
/// These tests therefore assert against behaviour, not structure: that the critics move at all, that they
/// rank a rewarded action above a punished one, and that the policy follows them. They reach the critics by
/// reflection rather than through new public API so that the whole file compiles and runs against the
/// pre-fix commit, where it fails.
/// </para>
/// </remarks>
public sealed class FinancialSacLearnsFromRewardTests
{
    private const int StateSize = 4;
    private const int ActionSize = 1;

    /// <summary>The single state the toy task is played in.</summary>
    private static Vector<double> FixedState() => State(StateSize, salt: 1);

    private static Vector<double> Action(double value)
    {
        var v = new Vector<double>(ActionSize);
        v[0] = value;
        return v;
    }

    /// <summary>
    /// A one-state bandit: action +1 pays +1, action -1 pays -1. Terminal transitions, so the TD target is
    /// exactly the reward and the correct Q-ordering is unambiguous and bootstrap-free.
    /// </summary>
    private static FinancialSACAgent<double> CreateAgent(int seed, out Vector<double> state)
    {
        var options = (FinancialSACAgentOptions<double>)Options(Sac, StateSize, ActionSize, seed);
        options.BatchSize = 8;
        options.WarmupSteps = 0;
        options.LearningRate = 0.01;
        options.Tau = 0.05;
        options.AutoTuneAlpha = false;
        options.SACAlpha = 0.0; // Isolate the value signal: no entropy term in the target for this fixture.
        state = FixedState();
        return (FinancialSACAgent<double>)Create(Sac, options);
    }

    private static void StoreBanditExperience(FinancialSACAgent<double> agent, Vector<double> state, int pairs)
    {
        for (int i = 0; i < pairs; i++)
        {
            agent.StoreExperience(state, Action(1.0), 1.0, state, done: true);
            agent.StoreExperience(state, Action(-1.0), -1.0, state, done: true);
        }
    }

    private static INeuralNetwork<double> Network(FinancialSACAgent<double> agent, string fieldName)
    {
        var field = typeof(FinancialSACAgent<double>).GetField(
            fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (field is null)
        {
            throw new InvalidOperationException($"FinancialSACAgent has no field '{fieldName}'.");
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
        if (a.Length != b.Length)
        {
            return double.PositiveInfinity;
        }

        double worst = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        }

        return worst;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Sac_trains_its_twin_critics()
    {
        using var agent = CreateAgent(seed: 31, out var state);
        var critic1 = Network(agent, "_critic1");
        var critic2 = Network(agent, "_critic2");

        var before1 = critic1.GetParameters().Clone();
        var before2 = critic2.GetParameters().Clone();

        StoreBanditExperience(agent, state, pairs: 16);
        for (int i = 0; i < 20; i++)
        {
            agent.Train();
        }

        // Pre-fix these are exactly zero: Train() never touched either critic.
        Assert.True(MaxAbsoluteDifference(before1, critic1.GetParameters()) > 1e-12,
            "critic 1 was never updated by Train(): the reward cannot have entered the value function.");
        Assert.True(MaxAbsoluteDifference(before2, critic2.GetParameters()) > 1e-12,
            "critic 2 was never updated by Train(): the reward cannot have entered the value function.");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Twin_critics_are_four_independently_initialised_networks()
    {
        using var agent = CreateAgent(seed: 32, out var state);
        var critic1 = Network(agent, "_critic1");
        var critic2 = Network(agent, "_critic2");
        var target1 = Network(agent, "_targetCritic1");
        var target2 = Network(agent, "_targetCritic2");

        // Reference inequality: NeuralNetwork.InitializeLayers does Layers.AddRange(Architecture.Layers),
        // a REFERENCE copy, so two networks built from one architecture instance share mutable layers and
        // min(Q1, Q2) silently degenerates to min(Q, Q).
        var networks = new object[] { critic1, critic2, target1, target2 };
        for (int i = 0; i < networks.Length; i++)
        {
            for (int j = i + 1; j < networks.Length; j++)
            {
                Assert.False(ReferenceEquals(networks[i], networks[j]),
                    $"critic networks {i} and {j} are the same object.");
            }
        }

        // The online twins must also differ NUMERICALLY, or the pessimistic minimum is redundant.
        Assert.True(MaxAbsoluteDifference(critic1.GetParameters(), critic2.GetParameters()) > 1e-12,
            "the twin critics were initialised identically, so min(Q1, Q2) is just Q1.");

        // ... and must still differ after a shared update, proving the two are trained as separate networks.
        StoreBanditExperience(agent, state, pairs: 8);
        agent.Train();

        Assert.True(MaxAbsoluteDifference(critic1.GetParameters(), critic2.GetParameters()) > 1e-12,
            "the twin critics collapsed onto identical parameters after one update.");

        double q1 = EvaluateQ(critic1, state, Action(1.0));
        double q2 = EvaluateQ(critic2, state, Action(1.0));
        Assert.True(Math.Abs(q1 - q2) > 1e-12,
            $"both critics returned the same Q ({q1:R}); they are not independent estimates.");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Sac_critics_rank_the_rewarded_action_above_the_punished_one()
    {
        using var agent = CreateAgent(seed: 33, out var state);
        var critic1 = Network(agent, "_critic1");

        var good = Action(1.0);
        var bad = Action(-1.0);
        double gapBefore = EvaluateQ(critic1, state, good) - EvaluateQ(critic1, state, bad);

        StoreBanditExperience(agent, state, pairs: 32);
        var losses = new List<double>();
        for (int i = 0; i < 200; i++)
        {
            losses.Add(agent.Train());
        }

        double qGood = EvaluateQ(critic1, state, good);
        double qBad = EvaluateQ(critic1, state, bad);

        Assert.True(qGood > qBad,
            $"critic ranks the punished action at least as high as the rewarded one (Q(good)={qGood:F4}, Q(bad)={qBad:F4}).");
        Assert.True(qGood - qBad > gapBefore,
            $"the Q gap did not widen with training (before={gapBefore:F4}, after={qGood - qBad:F4}).");

        // The critic is regressing on terminal targets of exactly +1 and -1, so it should approach them.
        Assert.True(qGood > 0.0, $"Q(good)={qGood:F4} did not become positive despite a reward of +1.");
        Assert.True(qBad < 0.0, $"Q(bad)={qBad:F4} did not become negative despite a reward of -1.");

        // Critic loss must fall: average of the last fifth below the average of the first fifth.
        int window = Math.Max(1, losses.Count / 5);
        double first = 0.0;
        double last = 0.0;
        for (int i = 0; i < window; i++)
        {
            first += losses[i];
            last += losses[losses.Count - 1 - i];
        }

        Assert.True(last / window < first / window,
            $"training loss did not decrease (first {first / window:F4} -> last {last / window:F4}).");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Sac_policy_follows_the_reward_not_its_own_past_actions()
    {
        // Two identically-seeded agents trained on MIRRORED rewards. Both see exactly the same two actions
        // (+1 and -1) the same number of times; only the sign of the reward attached to them differs. An
        // agent that regresses onto the actions it took — what the pre-fix code did — converges to their
        // mean (0) in BOTH runs and cannot separate them. Only an agent that reads the reward can.
        double Learn(double rewardedAction, int seed)
        {
            using var agent = CreateAgent(seed, out var state);
            for (int i = 0; i < 32; i++)
            {
                agent.StoreExperience(state, Action(rewardedAction), 1.0, state, done: true);
                agent.StoreExperience(state, Action(-rewardedAction), -1.0, state, done: true);
            }

            for (int i = 0; i < 200; i++)
            {
                agent.Train();
            }

            return agent.SelectAction(state, training: false)[0];
        }

        double rewardingPlusOne = Learn(1.0, seed: 34);
        double rewardingMinusOne = Learn(-1.0, seed: 34);

        Assert.True(rewardingPlusOne > rewardingMinusOne + 0.2,
            $"the policy did not follow the reward: rewarding +1 produced {rewardingPlusOne:F4}, "
            + $"rewarding -1 produced {rewardingMinusOne:F4} (identical data, mirrored reward).");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Target_critics_follow_the_online_critics_through_tau()
    {
        using var agent = CreateAgent(seed: 35, out var state);
        var critic1 = Network(agent, "_critic1");
        var target1 = Network(agent, "_targetCritic1");

        // Constructed by a hard sync, so target and online start identical.
        Assert.True(MaxAbsoluteDifference(critic1.GetParameters(), target1.GetParameters()) < 1e-12,
            "target critic did not start as a copy of its online critic.");

        var targetBefore = target1.GetParameters().Clone();
        StoreBanditExperience(agent, state, pairs: 16);
        for (int i = 0; i < 10; i++)
        {
            agent.Train();
        }

        double targetMoved = MaxAbsoluteDifference(targetBefore, target1.GetParameters());
        double onlineVersusTarget = MaxAbsoluteDifference(critic1.GetParameters(), target1.GetParameters());

        // Pre-fix the soft update was an empty method, so the target never moved at all.
        Assert.True(targetMoved > 1e-12, "the target critic never moved: the soft update does nothing.");

        // But it must LAG: a target that tracks exactly is not a target network.
        Assert.True(onlineVersusTarget > 1e-12,
            "the target critic is identical to the online critic; tau is being ignored.");
    }
}
