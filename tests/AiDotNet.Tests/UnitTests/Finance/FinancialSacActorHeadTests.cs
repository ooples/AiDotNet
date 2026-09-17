using System;
using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models.Options;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// <see cref="FinancialSACAgent{T}"/> supports two policy heads. The default keeps the actor's output width
/// at <c>ActionSize</c> and learns one spread shared by every state; the opt-in
/// <see cref="SacActorHead.StateConditionedGaussian"/> is the head from the SAC paper, where the network
/// predicts the spread FROM the state and the actor is widened to <c>2 * ActionSize</c>.
/// </summary>
/// <remarks>
/// Both heads are exercised here, not just the default: the paper's head has to actually learn a
/// state-dependent spread, and both have to still learn from reward.
/// </remarks>
public sealed class FinancialSacActorHeadTests
{
    private const int StateSize = 4;
    private const int ActionSize = 1;

    private static Vector<double> Action(double value)
    {
        var v = new Vector<double>(ActionSize);
        v[0] = value;
        return v;
    }

    private static FinancialSACAgent<double> CreateAgent(SacActorHead head, int seed)
    {
        var options = (FinancialSACAgentOptions<double>)Options(Sac, StateSize, ActionSize, seed);
        options.BatchSize = 8;
        options.WarmupSteps = 0;
        options.LearningRate = 0.01;
        options.Tau = 0.05;
        options.AutoTuneAlpha = false;
        options.SACAlpha = 0.2;
        options.ActorHead = head;

        int actorOutputs = head == SacActorHead.StateConditionedGaussian ? ActionSize * 2 : ActionSize;
        return (FinancialSACAgent<double>)Create(Sac, options, Arch(StateSize, actorOutputs));
    }

    [Fact]
    [Trait("category", "unit")]
    public void The_default_head_is_state_independent_and_keeps_the_historical_actor_width()
    {
        var options = (FinancialSACAgentOptions<double>)Options(Sac, StateSize, ActionSize, seed: 81);
        Assert.Equal(SacActorHead.StateIndependentLogStd, options.ActorHead);

        // An ActionSize-wide actor architecture — what every existing caller builds — still works.
        using var agent = (FinancialSACAgent<double>)Create(Sac, options, Arch(StateSize, ActionSize));
        Assert.Equal(SacActorHead.StateIndependentLogStd, agent.ActorHead);

        var a = agent.PolicyStandardDeviationsFor(State(StateSize, salt: 1));
        var b = agent.PolicyStandardDeviationsFor(State(StateSize, salt: 2));
        Assert.Equal(a[0], b[0], 12); // state-independent by construction
    }

    [Fact]
    [Trait("category", "unit")]
    public void The_state_conditioned_head_rejects_an_actor_architecture_of_the_old_width()
    {
        var options = (FinancialSACAgentOptions<double>)Options(Sac, StateSize, ActionSize, seed: 82);
        options.ActorHead = SacActorHead.StateConditionedGaussian;

        var error = Assert.Throws<ArgumentException>(
            () => Create(Sac, options, Arch(StateSize, ActionSize)));

        Assert.Contains("StateConditionedGaussian", error.Message);
        Assert.Contains("2 * ActionSize", error.Message);
    }

    [Theory]
    [InlineData(SacActorHead.StateIndependentLogStd)]
    [InlineData(SacActorHead.StateConditionedGaussian)]
    [Trait("category", "unit")]
    public void Both_heads_follow_the_reward_rather_than_their_own_past_actions(SacActorHead head)
    {
        // The mirrored-reward control from the learning tests, run on each head: identical data, only the
        // sign of the reward flipped, so an agent that regresses onto its own actions cannot separate them.
        double Learn(double rewardedAction)
        {
            using var agent = CreateAgent(head, seed: 83);
            var state = State(StateSize, salt: 1);
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

        double rewardingPlusOne = Learn(1.0);
        double rewardingMinusOne = Learn(-1.0);

        Assert.True(rewardingPlusOne > rewardingMinusOne + 0.2,
            $"{head}: policy did not follow the reward — rewarding +1 gave {rewardingPlusOne:F4}, "
            + $"rewarding -1 gave {rewardingMinusOne:F4}.");
    }

    [Fact]
    [Trait("category", "unit")]
    public void The_state_conditioned_head_learns_a_wider_spread_where_the_action_does_not_matter()
    {
        // Two regions of the state space:
        //   CLEAN  — the reward depends sharply on the action (+1 pays +1, -1 pays -1), so the critic's
        //            value surface is steep in the action and being uncertain is expensive.
        //   NOISY  — the reward is the same whatever the action (both pay 0), so the value surface is flat
        //            in the action and there is nothing to be gained by committing.
        // A state-conditioned spread must end up WIDER in the noisy region than in the clean one. A
        // state-independent spread cannot express that difference at all.
        using var agent = CreateAgent(SacActorHead.StateConditionedGaussian, seed: 84);

        var clean = State(StateSize, salt: 1);
        var noisy = State(StateSize, salt: 9);

        for (int i = 0; i < 48; i++)
        {
            agent.StoreExperience(clean, Action(1.0), 1.0, clean, done: true);
            agent.StoreExperience(clean, Action(-1.0), -1.0, clean, done: true);
            agent.StoreExperience(noisy, Action(1.0), 0.0, noisy, done: true);
            agent.StoreExperience(noisy, Action(-1.0), 0.0, noisy, done: true);
        }

        for (int i = 0; i < 300; i++)
        {
            agent.Train();
        }

        double noisySpread = agent.PolicyStandardDeviationsFor(noisy)[0];
        double cleanSpread = agent.PolicyStandardDeviationsFor(clean)[0];

        Assert.True(noisySpread > cleanSpread,
            $"the learned spread is not state-dependent: noisy region {noisySpread:F4} vs clean region "
            + $"{cleanSpread:F4} (a state-conditioned head must widen where the action does not matter).");
    }

    [Fact]
    [Trait("category", "unit")]
    public void The_state_independent_head_cannot_vary_its_spread_by_state()
    {
        // The counterpart of the test above, pinning the documented limitation of the default head so the
        // difference between the two options is not just an assertion in a doc comment.
        using var agent = CreateAgent(SacActorHead.StateIndependentLogStd, seed: 85);

        var clean = State(StateSize, salt: 1);
        var noisy = State(StateSize, salt: 9);

        for (int i = 0; i < 48; i++)
        {
            agent.StoreExperience(clean, Action(1.0), 1.0, clean, done: true);
            agent.StoreExperience(clean, Action(-1.0), -1.0, clean, done: true);
            agent.StoreExperience(noisy, Action(1.0), 0.0, noisy, done: true);
            agent.StoreExperience(noisy, Action(-1.0), 0.0, noisy, done: true);
        }

        for (int i = 0; i < 100; i++)
        {
            agent.Train();
        }

        Assert.Equal(
            agent.PolicyStandardDeviationsFor(noisy)[0],
            agent.PolicyStandardDeviationsFor(clean)[0],
            12);
    }
}
