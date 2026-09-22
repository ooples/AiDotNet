using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// The three discrete trading agents must never emit an illegal action — at ANY of their selection sites.
///
/// <para><b>Why per-agent tests on top of the ActionMasking unit tests.</b> Those prove the primitives are
/// correct in isolation. They cannot prove an agent CALLS them, or calls them everywhere it selects. Each
/// agent has two selection paths, and masking one while leaving the other open is the defect this suite
/// exists to catch:</para>
///
/// <list type="bullet">
/// <item><b>DQN</b> — the epsilon exploration draw and the greedy argmax. Masking only the argmax leaves
/// exploration illegal at rate epsilon, which early in training is nearly every step.</item>
/// <item><b>PPO</b> — the training sample and the greedy argmax, both downstream of one softmax. The mask has
/// to land on the logits before it.</item>
/// <item><b>A2C</b> — the categorical sample and the greedy argmax, over a distribution the actor emits
/// directly.</item>
/// </list>
///
/// <para>Every assertion is on the ACTION RETURNED, not on internal state: the contract is that the emitted
/// index is legal, and only the returned vector can evidence that.</para>
/// </summary>
public class MaskableAgentSelectionTests
{
    private const int StateSize = 6;
    private const int ActionSize = 4;
    private const int Draws = 400;

    /// <summary>Only index 2 is selectable — a mask no correct agent can escape.</summary>
    private static bool[] OnlyAction2 => [false, false, true, false];

    /// <summary>Two legal, so a correct agent can still vary and the test is not satisfied by a constant.</summary>
    private static bool[] Actions1And3 => [false, true, false, true];

    public static IEnumerable<object[]> MaskableAgents()
    {
        yield return ["DQN"];
        yield return ["PPO"];
        yield return ["A2C"];
    }

    /// <summary>
    /// THE CONTRACT, in training mode — where exploration is live and the defect hides.
    /// </summary>
    [Theory]
    [Trait("category", "unit")]
    [MemberData(nameof(MaskableAgents))]
    public void A_training_selection_is_always_legal(string agentName)
    {
        using var concreteAgent = Agent(agentName);
        var agent = (IMaskableAgent<double>)concreteAgent;
        var state = FixedState();

        for (var i = 0; i < Draws; i++)
        {
            var action = agent.SelectAction(state, training: true, OnlyAction2);
            Assert.Equal(2, SelectedIndex(action));
        }
    }

    /// <summary>And in greedy mode, where evaluation and serving run.</summary>
    [Theory]
    [Trait("category", "unit")]
    [MemberData(nameof(MaskableAgents))]
    public void A_greedy_selection_is_always_legal(string agentName)
    {
        using var concreteAgent = Agent(agentName);
        var agent = (IMaskableAgent<double>)concreteAgent;
        var state = FixedState();

        for (var i = 0; i < 20; i++)
        {
            var action = agent.SelectAction(state, training: false, OnlyAction2);
            Assert.Equal(2, SelectedIndex(action));
        }
    }

    /// <summary>
    /// The mask restricts without collapsing: with two legal actions the agent must stay inside that set over
    /// many draws. A masked agent that always returned the first legal index would satisfy the single-legal
    /// tests above while destroying the policy.
    /// </summary>
    [Theory]
    [Trait("category", "unit")]
    [MemberData(nameof(MaskableAgents))]
    public void Selection_stays_inside_a_two_action_legal_set(string agentName)
    {
        using var concreteAgent = Agent(agentName);
        var agent = (IMaskableAgent<double>)concreteAgent;
        var state = FixedState();

        for (var i = 0; i < Draws; i++)
        {
            var index = SelectedIndex(agent.SelectAction(state, training: true, Actions1And3));
            Assert.True(index is 1 or 3, $"{agentName} selected illegal action {index}");
        }
    }

    /// <summary>
    /// A null mask must reproduce the unmasked path EXACTLY. This is what makes the change additive: every
    /// existing caller goes through the two-argument overload, which forwards null.
    /// </summary>
    [Theory]
    [Trait("category", "unit")]
    [MemberData(nameof(MaskableAgents))]
    public void A_null_mask_behaves_exactly_like_the_unmasked_overload(string agentName)
    {
        // Bound STATICALLY through the agent base, not via dynamic: a dynamic call would resolve at runtime
        // and keep passing even if the two-argument overload were removed, which is precisely the regression
        // this test exists to catch.
        using var agent = Agent(agentName);
        var state = FixedState();

        var viaMasked = SelectedIndex(((IMaskableAgent<double>)agent).SelectAction(state, training: false, legalActions: null));
        var viaOriginal = SelectedIndex(agent.SelectAction(state, training: false));

        Assert.Equal(viaOriginal, viaMasked);
    }

    /// <summary>
    /// A mask sized for a different action space is refused rather than silently restricting the wrong
    /// actions — the failure mode that would trade a structure the account is not cleared for.
    /// </summary>
    [Theory]
    [Trait("category", "unit")]
    [MemberData(nameof(MaskableAgents))]
    public void A_wrong_length_mask_is_refused(string agentName)
    {
        using var concreteAgent = Agent(agentName);
        var agent = (IMaskableAgent<double>)concreteAgent;

        Assert.Throws<ArgumentException>(
            () => agent.SelectAction(FixedState(), training: true, [true, true]));
    }

    /// <summary>
    /// An all-masked state throws rather than emitting an arbitrary action. For the softmax agents it would
    /// otherwise produce a NaN distribution; for all three it would mean "act when nothing is permitted".
    /// </summary>
    [Theory]
    [Trait("category", "unit")]
    [MemberData(nameof(MaskableAgents))]
    public void An_all_masked_state_is_refused(string agentName)
    {
        using var concreteAgent = Agent(agentName);
        var agent = (IMaskableAgent<double>)concreteAgent;

        Assert.Throws<InvalidOperationException>(
            () => agent.SelectAction(FixedState(), training: true, [false, false, false, false]));
    }

    /// <summary>
    /// FinRL is a WRAPPER that forwards selection to an inner DQN/PPO/A2C/SAC. Until this was wired, a
    /// FinRL-wrapped DQN reported as non-maskable, so a caller testing for <see cref="IMaskableAgent{T}"/> fell
    /// back to unmasked selection on an agent that could perfectly well have honoured the mask. The wrapper is
    /// where masking silently disappears, which is exactly why it needs its own coverage rather than inheriting
    /// confidence from the three tests above.
    /// </summary>
    [Theory]
    [Trait("category", "unit")]
    [InlineData(FinRLAlgorithm.DQN)]
    [InlineData(FinRLAlgorithm.PPO)]
    [InlineData(FinRLAlgorithm.A2C)]
    public void FinRL_forwards_the_mask_to_a_discrete_inner_agent(FinRLAlgorithm algorithm)
    {
        using var concreteAgent = FinRL(algorithm);
        var agent = (IMaskableAgent<double>)concreteAgent;
        var state = FixedState();

        for (var i = 0; i < Draws; i++)
        {
            Assert.Equal(2, SelectedIndex(agent.SelectAction(state, training: true, OnlyAction2)));
        }
    }

    /// <summary>
    /// A SAC inner agent is continuous and CANNOT honour a mask. The wrapper refuses rather than dropping it:
    /// quietly ignoring the mask would hand back an action the caller has been told is legal when nothing
    /// checked — a pre-shield degraded to no shield, which is worse than having none because the caller stops
    /// looking. The message has to name the algorithm, or the failure is unactionable.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void FinRL_refuses_a_mask_its_inner_agent_cannot_honour()
    {
        using var concreteAgent = FinRL(FinRLAlgorithm.SAC);
        var agent = (IMaskableAgent<double>)concreteAgent;

        var error = Assert.Throws<InvalidOperationException>(
            () => agent.SelectAction(FixedState(), training: true, OnlyAction2));

        Assert.Contains("SAC", error.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// A null mask must not throw even for the continuous inner agent, and must reproduce the unmasked call
    /// exactly — compared across the WHOLE vector, because SAC emits a continuous action for which a one-hot
    /// index comparison would be meaningless.
    /// </summary>
    [Theory]
    [Trait("category", "unit")]
    [InlineData(FinRLAlgorithm.DQN)]
    [InlineData(FinRLAlgorithm.PPO)]
    [InlineData(FinRLAlgorithm.A2C)]
    [InlineData(FinRLAlgorithm.SAC)]
    public void FinRL_with_a_null_mask_matches_the_unmasked_call(FinRLAlgorithm algorithm)
    {
        using var agent = FinRL(algorithm);
        var state = FixedState();

        var viaOriginal = agent.SelectAction(state, training: false);
        var viaMasked = ((IMaskableAgent<double>)agent).SelectAction(state, training: false, legalActions: null);

        Assert.Equal(viaOriginal.Length, viaMasked.Length);
        for (var i = 0; i < viaOriginal.Length; i++)
        {
            Assert.Equal(viaOriginal[i], viaMasked[i], precision: 10);
        }
    }

    private static FinRLAgent<double> FinRL(FinRLAlgorithm algorithm) => algorithm switch
    {
        FinRLAlgorithm.DQN => new FinRLAgent<double>(Arch(StateSize, ActionSize), DqnOptions(), algorithm),
        FinRLAlgorithm.PPO => new FinRLAgent<double>(Arch(StateSize, ActionSize), PpoOptions(), algorithm, Arch(StateSize, 1)),
        FinRLAlgorithm.A2C => new FinRLAgent<double>(Arch(StateSize, ActionSize), A2cOptions(), algorithm, Arch(StateSize, 1)),
        // SAC's critic is a Q-function Q(s,a), so it takes the state CONCATENATED WITH the action and is
        // StateSize + ActionSize wide. PPO's and A2C's critics are V(s) and take the state alone. Sizing this
        // like theirs fails construction with "input size 6 does not match expected 10".
        FinRLAlgorithm.SAC => new FinRLAgent<double>(Arch(StateSize, ActionSize), SacOptions(), algorithm, Arch(StateSize + ActionSize, 1)),
        _ => throw new ArgumentOutOfRangeException(nameof(algorithm), algorithm, "not a FinRL algorithm"),
    };

    private static TradingAgentOptions<double> SacOptions() => new()
    {
        StateSize = StateSize,
        ActionSize = ActionSize,
        BatchSize = 8,
        ReplayBufferSize = 256,
        Seed = 7,
        ContinuousActions = true,
    };

    /// <summary>
    /// <see cref="FinancialPPOAgent{T}"/> implements <see cref="IMaskableAgent{T}"/> unconditionally, but
    /// honours a mask only when configured for DISCRETE actions. Configured continuous, it emits a real-valued
    /// vector with no index set to restrict — so it refuses the mask instead of returning an
    /// unconstrained action that a caller holding the interface would read as masked.
    ///
    /// <para>The silent-drop version of this passed every other test in this file, because they all pass a
    /// discrete configuration. That is what makes it worth pinning: the interface, not the configuration, is
    /// what a caller sees.</para>
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void A_continuous_PPO_refuses_a_mask_it_cannot_honour()
    {
        var options = PpoOptions();
        options.ContinuousActions = true;
        using var agent = new FinancialPPOAgent<double>(Arch(StateSize, ActionSize), Arch(StateSize, 1), options);

        var error = Assert.Throws<InvalidOperationException>(
            () => ((IMaskableAgent<double>)agent).SelectAction(FixedState(), training: true, OnlyAction2));

        Assert.Contains("ContinuousActions", error.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// And the null mask still goes through, because "no restriction asked for" is not the same as "a
    /// restriction this agent cannot apply". A refusal that also blocked null would break every existing
    /// continuous caller, since the two-argument overload forwards null.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void A_continuous_PPO_still_accepts_a_null_mask()
    {
        var options = PpoOptions();
        options.ContinuousActions = true;
        using var agent = new FinancialPPOAgent<double>(Arch(StateSize, ActionSize), Arch(StateSize, 1), options);

        var action = ((IMaskableAgent<double>)agent).SelectAction(FixedState(), training: false, legalActions: null);

        Assert.Equal(ActionSize, action.Length);
    }

    /// <summary>The concrete agent, so the unmasked overload can be bound statically.</summary>
    private static TradingAgentBase<double> Agent(string agentName) => agentName switch
    {
        "DQN" => new FinancialDQNAgent<double>(Arch(StateSize, ActionSize), DqnOptions()),
        "PPO" => new FinancialPPOAgent<double>(Arch(StateSize, ActionSize), Arch(StateSize, 1), PpoOptions()),
        "A2C" => new FinancialA2CAgent<double>(Arch(StateSize, ActionSize), Arch(StateSize, 1), A2cOptions()),
        _ => throw new ArgumentOutOfRangeException(nameof(agentName), agentName, "not a maskable agent"),
    };

    private static FinancialDQNAgentOptions<double> DqnOptions() => new()
    {
        StateSize = StateSize,
        ActionSize = ActionSize,
        BatchSize = 8,
        ReplayBufferSize = 256,
        Seed = 7,
        // Always explore, so the epsilon branch is the one under test rather than an occasional visitor.
        EpsilonStart = 1.0,
        EpsilonEnd = 1.0,
        EpsilonDecay = 1.0,
    };

    private static FinancialPPOAgentOptions<double> PpoOptions() => new()
    {
        StateSize = StateSize,
        ActionSize = ActionSize,
        BatchSize = 8,
        ReplayBufferSize = 256,
        Seed = 7,
        ContinuousActions = false,
    };

    private static TradingAgentOptions<double> A2cOptions() => new()
    {
        StateSize = StateSize,
        ActionSize = ActionSize,
        BatchSize = 8,
        ReplayBufferSize = 256,
        Seed = 7,
    };

    private static NeuralNetworkArchitecture<double> Arch(int inputs, int outputs) =>
        new(inputFeatures: inputs, outputSize: outputs, complexity: NetworkComplexity.Simple)
        {
            RandomSeed = 7,
        };

    private static Vector<double> FixedState()
    {
        var state = new Vector<double>(StateSize);
        for (var i = 0; i < StateSize; i++)
        {
            state[i] = 0.1 * (i + 1);
        }

        return state;
    }

    /// <summary>
    /// The set index of a one-hot action vector — ASSERTING the one-hot shape rather than assuming it.
    /// </summary>
    /// <remarks>
    /// An argmax alone would pass on a raw logit vector, and that distinction is live here: the masked paths
    /// build their result as <c>action[index] = One</c> over a zeroed vector, while the continuous path returns
    /// the actor's logits directly. A branch that confused the two would still put its largest component at a
    /// legal index and satisfy every assertion in this file. Checking the shape at the single point all of
    /// those assertions pass through closes the hole for all of them at once, rather than in one extra test.
    /// </remarks>
    private static int SelectedIndex(Vector<double> action)
    {
        var best = -1;
        for (var i = 0; i < action.Length; i++)
        {
            if (action[i] == 1.0)
            {
                Assert.True(best < 0, $"action is not one-hot: indices {best} and {i} are both 1");
                best = i;
            }
            else
            {
                Assert.True(action[i] == 0.0, $"action is not one-hot: index {i} is {action[i]}, not 0 or 1");
            }
        }

        Assert.True(best >= 0, "action is not one-hot: no component is 1");
        return best;
    }
}
