using System;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// A mask that reaches only the SELECTION is half a feature. PPO re-evaluates the policy at the state it
/// acted in, and if that re-evaluation is unrestricted while the action was drawn from the restricted
/// distribution, the importance ratio divides two different policies and the entropy bonus pays the actor
/// to spread probability onto actions the environment refuses.
///
/// <para>Nothing about that failure is visible from outside. Training runs, the loss is a plausible number,
/// and every emitted action is still legal because selection masks correctly — only the LEARNING is wrong.
/// So these tests assert on the loss PPO actually computed, which is the one observable that separates the
/// two cases.</para>
///
/// <para>Selection-side legality is covered by <see cref="MaskableAgentSelectionTests"/>; nothing here
/// restates it.</para>
/// </summary>
public class PpoMaskedUpdateTests
{
    private const int StateSize = 6;
    private const int ActionSize = 4;

    /// <summary>Eight steps: advantage normalisation is skipped on shorter rollouts.</summary>
    private const int RolloutLength = 8;

    /// <summary>One action legal, so the masked policy is a point mass.</summary>
    private static bool[] OneLegal() => [false, false, true, false];

    private static bool[] TwoLegal() => [false, true, false, true];

    private static bool[] NothingForbidden() => [true, true, true, true];

    /// <summary>
    /// THE INVARIANT. With exactly one action legal at every step the whole policy loss must be zero, and
    /// each term is zero for its own reason:
    ///
    /// <list type="bullet">
    /// <item>The masked distribution is a point mass, so the behaviour log-probability is log 1 = 0. The
    /// update re-derives it under the same mask against weights that have not moved yet, so the new
    /// log-probability is also 0 and the ratio is exactly 1 — the surrogate collapses to the advantage.</item>
    /// <item>Advantages are normalised across the rollout, so they sum to zero, and one epoch over one
    /// mini-batch means a single update sees all eight of them at once.</item>
    /// <item>The entropy of a point mass is zero, so the exploration bonus contributes nothing.</item>
    /// </list>
    ///
    /// <para>Drop the mask from the update and all three collapse together: the new log-probability becomes
    /// log p(a) over four near-uniform actions — about -1.39 — so the ratio lands near 0.25, clipping bites
    /// asymmetrically on positive and negative advantages, and the entropy term pays out on the order of
    /// log 4. The loss is then a sizeable number rather than zero, which is why the tolerance below can be
    /// tight without being brittle.</para>
    ///
    /// <para><c>ValueCoefficient = 0</c> so the returned figure is the POLICY loss alone. The critic still
    /// trains; it just does not contribute to what is asserted.</para>
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void A_rollout_under_one_legal_action_updates_at_a_ratio_of_exactly_one()
    {
        using var agent = Agent();
        var loss = RunRollout(agent, _ => OneLegal());

        Assert.Equal(0.0, loss, tolerance: 1e-6);
    }

    /// <summary>
    /// The mask is copied on the way in, so a caller reusing one array across steps cannot retroactively
    /// rewrite what was legal at a step already stored.
    ///
    /// <para>Every array handed to a selection here is overwritten to all-legal immediately afterwards.
    /// Retaining the caller's reference instead of a copy would make each stored step read back as
    /// unrestricted, and the invariant above would break — which is precisely what this asserts cannot
    /// happen.</para>
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void A_caller_reusing_its_mask_array_cannot_rewrite_a_stored_step()
    {
        using var agent = Agent();
        var loss = RunRollout(
            agent,
            _ => OneLegal(),
            afterSelect: mask =>
            {
                for (var i = 0; i < mask!.Length; i++)
                {
                    mask[i] = true;
                }
            });

        Assert.Equal(0.0, loss, tolerance: 1e-6);
    }

    /// <summary>
    /// A step whose mask cannot be recovered is refused, not guessed.
    ///
    /// <para>The selection cache is the only record of which actions were legal. When
    /// <c>StoreExperience</c> cannot match its arguments to the preceding selection, the two honest options
    /// are to refuse or to take the behaviour log-probability from the unrestricted distribution — and the
    /// second is the very defect this file exists to prevent, arriving silently through a side door.</para>
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void A_step_whose_mask_cannot_be_recovered_is_refused()
    {
        using var agent = Agent();
        var action = ((IMaskableAgent<double>)agent).SelectAction(State(0), training: true, OneLegal());

        var error = Assert.Throws<InvalidOperationException>(
            () => agent.StoreExperience(State(99), action, 0.5, State(100), done: true));

        Assert.Contains("mask", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// A mask that forbids nothing must be indistinguishable from no mask at all: same seed, same
    /// trajectory, same loss.
    ///
    /// <para>This is the regression guard on the masking path itself. It proves the added bias tensor is a
    /// true no-op when every action is legal, so the agents and callers that never mask pay nothing for the
    /// feature.</para>
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void A_mask_that_forbids_nothing_trains_identically_to_no_mask()
    {
        using var maskedAgent = Agent();
        using var unmaskedAgent = Agent();
        var masked = RunRollout(maskedAgent, _ => NothingForbidden());
        var unmasked = RunRollout(unmaskedAgent, _ => null);

        Assert.Equal(unmasked, masked, tolerance: 1e-9);
    }

    /// <summary>
    /// Negative infinity reaches the loss as an additive logit bias, so it has to survive the gradient tape
    /// and not merely the forward pass.
    ///
    /// <para>The softmax drives the masked entries to exactly zero probability, the log-probability helper's
    /// 1e-8 floor keeps the following log finite, and both helpers then multiply those entries by zero. If
    /// any link in that chain were wrong the loss would come back NaN rather than merely incorrect — and it
    /// would do so on the SECOND round, once poisoned weights had been written, which is why this trains
    /// repeatedly instead of once.</para>
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Repeated_training_under_a_partial_mask_stays_finite()
    {
        using var agent = Agent();

        for (var round = 0; round < 3; round++)
        {
            var loss = RunRollout(agent, _ => TwoLegal());

            // Not double.IsFinite: the test project also targets net471, where it does not exist.
            Assert.True(
                !double.IsNaN(loss) && !double.IsInfinity(loss),
                $"round {round} produced a non-finite loss ({loss}): the masked logits did not survive the tape");
        }
    }

    /// <summary>
    /// One rollout of <see cref="RolloutLength"/> masked steps, followed by the update that consumes it.
    /// </summary>
    /// <param name="maskFor">The mask in force at a given step, or <see langword="null"/> for no mask.</param>
    /// <param name="afterSelect">
    /// Runs between selection and storage, so a test can disturb the array the caller passed.
    /// </param>
    private static double RunRollout(
        FinancialPPOAgent<double> agent,
        Func<int, bool[]?> maskFor,
        Action<bool[]?>? afterSelect = null)
    {
        for (var step = 0; step < RolloutLength; step++)
        {
            var state = State(step);
            var mask = maskFor(step);

            var action = ((IMaskableAgent<double>)agent).SelectAction(state, training: true, mask);
            afterSelect?.Invoke(mask);

            agent.StoreExperience(state, action, Reward(step), State(step + 1), done: step == RolloutLength - 1);
        }

        return agent.Train();
    }

    private static FinancialPPOAgent<double> Agent()
    {
        var options = new FinancialPPOAgentOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            BatchSize = RolloutLength,
            ReplayBufferSize = 256,
            Seed = 7,
            ContinuousActions = false,

            // One epoch over one mini-batch: a single update across the whole rollout, so the ratio is read
            // against the weights the actions were drawn from and the normalised advantages still sum to zero.
            NumEpochs = 1,
            NumMiniBatches = 1,

            // So Train() returns the policy loss alone.
            ValueCoefficient = 0.0,
        };

        return new FinancialPPOAgent<double>(Arch(StateSize, ActionSize), Arch(StateSize, 1), options);
    }

    private static NeuralNetworkArchitecture<double> Arch(int inputs, int outputs) =>
        new(inputFeatures: inputs, outputSize: outputs, complexity: NetworkComplexity.Simple)
        {
            RandomSeed = 7,
        };

    /// <summary>Distinct per step, so the rollout is eight states rather than one state eight times.</summary>
    private static Vector<double> State(int step)
    {
        var state = new Vector<double>(StateSize);
        for (var i = 0; i < StateSize; i++)
        {
            state[i] = (0.1 * (i + 1)) + (0.05 * step);
        }

        return state;
    }

    /// <summary>Varied, so advantages have a non-zero spread and normalisation actually does something.</summary>
    private static double Reward(int step) => 0.5 - (0.25 * (step % 3));
}
