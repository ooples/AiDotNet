using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// <see cref="FinancialDQNAgent{T}"/> never annealed its exploration rate.
///
/// <para><c>SelectAction</c> compared its random draw against <c>TradingOptions.EpsilonStart</c> — a constant
/// — so the behaviour policy stayed at its initial value for the entire run. At the shipped default of 1.0
/// that is a uniformly random action on EVERY training step: the Q-network learned from experience the agent
/// never acted on, and its learning curve was noise. <c>EpsilonEnd</c> and <c>EpsilonDecay</c> were declared
/// on <c>TradingAgentOptions</c>, were validated against each other in <c>Validate()</c>, were plumbed through
/// <c>TradingAgentBase.CreateBaseOptions</c> — and were read by nothing.</para>
///
/// <para><b>These tests assert on BEHAVIOUR, not on the field.</b> An earlier draft asserted only on
/// <c>GetMetrics()["Epsilon"]</c> and passed 5/5 against deliberately re-broken code, because the metric
/// reports the field while <c>SelectAction</c> read the option — the two are independent. The decisive
/// assertions below drive <c>SelectAction(training: true)</c> and measure how often it returns the greedy
/// action. With <c>EpsilonStart = 1.0</c> and the annealed rate at a 0.01 floor, reading the option gives a
/// uniform spread over the action space and reading the annealed rate gives ~99% greedy. Nothing else
/// separates the two.</para>
/// </summary>
public sealed class FinancialDqnExplorationScheduleTests
{
    private const int StateSize = 4;
    private const int ActionSize = 3;
    private const int BatchSize = 8;
    private const int Draws = 600;

    /// <summary>
    /// THE DEFECT, stated as behaviour. After annealing to the floor the agent must act GREEDILY almost
    /// always. Reading EpsilonStart (1.0) instead makes every draw random, so the greedy share collapses to
    /// roughly 1/ActionSize.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void After_annealing_the_training_policy_is_almost_always_greedy()
    {
        // Start 1.0 so the OPTION says "always explore"; Decay 0.01 drives the ANNEALED rate to the 0.01
        // floor within two updates. The two now disagree by a factor of 100.
        var agent = Agent(epsilonStart: 1.0, epsilonEnd: 0.01, epsilonDecay: 0.01);
        DriveTraining(agent, updates: 4);

        var state = FixedState();
        var greedy = agent.SelectAction(state, training: false);
        var greedyShare = GreedyShare(agent, state, greedy);

        // Uniform-random over 3 actions would land near 0.33 (and does, under the defect).
        Assert.True(greedyShare > 0.90,
            $"training policy was greedy only {greedyShare:P0} of {Draws} draws — exploration is not "
            + "annealing, so SelectAction is reading EpsilonStart rather than the decayed rate.");
    }

    /// <summary>
    /// The control that makes the test above meaningful: with the rate genuinely HIGH, the same measurement
    /// must show a near-uniform spread. Without this, a SelectAction that ignored epsilon entirely and always
    /// went greedy would also pass.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void A_high_exploration_rate_really_does_produce_random_actions()
    {
        // Start == End means the anneal has nowhere to go: the rate stays at 1.0 by both readings.
        var agent = Agent(epsilonStart: 1.0, epsilonEnd: 1.0, epsilonDecay: 1.0);
        DriveTraining(agent, updates: 4);

        var state = FixedState();
        var greedy = agent.SelectAction(state, training: false);
        var greedyShare = GreedyShare(agent, state, greedy);

        // ~1/3 by chance, plus the random draw sometimes picking the greedy action anyway.
        Assert.True(greedyShare < 0.60,
            $"greedy {greedyShare:P0} of {Draws} draws at a 1.0 exploration rate — SelectAction is not "
            + "exploring at all, so the measurement above proves nothing.");
    }

    /// <summary>The floor holds: exploration must not decay away to zero and freeze the policy.</summary>
    [Fact]
    [Trait("category", "unit")]
    public void Exploration_never_falls_below_the_configured_floor()
    {
        var agent = Agent(epsilonStart: 1.0, epsilonEnd: 0.25, epsilonDecay: 0.5);

        DriveTraining(agent, updates: 40); // 0.5^40 ~ 9e-13 without a floor.

        Assert.Equal(0.25, Epsilon(agent), 9);
    }

    /// <summary>
    /// A <c>Train()</c> that bails at the replay gate learned nothing, so exploration has not earned a
    /// reduction. Decaying there would anneal against wall-clock rather than against experience.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void A_train_call_that_does_not_update_does_not_decay()
    {
        var agent = Agent(epsilonStart: 1.0, epsilonEnd: 0.01, epsilonDecay: 0.5);

        StoreExperiences(agent, BatchSize - 1); // one short: every Train() returns at the gate.
        for (var i = 0; i < 5; i++)
        {
            agent.Train();
        }

        Assert.Equal(1.0, Epsilon(agent), 9);
    }

    /// <summary>
    /// The rate must be observable. GetTradingMetrics reports Sharpe, drawdown, cumulative return, win rate,
    /// trade count, portfolio value and initial capital — every one an OUTCOME, none of which separates a
    /// policy that learned from one acting at random. That is why this defect survived.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void The_exploration_rate_is_published_in_metrics()
    {
        var agent = Agent(epsilonStart: 0.7, epsilonEnd: 0.01, epsilonDecay: 0.99);

        Assert.True(agent.GetMetrics().ContainsKey("Epsilon"), "GetMetrics does not publish Epsilon");
        Assert.Equal(0.7, Epsilon(agent), 9);

        DriveTraining(agent, updates: 10);
        Assert.True(Epsilon(agent) < 0.7, "the published rate did not move while the agent trained");
    }

    /// <summary>
    /// Greedy (serving) selection must not consult epsilon at all — which is why the rate is training-only
    /// state that need not survive serialization. Every serving path passes training: false.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Greedy_selection_is_unaffected_by_the_exploration_rate()
    {
        var agent = Agent(epsilonStart: 1.0, epsilonEnd: 1.0, epsilonDecay: 1.0);
        var state = FixedState();

        var first = agent.SelectAction(state, training: false);
        for (var draw = 0; draw < 50; draw++)
        {
            var again = agent.SelectAction(state, training: false);
            for (var i = 0; i < first.Length; i++)
            {
                Assert.Equal(first[i], again[i], 9);
            }
        }
    }

    /// <summary>Share of TRAINING draws that returned the same action greedy selection returns.</summary>
    private static double GreedyShare(FinancialDQNAgent<double> agent, Vector<double> state, Vector<double> greedy)
    {
        var matches = 0;
        for (var draw = 0; draw < Draws; draw++)
        {
            var action = agent.SelectAction(state, training: true);
            if (SameAction(action, greedy))
            {
                matches++;
            }
        }

        return (double)matches / Draws;
    }

    private static bool SameAction(Vector<double> a, Vector<double> b)
    {
        for (var i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }

        return true;
    }

    private static Vector<double> FixedState() => new(new[] { 0.10, 0.20, 0.30, 0.40 });

    private static double Epsilon(FinancialDQNAgent<double> agent) => agent.GetMetrics()["Epsilon"];

    private static void DriveTraining(FinancialDQNAgent<double> agent, int updates)
    {
        StoreExperiences(agent, BatchSize * 2);
        for (var i = 0; i < updates; i++)
        {
            agent.Train();
        }
    }

    private static void StoreExperiences(FinancialDQNAgent<double> agent, int count)
    {
        for (var i = 0; i < count; i++)
        {
            var scale = (i % 7) * 0.1;
            var state = new Vector<double>(new[] { scale, scale + 0.1, scale + 0.2, scale + 0.3 });
            var next = new Vector<double>(new[] { scale + 0.05, scale + 0.15, scale + 0.25, scale + 0.35 });
            var action = new Vector<double>(ActionSize);
            action[i % ActionSize] = 1.0;
            agent.StoreExperience(state, action, reward: (i % 3) - 1.0, nextState: next, done: false);
        }
    }

    private static FinancialDQNAgent<double> Agent(double epsilonStart, double epsilonEnd, double epsilonDecay)
    {
        var options = new FinancialDQNAgentOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            BatchSize = BatchSize,
            ReplayBufferSize = 512,
            Seed = 7,
            // These tests are about the ANNEAL, not about the warmup gate. At the shipped default of
            // 1000 (capped to ReplayBufferSize) every Train() below returns at TradingAgentBase.IsInWarmup,
            // _updateCount stays 0, and CurrentEpsilon reports EpsilonStart forever -- which looks exactly
            // like the defect these tests exist to catch. The batch gate is still in force and is covered
            // on its own by A_train_call_that_does_not_update_does_not_decay.
            WarmupSteps = 0,
            EpsilonStart = epsilonStart,
            EpsilonEnd = epsilonEnd,
            EpsilonDecay = epsilonDecay,
        };

        return new FinancialDQNAgent<double>(CreateArchitecture(StateSize, ActionSize), options);
    }

    private static NeuralNetworkArchitecture<double> CreateArchitecture(int inputSize, int outputSize)
        => new(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: inputSize,
            outputSize: outputSize);
}
