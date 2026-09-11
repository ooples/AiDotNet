using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// These agents could not learn a policy, for reasons independent of any environment they were given.
///
/// <para>DQN's behaviour policy was 100% uniform random for the entire run and its target network almost never
/// synced; A2C treated unbounded regression outputs as a probability distribution, so exploration collapsed
/// onto the first or last action; and SAC's exploration noise had a positive mean, so every exploratory action
/// was pushed one way. Each is asserted here directly rather than inferred from a learning curve, because a
/// learning curve produced by a uniformly random policy looks like noise either way.</para>
/// </summary>
public class TradingAgentLearningTests
{
    private const int StateSize = 10;
    private const int ActionSize = 3;

    /// <summary>The ACTOR: state in, one output per action.</summary>
    private static NeuralNetworkArchitecture<double> Actor() => Architecture(StateSize, ActionSize);

    /// <summary>The CRITIC: a value function, so ONE output. SAC's critic is a Q-network and takes the action
    /// alongside the state, which is why its input is state + action wide.</summary>
    private static NeuralNetworkArchitecture<double> Critic(bool takesAction = false) =>
        Architecture(takesAction ? StateSize + ActionSize : StateSize, 1);

    private static NeuralNetworkArchitecture<double> Architecture(int inputSize, int outputSize) =>
        new(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize) { RandomSeed = 271828 };

    private static Vector<double> State(int seed)
    {
        var random = new Random(seed);
        var state = new Vector<double>(StateSize);
        for (var i = 0; i < StateSize; i++)
        {
            state[i] = (random.NextDouble() * 2.0) - 1.0;
        }

        return state;
    }

    private static void TrainSteps(FinancialDQNAgent<double> agent, int steps)
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        for (var step = 0; step < steps; step++)
        {
            var state = State(step);
            var action = agent.SelectTradingAction(state, training: true);
            agent.StoreTradingExperience(
                state, action, numOps.FromDouble(0.01), State(step + 1), done: false, pnl: numOps.FromDouble(0.01));
            _ = agent.TrainOnExperiences();
        }
    }

    [Fact]
    public void Dqn_epsilon_decays_from_start_toward_end()
    {
        // THE DEFECT. SelectAction compared against TradingOptions.EpsilonStart, which defaults to 1.0 and was
        // never decayed - EpsilonEnd and EpsilonDecay were declared, validated against each other, and read by
        // nobody. The behaviour policy was uniform random for the whole run, so the network's own Q-values were
        // never once acted on during training.
        var options = new FinancialDQNAgentOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            EpsilonStart = 1.0,
            EpsilonEnd = 0.05,
            EpsilonDecay = 0.9,
            BatchSize = 4,
            TargetUpdateFrequency = 10,
        };

        using var agent = new FinancialDQNAgent<double>(Actor(), options);

        Assert.Equal(1.0, agent.Epsilon, 9);

        TrainSteps(agent, 40);

        Assert.True(agent.Epsilon < 1.0, $"epsilon never moved from EpsilonStart (still {agent.Epsilon})");
        Assert.True(agent.Epsilon >= options.EpsilonEnd, $"epsilon fell below EpsilonEnd ({agent.Epsilon})");
    }

    [Theory]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(25)]
    public void Dqn_synchronises_its_target_network_on_an_exact_schedule(int frequency)
    {
        // THE FIX WITH THE LARGEST EFFECT ON LEARNING, AND NOTHING ASSERTED IT.
        //
        // The condition was `rng.Next(TargetUpdateFrequency) == 0`: a coin flip with probability 1/N per
        // step rather than a schedule, giving roughly 0.6 expected syncs across an entire run. The TD target
        // was therefore computed from the very network being updated in the same batch, which is the thing a
        // target network exists to prevent.
        //
        // TargetUpdateFrequency = 10 appeared in four tests as CONFIGURATION and no test observed what it
        // did, so a regression to the coin flip would have left every assertion in this file passing.
        //
        // Counting syncs alone would not discriminate either: a coin flip with p = 1/N also averages
        // steps/N syncs. What separates them is EXACTNESS - the deterministic condition satisfies the
        // identity below on every run, and a probabilistic one satisfies it only by chance.
        var options = new FinancialDQNAgentOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            BatchSize = 4,
            TargetUpdateFrequency = frequency,
        };

        using var agent = new FinancialDQNAgent<double>(Actor(), options);

        // Before any training the constructor has synced once, so the two networks start equal.
        Assert.Equal(1.0, agent.GetTradingMetrics()["TargetSyncCount"], 9);

        TrainSteps(agent, 120);

        var metrics = agent.GetTradingMetrics();
        var trainingSteps = (int)metrics["TrainingSteps"];
        var syncs = (int)metrics["TargetSyncCount"];

        Assert.True(trainingSteps > frequency, $"only {trainingSteps} training steps; too few to cross the schedule");
        Assert.Equal(1 + (trainingSteps / frequency), syncs);
    }

    [Fact]
    public void Dqn_epsilon_decay_of_one_holds_the_rate_instead_of_annealing()
    {
        // EpsilonDecay = 1.0 is a legitimate configuration meaning "hold the exploration rate". It must not be
        // mistaken for "no schedule configured" and annealed anyway - a fixed-epsilon run is how you isolate
        // whether exploration or learning is the thing that changed.
        var options = new FinancialDQNAgentOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            EpsilonStart = 0.3,
            EpsilonEnd = 0.01,
            EpsilonDecay = 1.0,
            BatchSize = 4,
            TargetUpdateFrequency = 10,
        };

        using var agent = new FinancialDQNAgent<double>(Actor(), options);
        TrainSteps(agent, 20);

        Assert.Equal(0.3, agent.Epsilon, 9);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    public void Dqn_direct_gradients_share_the_training_and_target_schedule(int frequency)
    {
        var options = new FinancialDQNAgentOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            BatchSize = 1,
            TargetUpdateFrequency = frequency,
            EpsilonStart = 0.8,
            EpsilonEnd = 0.01,
            EpsilonDecay = 0.5,
            Seed = 99,
        };
        using var agent = new FinancialDQNAgent<double>(Actor(), options);
        var onlineField = typeof(FinancialDQNAgent<double>).GetField("_qNetwork", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("The online-network regression probe must observe the actual online network.");
        var online = Assert.IsAssignableFrom<INeuralNetwork<double>>(onlineField.GetValue(agent));
        var targetField = typeof(FinancialDQNAgent<double>).GetField("_targetNetwork", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("The target-network regression probe must observe the actual target network.");
        var target = Assert.IsAssignableFrom<INeuralNetwork<double>>(targetField.GetValue(agent));
        var initial = online.GetParameters().ToArray();
        Assert.NotEmpty(initial);
        Assert.Equal(initial, target.GetParameters().ToArray());

        // A rejected update must not consume a training step or alter either network.
        Assert.Throws<ArgumentException>(() => agent.ApplyGradients(new Vector<double>(1), 0.01));
        Assert.Equal(0.0, agent.GetTradingMetrics()["TrainingSteps"]);
        Assert.Equal(1.0, agent.GetTradingMetrics()["TargetSyncCount"]);
        Assert.Equal(options.EpsilonStart, agent.Epsilon);
        Assert.Equal(initial, online.GetParameters().ToArray());
        Assert.Equal(initial, target.GetParameters().ToArray());

        for (var step = 1; step <= frequency + 1; step++)
        {
            var onlineBefore = online.GetParameters().ToArray();
            var targetBefore = target.GetParameters().ToArray();
            agent.ApplyGradients(new Vector<double>(Enumerable.Repeat(0.25, initial.Length).ToArray()), 0.01);
            var onlineAfter = online.GetParameters().ToArray();
            Assert.False(onlineBefore.SequenceEqual(onlineAfter), "the direct update must really change the online weights");
            var metrics = agent.GetTradingMetrics();
            Assert.Equal((double)step, metrics["TrainingSteps"]);
            Assert.Equal((double)(1 + step / frequency), metrics["TargetSyncCount"]);
            Assert.Equal(Math.Max(options.EpsilonEnd, options.EpsilonStart * Math.Pow(options.EpsilonDecay, step)), agent.Epsilon, 12);
            if (step % frequency == 0)
            {
                Assert.Equal(onlineAfter, target.GetParameters().ToArray());
            }
            else
            {
                Assert.Equal(targetBefore, target.GetParameters().ToArray());
                Assert.False(onlineAfter.SequenceEqual(target.GetParameters().ToArray()));
            }
        }

        // A replay update must advance the same schedule, not restart a second counter.
        var targetBeforeReplay = target.GetParameters().ToArray();
        TrainSteps(agent, 1);
        var completed = frequency + 2;
        Assert.Equal((double)completed, agent.GetTradingMetrics()["TrainingSteps"]);
        Assert.Equal((double)(1 + completed / frequency), agent.GetTradingMetrics()["TargetSyncCount"]);
        Assert.Equal(
            completed % frequency == 0 ? online.GetParameters().ToArray() : targetBeforeReplay,
            target.GetParameters().ToArray());
    }

    [Fact]
    public void Dqn_publishes_the_exploration_rate_in_its_trading_metrics()
    {
        // The Epsilon property serves anything holding the concrete agent. A harness that collects metrics
        // GENERICALLY - one row per agent, which is how this defect was found downstream - sees only the
        // dictionary, and there an agent that annealed and one that never explored look identical.
        var options = new FinancialDQNAgentOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            EpsilonStart = 0.8,
            EpsilonEnd = 0.02,
            EpsilonDecay = 0.5,
            BatchSize = 4,
            TargetUpdateFrequency = 10,
        };

        // Enough steps to fill the replay buffer and apply real updates: below BatchSize nothing trains, so
        // nothing decays, and the assertion below would be vacuous rather than wrong.
        using var agent = new FinancialDQNAgent<double>(Actor(), options);
        TrainSteps(agent, 20);

        var metrics = agent.GetTradingMetrics();
        Assert.True(metrics.ContainsKey("Epsilon"), "the agent does not publish its exploration rate");
        Assert.Equal(agent.Epsilon, Convert.ToDouble(metrics["Epsilon"]), 9);
        Assert.True(agent.Epsilon < 0.8, "epsilon did not decay, so the published value proves nothing");
    }

    [Fact]
    public void Dqn_epsilon_is_monotonically_non_increasing()
    {
        var options = new FinancialDQNAgentOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            EpsilonStart = 1.0,
            EpsilonEnd = 0.01,
            EpsilonDecay = 0.95,
            BatchSize = 4,
            TargetUpdateFrequency = 10,
        };

        using var agent = new FinancialDQNAgent<double>(Actor(), options);
        var curve = new List<double> { agent.Epsilon };
        for (var i = 0; i < 30; i++)
        {
            TrainSteps(agent, 1);
            curve.Add(agent.Epsilon);
        }

        for (var i = 1; i < curve.Count; i++)
        {
            Assert.True(curve[i] <= curve[i - 1], $"epsilon rose at step {i}: {curve[i - 1]} -> {curve[i]}");
        }

        Assert.True(curve[^1] < curve[0]);
    }

    [Fact]
    public void A2c_samples_from_a_real_distribution_rather_than_falling_through()
    {
        // THE DEFECT. The actor is built with NeuralNetworkTaskType.Regression, so its outputs are unbounded
        // reals with an identity activation - and they were fed straight into an inverse-CDF sampler as if they
        // were probabilities:
        //
        //     cumulative += probs[i];  if (r < cumulative) return i;   ... return LAST index
        //
        // A freshly initialised network produces small outputs that do not sum to 1, so the running cumulative
        // rarely reaches a uniform r in [0,1) and the loop FALLS OFF THE END - returning the last action almost
        // every time. A negative output additionally makes the cumulative non-monotonic, so an action can be
        // skipped entirely. Exploration was therefore all but deterministic on the last tier, and a trading
        // agent whose middle action is "hold" could not choose to do nothing.
        //
        // A FIXED state is used deliberately: varying it per sample lets different states land on different
        // actions and hides the fall-through behind that variation. With one state, sampling either explores
        // or it does not.
        var options = new FinancialA2CAgentOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            Seed = 4242,
        };

        using var agent = new FinancialA2CAgent<double>(Actor(), Critic(), options);

        // Fix the actual policy, not merely its initialization seed: zero logits give a known
        // uniform distribution. A direct-logit sampler would fall through to the last action.
        var parameters = agent.GetParameters();
        Assert.True(parameters.Length > 0);
        agent.SetParameters(new Vector<double>(new double[parameters.Length]));
        Assert.All(agent.GetParameters().ToArray(), value => Assert.Equal(0.0, value));
        var expectedRandom = RandomHelper.CreateSeededRandom(4242);

        var state = State(11);
        var counts = new int[ActionSize];
        const int samples = 600;
        for (var i = 0; i < samples; i++)
        {
            var action = agent.SelectTradingAction(state, training: true);
            var expectedAction = (int)(expectedRandom.NextDouble() * ActionSize);
            Assert.Equal(1.0, action[expectedAction]);
            Assert.Equal(1.0, action.ToArray().Sum());
            for (var a = 0; a < action.Length; a++)
            {
                if (action[a] != 0.0)
                {
                    counts[a]++;
                }
            }
        }

        var share = counts.Select(c => (double)c / samples).ToArray();
        var dominant = share.Max();

        Assert.True(
            dominant < 0.95,
            $"one action took {dominant:P0} of {samples} draws from a SINGLE state "
            + $"({string.Join(", ", share.Select((v, i) => $"a{i}={v:P0}"))}) - that is the sampler falling "
            + "through, not sampling");

        // And every action must be reachable at all.
        Assert.All(counts, c => Assert.True(c > 0, $"an action was never chosen in {samples} draws"));
    }

    [Fact]
    public void Sac_exploration_noise_is_centred_on_the_policy()
    {
        // THE DEFECT. The noise was `NextDouble() * 0.1` - uniform on [0, 0.1), mean +0.05, never negative. For
        // an agent whose action is a signed position that means it explored only the long side, and the bias
        // does not average out over a run: it accumulates into the experience the critics learn from.
        var options = new FinancialSACAgentOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            Seed = 909,
        };

        using var agent = new FinancialSACAgent<double>(Actor(), Critic(takesAction: true), options);

        var state = State(7);
        var deterministic = agent.SelectTradingAction(state, training: false);

        var deltas = new List<double>();
        for (var i = 0; i < 800; i++)
        {
            var explored = agent.SelectTradingAction(state, training: true);
            for (var a = 0; a < explored.Length; a++)
            {
                deltas.Add(explored[a] - deterministic[a]);
            }
        }

        Assert.Contains(deltas, d => d < 0);   // it can explore DOWNWARD at all
        Assert.Contains(deltas, d => d > 0);

        // Symmetric: the mean perturbation is near zero, not near the old +0.05.
        var mean = deltas.Average();
        Assert.True(Math.Abs(mean) < 0.01, $"exploration noise has a mean of {mean:F4}; it must be centred on the policy");
    }
}
