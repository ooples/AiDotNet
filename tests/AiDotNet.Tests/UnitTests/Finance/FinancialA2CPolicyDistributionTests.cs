using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Regression tests for <see cref="FinancialA2CAgent{T}"/>'s policy distribution. The actor is a regression
/// network with a linear (identity) output, so its outputs are LOGITS — they can be negative and do not sum to
/// one. Sampling from them as if they were probabilities collapsed exploration onto an edge tier, and training
/// them by MSE towards the sampled one-hot action ignored the advantage entirely (a punished action was
/// reinforced exactly like a rewarded one).
/// </summary>
public sealed class FinancialA2CPolicyDistributionTests
{
    private const int StateSize = 4;
    private const int ActionSize = 3;

    private static FinancialA2CAgent<double> CreateZeroedAgent(int seed, out Vector<double> state)
    {
        var options = Options(A2C, StateSize, ActionSize, seed);
        options.BatchSize = 1;
        options.WarmupSteps = 0;
        options.EntropyCoefficient = 0.0;
        var agent = (FinancialA2CAgent<double>)Create(A2C, options);

        state = State(StateSize, salt: 1);
        agent.SelectAction(state, training: false); // materialize lazily-shaped layers
        // All-zero weights make every actor output (logit) exactly 0 for every state, so the policy
        // must be the uniform distribution and the critic must predict V(s) = 0.
        agent.SetParameters(new Vector<double>((int)agent.ParameterCount));
        return agent;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Training_actions_are_sampled_from_the_softmax_of_the_actor_logits()
    {
        using var agent = CreateZeroedAgent(seed: 11, out var state);

        const int draws = 3000;
        var counts = new int[ActionSize];
        for (int i = 0; i < draws; i++)
        {
            var action = agent.SelectAction(state, training: true);
            Assert.Equal(ActionSize, action.Length);
            counts[ArgMax(action)]++;
        }

        // softmax([0, 0, 0]) = [1/3, 1/3, 1/3]. Treating the raw zeros as probabilities never sampled the
        // first or the interior tier at all (the cumulative sum never exceeds the uniform draw).
        for (int a = 0; a < ActionSize; a++)
        {
            double frequency = counts[a] / (double)draws;
            Assert.True(frequency > 0.28 && frequency < 0.39,
                $"action {a} sampled with frequency {frequency:F3}; counts = [{string.Join(", ", counts)}]");
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Actor_update_moves_probability_away_from_an_action_with_negative_advantage()
    {
        using var agent = CreateZeroedAgent(seed: 3, out var state);
        const int punished = 1;

        // Terminal transition, reward -1, V(s) = 0  =>  advantage = -1 for the interior action.
        agent.StoreExperience(state, SampleActionForIndex(agent, state, punished), -1.0, state, done: true);
        agent.Train();

        // grad of -A * log softmax(z)[a] with A < 0 lowers z[a] and raises every other logit, so the
        // greedy action must leave the punished tier. MSE-to-one-hot raised z[a] instead.
        int greedy = ArgMax(agent.SelectAction(state, training: false));
        Assert.NotEqual(punished, greedy);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Actor_update_moves_probability_towards_an_action_with_positive_advantage()
    {
        using var agent = CreateZeroedAgent(seed: 4, out var state);
        const int rewarded = 2;

        agent.StoreExperience(state, SampleActionForIndex(agent, state, rewarded), 1.0, state, done: true);
        agent.Train();

        Assert.Equal(rewarded, ArgMax(agent.SelectAction(state, training: false)));
    }

    [Fact]
    [Trait("category", "unit")]
    public void Half_extreme_logits_preserve_finite_losses_gradients_and_rollout_readiness()
    {
        var actorLayer = new DenseLayer<Half>(2, (IActivationFunction<Half>)new IdentityActivation<Half>());
        var criticLayer = new DenseLayer<Half>(1, (IActivationFunction<Half>)new IdentityActivation<Half>());
        var actorArch = new NeuralNetworkArchitecture<Half>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 2);
        var criticArch = new NeuralNetworkArchitecture<Half>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        actorArch.Layers.Add(actorLayer);
        criticArch.Layers.Add(criticLayer);

        var options = new FinancialA2CAgentOptions<Half>
        {
            StateSize = 1,
            ActionSize = 2,
            HiddenLayers = Array.Empty<int>(),
            BatchSize = 2,
            NSteps = 2,
            WarmupSteps = 0,
            EntropyCoefficient = 0.01,
            LearningRate = (Half)0.05,
            InitialCapital = (Half)1000,
            Seed = 17,
        };

        using var agent = new FinancialA2CAgent<Half>(actorArch, criticArch, options);
        var state = new Vector<Half>(new[] { (Half)1 });
        agent.SelectAction(state, training: false);

        // Configure exact [0, -100] actor logits on state = [1] and V(s) = 0 on the critic.
        actorLayer.SetParameters(new Vector<Half>(new[] { (Half)0, (Half)(-100), (Half)0, (Half)0 }));
        criticLayer.SetParameters(new Vector<Half>(new[] { (Half)0, (Half)0 }));

        // Step 1 of 2: rollout readiness holds back autonomous training until BatchSize / NSteps is met.
        var firstAction = agent.SelectAction(state, training: true);
        Assert.Equal((Half)1, firstAction[0]);
        Assert.Equal((Half)0, firstAction[1]);
        agent.StoreExperience(state, firstAction, (Half)1, state, done: false);
        Assert.Equal((Half)0, agent.Train());

        // Step 2 of 2: rollout becomes ready and trains without 0 * (-Infinity) = NaN in Half.
        var secondAction = agent.SelectAction(state, training: true);
        agent.StoreExperience(state, secondAction, (Half)1, state, done: true);
        Half rolloutLoss = agent.Train();
        Assert.False(Half.IsNaN(rolloutLoss));
        Assert.False(Half.IsInfinity(rolloutLoss));
        Assert.All(agent.GetParameters().ToArray(), value =>
        {
            Assert.False(Half.IsNaN(value));
            Assert.False(Half.IsInfinity(value));
        });
        // Consumed rollout is drained after the update.
        Assert.Equal((Half)0, agent.Train());

        // Reset actor to [0, -100] and critic to 0, then apply a supervised reward to the -100 action:
        // stable TensorLogSoftmax + TensorWhere must keep logProb[1] = -100 finite and propagate a non-zero
        // gradient that raises logit[1] relative to logit[0].
        actorLayer.SetParameters(new Vector<Half>(new[] { (Half)0, (Half)(-100), (Half)0, (Half)0 }));
        criticLayer.SetParameters(new Vector<Half>(new[] { (Half)0, (Half)0 }));
        var beforeWeights = actorLayer.GetParameters().ToArray();

        agent.Train(state, new Vector<Half>(new[] { (Half)0, (Half)1 }));
        Assert.All(agent.GetMetrics().Values, metric =>
        {
            Assert.False(Half.IsNaN(metric));
            Assert.False(Half.IsInfinity(metric));
        });

        var afterWeights = actorLayer.GetParameters().ToArray();
        Assert.All(afterWeights, value =>
        {
            Assert.False(Half.IsNaN(value));
            Assert.False(Half.IsInfinity(value));
        });
        Assert.True((double)afterWeights[0] < (double)beforeWeights[0], $"Expected weight[0] to decrease from {beforeWeights[0]}, got {afterWeights[0]}.");
        Assert.True((double)afterWeights[3] > (double)beforeWeights[3], $"Expected bias[1] to increase from {beforeWeights[3]}, got {afterWeights[3]}.");
        double gapBefore = ((double)beforeWeights[1] + (double)beforeWeights[3]) - ((double)beforeWeights[0] + (double)beforeWeights[2]);
        double gapAfter = ((double)afterWeights[1] + (double)afterWeights[3]) - ((double)afterWeights[0] + (double)afterWeights[2]);
        Assert.True(gapAfter > gapBefore, $"Expected logit[1] - logit[0] to increase from {gapBefore}, got {gapAfter}.");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Half_masked_action_with_extreme_bias_is_excluded_and_receives_zero_gradient()
    {
        var actorLayer = new DenseLayer<Half>(3, (IActivationFunction<Half>)new IdentityActivation<Half>());
        var criticLayer = new DenseLayer<Half>(1, (IActivationFunction<Half>)new IdentityActivation<Half>());
        var actorArch = new NeuralNetworkArchitecture<Half>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 3);
        var criticArch = new NeuralNetworkArchitecture<Half>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        actorArch.Layers.Add(actorLayer);
        criticArch.Layers.Add(criticLayer);

        var options = new FinancialA2CAgentOptions<Half>
        {
            StateSize = 1,
            ActionSize = 3,
            HiddenLayers = Array.Empty<int>(),
            BatchSize = 1,
            NSteps = 1,
            WarmupSteps = 0,
            EntropyCoefficient = 0.01,
            LearningRate = (Half)0.1,
            InitialCapital = (Half)1000,
            Seed = 29,
        };

        using var agent = new FinancialA2CAgent<Half>(actorArch, criticArch, options);
        var state = new Vector<Half>(new[] { (Half)1 });
        agent.SelectAction(state, training: false);
        bool[] legalActions = { true, true, false };

        // Illegal action 2 has raw logit +50 (would dominate unmasked), while legal actions 0 and 1 have [0, -1].
        actorLayer.SetParameters(new Vector<Half>(new[]
        {
            (Half)0, (Half)(-1), (Half)50,
            (Half)0, (Half)0, (Half)0
        }));
        criticLayer.SetParameters(new Vector<Half>(new[] { (Half)0, (Half)0 }));

        for (int i = 0; i < 32; i++)
        {
            var evalAction = agent.SelectAction(state, training: false, legalActions);
            Assert.Equal((Half)0, evalAction[2]);
        }

        Vector<Half>? chosen = null;
        for (int draw = 0; draw < 128; draw++)
        {
            var candidate = agent.SelectAction(state, training: true, legalActions);
            Assert.Equal((Half)0, candidate[2]);
            if (candidate[0] == (Half)1)
            {
                chosen = candidate;
                break;
            }
        }

        Assert.NotNull(chosen);
        var beforeActor = actorLayer.GetParameters().ToArray();

        agent.StoreExperience(state, chosen, (Half)2, state, done: true);
        Half loss = agent.Train();

        Assert.False(Half.IsNaN(loss));
        Assert.False(Half.IsInfinity(loss));

        var afterActor = actorLayer.GetParameters().ToArray();
        Assert.All(afterActor, value =>
        {
            Assert.False(Half.IsNaN(value));
            Assert.False(Half.IsInfinity(value));
        });

        // Legal action 0 was rewarded with positive advantage, so legal weights move; masked action 2 gets zero gradient.
        double legalGapBefore = ((double)beforeActor[0] + (double)beforeActor[3]) - ((double)beforeActor[1] + (double)beforeActor[4]);
        double legalGapAfter = ((double)afterActor[0] + (double)afterActor[3]) - ((double)afterActor[1] + (double)afterActor[4]);
        Assert.True(legalGapAfter > legalGapBefore, $"Expected legal logit[0] - logit[1] to increase from {legalGapBefore}, got {legalGapAfter}.");
        Assert.Equal(beforeActor[2], afterActor[2]);
        Assert.Equal(beforeActor[5], afterActor[5]);
    }
}
