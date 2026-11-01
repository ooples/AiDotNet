using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ReinforcementLearning;

/// <summary>
/// Chen et al. 2021, Decision Transformer: (return-to-go, state, action) tokens per timestep with a learned timestep
/// embedding, a causally masked GPT, actions predicted from state tokens, and return-conditioned acting (section 3,
/// appendix A.2) (#2397).
/// </summary>
public class DecisionTransformerPaperTests
{
    private static Vector<double> V(params double[] values) => new(values);

    private static DecisionTransformerOptions<double> Small(bool continuous = true) => new()
    {
        StateSize = 2, ActionSize = 2, EmbeddingDim = 16, NumLayers = 2, NumHeads = 2, ContextLength = 4,
        DropoutRate = 0.0, MaxEpisodeLength = 16, IsContinuous = continuous, Seed = 1,
    };

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheGymModel()
    {
        await Task.Yield();
        var o = new DecisionTransformerOptions<double>();
        Assert.Equal((3, 1, 128, 20), (o.NumLayers, o.NumHeads, o.EmbeddingDim, o.ContextLength));
        Assert.Equal(0.1, o.DropoutRate);
        Assert.Equal(100000, o.LearningRateWarmupSteps);
        Assert.Equal(1000, o.MaxEpisodeLength);
    }

    [Fact(Timeout = 60000)]
    public async Task Network_IsACausalGptOverReturnStateActionTokens()
    {
        await Task.Yield();
        var agent = new DecisionTransformerAgent<double>(Small());
        var network = agent.Network;
        Assert.Equal(2, network.Blocks.Count);
        Assert.All(network.Blocks, b => Assert.True(Assert.IsType<MultiHeadAttentionLayer<double>>(b.AttentionLayer).UseCausalMask));

        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(2);
        var input = new Tensor<double>(new[] { 4, network.StepWidth });
        for (int t = 0; t < 4; t++)
        {
            for (int j = 0; j < network.StepWidth - 1; j++) input[t, j] = rng.NextDouble() - 0.5;
            input[t, network.StepWidth - 1] = t;
        }
        var baseline = network.Predict(input);
        Assert.Equal(new[] { 4, 2 }, baseline.Shape.ToArray());

        // a_t is predicted from s_t: changing the action a_2 (a later token) leaves the predictions up to t = 2 alone and
        // changes the one at t = 3, which sees it.
        var changed = input.Clone();
        changed[2, 1 + 2] += 1.0;
        var output = network.Predict(changed);
        for (int t = 0; t <= 2; t++) for (int j = 0; j < 2; j++) Assert.Equal(baseline[t, j], output[t, j], 12);
        Assert.NotEqual(baseline[3, 0], output[3, 0], 9);

        // The return-to-go R_1 precedes s_1 and conditions its prediction.
        changed = input.Clone();
        changed[1, 0] += 1.0;
        output = network.Predict(changed);
        Assert.Equal(baseline[0, 0], output[0, 0], 12);
        Assert.NotEqual(baseline[1, 0], output[1, 0], 9);

        // Continuous actions are tanh outputs.
        Assert.All(Enumerable.Range(0, baseline.Length), i => Assert.InRange(baseline[i], -1.0, 1.0));
    }

    [Fact(Timeout = 300000)]
    public async Task ReturnConditioning_SelectsTheBehaviourThatEarnedTheReturn()
    {
        await Task.Yield();
        // One-step episodes from two behaviours: action +0.5 earned a return of 1, action -0.5 earned 0. Behaviour cloning
        // conditioned on the return separates them (section 4).
        var agent = new DecisionTransformerAgent<double>(new DecisionTransformerOptions<double>
        {
            StateSize = 2, ActionSize = 2, EmbeddingDim = 16, NumLayers = 2, NumHeads = 2, ContextLength = 4,
            DropoutRate = 0.0, MaxEpisodeLength = 16, BatchSize = 32, LearningRateWarmupSteps = 1, Seed = 3,
        });
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(4);
        var data = new List<List<(Vector<double>, Vector<double>, double)>>();
        for (int i = 0; i < 200; i++)
        {
            bool good = i % 2 == 0;
            double a = good ? 0.5 : -0.5;
            data.Add(new List<(Vector<double>, Vector<double>, double)> { (V(rng.NextDouble(), rng.NextDouble()), V(a, a), good ? 1.0 : 0.0) });
        }
        agent.LoadOfflineData(data);
        double first = agent.Train();
        double last = first;
        for (int step = 0; step < 400; step++) last = agent.Train();
        Assert.True(last < 0.1 * first, $"loss {first:G4} -> {last:G4}");

        agent.ResetEpisode();
        var high = agent.SelectActionWithReturn(V(0.3, 0.6), 1.0, training: false);
        agent.ResetEpisode();
        var low = agent.SelectActionWithReturn(V(0.3, 0.6), 0.0, training: false);
        Assert.True(high[0] > 0.3 && low[0] < -0.3, $"return 1 -> {high[0]:F3}, return 0 -> {low[0]:F3}");
    }

    [Fact(Timeout = 120000)]
    public async Task Acting_DecrementsTheReturnToGoByEachReward()
    {
        await Task.Yield();
        // Algorithm 1's evaluation loop: R <- R - r after each step, and the context keeps the last K timesteps.
        var agent = new DecisionTransformerAgent<double>(new DecisionTransformerOptions<double>
        {
            StateSize = 2, ActionSize = 2, EmbeddingDim = 16, NumLayers = 1, NumHeads = 1, ContextLength = 2,
            DropoutRate = 0.0, MaxEpisodeLength = 16, TargetReturn = 5.0, Seed = 5,
        });
        var s = V(0.1, 0.2);
        var expected = agent.SelectActionWithReturn(s, 5.0, training: false);
        agent.ResetEpisode();
        Assert.Equal(expected[0], agent.SelectAction(s, training: false)[0], 12);
        agent.StoreExperience(s, expected, 2.0, s, done: false);
        // The next step conditions on 5 - 2 = 3.
        var second = agent.SelectAction(s, training: false);
        agent.ResetEpisode();
        agent.SelectActionWithReturn(s, 5.0, training: false);
        Assert.Equal(second[0], agent.SelectActionWithReturn(s, 3.0, training: false)[0], 12);
    }

    [Fact(Timeout = 300000)]
    public async Task Discrete_LearnsWithCrossEntropy()
    {
        await Task.Yield();
        var agent = new DecisionTransformerAgent<double>(new DecisionTransformerOptions<double>
        {
            StateSize = 2, ActionSize = 2, EmbeddingDim = 16, NumLayers = 1, NumHeads = 2, ContextLength = 3,
            DropoutRate = 0.0, MaxEpisodeLength = 16, BatchSize = 32, LearningRateWarmupSteps = 1, IsContinuous = false, Seed = 6,
        });
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7);
        var data = new List<List<(Vector<double>, Vector<double>, double)>>();
        for (int i = 0; i < 100; i++)
        {
            var episode = new List<(Vector<double>, Vector<double>, double)>();
            for (int t = 0; t < 3; t++)
            {
                var state = V(rng.NextDouble(), rng.NextDouble());
                int best = state[0] > state[1] ? 0 : 1;
                episode.Add((state, best == 0 ? V(1, 0) : V(0, 1), 1.0));
            }
            data.Add(episode);
        }
        agent.LoadOfflineData(data);
        double first = agent.Train(), last = first;
        for (int step = 0; step < 300; step++) last = agent.Train();
        Assert.True(last < 0.5 * first, $"cross-entropy {first:G4} -> {last:G4}");
        agent.ResetEpisode();
        var action = agent.SelectActionWithReturn(V(0.9, 0.1), 3.0, training: false);
        Assert.Equal(new[] { 1.0, 0.0 }, new[] { action[0], action[1] });
    }
}
