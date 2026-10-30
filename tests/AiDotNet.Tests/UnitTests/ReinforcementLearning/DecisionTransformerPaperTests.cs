using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Decision Transformer (Chen et al. 2021) against its paper: returns-to-go, the causal (R̂, s, a) token model,
/// return conditioning, the offline buffer, and every option the agent reads.
/// </summary>
public class DecisionTransformerPaperTests
{
    private const int StateSize = 2;
    private const int ActionSize = 2;

    private static DecisionTransformerOptions<double> Small(
        int numHeads = 2,
        double dropoutRate = 0.0,
        int bufferSize = 1000,
        double? targetReturn = null,
        double returnScale = 1.0,
        bool isContinuous = true,
        int actionSize = ActionSize,
        Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>? optimizerFactory = null) =>
        new DecisionTransformerOptions<double>
        {
            StateSize = StateSize,
            ActionSize = actionSize,
            IsContinuous = isContinuous,
            EmbeddingDim = 16,
            NumLayers = 1,
            NumHeads = numHeads,
            ContextLength = 4,
            DropoutRate = dropoutRate,
            MaxEpisodeLength = 16,
            BufferSize = bufferSize,
            TargetReturn = targetReturn,
            ReturnScale = returnScale,
            BatchSize = 4,
            LearningRate = 1e-3,
            OptimizerFactory = optimizerFactory,
            Seed = 7,
        };

    private static Vector<double> V(params double[] values) => new Vector<double>(values);

    private static List<(Vector<double> state, Vector<double> action, double reward)> Trajectory(
        double[] rewards, Vector<double> action)
    {
        var steps = new List<(Vector<double>, Vector<double>, double)>();
        for (int t = 0; t < rewards.Length; t++)
            steps.Add((V(0.1 * t, -0.2 * t), action, rewards[t]));
        return steps;
    }

    private static double MaxAbsDifference(Vector<double> a, Vector<double> b)
    {
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    // ── Returns-to-go (Sec. 3: R̂_t = Σ_{t'≥t} r_t') ───────────────────────────────────────────────────────

    /// <summary>
    /// The return-to-go of each stored step is the undiscounted sum of the rewards from that step to the end.
    /// Fails if the sum runs forwards, is off by one, or is discounted.
    /// </summary>
    [Fact]
    public void LoadedTrajectory_StoresSuffixSumReturnsToGo()
    {
        using var agent = new DecisionTransformerAgent<double>(Small());
        agent.LoadOfflineData(new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 1.0, 2.0, 3.0 }, V(0.1, 0.2)),
        });

        Assert.Equal(new[] { 6.0, 5.0, 3.0 }, agent.GetStoredReturnsToGo(0).ToArray());
    }

    /// <summary>
    /// Algorithm 1: at evaluation the next return-to-go is the previous one minus the reward received, starting
    /// from the target return. Fails if StoreExperience does not decrement the running return-to-go (or adds the
    /// reward instead), or if the first step is not conditioned on TargetReturn.
    /// </summary>
    [Fact]
    public void Evaluation_DecrementsReturnToGoByObservedReward()
    {
        using var agent = new DecisionTransformerAgent<double>(Small(targetReturn: 10.0));

        var s0 = V(0.3, -0.1);
        var s1 = V(0.5, 0.2);
        var a0 = agent.SelectAction(s0, training: false);
        agent.StoreExperience(s0, a0, 3.0, s1, done: false);
        agent.SelectAction(s1, training: false);
        var a1 = agent.SelectAction(V(0.7, 0.4), training: false);
        Assert.Equal(ActionSize, a1.Length);

        // Step 0 starts at the target 10; reward 3 leaves 7 for step 1; no reward was stored after step 1.
        Assert.Equal(new[] { 10.0, 7.0, 7.0 }, agent.CurrentContextReturnsToGo.ToArray());
    }

    /// <summary>
    /// When TargetReturn is unset the agent conditions on the best episode return it has stored. Two agents with
    /// identical weights, one told that target explicitly, must act identically. Fails if the default ignores the
    /// buffer (it would condition on zero instead of 9).
    /// </summary>
    [Fact]
    public void UnsetTargetReturn_ConditionsOnBestStoredEpisodeReturn()
    {
        using var implicitTarget = new DecisionTransformerAgent<double>(Small());
        implicitTarget.LoadOfflineData(new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 1.0, 1.0 }, V(0.1, 0.1)),
            Trajectory(new[] { 4.0, 5.0 }, V(0.1, 0.1)),
        });

        using var explicitTarget = new DecisionTransformerAgent<double>(Small(targetReturn: 9.0));
        using var zeroTarget = new DecisionTransformerAgent<double>(Small(targetReturn: 0.0));
        explicitTarget.SetParameters(implicitTarget.GetParameters());
        zeroTarget.SetParameters(implicitTarget.GetParameters());

        var state = V(0.2, -0.4);
        Assert.True(MaxAbsDifference(implicitTarget.Predict(state), explicitTarget.Predict(state)) < 1e-12);
        Assert.True(MaxAbsDifference(implicitTarget.Predict(state), zeroTarget.Predict(state)) > 1e-9);
    }

    // ── Causal transformer over (R̂, s, a) tokens (Sec. 3) ─────────────────────────────────────────────────

    /// <summary>
    /// The action at timestep t is read from the state token s_t, which may attend only to itself and earlier
    /// tokens. Changing a_1 or R̂_2 must leave the predictions at s_0 and s_1 bit-for-bit unchanged and move the
    /// prediction at s_2. Fails if the causal mask is off (s_0 would see the future) or if the action is read
    /// from a token after a_t.
    /// </summary>
    [Fact]
    public void StateTokenPrediction_IsCausal()
    {
        using var agent = new DecisionTransformerAgent<double>(Small());

        var returns = new List<double> { 3.0, 2.0, 1.0 };
        var states = new List<Vector<double>> { V(0.1, 0.2), V(-0.3, 0.4), V(0.5, -0.6) };
        var actions = new List<Vector<double>?> { V(0.2, -0.2), V(0.4, 0.1), V(-0.5, 0.3) };
        var baseline = agent.PredictContext(returns, states, actions, 0);

        var changedAction = new List<Vector<double>?> { actions[0], V(-0.9, 0.9), actions[2] };
        var afterAction = agent.PredictContext(returns, states, changedAction, 0);
        Assert.True(MaxAbsDifference(baseline[0], afterAction[0]) < 1e-12, "s_0 saw a later action.");
        Assert.True(MaxAbsDifference(baseline[1], afterAction[1]) < 1e-12, "s_1 saw its own action a_1.");
        Assert.True(MaxAbsDifference(baseline[2], afterAction[2]) > 1e-9, "s_2 did not see the earlier action a_1.");

        var changedReturn = new List<double> { 3.0, 2.0, -5.0 };
        var afterReturn = agent.PredictContext(changedReturn, states, actions, 0);
        Assert.True(MaxAbsDifference(baseline[0], afterReturn[0]) < 1e-12, "s_0 saw a later return-to-go.");
        Assert.True(MaxAbsDifference(baseline[1], afterReturn[1]) < 1e-12, "s_1 saw a later return-to-go.");
        Assert.True(MaxAbsDifference(baseline[2], afterReturn[2]) > 1e-9, "s_2 ignored its own return-to-go.");
    }

    /// <summary>
    /// NumHeads configures every block's causal self-attention, and the head count changes the function
    /// computed from the same weights. Fails if the option is not passed to the attention layer.
    /// </summary>
    [Fact]
    public void NumHeads_SetsCausalAttentionHeadCount()
    {
        using var oneHead = new DecisionTransformerAgent<double>(Small(numHeads: 1));
        using var fourHeads = new DecisionTransformerAgent<double>(Small(numHeads: 4));

        foreach (var (agent, heads) in new[] { (oneHead, 1), (fourHeads, 4) })
        {
            var network = (NeuralNetworkBase<double>)agent.TransformerNetwork;
            var blocks = network.Layers.OfType<TransformerEncoderBlock<double>>().ToList();
            Assert.Single(blocks);
            var attention = Assert.IsType<MultiHeadAttentionLayer<double>>(blocks[0].AttentionLayer);
            Assert.Equal(heads, attention.HeadCount);
            Assert.True(attention.UseCausalMask);
        }

        var weights = oneHead.GetParameters();
        Assert.Equal(weights.Length, fourHeads.GetParameters().Length);
        fourHeads.SetParameters(weights);

        var state = V(0.6, -0.8);
        Assert.True(MaxAbsDifference(oneHead.Predict(state), fourHeads.Predict(state)) > 1e-9,
            "One and four heads computed the same function from the same weights.");
    }

    /// <summary>
    /// DropoutRate is applied during training only: two training-mode forwards differ when it is positive and are
    /// identical when it is zero, while Predict stays deterministic. Fails if dropout is never built into the
    /// network, or if it leaks into action selection.
    /// </summary>
    [Fact]
    public void DropoutRate_IsActiveInTrainingOnly()
    {
        static (double TrainingSpread, double PredictSpread) Spread(double rate)
        {
            using var agent = new DecisionTransformerAgent<double>(Small(dropoutRate: rate));
            var network = (NeuralNetworkBase<double>)agent.TransformerNetwork;

            var rng = RandomHelper.CreateSeededRandom(11);
            int features = network.Architecture.InputSize;
            var tokens = new Tensor<double>(new[] { 1, 12, features });
            for (int i = 0; i < tokens.Length; i++) tokens.SetFlat(i, rng.NextDouble());

            network.SetTrainingMode(true);
            var first = network.ForwardForTraining(tokens).ToVector();
            var second = network.ForwardForTraining(tokens).ToVector();
            network.SetTrainingMode(false);

            var state = V(0.4, 0.9);
            return (MaxAbsDifference(first, second), MaxAbsDifference(agent.Predict(state), agent.Predict(state)));
        }

        var withDropout = Spread(0.5);
        var withoutDropout = Spread(0.0);

        Assert.True(withDropout.TrainingSpread > 1e-6, "DropoutRate 0.5 left training forwards identical.");
        Assert.Equal(0.0, withoutDropout.TrainingSpread);
        Assert.Equal(0.0, withDropout.PredictSpread);
    }

    // ── Offline buffer ───────────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// BufferSize caps the stored transitions, evicting whole trajectories oldest first and keeping only the most
    /// recent steps of a trajectory longer than the whole buffer. Fails if the capacity is ignored.
    /// </summary>
    [Fact]
    public void BufferSize_EvictsOldestTrajectoriesFirst()
    {
        using var agent = new DecisionTransformerAgent<double>(Small(bufferSize: 5));
        agent.LoadOfflineData(new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 1.0, 1.0, 1.0 }, V(0.1, 0.1)),
            Trajectory(new[] { 2.0, 2.0, 2.0 }, V(0.1, 0.1)),
        });

        Assert.Equal(1, agent.StoredTrajectoryCount);
        Assert.Equal(3, agent.StoredTransitionCount);
        Assert.Equal(new[] { 6.0, 4.0, 2.0 }, agent.GetStoredReturnsToGo(0).ToArray());
        Assert.Equal(3.0, agent.GetMetrics()["buffer_size"]);

        agent.LoadOfflineData(new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, V(0.1, 0.1)),
        });

        Assert.Equal(1, agent.StoredTrajectoryCount);
        Assert.Equal(5, agent.StoredTransitionCount);
        // The last five steps of a seven-step episode, with returns-to-go of the whole episode.
        Assert.Equal(new[] { 5.0, 4.0, 3.0, 2.0, 1.0 }, agent.GetStoredReturnsToGo(0).ToArray());
    }

    /// <summary>
    /// Transitions arriving through StoreExperience become a trajectory when the episode ends. Fails if the online
    /// path leaves the buffer empty (the agent would never train from its own experience).
    /// </summary>
    [Fact]
    public void StoreExperience_CommitsEpisodeOnDone()
    {
        using var agent = new DecisionTransformerAgent<double>(Small());
        agent.StoreExperience(V(0.1, 0.1), V(0.2, 0.2), 1.0, V(0.2, 0.2), done: false);
        Assert.Equal(0, agent.StoredTrajectoryCount);
        agent.StoreExperience(V(0.2, 0.2), V(0.3, 0.3), 2.0, V(0.3, 0.3), done: true);

        Assert.Equal(1, agent.StoredTrajectoryCount);
        Assert.Equal(new[] { 3.0, 2.0 }, agent.GetStoredReturnsToGo(0).ToArray());
    }

    // ── Training (Sec. 3: supervised loss on the predicted actions) ─────────────────────────────────────

    /// <summary>
    /// The paper's central claim: trained on a high-return trajectory that took one action and a low-return one
    /// that took the opposite action from the same states, the model reproduces whichever behaviour matches the
    /// return it is conditioned on. Fails if the return-to-go token does not reach the prediction, or if the
    /// regression loss does not train the action head.
    /// </summary>
    [Fact]
    public void ReturnConditioning_SelectsTheMatchingBehaviour()
    {
        using var trained = new DecisionTransformerAgent<double>(Small());
        trained.LoadOfflineData(new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 1.0, 1.0, 1.0, 1.0 }, V(0.5, 0.5)),
            Trajectory(new[] { 0.0, 0.0, 0.0, 0.0 }, V(-0.5, -0.5)),
        });

        double firstLoss = trained.Train();
        double lastLoss = firstLoss;
        for (int i = 0; i < 600; i++) lastLoss = trained.Train();
        Assert.True(lastLoss < firstLoss, $"Training did not reduce the loss ({firstLoss} -> {lastLoss}).");

        using var high = new DecisionTransformerAgent<double>(Small(targetReturn: 4.0));
        using var low = new DecisionTransformerAgent<double>(Small(targetReturn: 0.0));
        high.SetParameters(trained.GetParameters());
        low.SetParameters(trained.GetParameters());

        var start = V(0.0, 0.0);
        var highAction = high.Predict(start);
        var lowAction = low.Predict(start);
        Assert.True(highAction[0] > 0.2 && highAction[1] > 0.2, $"Target 4 gave {highAction[0]}, {highAction[1]}.");
        Assert.True(lowAction[0] < -0.2 && lowAction[1] < -0.2, $"Target 0 gave {lowAction[0]}, {lowAction[1]}.");
    }

    /// <summary>
    /// Discrete actions train with cross-entropy (the paper's Atari setting): the logit of the dataset action must
    /// come to dominate and the greedy action becomes its one-hot. Fails if the cross-entropy sign is flipped or the
    /// discrete head is trained as a regression.
    /// </summary>
    [Fact]
    public void DiscreteActions_TrainWithCrossEntropy()
    {
        using var agent = new DecisionTransformerAgent<double>(Small(isContinuous: false, actionSize: 3));
        var chosen = V(0.0, 0.0, 1.0);
        agent.LoadOfflineData(new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 1.0, 1.0, 1.0, 1.0 }, chosen),
        });

        var state = V(0.0, 0.0);
        double Margin()
        {
            var logits = agent.PredictContext(new List<double> { 4.0 }, new List<Vector<double>> { state },
                new List<Vector<double>?> { null }, 0)[0];
            return logits[2] - Math.Max(logits[0], logits[1]);
        }

        double before = Margin();
        for (int i = 0; i < 200; i++) agent.Train();
        double after = Margin();

        Assert.True(after > before && after > 0.0, $"Dataset action's logit margin went {before} -> {after}.");
        Assert.Equal(new[] { 0.0, 0.0, 1.0 }, agent.Predict(state).ToArray());
    }

    /// <summary>
    /// ReturnScale divides every return-to-go before it is embedded, so target R at scale c must act exactly like
    /// target R / c at scale 1 with the same weights. Fails if the scale is ignored.
    /// </summary>
    [Fact]
    public void ReturnScale_DividesReturnsBeforeEmbedding()
    {
        using var scaled = new DecisionTransformerAgent<double>(Small(targetReturn: 8.0, returnScale: 4.0));
        using var equivalent = new DecisionTransformerAgent<double>(Small(targetReturn: 2.0));
        using var unscaled = new DecisionTransformerAgent<double>(Small(targetReturn: 8.0));
        equivalent.SetParameters(scaled.GetParameters());
        unscaled.SetParameters(scaled.GetParameters());

        var state = V(-0.3, 0.7);
        Assert.True(MaxAbsDifference(scaled.Predict(state), equivalent.Predict(state)) < 1e-12);
        Assert.True(MaxAbsDifference(scaled.Predict(state), unscaled.Predict(state)) > 1e-9);
    }

    /// <summary>
    /// OptimizerFactory supplies the transformer's optimizer: invoked once, and a step six orders of magnitude
    /// faster moves the weights about that much further. Fails if the agent builds its own optimizer instead.
    /// </summary>
    [Fact]
    public void OptimizerFactory_DrivesTraining()
    {
        static AdamOptimizer<double, Tensor<double>, Tensor<double>> Adam(double rate) =>
            new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
                {
                    InitialLearningRate = rate,
                    UseAdaptiveLearningRate = false,
                });

        double Displacement(double rate, out int invocations)
        {
            int calls = 0;
            using var agent = new DecisionTransformerAgent<double>(Small(optimizerFactory: () =>
            {
                calls++;
                return Adam(rate);
            }));
            agent.LoadOfflineData(new List<List<(Vector<double>, Vector<double>, double)>>
            {
                Trajectory(new[] { 1.0, 0.0, 1.0, 0.0 }, V(0.3, -0.3)),
            });

            agent.Train();
            var before = agent.GetParameters();
            agent.Train();
            var after = agent.GetParameters();
            invocations = calls;

            double sum = 0.0;
            for (int i = 0; i < before.Length; i++) sum += (after[i] - before[i]) * (after[i] - before[i]);
            return Math.Sqrt(sum);
        }

        double fast = Displacement(1e-3, out int fastCalls);
        double slow = Displacement(1e-9, out int slowCalls);

        Assert.Equal(1, fastCalls);
        Assert.Equal(1, slowCalls);
        Assert.True(fast > 1e3 * slow, $"OptimizerFactory ignored: displacement {fast} at 1e-3 vs {slow} at 1e-9.");
    }

    /// <summary>
    /// A clone acts like its source, including the default conditioning target learned from the buffer. Fails if
    /// the best stored return is not carried with the agent's state.
    /// </summary>
    [Fact]
    public void Clone_KeepsTheDefaultConditioningTarget()
    {
        using var agent = new DecisionTransformerAgent<double>(Small());
        agent.LoadOfflineData(new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 3.0, 4.0 }, V(0.2, -0.2)),
        });
        agent.Train();

        var clone = agent.Clone();
        var state = V(0.5, 0.5);
        Assert.True(MaxAbsDifference(agent.Predict(state), clone.Predict(state)) < 1e-12);
        (clone as IDisposable)?.Dispose();
    }

    /// <summary>Out-of-range options are rejected at construction rather than failing deep inside training.</summary>
    [Fact]
    public void InvalidOptions_AreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DecisionTransformerAgent<double>(Small(numHeads: 3)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DecisionTransformerAgent<double>(Small(dropoutRate: 1.0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DecisionTransformerAgent<double>(Small(bufferSize: 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DecisionTransformerAgent<double>(Small(returnScale: 0.0)));
    }
}
