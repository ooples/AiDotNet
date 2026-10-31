using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
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
        Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>? optimizerFactory = null,
        int warmupSteps = 0,
        double maxGradientNorm = 0.25,
        double weightDecay = 1e-4,
        bool normalizeStates = true) =>
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
            WarmupSteps = warmupSteps,
            MaxGradientNorm = maxGradientNorm,
            WeightDecay = weightDecay,
            NormalizeStates = normalizeStates,
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

    private static List<List<(Vector<double>, Vector<double>, double)>> Dataset() =>
        new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 1.0, 0.0, 1.0, 0.0 }, V(0.3, -0.3)),
        };

    /// <summary>Euclidean norm of the weight change made by one default-optimizer update.</summary>
    private static (double Norm, double DotWithWeights) FirstUpdate(DecisionTransformerOptions<double> options)
    {
        using var agent = new DecisionTransformerAgent<double>(options);
        agent.LoadOfflineData(Dataset());
        var before = agent.GetParameters();
        agent.Train();
        var after = agent.GetParameters();

        double sum = 0.0, dot = 0.0;
        for (int i = 0; i < before.Length; i++)
        {
            double delta = after[i] - before[i];
            sum += delta * delta;
            dot += delta * before[i];
        }

        return (Math.Sqrt(sum), dot);
    }

    // ── Paper hyperparameters (Appendix A.2.1, Table 9) ─────────────────────────────────────────────────

    /// <summary>
    /// Default options are Table 9's Gym hyperparameters. Fails if any default drifts from the table.
    /// </summary>
    [Fact]
    public void DefaultOptions_AreTheTable9GymHyperparameters()
    {
        var options = new DecisionTransformerOptions<double>();
        Assert.Equal(1e-4, options.LearningRate);
        Assert.Equal(64, options.BatchSize);
        Assert.Equal(0.25, options.MaxGradientNorm);
        Assert.Equal(100000, options.WarmupSteps);
        Assert.Equal(1e-4, options.WeightDecay);
        Assert.Equal(3, options.NumLayers);
        Assert.Equal(1, options.NumHeads);
        Assert.Equal(128, options.EmbeddingDim);
        Assert.Equal(20, options.ContextLength);
        Assert.Equal(0.1, options.DropoutRate);
        Assert.True(options.NormalizeStates);
    }

    /// <summary>
    /// The default optimizer is Table 9's: AdamW with PyTorch's defaults, the configured weight decay, global-norm
    /// clipping at MaxGradientNorm, and a per-update warmup schedule; the feed-forward nonlinearity is ReLU. Fails
    /// if the agent falls back to Adam, drops the clip or the schedule, or keeps GELU.
    /// </summary>
    [Fact]
    public void DefaultOptimizer_IsTable9AdamWWithClippingAndWarmup()
    {
        using var agent = new DecisionTransformerAgent<double>(Small(warmupSteps: 10, maxGradientNorm: 0.25));
        var optimizer = Assert.IsType<AdamWOptimizer<double, Tensor<double>, Tensor<double>>>(agent.TransformerOptimizer);
        var options = Assert.IsType<AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());

        Assert.Equal(1e-4, options.WeightDecay);
        Assert.Equal(0.9, options.Beta1);
        Assert.Equal(0.999, options.Beta2);
        Assert.Equal(1e-8, options.Epsilon);
        Assert.True(options.EnableGradientClipping);
        Assert.Equal(GradientClippingMethod.ByNorm, options.GradientClippingMethod);
        Assert.Equal(0.25, options.MaxGradientNorm);
        Assert.IsType<LinearWarmupScheduler>(options.LearningRateScheduler);
        Assert.Equal(SchedulerStepMode.StepPerBatch, options.SchedulerStepMode);

        var network = (NeuralNetworkBase<double>)agent.TransformerNetwork;
        var block = network.Layers.OfType<TransformerEncoderBlock<double>>().First();
        Assert.IsType<ReLUActivation<double>>(block.FfnUpLayer.ScalarActivation);
    }

    /// <summary>
    /// The learning rate follows the reference LambdaLR schedule lr * min((s + 1) / W, 1): the first update uses
    /// lr / W and update s uses lr * (s + 1) / W until it reaches lr. Fails if the warmup starts at zero, ends one
    /// step late, or is not stepped once per update.
    /// </summary>
    [Fact]
    public void Warmup_FollowsTheReferenceLinearSchedule()
    {
        const int warmup = 4;
        const double rate = 1e-3;
        using var agent = new DecisionTransformerAgent<double>(Small(warmupSteps: warmup));
        agent.LoadOfflineData(Dataset());
        var optimizer = Assert.IsAssignableFrom<GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>>>(
            agent.TransformerOptimizer);

        for (int update = 0; update < 7; update++)
        {
            double expected = rate * Math.Min((update + 1.0) / warmup, 1.0);
            Assert.True(Math.Abs(optimizer.GetCurrentLearningRate() - expected) < 1e-15,
                $"Update {update} ran at {optimizer.GetCurrentLearningRate():R}, expected {expected:R}.");
            agent.Train();
        }
    }

    /// <summary>
    /// MaxGradientNorm bounds the global gradient norm of the default optimizer. Clipped to 1e-12, every gradient
    /// coordinate falls far below Adam's epsilon, so the first update shrinks by orders of magnitude compared with
    /// the unclipped one. Fails if the bound is not handed to the optimizer.
    /// </summary>
    [Fact]
    public void MaxGradientNorm_ClipsTheDefaultOptimizerStep()
    {
        double unclipped = FirstUpdate(Small(maxGradientNorm: 0.0, weightDecay: 0.0)).Norm;
        double clipped = FirstUpdate(Small(maxGradientNorm: 1e-12, weightDecay: 0.0)).Norm;

        Assert.True(unclipped > 0.0);
        Assert.True(unclipped > 100.0 * clipped, $"Clipping had no effect: {unclipped:R} vs {clipped:R}.");
    }

    /// <summary>
    /// WeightDecay is AdamW's decoupled decay: with the gradient step clipped to nothing, an update with decay 0.5
    /// pulls the weights towards zero (the change points against the weights) by far more than the update without
    /// decay. Fails if the decay is not passed to the optimizer or the optimizer is not AdamW.
    /// </summary>
    [Fact]
    public void WeightDecay_ShrinksWeightsTowardZero()
    {
        var withDecay = FirstUpdate(Small(maxGradientNorm: 1e-12, weightDecay: 0.5));
        var withoutDecay = FirstUpdate(Small(maxGradientNorm: 1e-12, weightDecay: 0.0));

        Assert.True(withDecay.Norm > 100.0 * withoutDecay.Norm,
            $"Weight decay had no effect: {withDecay.Norm:R} vs {withoutDecay.Norm:R}.");
        Assert.True(withDecay.DotWithWeights < 0.0, "Weight decay did not move the weights towards zero.");
    }

    /// <summary>
    /// A supplied OptimizerFactory owns its own settings: the agent's MaxGradientNorm, WeightDecay and WarmupSteps
    /// are not applied to the optimizer it returns. Fails if the agent wraps or reconfigures the factory's optimizer.
    /// </summary>
    [Fact]
    public void OptimizerFactory_OwnsItsOwnClippingAndSchedule()
    {
        var supplied = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 1e-3,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
            });
        using var agent = new DecisionTransformerAgent<double>(
            Small(optimizerFactory: () => supplied, warmupSteps: 10, maxGradientNorm: 0.25));

        Assert.Same(supplied, agent.TransformerOptimizer);
        Assert.False(supplied.GetOptions() is GradientBasedOptimizerOptions<double, Tensor<double>, Tensor<double>> o
            && (o.EnableGradientClipping || o.LearningRateScheduler is not null));
    }

    // ── State normalisation (reference implementation: (s - mean) / (std + 1e-6)) ───────────────────────

    /// <summary>
    /// The state statistics are the dataset's per-dimension mean and population standard deviation plus 1e-6, and
    /// they are refreshed when StoreExperience commits an episode. Fails on a sample (n - 1) deviation, a missing
    /// epsilon, or statistics that go stale when the buffer changes.
    /// </summary>
    [Fact]
    public void StateStatistics_AreTheBufferMeanAndPopulationStd()
    {
        using var agent = new DecisionTransformerAgent<double>(Small());
        Assert.Equal(new[] { 0.0, 0.0 }, agent.GetStateMean());
        Assert.Equal(new[] { 1.0, 1.0 }, agent.GetStateStd());

        // States (0, 0), (0.1, -0.2), (0.2, -0.4).
        agent.LoadOfflineData(new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 1.0, 1.0, 1.0 }, V(0.1, 0.1)),
        });
        double std0 = Math.Sqrt(0.02 / 3.0);
        Assert.True(Math.Abs(agent.GetStateMean()[0] - 0.1) < 1e-15 && Math.Abs(agent.GetStateMean()[1] + 0.2) < 1e-15);
        Assert.True(Math.Abs(agent.GetStateStd()[0] - (std0 + 1e-6)) < 1e-15);
        Assert.True(Math.Abs(agent.GetStateStd()[1] - (2.0 * std0 + 1e-6)) < 1e-15);

        // One more episode of a single state (1, 1): four states in all.
        agent.StoreExperience(V(1.0, 1.0), V(0.1, 0.1), 1.0, V(1.0, 1.0), done: true);
        Assert.True(Math.Abs(agent.GetStateMean()[0] - 1.3 / 4.0) < 1e-15);
        Assert.True(Math.Abs(agent.GetStateMean()[1] - 0.4 / 4.0) < 1e-15);
    }

    /// <summary>
    /// The state embedding sees (s - mean) / std: with the same weights and target, the normalising agent's action
    /// for s equals the non-normalising agent's action for the standardised state, and differs from its action for
    /// the raw state. Fails if normalisation is skipped, inverted, or applied with other statistics.
    /// </summary>
    [Fact]
    public void NormalizeStates_StandardisesStatesBeforeEmbedding()
    {
        using var normalising = new DecisionTransformerAgent<double>(Small(targetReturn: 2.0));
        using var raw = new DecisionTransformerAgent<double>(Small(targetReturn: 2.0, normalizeStates: false));
        normalising.LoadOfflineData(new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 1.0, 1.0, 1.0 }, V(0.1, 0.1)),
        });
        raw.SetParameters(normalising.GetParameters());

        var state = V(0.7, -0.9);
        var standardised = V(
            (state[0] - normalising.GetStateMean()[0]) / normalising.GetStateStd()[0],
            (state[1] - normalising.GetStateMean()[1]) / normalising.GetStateStd()[1]);

        Assert.True(MaxAbsDifference(normalising.Predict(state), raw.Predict(standardised)) < 1e-12);
        Assert.True(MaxAbsDifference(normalising.Predict(state), raw.Predict(state)) > 1e-9);
    }

    /// <summary>
    /// The statistics are model state: a clone normalises with the source's mean and deviation even though the
    /// trajectory buffer itself is not copied. Fails if the statistics are left out of serialization.
    /// </summary>
    [Fact]
    public void StateStatistics_TravelWithClones()
    {
        using var agent = new DecisionTransformerAgent<double>(Small(targetReturn: 2.0));
        agent.LoadOfflineData(new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 1.0, 1.0, 1.0 }, V(0.1, 0.1)),
        });

        var clone = Assert.IsType<DecisionTransformerAgent<double>>(agent.Clone());
        try
        {
            Assert.Equal(agent.GetStateMean(), clone.GetStateMean());
            Assert.Equal(agent.GetStateStd(), clone.GetStateStd());
            var state = V(0.7, -0.9);
            Assert.True(MaxAbsDifference(agent.Predict(state), clone.Predict(state)) < 1e-12);
        }
        finally
        {
            clone.Dispose();
        }
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

    /// <summary>
    /// DropoutRate also reaches the attention probabilities of every block (GPT-2's attn_pdrop). Fails if the
    /// attention layers are built without attention dropout.
    /// </summary>
    [Fact]
    public void DropoutRate_ReachesTheAttentionProbabilities()
    {
        using var agent = new DecisionTransformerAgent<double>(Small(dropoutRate: 0.1));
        var network = (NeuralNetworkBase<double>)agent.TransformerNetwork;
        foreach (var block in network.Layers.OfType<TransformerEncoderBlock<double>>())
        {
            var attention = Assert.IsType<MultiHeadAttentionLayer<double>>(block.AttentionLayer);
            Assert.Equal(0.1, attention.AttentionDropoutRate);
        }
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
        var dataset = new List<List<(Vector<double>, Vector<double>, double)>>
        {
            Trajectory(new[] { 1.0, 1.0, 1.0, 1.0 }, V(0.5, 0.5)),
            Trajectory(new[] { 0.0, 0.0, 0.0, 0.0 }, V(-0.5, -0.5)),
        };
        using var trained = new DecisionTransformerAgent<double>(Small());
        trained.LoadOfflineData(dataset);

        double firstLoss = trained.Train();
        double lastLoss = firstLoss;
        for (int i = 0; i < 600; i++) lastLoss = trained.Train();
        Assert.True(lastLoss < firstLoss, $"Training did not reduce the loss ({firstLoss} -> {lastLoss}).");

        using var high = new DecisionTransformerAgent<double>(Small(targetReturn: 4.0));
        using var low = new DecisionTransformerAgent<double>(Small(targetReturn: 0.0));
        // The same dataset gives the conditioned copies the same state-normalisation statistics.
        high.LoadOfflineData(dataset);
        low.LoadOfflineData(dataset);
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
        Assert.Throws<ArgumentOutOfRangeException>(() => new DecisionTransformerAgent<double>(Small(maxGradientNorm: -1.0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DecisionTransformerAgent<double>(Small(weightDecay: -1e-4)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DecisionTransformerAgent<double>(Small(warmupSteps: -1)));
    }
}
