using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.Agents.Dreamer;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Checks DreamerAgent against Hafner et al. (2020), "Dream to Control": the RSSM shapes, the world-model objective
/// of Equation 10 with its free nats, KL, reward and discount weights, the V-lambda targets of Equation 6, sequence
/// sampling, and the history-filtering policy.
/// </summary>
public class DreamerPaperTests
{
    private const int ObservationSize = 4;
    private const int ActionSize = 2;

    private static DreamerOptions<double> SmallOptions(
        double klScale = 1.0,
        double freeNats = 3.0,
        double rewardScale = 1.0,
        double continueScale = 10.0,
        double lambda = 0.95,
        int batchLength = 10,
        int deterministicSize = 12,
        int stochasticSize = 4,
        double actorLearningRate = 8e-5,
        double valueLearningRate = 8e-5,
        Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>? optimizerFactory = null)
        => new DreamerOptions<double>
        {
            ObservationSize = ObservationSize,
            ActionSize = ActionSize,
            DeterministicSize = deterministicSize,
            StochasticSize = stochasticSize,
            HiddenSize = 16,
            BatchLength = batchLength,
            ImaginationHorizon = 4,
            BatchSize = 6,
            ReplayBufferSize = 512,
            KLScale = klScale,
            FreeNats = freeNats,
            RewardScale = rewardScale,
            ContinueScale = continueScale,
            Lambda = lambda,
            LearningRate = 3e-3,
            ActorLearningRate = actorLearningRate,
            ValueLearningRate = valueLearningRate,
            OptimizerFactory = optimizerFactory,
            Seed = 17,
        };

    /// <summary>
    /// Stores <paramref name="count"/> transitions in episodes of <paramref name="episodeLength"/> steps, from a
    /// fixed generator: a slowly drifting observation and a reward that depends on the action.
    /// </summary>
    private static void Feed(DreamerAgent<double> agent, int count, int episodeLength, int seed = 3)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var state = new Vector<double>(ObservationSize);
        for (int t = 0; t < count; t++)
        {
            var action = new Vector<double>(ActionSize);
            for (int j = 0; j < ActionSize; j++) action[j] = rng.NextDouble() * 2.0 - 1.0;
            var next = new Vector<double>(ObservationSize);
            for (int j = 0; j < ObservationSize; j++) next[j] = 0.8 * state[j] + 0.3 * action[j % ActionSize] + 0.1 * (rng.NextDouble() - 0.5);
            double reward = action[0] * state[0] + 0.5;
            bool done = (t + 1) % episodeLength == 0;
            agent.StoreExperience(state, action, reward, next, done);
            state = done ? new Vector<double>(ObservationSize) : next;
        }
    }

    private static DreamerAgent<double> Trained(DreamerOptions<double> options, int steps, int episodeLength = 20)
    {
        var agent = new DreamerAgent<double>(options);
        Feed(agent, 120, episodeLength);
        for (int i = 0; i < steps; i++) agent.Train();
        return agent;
    }

    private static double[] NetworkParameters(DreamerAgent<double> agent, int index)
    {
        var parameters = agent.GetNetworks()[index].GetParameters();
        var copy = new double[parameters.Length];
        for (int i = 0; i < copy.Length; i++) copy[i] = parameters[i];
        return copy;
    }

    private static double MaxAbsDifference(double[] left, double[] right)
    {
        Assert.Equal(left.Length, right.Length);
        double worst = 0.0;
        for (int i = 0; i < left.Length; i++) worst = Math.Max(worst, Math.Abs(left[i] - right[i]));
        return worst;
    }

    private static Tensor<double> Column(params double[] values) => new Tensor<double>(values, [values.Length, 1]);

    private static Tensor<double> Matrix1(params double[] values) => new Tensor<double>(values, [1, values.Length]);

    // Network order (DreamerAgent.RegisterComponents): encoder 0, transition input 1, GRU gates 2, GRU candidate 3,
    // prior 4, posterior 5, observation 6, reward 7, discount 8, action 9, value 10.
    private const int GateNetwork = 2;
    private const int CandidateNetwork = 3;
    private const int PriorNetwork = 4;
    private const int PosteriorNetwork = 5;
    private const int TransitionInputNetwork = 1;
    private const int RewardNetwork = 7;
    private const int ContinueNetwork = 8;
    private const int ActorNetwork = 9;
    private const int ValueNetwork = 10;

    [Fact]
    public void GaussianKl_MatchesTheClosedFormForDiagonalGaussians()
    {
        using var agent = new DreamerAgent<double>(SmallOptions());
        double[] qMean = { 0.5, -1.0, 0.2 }, qStd = { 0.3, 2.0, 1.0 };
        double[] pMean = { 0.0, 0.5, 0.2 }, pStd = { 1.0, 0.5, 1.0 };

        double expected = 0.0;
        for (int i = 0; i < 3; i++)
        {
            expected += Math.Log(pStd[i] / qStd[i])
                + (qStd[i] * qStd[i] + (qMean[i] - pMean[i]) * (qMean[i] - pMean[i])) / (2.0 * pStd[i] * pStd[i])
                - 0.5;
        }

        var kl = agent.GaussianKl(Matrix1(qMean), Matrix1(qStd), Matrix1(pMean), Matrix1(pStd)).ToVector();
        Assert.Equal(expected, kl[0], 1e-10);

        var self = agent.GaussianKl(Matrix1(qMean), Matrix1(qStd), Matrix1(qMean), Matrix1(qStd)).ToVector();
        Assert.Equal(0.0, self[0], 1e-12);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.7)]
    [InlineData(0.95)]
    [InlineData(1.0)]
    public void LambdaReturns_MatchEquation6(double lambda)
    {
        using var agent = new DreamerAgent<double>(SmallOptions());
        const double gamma = 0.9;
        double[] r = { 0.3, -1.2, 2.0, 0.7, 0.1 };   // r(s_tau), tau = t..t+H with H = 4
        double[] v = { 1.5, 0.4, -0.6, 2.2, 3.1 };   // v(s_tau)
        int horizon = r.Length - 1;

        var rewards = new Tensor<double>[r.Length];
        var values = new Tensor<double>[r.Length];
        var discounts = new Tensor<double>[r.Length];
        for (int k = 0; k < r.Length; k++)
        {
            rewards[k] = Column(r[k]);
            values[k] = Column(v[k]);
            discounts[k] = Column(gamma);
        }

        var returns = agent.ComputeLambdaReturns(rewards, values, discounts, lambda);

        // Equation 5: V_N^k(s_tau) = sum_{n=tau}^{h-1} gamma^{n-tau} r_n + gamma^{h-tau} v(s_h), h = min(tau+k, t+H).
        double NStep(int tau, int k)
        {
            int h = Math.Min(tau + k, horizon);
            double sum = 0.0;
            for (int n = tau; n < h; n++) sum += Math.Pow(gamma, n - tau) * r[n];
            return sum + Math.Pow(gamma, h - tau) * v[h];
        }

        for (int tau = 0; tau < horizon; tau++)
        {
            // Equation 6: V_lambda = (1 - lambda) sum_{n=1}^{H-1} lambda^{n-1} V_N^n + lambda^{H-1} V_N^H.
            double expected = 0.0;
            for (int n = 1; n <= horizon - 1; n++) expected += (1.0 - lambda) * Math.Pow(lambda, n - 1) * NStep(tau, n);
            expected += Math.Pow(lambda, horizon - 1) * NStep(tau, horizon);
            Assert.Equal(expected, returns[tau].ToVector()[0], 1e-10);
        }

        Assert.Equal(v[horizon], returns[horizon].ToVector()[0], 1e-12);
    }

    [Fact]
    public void LambdaReturns_AtTheExtremesAreTheOneStepAndFullHorizonEstimates()
    {
        using var agent = new DreamerAgent<double>(SmallOptions());
        const double gamma = 0.5;
        var rewards = new[] { Column(1.0), Column(2.0), Column(4.0) };
        var values = new[] { Column(10.0), Column(20.0), Column(40.0) };
        var discounts = new[] { Column(gamma), Column(gamma), Column(gamma) };

        // lambda = 0: the one-step estimate r_t + gamma v(s_{t+1}).
        var oneStep = agent.ComputeLambdaReturns(rewards, values, discounts, 0.0);
        Assert.Equal(1.0 + gamma * 20.0, oneStep[0].ToVector()[0], 1e-12);

        // lambda = 1: rewards to the horizon plus the bootstrap, V_N^H (Equation 5 with k = H).
        var full = agent.ComputeLambdaReturns(rewards, values, discounts, 1.0);
        Assert.Equal(1.0 + gamma * 2.0 + gamma * gamma * 40.0, full[0].ToVector()[0], 1e-12);
    }

    [Fact]
    public void DeterministicAndStochasticSizes_ShapeTheRecurrentStateSpaceModel()
    {
        const int d = 7, s = 3, hidden = 16;
        var agent = Trained(SmallOptions(deterministicSize: d, stochasticSize: s), steps: 1);
        var networks = agent.GetNetworks();

        // Transition input [z, a] -> D; GRU gates [x, h] -> 2D; candidate [x, r*h] -> D.
        Assert.Equal((s + ActionSize) * d + d, (int)networks[TransitionInputNetwork].ParameterCount);
        Assert.Equal(2 * d * 2 * d + 2 * d, (int)networks[GateNetwork].ParameterCount);
        Assert.Equal(2 * d * d + d, (int)networks[CandidateNetwork].ParameterCount);
        // Prior h -> D -> (mean, std) of S; posterior [h, e] -> D -> (mean, std) of S.
        Assert.Equal(d * d + d + d * 2 * s + 2 * s, (int)networks[PriorNetwork].ParameterCount);
        Assert.Equal((d + hidden) * d + d + d * 2 * s + 2 * s, (int)networks[PosteriorNetwork].ParameterCount);
        // The action and value models read the full model state s_t = (h_t, z_t).
        Assert.Equal((d + s) * hidden + hidden + 2 * (hidden * hidden + hidden) + hidden * 2 * ActionSize + 2 * ActionSize,
            (int)networks[ActorNetwork].ParameterCount);
        Assert.Equal((d + s) * hidden + hidden + 2 * (hidden * hidden + hidden) + hidden + 1,
            (int)networks[ValueNetwork].ParameterCount);
    }

    [Fact]
    public void BatchLength_BoundsTheSequencesTheWorldModelIsUnrolledOver()
    {
        var shortSequences = new DreamerAgent<double>(SmallOptions(batchLength: 5));
        Feed(shortSequences, 60, episodeLength: 30);
        shortSequences.Train();
        Assert.Equal(5, shortSequences.LastSequenceLength);

        // Longer than every stored episode: the agent trains on the longest within-episode runs it has.
        var longSequences = new DreamerAgent<double>(SmallOptions(batchLength: 50));
        Feed(longSequences, 60, episodeLength: 8);
        longSequences.Train();
        Assert.Equal(8, longSequences.LastSequenceLength);
    }

    [Fact]
    public void SampledSequences_NeverCrossAnEpisodeBoundary()
    {
        var replay = new DreamerAgent<double>.SequenceReplay(64);
        for (int t = 0; t < 24; t++)
        {
            replay.Add(new Experience<double, Vector<double>, Vector<double>>(
                new Vector<double>(1), new Vector<double>(1), 0.0, new Vector<double>(1), (t + 1) % 8 == 0));
        }

        var rng = RandomHelper.CreateSeededRandom(5);
        foreach (int start in replay.SampleStarts(8, 300, rng)) Assert.Equal(0, start % 8);
        foreach (int start in replay.SampleStarts(3, 300, rng)) Assert.True(start % 8 <= 5, $"Start {start} runs past its episode's end.");
        Assert.Equal(8, replay.LongestSequence(50));
    }

    [Fact]
    public void SequenceReplay_OverwritesTheOldestTransitionWhenFull()
    {
        var replay = new DreamerAgent<double>.SequenceReplay(4);
        for (int t = 0; t < 6; t++)
        {
            replay.Add(new Experience<double, Vector<double>, Vector<double>>(
                new Vector<double>(new[] { (double)t }), new Vector<double>(1), t, new Vector<double>(1), false));
        }

        Assert.Equal(4, replay.Count);
        for (int i = 0; i < 4; i++) Assert.Equal(i + 2.0, replay.At(i).Reward);
    }

    [Fact]
    public void FreeNats_StopTheKlFromTrainingThePrior()
    {
        // The prior p(z_t | h_t) appears in the world-model objective only through the KL term, so free nats above
        // any reachable divergence must leave it untouched while a zero threshold trains it.
        var clipped = new DreamerAgent<double>(SmallOptions(freeNats: 1e6));
        var open = new DreamerAgent<double>(SmallOptions(freeNats: 0.0));
        Feed(clipped, 120, 20);
        Feed(open, 120, 20);
        clipped.Train();
        open.Train();
        var clippedBefore = NetworkParameters(clipped, PriorNetwork);
        var openBefore = NetworkParameters(open, PriorNetwork);
        for (int i = 0; i < 3; i++)
        {
            clipped.Train();
            open.Train();
        }

        Assert.Equal(0.0, MaxAbsDifference(clippedBefore, NetworkParameters(clipped, PriorNetwork)));
        Assert.True(MaxAbsDifference(openBefore, NetworkParameters(open, PriorNetwork)) > 0.0,
            "With zero free nats the KL term must train the prior.");
    }

    [Fact]
    public void KLScale_LargerBetaGivesASmallerPosteriorPriorDivergence()
    {
        var unregularized = Trained(SmallOptions(klScale: 0.0, freeNats: 0.0), steps: 40);
        var regularized = Trained(SmallOptions(klScale: 10.0, freeNats: 0.0), steps: 40);

        double free = unregularized.LastWorldModelTerms.Divergence;
        double penalized = regularized.LastWorldModelTerms.Divergence;
        Assert.True(penalized < free,
            $"beta = 10 should pull the posterior toward the prior: KL {penalized} vs {free} at beta = 0.");
    }

    [Fact]
    public void RewardScale_WeightsTheRewardLikelihood()
    {
        var ignored = Trained(SmallOptions(rewardScale: 0.0), steps: 1);
        var rewardBefore = NetworkParameters(ignored, RewardNetwork);
        for (int i = 0; i < 3; i++) ignored.Train();
        Assert.Equal(0.0, MaxAbsDifference(rewardBefore, NetworkParameters(ignored, RewardNetwork)));

        var weighted = Trained(SmallOptions(rewardScale: 1.0), steps: 40);
        var unweighted = Trained(SmallOptions(rewardScale: 0.0), steps: 40);
        Assert.True(weighted.LastWorldModelTerms.Reward < unweighted.LastWorldModelTerms.Reward,
            $"Reward error with RewardScale 1 ({weighted.LastWorldModelTerms.Reward}) should be below the untrained "
            + $"reward model's ({unweighted.LastWorldModelTerms.Reward}).");
    }

    [Fact]
    public void ContinueScale_TrainsTheDiscountPredictorAndZeroDisablesIt()
    {
        var disabled = Trained(SmallOptions(continueScale: 0.0), steps: 1, episodeLength: 5);
        var continueBefore = NetworkParameters(disabled, ContinueNetwork);
        for (int i = 0; i < 3; i++) disabled.Train();
        Assert.Equal(0.0, MaxAbsDifference(continueBefore, NetworkParameters(disabled, ContinueNetwork)));
        Assert.Equal(0.0, disabled.LastWorldModelTerms.Continue);

        // Enabled, the binary classifier fits the soft labels 0 and gamma (Appendix A): its loss falls from the
        // untrained log 2 toward the labels' entropy.
        var enabled = Trained(SmallOptions(continueScale: 10.0), steps: 1, episodeLength: 5);
        double initial = enabled.LastWorldModelTerms.Continue;
        for (int i = 0; i < 40; i++) enabled.Train();
        double fitted = enabled.LastWorldModelTerms.Continue;
        Assert.True(fitted < initial, $"The discount predictor did not learn: loss {initial} -> {fitted}.");
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1e6)]
    public void WorldModelObjective_IsEquation10WithTheConfiguredWeights(double freeNats)
    {
        const double klScale = 0.5, rewardScale = 2.0, continueScale = 3.0;
        var agent = Trained(SmallOptions(klScale: klScale, freeNats: freeNats, rewardScale: rewardScale, continueScale: continueScale), steps: 1, episodeLength: 7);
        var terms = agent.LastWorldModelTerms;

        double expected = terms.Reconstruction
            + rewardScale * terms.Reward
            + klScale * Math.Max(terms.Divergence, freeNats)
            + continueScale * terms.Continue;
        Assert.Equal(expected, terms.Total, 6);
        Assert.True(terms.Continue > 0.0 && terms.Reconstruction > 0.0 && terms.Divergence >= 0.0);
    }

    [Fact]
    public void Lambda_ReachesTheValueTargets()
    {
        var oneStep = Trained(SmallOptions(lambda: 0.0), steps: 2);
        var fullHorizon = Trained(SmallOptions(lambda: 1.0), steps: 2);
        Assert.True(MaxAbsDifference(NetworkParameters(oneStep, ValueNetwork), NetworkParameters(fullHorizon, ValueNetwork)) > 0.0,
            "Two agents that differ only in lambda trained identical value models, so lambda never reached the targets.");
        // Lambda does not touch the world model, which is trained first on the same data and seed.
        Assert.Equal(0.0, MaxAbsDifference(NetworkParameters(oneStep, PriorNetwork), NetworkParameters(fullHorizon, PriorNetwork)));
    }

    [Fact]
    public void Acting_FiltersTheEpisodeHistory_AndPredictIsTheFirstStepOfAnEpisode()
    {
        var agent = Trained(SmallOptions(), steps: 2);
        var observation = new Vector<double>(new[] { 0.4, -0.2, 0.9, 0.1 });
        var other = new Vector<double>(new[] { -0.8, 0.6, -0.3, 0.5 });

        agent.ResetEpisode();
        var fresh = agent.SelectAction(observation, training: false);
        var predicted = agent.Predict(observation);
        for (int i = 0; i < ActionSize; i++) Assert.Equal(predicted[i], fresh[i]);

        // The same observation after a different history is a different model state, so a different action.
        agent.ResetEpisode();
        agent.SelectAction(other, training: false);
        var afterHistory = agent.SelectAction(observation, training: false);
        bool differs = false;
        for (int i = 0; i < ActionSize; i++) differs |= Math.Abs(afterHistory[i] - fresh[i]) > 1e-12;
        Assert.True(differs, "The action ignored the episode history the RSSM filters.");

        for (int i = 0; i < ActionSize; i++) Assert.InRange(afterHistory[i], -1.0, 1.0);
    }

    [Fact]
    public void Options_DefaultToThePaperHyperparameters()
    {
        var options = new DreamerOptions<double>();
        Assert.Equal(200, options.DeterministicSize);
        Assert.Equal(30, options.StochasticSize);
        Assert.Equal(300, options.HiddenSize);
        Assert.Equal(50, options.BatchLength);
        Assert.Equal(50, options.BatchSize);
        Assert.Equal(15, options.ImaginationHorizon);
        Assert.Equal(0.95, options.Lambda);
        Assert.Equal(1.0, options.KLScale);
        Assert.Equal(3.0, options.FreeNats);
        Assert.Equal(1.0, options.RewardScale);
        Assert.Equal(10.0, options.ContinueScale);
        // Appendix A: 6e-4 for the world model, 8e-5 for the value and action models.
        Assert.Equal(6e-4, options.LearningRate);
        Assert.Equal(8e-5, options.ActorLearningRate);
        Assert.Equal(8e-5, options.ValueLearningRate);
    }

    /// <summary>
    /// One agent per rate, identical except for <paramref name="model"/>'s learning rate, trained on the same data:
    /// the displacement of every network on the second update (the first sizes the lazy layers), and every network's
    /// parameters after the first update.
    /// </summary>
    private static (double[] Displacement, double[][] AfterFirst) TrainWithRate(int model, double rate)
    {
        var options = model == ActorNetwork
            ? SmallOptions(actorLearningRate: rate)
            : SmallOptions(valueLearningRate: rate);
        var agent = new DreamerAgent<double>(options);
        Feed(agent, 120, 20);
        agent.Train();
        var afterFirst = new double[11][];
        for (int i = 0; i < afterFirst.Length; i++) afterFirst[i] = NetworkParameters(agent, i);
        agent.Train();
        var displacement = new double[afterFirst.Length];
        for (int i = 0; i < afterFirst.Length; i++)
        {
            var after = NetworkParameters(agent, i);
            double sum = 0.0;
            for (int j = 0; j < after.Length; j++) sum += (after[j] - afterFirst[i][j]) * (after[j] - afterFirst[i][j]);
            displacement[i] = Math.Sqrt(sum);
        }
        return (displacement, afterFirst);
    }

    [Theory]
    [InlineData(ActorNetwork)]
    [InlineData(ValueNetwork)]
    public void ActorAndValueLearningRates_EachDriveOnlyTheirOwnModel(int model)
    {
        var fast = TrainWithRate(model, 1e-3);
        var slow = TrainWithRate(model, 1e-9);

        // An Adam step moves each weight by about its rate, so a rate a million times larger must move the model
        // about that much further. A model that ignored its option would train at the same rate in both agents.
        Assert.True(fast.Displacement[model] > 1e3 * slow.Displacement[model],
            $"Network {model} moved {fast.Displacement[model]} at rate 1e-3 and {slow.Displacement[model]} at 1e-9.");

        // The rate is that model's alone: after the first update every other network is bit-identical. (The world
        // model trains before behaviour, and the value targets are formed before the action model steps.)
        for (int i = 0; i < 11; i++)
        {
            if (i == model) continue;
            Assert.Equal(0.0, MaxAbsDifference(fast.AfterFirst[i], slow.AfterFirst[i]));
        }
        Assert.True(MaxAbsDifference(fast.AfterFirst[model], slow.AfterFirst[model]) > 0.0);
    }

    /// <summary>Plain SGD, p -= lr * g: no momentum, no regularization, no clipping of its own.</summary>
    private static IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> PlainSgd(double rate) =>
        new GradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new GradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = rate,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
                RegularizationOptions = new RegularizationOptions { Type = RegularizationType.None, Strength = 0.0 },
            });

    [Fact]
    public void GradientClipping_BoundsTheWorldModelAsOneGroup()
    {
        // Appendix A clips gradient norms at 100, and the authors' implementation clips each optimizer's whole
        // variable set, so the world model's networks share ONE norm. With plain SGD the world model's step is then
        // exactly lr * 100 in total whenever the joint gradient exceeds 100, which a large reward weight guarantees.
        // Clipping each network on its own bounds every network separately instead: the joint step is then
        // lr * sqrt(sum_i min(|g_i|, 100)^2), which is larger as soon as two networks receive gradient (and is the
        // unclipped lr * |g| when each network alone stays under 100 while their joint norm does not).
        const double Rate = 1e-4;
        var agent = new DreamerAgent<double>(SmallOptions(rewardScale: 1e4, optimizerFactory: () => PlainSgd(Rate)));
        Feed(agent, 120, 20);
        agent.Train();   // sizes the lazy layers

        var worldModel = new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
        var before = new double[worldModel.Length][];
        for (int k = 0; k < worldModel.Length; k++) before[k] = NetworkParameters(agent, worldModel[k]);
        agent.Train();

        double total = 0.0;
        int moved = 0;
        for (int k = 0; k < worldModel.Length; k++)
        {
            var after = NetworkParameters(agent, worldModel[k]);
            double own = 0.0;
            for (int j = 0; j < after.Length; j++) own += (after[j] - before[k][j]) * (after[j] - before[k][j]);
            total += own;
            if (own > 0.0) moved++;
        }
        total = Math.Sqrt(total);

        Assert.True(moved >= 2, $"Only {moved} world-model network(s) received gradient, so the test cannot tell a group clip from a per-network one.");
        Assert.True(Math.Abs(total - Rate * 100.0) <= 1e-6 * Rate * 100.0,
            $"The world model moved {total} on one SGD step at lr {Rate}; one group clipped at norm 100 moves exactly {Rate * 100.0}.");
    }

    public static IEnumerable<object[]> InvalidOptions()
    {
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, DeterministicSize = 0 } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, StochasticSize = 0 } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, HiddenSize = 0 } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, BatchLength = 0 } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, ImaginationHorizon = 0 } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, KLScale = -1.0 } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, FreeNats = -0.5 } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, RewardScale = double.NaN } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, ContinueScale = double.PositiveInfinity } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, Lambda = 1.5 } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, ActorLearningRate = 0.0 } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, ValueLearningRate = -1e-4 } };
        yield return new object[] { new DreamerOptions<double> { ObservationSize = 4, ActionSize = 2, ActorLearningRate = double.NaN } };
    }

    [Theory]
    [MemberData(nameof(InvalidOptions))]
    public void Constructor_RejectsOutOfRangeOptions(DreamerOptions<double> options)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DreamerAgent<double>(options));
    }
}
