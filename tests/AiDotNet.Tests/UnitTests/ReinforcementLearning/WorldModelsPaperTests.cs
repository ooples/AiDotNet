using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.WorldModels;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Checks WorldModelsAgent against Ha &amp; Schmidhuber 2018 ("Recurrent World Models Facilitate Policy Evolution"):
/// V's ELBO, M's mixture-density loss and temperature sampling, the linear controller, the dream rollout, the
/// V -> M -> C training cycle and the CMA-ES budget, each driven by the option that names it.
/// </summary>
public class WorldModelsPaperTests
{
    private const int ObservationSize = 4;
    private const int ActionSize = 2;

    private static WorldModelsOptions<double> Small(
        int latent = 2,
        int hidden = 4,
        int mixtures = 2,
        int rnnLayers = 1,
        double temperature = 1.0,
        double vaeBeta = 1.0,
        int vaeEpochs = 1,
        int rnnEpochs = 1,
        int sequenceLength = 1000,
        int generations = 1,
        int population = 4,
        int evaluationRollouts = 1,
        int rolloutLength = 2,
        int batchSize = 8,
        double learningRate = 0.001,
        int seed = 17) => new WorldModelsOptions<double>
        {
            ObservationWidth = ObservationSize,
            ObservationHeight = 1,
            ObservationChannels = 1,
            ActionSize = ActionSize,
            LatentSize = latent,
            RNNHiddenSize = hidden,
            NumMixtures = mixtures,
            RNNLayers = rnnLayers,
            Temperature = temperature,
            VAEBeta = vaeBeta,
            VAEEncoderChannels = new List<int> { 8 },
            VAEEpochs = vaeEpochs,
            RNNEpochs = rnnEpochs,
            RNNSequenceLength = sequenceLength,
            ControllerGenerations = generations,
            ControllerPopulationSize = population,
            ControllerEvaluationRollouts = evaluationRollouts,
            RolloutLength = rolloutLength,
            BatchSize = batchSize,
            ReplayBufferSize = 256,
            LearningRate = learningRate,
            Seed = seed,
        };

    private static Vector<double> Uniform(Random rng, int length)
    {
        var v = new Vector<double>(length);
        for (int i = 0; i < length; i++) v[i] = rng.NextDouble();
        return v;
    }

    private static void Fill(WorldModelsAgent<double> agent, int count, int seed, int? doneAt = null)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var state = Uniform(rng, ObservationSize);
        for (int t = 0; t < count; t++)
        {
            var next = Uniform(rng, ObservationSize);
            var action = new Vector<double>(new[] { rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1 });
            agent.StoreExperience(state, action, action[0], next, doneAt == t);
            state = next;
        }
    }

    private static double[] Snapshot(Vector<double> v) => Enumerable.Range(0, v.Length).Select(i => v[i]).ToArray();

    private static bool Differs(double[] a, Vector<double> b)
        => a.Length != b.Length || Enumerable.Range(0, a.Length).Any(i => a[i] != b[i]);

    /// <summary>Zeroes M's head weights and sets its biases, so every head output is that constant.</summary>
    private static void PinHead(WorldModelsAgent<double> agent, int rewardIndex, double reward, double doneLogit)
    {
        var weights = agent.MixtureHead.GetWeights();
        for (int i = 0; i < weights.Length; i++) weights[i] = 0.0;
        var biases = agent.MixtureHead.GetBiases();
        for (int i = 0; i < biases.Length; i++) biases[i] = 0.0;
        biases[rewardIndex] = reward;
        biases[rewardIndex + 1] = doneLogit;
    }

    // ── C: the controller ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Controller_IsThePapersLinearLayer_With867ParametersForCarRacing()
    {
        // Section "Controller (C) Model": a_t = W_c [z_t h_t] + b_c with z in R^32, h in R^256 and a 3-d action,
        // which the paper counts as 867 parameters.
        var agent = new WorldModelsAgent<double>(new WorldModelsOptions<double> { ActionSize = 3 });
        Assert.Equal(867, agent.ControllerParameterLength);
    }

    [Fact]
    public void ControllerLayers_InsertsHiddenTanhLayers()
    {
        var agent = new WorldModelsAgent<double>(new WorldModelsOptions<double>
        {
            ActionSize = 3,
            ControllerLayers = new List<int> { 40 },
        });
        Assert.Equal((288 * 40 + 40) + (40 * 3 + 3), agent.ControllerParameterLength);
    }

    // ── V: the VAE ────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void VaeObjective_IsReconstructionPlusBetaTimesKl()
    {
        var agent = new WorldModelsAgent<double>(Small(latent: 1));
        var reconstruction = new Tensor<double>(new[] { 1, 2 }); reconstruction[0] = 0.2; reconstruction[1] = 0.7;
        var observation = new Tensor<double>(new[] { 1, 2 }); observation[0] = 0.0; observation[1] = 1.0;
        var mu = new Tensor<double>(new[] { 1, 1 }); mu[0] = 0.5;
        var logVar = new Tensor<double>(new[] { 1, 1 }); logVar[0] = 0.3;

        double l2 = 0.2 * 0.2 + 0.3 * 0.3;
        double kl = -0.5 * (1.0 + 0.3 - 0.25 - Math.Exp(0.3));

        foreach (double beta in new[] { 0.0, 1.0, 2.5 })
        {
            double value = agent.VaeObjective(reconstruction, observation, mu, logVar, beta, 1.0)[0];
            Assert.Equal(l2 + beta * kl, value, 10);
        }
    }

    [Fact]
    public void VAEBeta_PullsThePosteriorTowardsThePrior()
    {
        double MeanKl(double beta)
        {
            var agent = new WorldModelsAgent<double>(Small(latent: 3, vaeBeta: beta, vaeEpochs: 1000, learningRate: 0.01));
            Fill(agent, 16, seed: 3);
            for (int i = 0; i < 80; i++) agent.Train();
            Assert.Equal(WorldModelsTrainingStage.Vision, agent.CurrentStage);

            var rng = RandomHelper.CreateSeededRandom(3);
            var observations = new Tensor<double>(new[] { 16, ObservationSize });
            for (int i = 0; i < observations.Length; i++) observations[i] = rng.NextDouble();
            var (mean, logVar) = agent.Encode(observations);
            double kl = 0.0;
            for (int i = 0; i < mean.Length; i++)
                kl += -0.5 * (1.0 + logVar[i] - mean[i] * mean[i] - Math.Exp(logVar[i]));
            return kl / 16.0;
        }

        double free = MeanKl(0.0);
        double weighted = MeanKl(50.0);
        Assert.True(weighted < 0.5 * free,
            $"A KL weight of 50 should pull q(z|x) much closer to N(0, I) than a weight of 0: KL {weighted} vs {free}.");
    }

    // ── M: the MDN-RNN ────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void MdnObjective_IsMixtureNllPlusRewardAndDoneLosses_OverValidStepsOnly()
    {
        var agent = new WorldModelsAgent<double>(Small(latent: 1, mixtures: 2));
        // Layout per step: logits(2) | means(2) | log sigmas(2) | reward | done logit.
        double[] step = { 0.0, 1.0, 0.5, -1.0, 0.0, Math.Log(0.5), 0.3, 0.2 };
        var output = new Tensor<double>(new[] { 1, 2, 8 });
        for (int j = 0; j < 8; j++) { output[j] = step[j]; output[8 + j] = 7.0 * (j + 1); }
        var z = new Tensor<double>(new[] { 1, 2, 1 }); z[0] = 0.2; z[1] = 5.0;
        var reward = new Tensor<double>(new[] { 1, 2, 1 }); reward[0] = 1.0; reward[1] = -3.0;
        var done = new Tensor<double>(new[] { 1, 2, 1 }); done[0] = 1.0; done[1] = 0.0;
        var mask = new Tensor<double>(new[] { 1, 2, 1 }); mask[0] = 1.0; mask[1] = 0.0;

        double Density(double x, double m, double s) => Math.Exp(-0.5 * ((x - m) / s) * ((x - m) / s)) / (s * Math.Sqrt(2 * Math.PI));
        double p0 = 1.0 / (1.0 + Math.E), p1 = Math.E / (1.0 + Math.E);
        double nll = -Math.Log(p0 * Density(0.2, 0.5, 1.0) + p1 * Density(0.2, -1.0, 0.5));
        double rewardLoss = (0.3 - 1.0) * (0.3 - 1.0);
        double doneLoss = Math.Log(1.0 + Math.Exp(0.2)) - 0.2;

        double value = agent.MdnObjective(output, z, reward, done, mask, 1.0)[0];
        Assert.Equal(nll + rewardLoss + doneLoss, value, 9);
    }

    [Fact]
    public void Temperature_FlattensTheMixtureWeights_AndScalesSigmaBySqrtTau()
    {
        // Two components: logits (0, 1), means (-10, +10), sigma 0.01. Component 1 is drawn with probability
        // e^(1/tau) / (1 + e^(1/tau)).
        double PositiveFraction(double tau)
        {
            var agent = new WorldModelsAgent<double>(Small(latent: 1, mixtures: 2, temperature: tau));
            var output = new Tensor<double>(new[] { 8 });
            output[0] = 0.0; output[1] = 1.0; output[2] = -10.0; output[3] = 10.0;
            output[4] = Math.Log(0.01); output[5] = Math.Log(0.01);
            int positive = 0;
            for (int i = 0; i < 4000; i++) if (agent.SampleMixtureLatent(output, 0, 0) > 0) positive++;
            return positive / 4000.0;
        }

        Assert.InRange(PositiveFraction(0.25), Math.Exp(4.0) / (1.0 + Math.Exp(4.0)) - 0.02,
            Math.Exp(4.0) / (1.0 + Math.Exp(4.0)) + 0.02);
        Assert.InRange(PositiveFraction(4.0), Math.Exp(0.25) / (1.0 + Math.Exp(0.25)) - 0.04,
            Math.Exp(0.25) / (1.0 + Math.Exp(0.25)) + 0.04);

        // One component, mean 0, sigma 1: the sample variance is tau.
        double Variance(double tau)
        {
            var agent = new WorldModelsAgent<double>(Small(latent: 1, mixtures: 1, temperature: tau));
            var output = new Tensor<double>(new[] { 5 });
            var samples = Enumerable.Range(0, 4000).Select(_ => agent.SampleMixtureLatent(output, 0, 0)).ToArray();
            double m = samples.Average();
            return samples.Sum(s => (s - m) * (s - m)) / samples.Length;
        }

        Assert.InRange(Variance(1.0), 0.9, 1.1);
        Assert.InRange(Variance(4.0), 3.6, 4.4);
    }

    [Fact]
    public void RNNLayers_StacksLstmLayers()
    {
        int MemoryParameters(int layers)
        {
            var agent = new WorldModelsAgent<double>(Small(hidden: 4, rnnLayers: layers));
            Fill(agent, 4, seed: 5);
            agent.DreamReturn(agent.GetControllerParameters()); // materialises M's lazy layers
            return agent.GetNetworks()[2].GetParameters().Length;
        }

        // A stacked LSTM layer of H units fed by H units adds 4H(H + H) weights and 4H biases.
        Assert.Equal(8 * 4 * 4 + 4 * 4, MemoryParameters(2) - MemoryParameters(1));
    }

    [Fact]
    public void RNNSequenceLength_CutsTheTimelineIntoWindowsThatStopAtEpisodeEnds()
    {
        var shortWindows = new WorldModelsAgent<double>(Small(sequenceLength: 4));
        Fill(shortWindows, 16, seed: 7, doneAt: 5);
        // [0..3] [4..5] [6..9] [10..13] [14..15]
        Assert.Equal(5, shortWindows.SequenceWindowCount);

        var longWindows = new WorldModelsAgent<double>(Small(sequenceLength: 16));
        Fill(longWindows, 16, seed: 7, doneAt: 5);
        // [0..5] [6..15]
        Assert.Equal(2, longWindows.SequenceWindowCount);
    }

    // ── The dream ─────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Dream_AccumulatesPredictedReward_ForRolloutLengthSteps_AndStopsWhenDoneIsLikely()
    {
        const int latent = 2, mixtures = 2;
        int rewardIndex = 3 * latent * mixtures;
        var agent = new WorldModelsAgent<double>(Small(latent: latent, mixtures: mixtures, rolloutLength: 7, evaluationRollouts: 3));
        Fill(agent, 4, seed: 11);
        var controller = agent.GetControllerParameters();
        agent.DreamReturn(controller); // materialises M's lazy layers

        PinHead(agent, rewardIndex, reward: 1.0, doneLogit: -1.0);
        long before = agent.DreamStepsSimulated;
        Assert.Equal(7.0, agent.DreamReturn(controller), 12);
        Assert.Equal(7 * 3, agent.DreamStepsSimulated - before);

        // P(done) = sigmoid(1) > 0.5: every rollout collects its first reward and ends.
        PinHead(agent, rewardIndex, reward: 1.0, doneLogit: 1.0);
        Assert.Equal(1.0, agent.DreamReturn(controller), 12);
    }

    // ── The V -> M -> C cycle ─────────────────────────────────────────────────────────────────────

    [Fact]
    public void Train_RunsVForVAEEpochs_ThenMForRNNEpochs_ThenOneCmaEsRun_ThenRestarts()
    {
        var agent = new WorldModelsAgent<double>(Small(vaeEpochs: 2, rnnEpochs: 3, sequenceLength: 4, batchSize: 8));
        Fill(agent, 16, seed: 13);
        agent.DreamReturn(agent.GetControllerParameters()); // materialises every network
        var networks = agent.GetNetworks();
        var encoder = Snapshot(networks[0].GetParameters());
        var memory = Snapshot(networks[2].GetParameters());

        // 16 transitions at a batch of 8: two calls per V epoch, four calls for two epochs.
        for (int call = 0; call < 4; call++)
        {
            Assert.Equal(WorldModelsTrainingStage.Vision, agent.CurrentStage);
            agent.Train();
        }
        Assert.True(Differs(encoder, networks[0].GetParameters()), "V did not train in the Vision stage.");
        Assert.False(Differs(memory, networks[2].GetParameters()), "M moved while V was training.");

        // Four windows of 4 fit one batch: one call per M epoch, three calls for three epochs.
        encoder = Snapshot(networks[0].GetParameters());
        for (int call = 0; call < 3; call++)
        {
            Assert.Equal(WorldModelsTrainingStage.Memory, agent.CurrentStage);
            agent.Train();
        }
        Assert.True(Differs(memory, networks[2].GetParameters()), "M did not train in the Memory stage.");
        Assert.False(Differs(encoder, networks[0].GetParameters()), "V moved while M was training.");

        Assert.Equal(WorldModelsTrainingStage.Controller, agent.CurrentStage);
        memory = Snapshot(networks[2].GetParameters());
        int evaluations = agent.DreamEvaluations;
        agent.Train();
        Assert.True(agent.DreamEvaluations > evaluations, "The Controller stage did not evolve C in the dream.");
        Assert.False(Differs(memory, networks[2].GetParameters()), "M moved while C was evolving.");
        Assert.Equal(WorldModelsTrainingStage.Vision, agent.CurrentStage);
    }

    [Fact]
    public void ControllerStage_SpendsGenerationsTimesPopulationEvaluations_EachOverTheConfiguredRollouts()
    {
        const int latent = 2, mixtures = 2;
        var agent = new WorldModelsAgent<double>(Small(latent: latent, mixtures: mixtures, generations: 3, population: 5,
            evaluationRollouts: 2, rolloutLength: 3, batchSize: 8));
        Fill(agent, 8, seed: 19);
        agent.Train(); // V: one epoch of 8
        agent.Train(); // M: one epoch of one window
        Assert.Equal(WorldModelsTrainingStage.Controller, agent.CurrentStage);

        PinHead(agent, 3 * latent * mixtures, reward: 0.5, doneLogit: -1.0);
        int evaluations = agent.DreamEvaluations;
        long steps = agent.DreamStepsSimulated;
        double loss = agent.Train();

        // CMA-ES scores its start point, then generations x population candidates; the agent rescores the winner.
        int expectedEvaluations = 1 + 3 * 5 + 1;
        Assert.Equal(expectedEvaluations, agent.DreamEvaluations - evaluations);
        Assert.Equal(expectedEvaluations * 2L * 3L, agent.DreamStepsSimulated - steps);
        // Every dream step pays 0.5 for 3 steps: the return is 1.5, reported as the loss -1.5.
        Assert.Equal(-1.5, loss, 12);
    }

    // ── Validation ────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Constructor_RejectsOutOfRangeOptions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new WorldModelsAgent<double>(Small(temperature: 0.0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WorldModelsAgent<double>(Small(vaeBeta: -1.0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WorldModelsAgent<double>(Small(population: 3)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WorldModelsAgent<double>(Small(rnnLayers: 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WorldModelsAgent<double>(Small(generations: 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WorldModelsAgent<double>(Small(rolloutLength: 0)));
    }
}
