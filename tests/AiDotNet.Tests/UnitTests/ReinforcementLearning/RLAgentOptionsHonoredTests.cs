using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Agents.A2C;
using AiDotNet.ReinforcementLearning.Agents.A3C;
using AiDotNet.ReinforcementLearning.Agents.CQL;
using AiDotNet.ReinforcementLearning.Agents.Dreamer;
using AiDotNet.ReinforcementLearning.Agents.MADDPG;
using AiDotNet.ReinforcementLearning.Agents.MuZero;
using AiDotNet.ReinforcementLearning.Agents.Rainbow;
using AiDotNet.ReinforcementLearning.Agents.SAC;
using AiDotNet.ReinforcementLearning.Agents.TRPO;
using AiDotNet.ReinforcementLearning.Agents.WorldModels;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Each test fails if the agent ignores the option it names (issue #2090): the option must change what the
/// update does, by the amount the paper says it should.
/// </summary>
public class RLAgentOptionsHonoredTests
{
    private const int StateSize = 4;

    private static List<(Vector<double> State, Vector<double> Action, double Reward, Vector<double> Next, bool Done)>
        DiscreteData(int count, int actionSize, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var data = new List<(Vector<double>, Vector<double>, double, Vector<double>, bool)>();
        for (int t = 0; t < count; t++)
        {
            var state = new Vector<double>(StateSize);
            var next = new Vector<double>(StateSize);
            for (int i = 0; i < StateSize; i++)
            {
                state[i] = rng.NextDouble() * 2.0 - 1.0;
                next[i] = rng.NextDouble() * 2.0 - 1.0;
            }
            int chosen = rng.Next(actionSize);
            var action = new Vector<double>(actionSize);
            action[chosen] = 1.0;
            // Action 0 pays off when state[0] is positive, so the advantages carry a learnable signal.
            double reward = (chosen == 0 ? 1.0 : -1.0) * state[0];
            data.Add((state, action, reward, next, t % 8 == 7));
        }
        return data;
    }

    private static double MeanKl(IReadOnlyList<Vector<double>> before, IReadOnlyList<Vector<double>> after)
    {
        double total = 0.0;
        for (int s = 0; s < before.Count; s++)
        {
            for (int k = 0; k < before[s].Length; k++)
            {
                total += before[s][k] * Math.Log(before[s][k] / after[s][k]);
            }
        }
        return total / before.Count;
    }

    private static double TrpoUpdateKl(double maxKl, out double reportedKl)
    {
        var agent = new TRPOAgent<double>(new TRPOOptions<double>
        {
            StateSize = StateSize,
            ActionSize = 2,
            IsContinuous = false,
            StepsPerUpdate = 100_000,
            MaxKL = maxKl,
            PolicyHiddenLayers = new List<int> { 16 },
            ValueHiddenLayers = new List<int> { 16 },
            Seed = 3,
        });
        var data = DiscreteData(64, 2, seed: 11);
        foreach (var (state, action, reward, next, done) in data) agent.StoreExperience(state, action, reward, next, done);

        var before = data.Select(d => agent.PolicyDistribution(d.State)).ToList();
        agent.Train();
        var after = data.Select(d => agent.PolicyDistribution(d.State)).ToList();
        reportedKl = agent.LastPolicyKL;
        return MeanKl(before, after);
    }

    [Fact]
    public void TRPO_PolicyStepStaysInsideMaxKL_AndShrinksWithIt()
    {
        double wideKl = TrpoUpdateKl(0.01, out double wideReported);
        double narrowKl = TrpoUpdateKl(0.0005, out double narrowReported);

        // The update moved the policy, and its KL over the update's states respects the trust region.
        Assert.True(wideKl > 0.0, $"The TRPO step did not change the policy (KL {wideKl}).");
        Assert.True(narrowKl > 0.0, $"The TRPO step did not change the policy (KL {narrowKl}).");
        Assert.True(wideKl <= 0.01 * (1 + 1e-6), $"KL {wideKl} exceeds MaxKL 0.01.");
        Assert.True(narrowKl <= 0.0005 * (1 + 1e-6), $"KL {narrowKl} exceeds MaxKL 0.0005.");
        Assert.True(narrowKl < wideKl, $"A smaller MaxKL must give a smaller step: {narrowKl} vs {wideKl}.");

        // The agent's own accounting agrees with the independently computed divergence.
        Assert.Equal(wideKl, wideReported, 6);
        Assert.Equal(narrowKl, narrowReported, 6);
    }

    private static double TrpoValueDisplacement(double valueLearningRate, int valueIterations)
    {
        var agent = new TRPOAgent<double>(new TRPOOptions<double>
        {
            StateSize = StateSize,
            ActionSize = 2,
            StepsPerUpdate = 100_000,
            ValueLearningRate = valueLearningRate,
            ValueIterations = valueIterations,
            PolicyHiddenLayers = new List<int> { 8 },
            ValueHiddenLayers = new List<int> { 8 },
            Seed = 5,
        });
        foreach (var (state, action, reward, next, done) in DiscreteData(32, 2, seed: 13))
            agent.StoreExperience(state, action, reward, next, done);

        var before = agent.GetValueNetworkParameters();
        agent.Train();
        var after = agent.GetValueNetworkParameters();
        double sum = 0.0;
        for (int i = 0; i < before.Length; i++) sum += (after[i] - before[i]) * (after[i] - before[i]);
        return Math.Sqrt(sum);
    }

    [Fact]
    public void TRPO_ValueRegressionFollowsValueLearningRateAndIterations()
    {
        double fast = TrpoValueDisplacement(1e-2, 1);
        double slow = TrpoValueDisplacement(1e-5, 1);
        double many = TrpoValueDisplacement(1e-2, 5);

        // An Adam step moves each weight by about the learning rate, so a 1000x rate moves the value network
        // far more; and five regression passes move it further than one.
        Assert.True(fast > 100 * slow, $"ValueLearningRate ignored: displacement {fast} at 1e-2 vs {slow} at 1e-5.");
        Assert.True(many > 2 * fast, $"ValueIterations ignored: displacement {many} after 5 passes vs {fast} after 1.");
    }

    private static CQLAgent<double> TrainedCql(Action<CQLOptions<double>> configure)
    {
        var options = new CQLOptions<double>
        {
            StateSize = StateSize,
            ActionSize = 2,
            BatchSize = 8,
            BufferSize = 64,
            CQLNumActions = 4,
            PolicyHiddenLayers = new List<int> { 8 },
            QHiddenLayers = new List<int> { 8 },
            Seed = 17,
        };
        configure(options);
        var agent = new CQLAgent<double>(options);

        var rng = RandomHelper.CreateSeededRandom(23);
        var dataset = new List<(Vector<double>, Vector<double>, double, Vector<double>, bool)>();
        for (int t = 0; t < 32; t++)
        {
            var state = new Vector<double>(StateSize);
            var next = new Vector<double>(StateSize);
            for (int i = 0; i < StateSize; i++)
            {
                state[i] = rng.NextDouble() * 2.0 - 1.0;
                next[i] = rng.NextDouble() * 2.0 - 1.0;
            }
            var action = new Vector<double>(2);
            for (int k = 0; k < 2; k++) action[k] = rng.NextDouble() * 1.6 - 0.8;
            dataset.Add((state, action, rng.NextDouble(), next, t % 8 == 7));
        }
        agent.LoadOfflineData(dataset);
        agent.Train();
        return agent;
    }

    [Fact]
    public void CQL_LagrangeMultiplierMovesAgainstTheTargetActionGap()
    {
        const double qLearningRate = 3e-3;

        // gap - tau < 0 when tau is far above any gap: dual ascent lowers alpha'; far below, it raises it.
        var high = TrainedCql(o => { o.CQLLagrange = true; o.CQLTargetActionGap = 1e3; o.QLearningRate = qLearningRate; });
        var low = TrainedCql(o => { o.CQLLagrange = true; o.CQLTargetActionGap = -1e3; o.QLearningRate = qLearningRate; });
        var off = TrainedCql(o => { o.CQLLagrange = false; o.CQLTargetActionGap = -1e3; o.QLearningRate = qLearningRate; });

        Assert.True(high.ConservativeMultiplier < 1.0, $"alpha' should fall below 1 when tau exceeds the gap, was {high.ConservativeMultiplier}.");
        Assert.True(low.ConservativeMultiplier > 1.0, $"alpha' should rise above 1 when the gap exceeds tau, was {low.ConservativeMultiplier}.");
        Assert.Equal(1.0, off.ConservativeMultiplier);

        // One Adam step on log alpha' moves it by the critics' learning rate.
        Assert.Equal(qLearningRate, Math.Abs(Math.Log(high.ConservativeMultiplier)), 6);
        Assert.Equal(qLearningRate, Math.Abs(Math.Log(low.ConservativeMultiplier)), 6);
    }

    [Fact]
    public void CQL_EntropyTemperatureFollowsTargetEntropyAndAlphaLearningRate()
    {
        const double initial = 0.2;

        var raise = TrainedCql(o => { o.InitialTemperature = initial; o.TargetEntropy = 50.0; o.AlphaLearningRate = 1e-2; });
        var lower = TrainedCql(o => { o.InitialTemperature = initial; o.TargetEntropy = -50.0; o.AlphaLearningRate = 1e-2; });
        var slow = TrainedCql(o => { o.InitialTemperature = initial; o.TargetEntropy = -50.0; o.AlphaLearningRate = 1e-4; });

        // J(alpha) = -alpha (log pi + target): a target far above the policy's entropy raises alpha, far below lowers it.
        Assert.True(raise.EntropyTemperature > initial, $"alpha should rise toward a high target entropy, was {raise.EntropyTemperature}.");
        Assert.True(lower.EntropyTemperature < initial, $"alpha should fall toward a low target entropy, was {lower.EntropyTemperature}.");

        // One Adam step on log alpha moves it by AlphaLearningRate.
        Assert.Equal(1e-2, Math.Abs(Math.Log(lower.EntropyTemperature / initial)), 6);
        Assert.Equal(1e-4, Math.Abs(Math.Log(slow.EntropyTemperature / initial)), 6);
    }

    [Fact]
    public void Rainbow_NoisyNetSigmaSetsTheInitialNoiseScale()
    {
        RainbowDQNAgent<double> Build(double sigma) => new RainbowDQNAgent<double>(new RainbowDQNOptions<double>
        {
            StateSize = StateSize,
            ActionSize = 2,
            UseNoisyNetworks = true,
            NoisyNetSigma = sigma,
            SharedLayers = new List<int> { 32 },
            NumAtoms = 5,
        });

        var defaultSigma = Build(0.5).NoisyLayerSigmaInits;
        var smallSigma = Build(0.1).NoisyLayerSigmaInits;

        // Fortunato et al. 2018, Sec. 3.2: factorised noise starts at sigma_0 / sqrt(p) for p inputs.
        Assert.NotEmpty(defaultSigma);
        Assert.All(defaultSigma, s => Assert.Equal(0.5 / Math.Sqrt(32), s, 12));
        Assert.All(smallSigma, s => Assert.Equal(0.1 / Math.Sqrt(32), s, 12));
    }

    private static MuZeroAgent<double> SmallMuZero(
        double dirichletAlpha, double fraction,
        Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>? optimizerFactory = null) =>
        new MuZeroAgent<double>(new MuZeroOptions<double>
        {
            OptimizerFactory = optimizerFactory,
            ObservationSize = StateSize,
            ActionSize = 4,
            LatentStateSize = 6,
            RepresentationLayers = new List<int> { 8 },
            DynamicsLayers = new List<int> { 8 },
            PredictionLayers = new List<int> { 8 },
            RootDirichletAlpha = dirichletAlpha,
            RootExplorationFraction = fraction,
            BatchSize = 4,
            UnrollSteps = 2,
            TDSteps = 2,
            Seed = 29,
        });

    [Fact]
    public void MuZero_RootNoiseFollowsDirichletAlphaAndExplorationFraction()
    {
        var policy = new Vector<double>(new[] { 0.4, 0.3, 0.2, 0.1 });

        // Fraction 0 leaves the network's priors untouched.
        var untouched = SmallMuZero(0.3, 0.0).MixRootNoise(policy);
        for (int a = 0; a < 4; a++) Assert.Equal(policy[a], untouched[a], 12);

        // Fraction 1 is pure Dirichlet noise: a distribution, peaked for a small alpha and nearly flat for a large one.
        double MeanLargestShare(double alpha)
        {
            var agent = SmallMuZero(alpha, 1.0);
            double total = 0.0;
            for (int draw = 0; draw < 200; draw++)
            {
                var noise = agent.MixRootNoise(policy);
                Assert.Equal(1.0, noise.Sum(), 9);
                total += noise.Max();
            }
            return total / 200;
        }

        double sparse = MeanLargestShare(0.03);
        double flat = MeanLargestShare(100.0);
        Assert.True(sparse > 0.8, $"Dir(0.03) noise should concentrate on one action, mean largest share {sparse}.");
        Assert.True(flat < 0.35, $"Dir(100) noise should be nearly uniform over 4 actions, mean largest share {flat}.");
    }

    // The prediction head is linear, so MixRootNoise receives raw outputs that can be negative and need not sum to
    // one. PUCT needs a distribution: negatives clip to zero and the rest renormalize.
    [Fact]
    public void MuZero_RootPriorsFromRawHeadOutputs_AreADistribution()
    {
        var raw = new Vector<double>(new[] { 1.2, -0.5, 0.6, 0.2 });

        var clean = SmallMuZero(0.3, 0.0).MixRootNoise(raw);
        Assert.Equal(new[] { 0.6, 0.0, 0.3, 0.1 }, clean.Select(p => Math.Round(p, 12)).ToArray());

        var agent = SmallMuZero(0.3, 0.25);
        for (int draw = 0; draw < 50; draw++)
        {
            var mixed = agent.MixRootNoise(raw);
            Assert.Equal(1.0, mixed.Sum(), 9);
            Assert.All(mixed, p => Assert.True(p >= 0.0, $"negative prior {p}"));
        }

        // No positive output at all leaves a uniform prior rather than an all-zero one.
        var uniform = SmallMuZero(0.3, 0.0).PolicyPriors(new Vector<double>(new[] { -1.0, -2.0, 0.0, double.NaN }));
        Assert.All(uniform, p => Assert.Equal(0.25, p, 12));
    }

    [Fact]
    public void MuZero_TrainingRewritesReplayPrioritiesFromValueError()
    {
        var agent = SmallMuZero(0.3, 0.25);
        foreach (var (state, action, reward, next, done) in DiscreteData(16, 4, seed: 31))
            agent.StoreExperience(state, action, reward, next, done);

        // New transitions enter at the running maximum priority (1 before any update).
        Assert.All(agent.ReplayPriorities, p => Assert.Equal(1.0, p));

        agent.Train();

        // The sampled windows now carry |v(o_0) - z_0| + floor, which is not the placeholder 1.
        Assert.Contains(agent.ReplayPriorities, p => Math.Abs(p - 1.0) > 1e-9);
    }

    private static double A2CDisplacement(double valueLearningRate)
    {
        var agent = new A2CAgent<double>(new A2COptions<double>
        {
            StateSize = StateSize,
            ActionSize = 2,
            StepsPerUpdate = 16,
            PolicyLearningRate = 1e-7,
            ValueLearningRate = valueLearningRate,
            PolicyHiddenLayers = new List<int> { 8 },
            ValueHiddenLayers = new List<int> { 8 },
            Seed = 37,
        });
        // Storing runs both networks forward, which sizes their lazily built layers before the snapshot.
        foreach (var (state, action, reward, next, done) in DiscreteData(16, 2, seed: 41))
            agent.StoreExperience(state, action, reward, next, done);
        var before = agent.GetParameters();
        agent.Train();
        var after = agent.GetParameters();
        double sum = 0.0;
        for (int i = 0; i < before.Length; i++) sum += (after[i] - before[i]) * (after[i] - before[i]);
        return Math.Sqrt(sum);
    }

    [Fact]
    public void A2C_ValueNetworkTrainsAtValueLearningRate()
    {
        // With the policy rate negligible, the update's size is the value network's, which must scale with
        // ValueLearningRate rather than fall back to the network's built-in default.
        double fast = A2CDisplacement(1e-2);
        double slow = A2CDisplacement(1e-5);
        Assert.True(fast > 100 * slow, $"ValueLearningRate ignored: displacement {fast} at 1e-2 vs {slow} at 1e-5.");
    }

    // ── OptimizerFactory reaches every network ───────────────────────────────────────────────────────────

    private const double FastRate = 1e-3;
    private const double SlowRate = 1e-9;

    private static AdamOptimizer<double, Tensor<double>, Tensor<double>> Adam(double learningRate) =>
        new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = learningRate,
            UseAdaptiveLearningRate = false,
        });

    /// <summary>
    /// How far one update moves the agent's network weights. The first update sizes any lazily built layers, so
    /// the measured update is the second.
    /// </summary>
    private static double SecondUpdateDisplacement(DeepReinforcementLearningAgentBase<double> agent, Action feed)
    {
        feed();
        agent.Train();
        var before = agent.GetNetworkParameters();
        feed();
        agent.Train();
        var after = agent.GetNetworkParameters();

        Assert.Equal(before.Length, after.Length);
        double sum = 0.0;
        for (int i = 0; i < before.Length; i++) sum += (after[i] - before[i]) * (after[i] - before[i]);
        return Math.Sqrt(sum);
    }

    /// <summary>
    /// An Adam step moves each weight by about its learning rate, so a supplied optimizer six orders of magnitude
    /// faster must move the networks about that much further. An agent that ignores the option trains both runs
    /// with the same default and moves them about equally.
    /// </summary>
    private static void AssertOptimizerDrivesTraining(string agent, Func<double, double> displacementAtRate)
    {
        double fast = displacementAtRate(FastRate);
        double slow = displacementAtRate(SlowRate);
        Assert.True(fast > 0.0 && !double.IsNaN(fast), $"{agent}: the update did not move the networks ({fast}).");
        Assert.True(fast > 1e3 * slow,
            $"{agent}: OptimizerFactory ignored: displacement {fast} at {FastRate} vs {slow} at {SlowRate}.");
    }

    private static Action Feed(DeepReinforcementLearningAgentBase<double> agent, int count, int actionSize, int seed) => () =>
    {
        foreach (var (state, action, reward, next, done) in DiscreteData(count, actionSize, seed))
            agent.StoreExperience(state, action, reward, next, done);
    };

    [Fact]
    public void A3C_TrainsWithTheConfiguredOptimizer()
    {
        AssertOptimizerDrivesTraining("A3C", rate =>
        {
            var agent = new A3CAgent<double>(new A3COptions<double>
            {
                StateSize = StateSize,
                ActionSize = 2,
                PolicyHiddenLayers = new List<int> { 8 },
                ValueHiddenLayers = new List<int> { 8 },
                OptimizerFactory = () => Adam(rate),
            });
            return SecondUpdateDisplacement(agent, Feed(agent, 8, 2, seed: 43));
        });
    }

    [Fact]
    public void Rainbow_TrainsWithTheConfiguredOptimizer()
    {
        AssertOptimizerDrivesTraining("Rainbow", rate =>
        {
            var agent = new RainbowDQNAgent<double>(new RainbowDQNOptions<double>
            {
                StateSize = StateSize,
                ActionSize = 2,
                SharedLayers = new List<int> { 8 },
                NumAtoms = 5,
                NSteps = 1,
                UseNoisyNetworks = false,
                BatchSize = 8,
                ReplayBufferSize = 64,
                WarmupSteps = 0,
                TargetUpdateFrequency = 10_000,
                OptimizerFactory = () => Adam(rate),
            });
            return SecondUpdateDisplacement(agent, Feed(agent, 8, 2, seed: 47));
        });
    }

    [Fact]
    public void Dreamer_TrainsWithTheConfiguredOptimizer()
    {
        AssertOptimizerDrivesTraining("Dreamer", rate =>
        {
            var agent = new DreamerAgent<double>(new DreamerOptions<double>
            {
                ObservationSize = StateSize,
                ActionSize = 2,
                LatentSize = 6,
                HiddenSize = 8,
                ImaginationHorizon = 2,
                BatchSize = 8,
                ReplayBufferSize = 64,
                OptimizerFactory = () => Adam(rate),
            });
            return SecondUpdateDisplacement(agent, Feed(agent, 8, 2, seed: 53));
        });
    }

    [Fact]
    public void WorldModels_TrainsWithTheConfiguredOptimizer()
    {
        AssertOptimizerDrivesTraining("WorldModels", rate =>
        {
            var agent = new WorldModelsAgent<double>(new WorldModelsOptions<double>
            {
                ObservationWidth = StateSize,
                ObservationHeight = 1,
                ObservationChannels = 1,
                ActionSize = 2,
                LatentSize = 4,
                RNNHiddenSize = 4,
                BatchSize = 8,
                ReplayBufferSize = 64,
                OptimizerFactory = () => Adam(rate),
            });
            return SecondUpdateDisplacement(agent, Feed(agent, 8, 2, seed: 59));
        });
    }

    // Without a factory the networks train with Adam at LearningRate; before, they fell back to the network's own
    // default optimizer and LearningRate reached none of them.
    [Fact]
    public void Dreamer_WithoutAFactory_TrainsAtTheConfiguredLearningRate()
    {
        AssertOptimizerDrivesTraining("Dreamer", rate =>
        {
            var agent = new DreamerAgent<double>(new DreamerOptions<double>
            {
                ObservationSize = StateSize,
                ActionSize = 2,
                LatentSize = 6,
                HiddenSize = 8,
                ImaginationHorizon = 2,
                BatchSize = 8,
                ReplayBufferSize = 64,
                LearningRate = rate,
            });
            return SecondUpdateDisplacement(agent, Feed(agent, 8, 2, seed: 53));
        });
    }

    [Fact]
    public void WorldModels_WithoutAFactory_TrainsAtTheConfiguredLearningRate()
    {
        AssertOptimizerDrivesTraining("WorldModels", rate =>
        {
            var agent = new WorldModelsAgent<double>(new WorldModelsOptions<double>
            {
                ObservationWidth = StateSize,
                ObservationHeight = 1,
                ObservationChannels = 1,
                ActionSize = 2,
                LatentSize = 4,
                RNNHiddenSize = 4,
                BatchSize = 8,
                ReplayBufferSize = 64,
                LearningRate = rate,
            });
            return SecondUpdateDisplacement(agent, Feed(agent, 8, 2, seed: 59));
        });
    }

    [Fact]
    public void MuZero_TrainsWithTheConfiguredOptimizer()
    {
        AssertOptimizerDrivesTraining("MuZero", rate =>
        {
            var agent = SmallMuZero(0.3, 0.25, () => Adam(rate));
            return SecondUpdateDisplacement(agent, Feed(agent, 16, 4, seed: 61));
        });
    }

    // ── OptimizerFactory gives every network its own optimizer ───────────────────────────────────────────

    /// <summary>A factory that records every optimizer it hands out.</summary>
    private sealed class CountingFactory
    {
        public List<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>> Created { get; } = new();

        public IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> Create()
        {
            var optimizer = Adam(FastRate);
            Created.Add(optimizer);
            return optimizer;
        }
    }

    private static IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? TrainingOptimizerOf(INeuralNetwork<double> network) =>
        ((NeuralNetworkBase<double>)network).TrainingOptimizer;

    /// <summary>
    /// The factory ran once per network, and each network trains with the instance its own call returned. One
    /// shared instance would advance Adam's bias-correction step counter on every network's step.
    /// </summary>
    private static void AssertOneOptimizerPerNetwork(string agent, CountingFactory factory, IReadOnlyList<INeuralNetwork<double>> networks)
    {
        Assert.Equal(networks.Count, factory.Created.Count);
        var used = networks.Select(TrainingOptimizerOf).ToList();
        Assert.All(used, optimizer => Assert.Contains(factory.Created, created => ReferenceEquals(created, optimizer)));
        // ReferenceEqualityComparer is not available on net471, so compare pairwise.
        for (int i = 0; i < used.Count; i++)
            for (int j = i + 1; j < used.Count; j++)
                Assert.False(ReferenceEquals(used[i], used[j]), $"{agent}: networks {i} and {j} share one optimizer instance.");
    }

    [Fact]
    public void A3C_GivesEveryNetworkItsOwnOptimizer_IncludingEachWorkersLocalNetworks()
    {
        var factory = new CountingFactory();
        var agent = new A3CAgent<double>(new A3COptions<double>
        {
            StateSize = StateSize,
            ActionSize = 2,
            PolicyHiddenLayers = new List<int> { 8 },
            ValueHiddenLayers = new List<int> { 8 },
            OptimizerFactory = factory.Create,
        });
        AssertOneOptimizerPerNetwork("A3C", factory, agent.GetNetworks());

        var first = agent.WorkerNetworks(0);
        var second = agent.WorkerNetworks(1);
        var again = agent.WorkerNetworks(0);
        Assert.Same(first.Policy, again.Policy);

        var all = agent.GetNetworks().Concat(new[] { first.Policy, first.Value, second.Policy, second.Value }).ToList();
        AssertOneOptimizerPerNetwork("A3C", factory, all);
    }

    [Fact]
    public void Rainbow_GivesEveryNetworkItsOwnOptimizer()
    {
        var factory = new CountingFactory();
        var agent = new RainbowDQNAgent<double>(new RainbowDQNOptions<double>
        {
            StateSize = StateSize,
            ActionSize = 2,
            SharedLayers = new List<int> { 8 },
            NumAtoms = 5,
            OptimizerFactory = factory.Create,
        });
        AssertOneOptimizerPerNetwork("Rainbow", factory, agent.GetNetworks());
    }

    [Fact]
    public void Dreamer_GivesEveryNetworkItsOwnOptimizer()
    {
        var factory = new CountingFactory();
        var agent = new DreamerAgent<double>(new DreamerOptions<double>
        {
            ObservationSize = StateSize,
            ActionSize = 2,
            LatentSize = 6,
            HiddenSize = 8,
            OptimizerFactory = factory.Create,
        });
        AssertOneOptimizerPerNetwork("Dreamer", factory, agent.GetNetworks());
    }

    [Fact]
    public void WorldModels_GivesEveryNetworkItsOwnOptimizer()
    {
        var factory = new CountingFactory();
        var agent = new WorldModelsAgent<double>(new WorldModelsOptions<double>
        {
            ObservationWidth = StateSize,
            ObservationHeight = 1,
            ObservationChannels = 1,
            ActionSize = 2,
            LatentSize = 4,
            RNNHiddenSize = 4,
            OptimizerFactory = factory.Create,
        });
        AssertOneOptimizerPerNetwork("WorldModels", factory, agent.GetNetworks());
    }

    [Fact]
    public void MADDPG_GivesEveryNetworkItsOwnOptimizer()
    {
        var factory = new CountingFactory();
        var agent = new MADDPGAgent<double>(new MADDPGOptions<double>
        {
            NumAgents = 2,
            StateSize = StateSize,
            ActionSize = 2,
            OptimizerFactory = factory.Create,
        });
        AssertOneOptimizerPerNetwork("MADDPG", factory, agent.GetNetworks());
    }

    [Fact]
    public void MuZero_BuildsOneJointOptimizerPerAgent_EvenFromSharedOptions()
    {
        // MuZero's three networks train in one joint step, so each agent asks for exactly one optimizer, however
        // many steps it takes; two agents built from the same options get one each.
        var factory = new CountingFactory();
        var options = new MuZeroOptions<double>
        {
            ObservationSize = StateSize,
            ActionSize = 4,
            LatentStateSize = 6,
            RepresentationLayers = new List<int> { 8 },
            DynamicsLayers = new List<int> { 8 },
            PredictionLayers = new List<int> { 8 },
            BatchSize = 4,
            UnrollSteps = 2,
            TDSteps = 2,
            Seed = 29,
            OptimizerFactory = factory.Create,
        };
        foreach (var agent in new[] { new MuZeroAgent<double>(options), new MuZeroAgent<double>(options) })
        {
            var feed = Feed(agent, 16, 4, seed: 67);
            feed();
            agent.Train();
            feed();
            agent.Train();
        }

        Assert.Equal(2, factory.Created.Count);
        Assert.NotSame(factory.Created[0], factory.Created[1]);
    }

    // ── Options with no effect are not offered ───────────────────────────────────────────────────────────

    [Fact]
    public void TRPO_OffersNoOptimizer_BecauseItsPolicyStepIsTheNaturalGradient()
    {
        Assert.Null(typeof(TRPOOptions<double>).GetProperty("Optimizer"));
        Assert.All(typeof(TRPOAgent<double>).GetConstructors(),
            constructor => Assert.DoesNotContain(constructor.GetParameters(), p => p.Name == "optimizer"));
    }

    [Fact]
    public void ActorCriticOptions_OfferNoValueLossCoefficient_BecausePolicyAndValueTrainSeparately()
    {
        Assert.Null(typeof(A2COptions<double>).GetProperty("ValueLossCoefficient"));
        Assert.Null(typeof(A3COptions<double>).GetProperty("ValueLossCoefficient"));
        Assert.Null(typeof(PPOOptions<double>).GetProperty("ValueLossCoefficient"));
    }

    // ── TargetEntropy: null is -dim(A), and zero is a target like any other ──────────────────────────────

    [Fact]
    public void TargetEntropy_NullMeansMinusActionDimension_AndZeroIsHonoured()
    {
        SACAgent<double> Sac(double? target) => new SACAgent<double>(new SACOptions<double>
        {
            StateSize = StateSize,
            ActionSize = 3,
            ReplayBufferSize = 64,
            PolicyHiddenLayers = new List<int> { 8 },
            QHiddenLayers = new List<int> { 8 },
            TargetEntropy = target,
        });
        CQLAgent<double> Cql(double? target) => new CQLAgent<double>(new CQLOptions<double>
        {
            StateSize = StateSize,
            ActionSize = 3,
            BufferSize = 64,
            PolicyHiddenLayers = new List<int> { 8 },
            QHiddenLayers = new List<int> { 8 },
            TargetEntropy = target,
        });

        Assert.Equal(-3.0, Sac(null).ResolvedTargetEntropy);
        Assert.Equal(0.0, Sac(0.0).ResolvedTargetEntropy);
        Assert.Equal(1.5, Sac(1.5).ResolvedTargetEntropy);
        Assert.Equal(-3.0, Cql(null).ResolvedTargetEntropy);
        Assert.Equal(0.0, Cql(0.0).ResolvedTargetEntropy);
        Assert.Equal(1.5, Cql(1.5).ResolvedTargetEntropy);
    }

    // The discrete policy loss and entropy apply the softmax themselves, so the policy network must emit logits: a
    // softmax output layer made the agent train on softmax(π). Predict still reports π(·|s).
    [Fact]
    public void A3C_DiscretePolicyNetworkEmitsLogits_AndPredictReportsTheirSoftmax()
    {
        var agent = new A3CAgent<double>(new A3COptions<double> { StateSize = StateSize, ActionSize = 3, Seed = 5 });
        var state = new Vector<double>(new[] { 0.5, -1.0, 2.0, 0.25 });

        var logits = agent.GetNetworks()[0].Predict(Tensor<double>.FromVector(state)).ToVector();
        var policy = agent.Predict(state);

        Assert.Equal(3, logits.Length);
        Assert.NotEqual(1.0, logits.Sum(), 6);
        double max = logits.Max();
        double norm = logits.Sum(z => Math.Exp(z - max));
        for (int i = 0; i < logits.Length; i++)
            Assert.Equal(Math.Exp(logits[i] - max) / norm, policy[i], 12);
    }
}
