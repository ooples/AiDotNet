using System;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Policies;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class PoliciesIntegrationTests
{
    private const int StateSize = 3;
    private const int DiscreteActionSize = 3;
    private const int ContinuousActionSize = 2;

    [Fact(Timeout = 120000)]
    public async Task DiscretePolicy_SelectActionAndLogProb_ReturnValidOutputs()
    {
        var policy = new DiscretePolicy<double>(
            CreateNetwork(StateSize, DiscreteActionSize),
            DiscreteActionSize,
            new NoExploration<double>(),
            random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7));

        var state = CreateState(StateSize, 0.1);
        var action = policy.SelectAction(state, training: true);

        AssertOneHot(action, DiscreteActionSize);
        AssertFinite(policy.ComputeLogProb(state, action));
        Assert.Single(policy.GetNetworks());

        policy.Reset();
        policy.Dispose();
    }

    [Fact(Timeout = 120000)]
    public async Task ContinuousPolicy_SelectActionAndLogProb_ReturnValidOutputs()
    {
        var policy = new ContinuousPolicy<double>(
            CreateNetwork(StateSize, ContinuousActionSize * 2),
            ContinuousActionSize,
            new GaussianNoiseExploration<double>(initialStdDev: 0.0, noiseDecay: 1.0, minNoise: 0.0),
            useTanhSquashing: true,
            random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11));

        var state = CreateState(StateSize, 0.2);
        var action = policy.SelectAction(state, training: false);

        AssertVectorInRange(action, 0, ContinuousActionSize, -1.0, 1.0);
        AssertFinite(policy.ComputeLogProb(state, action));
        Assert.Single(policy.GetNetworks());

        policy.Reset();
        policy.Dispose();
    }

    [Fact(Timeout = 120000)]
    public async Task DeterministicPolicy_SelectActionAndLogProb_ReturnValidOutputs()
    {
        var policy = new DeterministicPolicy<double>(
            CreateNetwork(StateSize, ContinuousActionSize),
            ContinuousActionSize,
            new NoExploration<double>(),
            useTanhSquashing: true);

        var state = CreateState(StateSize, 0.3);
        var action = policy.SelectAction(state, training: false);

        AssertVectorInRange(action, 0, ContinuousActionSize, -1.0, 1.0);
        Assert.Equal(0.0, policy.ComputeLogProb(state, action));
        Assert.Single(policy.GetNetworks());

        policy.Reset();
        policy.Dispose();
    }

    [Fact(Timeout = 120000)]
    public async Task DeterministicPolicy_InvalidActionSize_Throws()
    {
        var policy = new DeterministicPolicy<double>(
            CreateNetwork(StateSize, outputSize: 1),
            actionSize: 2,
            new NoExploration<double>());

        var state = CreateState(StateSize, 0.1);

        Assert.Throws<ArgumentException>(() => policy.SelectAction(state, training: false));
    }

    [Fact(Timeout = 120000)]
    public async Task BetaPolicy_SelectActionAndLogProb_ReturnValidOutputs()
    {
        var policy = new BetaPolicy<double>(
            CreateNetwork(StateSize, ContinuousActionSize * 2),
            ContinuousActionSize,
            new NoExploration<double>(),
            actionMin: -1.0,
            actionMax: 1.0,
            random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(17));

        var state = CreateState(StateSize, 0.4);
        var action = policy.SelectAction(state, training: false);

        AssertVectorInRange(action, 0, ContinuousActionSize, -1.0, 1.0);
        AssertFinite(policy.ComputeLogProb(state, action));
        Assert.Single(policy.GetNetworks());

        policy.Reset();
        policy.Dispose();
    }

    [Fact(Timeout = 120000)]
    public async Task MixedPolicy_SelectActionAndLogProb_ReturnValidOutputs()
    {
        var policy = new MixedPolicy<double>(
            CreateNetwork(StateSize, DiscreteActionSize),
            CreateNetwork(StateSize, ContinuousActionSize * 2),
            DiscreteActionSize,
            ContinuousActionSize,
            new NoExploration<double>(),
            new NoExploration<double>(),
            sharedFeatures: false,
            random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(23));

        var state = CreateState(StateSize, 0.5);
        var action = policy.SelectAction(state, training: false);

        Assert.Equal(DiscreteActionSize + ContinuousActionSize, action.Length);
        AssertOneHot(action, 0, DiscreteActionSize);
        AssertFinite(action, DiscreteActionSize, ContinuousActionSize);
        AssertFinite(policy.ComputeLogProb(state, action));
        Assert.Equal(2, policy.GetNetworks().Count);

        policy.Reset();
        policy.Dispose();
    }

    [Fact(Timeout = 120000)]
    public async Task MultiModalPolicy_SelectActionAndLogProb_ReturnValidOutputs()
    {
        int components = 2;
        int outputSize = components * (1 + 2 * ContinuousActionSize);

        var policy = new MultiModalPolicy<double>(
            CreateNetwork(StateSize, outputSize),
            ContinuousActionSize,
            components,
            new NoExploration<double>(),
            random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(29));

        var state = CreateState(StateSize, 0.6);
        var action = policy.SelectAction(state, training: false);

        Assert.Equal(ContinuousActionSize, action.Length);
        AssertFinite(action, 0, ContinuousActionSize);
        AssertFinite(policy.ComputeLogProb(state, action));
        Assert.Single(policy.GetNetworks());

        policy.Reset();
        policy.Dispose();
    }

    [Fact(Timeout = 120000)]
    public async Task PolicyOptions_CanStoreCustomValues()
    {
        var loss = new MeanSquaredErrorLoss<double>();

        var discrete = new DiscretePolicyOptions<double>
        {
            StateSize = 4,
            ActionSize = 3,
            HiddenLayers = new[] { 8, 4 },
            LossFunction = loss,
            ExplorationStrategy = new NoExploration<double>(),
            Seed = 7
        };

        Assert.Equal(4, discrete.StateSize);
        Assert.Equal(3, discrete.ActionSize);
        Assert.Equal(new[] { 8, 4 }, discrete.HiddenLayers);
        Assert.Same(loss, discrete.LossFunction);
        Assert.IsType<NoExploration<double>>(discrete.ExplorationStrategy);
        Assert.Equal(7, discrete.Seed);

        var continuous = new ContinuousPolicyOptions<double>
        {
            StateSize = 5,
            ActionSize = 2,
            HiddenLayers = new[] { 16 },
            LossFunction = loss,
            ExplorationStrategy = new GaussianNoiseExploration<double>(initialStdDev: 0.0),
            UseTanhSquashing = true,
            Seed = 11
        };

        Assert.Equal(5, continuous.StateSize);
        Assert.Equal(2, continuous.ActionSize);
        Assert.Equal(new[] { 16 }, continuous.HiddenLayers);
        Assert.Same(loss, continuous.LossFunction);
        Assert.True(continuous.UseTanhSquashing);
        Assert.Equal(11, continuous.Seed);

        var deterministic = new DeterministicPolicyOptions<double>
        {
            StateSize = 3,
            ActionSize = 2,
            HiddenLayers = new[] { 32, 16 },
            LossFunction = loss,
            ExplorationStrategy = new NoExploration<double>(),
            UseTanhSquashing = false,
            Seed = 13
        };

        Assert.Equal(3, deterministic.StateSize);
        Assert.Equal(2, deterministic.ActionSize);
        Assert.Equal(new[] { 32, 16 }, deterministic.HiddenLayers);
        Assert.Same(loss, deterministic.LossFunction);
        Assert.False(deterministic.UseTanhSquashing);
        Assert.Equal(13, deterministic.Seed);

        var beta = new BetaPolicyOptions<double>
        {
            StateSize = 6,
            ActionSize = 2,
            HiddenLayers = new[] { 8, 8 },
            LossFunction = loss,
            ExplorationStrategy = new NoExploration<double>(),
            ActionMin = -2.0,
            ActionMax = 2.0,
            Seed = 17
        };

        Assert.Equal(6, beta.StateSize);
        Assert.Equal(2, beta.ActionSize);
        Assert.Equal(new[] { 8, 8 }, beta.HiddenLayers);
        Assert.Equal(-2.0, beta.ActionMin);
        Assert.Equal(2.0, beta.ActionMax);
        Assert.Equal(17, beta.Seed);

        var mixed = new MixedPolicyOptions<double>
        {
            StateSize = 4,
            DiscreteActionSize = 2,
            ContinuousActionSize = 1,
            HiddenLayers = new[] { 8 },
            LossFunction = loss,
            DiscreteExplorationStrategy = new NoExploration<double>(),
            ContinuousExplorationStrategy = new GaussianNoiseExploration<double>(initialStdDev: 0.0),
            SharedFeatures = true,
            Seed = 19
        };

        Assert.Equal(4, mixed.StateSize);
        Assert.Equal(2, mixed.DiscreteActionSize);
        Assert.Equal(1, mixed.ContinuousActionSize);
        Assert.Equal(new[] { 8 }, mixed.HiddenLayers);
        Assert.True(mixed.SharedFeatures);
        Assert.Equal(19, mixed.Seed);

        var multiModal = new MultiModalPolicyOptions<double>
        {
            StateSize = 4,
            ActionSize = 2,
            NumComponents = 4,
            HiddenLayers = new[] { 16, 8 },
            LossFunction = loss,
            ExplorationStrategy = new NoExploration<double>(),
            Seed = 23
        };

        Assert.Equal(4, multiModal.StateSize);
        Assert.Equal(2, multiModal.ActionSize);
        Assert.Equal(4, multiModal.NumComponents);
        Assert.Equal(new[] { 16, 8 }, multiModal.HiddenLayers);
        Assert.Equal(23, multiModal.Seed);
    }

    [Fact(Timeout = 120000)]
    public async Task ReinforcementLearningOptions_InitValues_AreStored()
    {
        var loss = new MeanSquaredErrorLoss<double>();

        var options = new ReinforcementLearningOptions<double>
        {
            LearningRate = 0.1,
            DiscountFactor = 0.9,
            LossFunction = loss,
            Seed = 5,
            BatchSize = 4,
            ReplayBufferSize = 10,
            TargetUpdateFrequency = 2,
            UsePrioritizedReplay = true,
            EpsilonStart = 0.8,
            EpsilonEnd = 0.2,
            EpsilonDecay = 0.5,
            WarmupSteps = 3,
            MaxGradientNorm = 1.5
        };

        Assert.Equal(0.1, options.LearningRate);
        Assert.Equal(0.9, options.DiscountFactor);
        Assert.Same(loss, options.LossFunction);
        Assert.Equal(5, options.Seed);
        Assert.Equal(4, options.BatchSize);
        Assert.Equal(10, options.ReplayBufferSize);
        Assert.Equal(2, options.TargetUpdateFrequency);
        Assert.True(options.UsePrioritizedReplay);
        Assert.Equal(0.8, options.EpsilonStart);
        Assert.Equal(0.2, options.EpsilonEnd);
        Assert.Equal(0.5, options.EpsilonDecay);
        Assert.Equal(3, options.WarmupSteps);
        Assert.Equal(1.5, options.MaxGradientNorm);
    }

    // ===== Options constructors =====
    // The policy options classes used to declare `new int? Seed`, hiding ModelOptions.Seed, and no policy read
    // them at all. Each policy now has an options constructor that reads them, and the seed is set here through
    // the inherited ModelOptions.Seed - the path a generic caller (cloning, AutoML) takes - so two identically
    // configured policies must draw identical action sequences.

    private const int OptionsSeed = 31;

    [Fact(Timeout = 120000)]
    public async Task DiscretePolicy_OptionsConstructor_ReadsActionSizeAndInheritedSeed()
    {
        await Task.Yield();
        var network = CreateNetwork(StateSize, DiscreteActionSize);
        DiscretePolicy<double> Build() => new(network, WithSeed(new DiscretePolicyOptions<double>
        {
            ActionSize = DiscreteActionSize,
            ExplorationStrategy = new NoExploration<double>(),
        }));

        AssertSameSampledActions(Build, DiscreteActionSize);
    }

    [Fact(Timeout = 120000)]
    public async Task ContinuousPolicy_OptionsConstructor_ReadsActionSizeAndInheritedSeed()
    {
        await Task.Yield();
        var network = CreateNetwork(StateSize, ContinuousActionSize * 2);
        ContinuousPolicy<double> Build() => new(network, WithSeed(new ContinuousPolicyOptions<double>
        {
            ActionSize = ContinuousActionSize,
            ExplorationStrategy = new NoExploration<double>(),
            UseTanhSquashing = true,
        }));

        AssertSameSampledActions(Build, ContinuousActionSize);
    }

    [Fact(Timeout = 120000)]
    public async Task DeterministicPolicy_OptionsConstructor_ReadsActionSizeAndInheritedSeed()
    {
        await Task.Yield();
        var network = CreateNetwork(StateSize, ContinuousActionSize);
        // The network's action is deterministic, so only the exploration noise can show the seed is read: Gaussian
        // noise draws from the policy's Random, which the options' Seed creates.
        DeterministicPolicy<double> Build(int seed)
        {
            var options = new DeterministicPolicyOptions<double>
            {
                ActionSize = ContinuousActionSize,
                ExplorationStrategy = new GaussianNoiseExploration<double>(initialStdDev: 0.5),
                UseTanhSquashing = false,
            };
            AiDotNet.Models.Options.ModelOptions inherited = options;
            inherited.Seed = seed;
            return new DeterministicPolicy<double>(network, options);
        }

        var actions = AssertSameSampledActions(() => Build(OptionsSeed), ContinuousActionSize);

        // A different seed must give different noise; a policy that ignored the seed would repeat the same actions.
        var state = CreateState(StateSize, 0.25);
        using var other = Build(OptionsSeed + 1);
        bool anyDifferent = false;
        foreach (var expected in actions)
        {
            var got = other.SelectAction(state, training: true);
            for (int i = 0; i < got.Length; i++) anyDifferent |= got[i] != expected[i];
        }
        Assert.True(anyDifferent, "A different Seed produced the same exploration noise, so the seed is not read.");
    }

    // The default exploration noise is created for the final ActionSize, not the size 1 it had before ActionSize was set.
    [Fact(Timeout = 120000)]
    public async Task DeterministicPolicyOptions_DefaultExploration_MatchesTheActionSize()
    {
        await Task.Yield();
        var network = CreateNetwork(StateSize, ContinuousActionSize);
        var options = new DeterministicPolicyOptions<double> { ActionSize = ContinuousActionSize };
        Assert.IsType<OrnsteinUhlenbeckNoise<double>>(options.ExplorationStrategy);

        using var policy = new DeterministicPolicy<double>(network, options);
        var action = policy.SelectAction(CreateState(StateSize, 0.25), training: true);
        Assert.Equal(ContinuousActionSize, action.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task BetaPolicy_OptionsConstructor_ReadsActionSizeBoundsAndInheritedSeed()
    {
        await Task.Yield();
        var network = CreateNetwork(StateSize, ContinuousActionSize * 2);
        BetaPolicy<double> Build() => new(network, WithSeed(new BetaPolicyOptions<double>
        {
            ActionSize = ContinuousActionSize,
            ExplorationStrategy = new NoExploration<double>(),
            ActionMin = -2.0,
            ActionMax = 2.0,
        }));

        var actions = AssertSameSampledActions(Build, ContinuousActionSize);
        foreach (var action in actions)
            AssertVectorInRange(action, 0, ContinuousActionSize, -2.0, 2.0);
    }

    [Fact(Timeout = 120000)]
    public async Task MixedPolicy_OptionsConstructor_ReadsActionSizesAndInheritedSeed()
    {
        await Task.Yield();
        var discrete = CreateNetwork(StateSize, DiscreteActionSize);
        var continuous = CreateNetwork(StateSize, ContinuousActionSize * 2);
        MixedPolicy<double> Build() => new(discrete, continuous, WithSeed(new MixedPolicyOptions<double>
        {
            DiscreteActionSize = DiscreteActionSize,
            ContinuousActionSize = ContinuousActionSize,
            DiscreteExplorationStrategy = new NoExploration<double>(),
            ContinuousExplorationStrategy = new NoExploration<double>(),
        }));

        AssertSameSampledActions(Build, DiscreteActionSize + ContinuousActionSize);
    }

    [Fact(Timeout = 120000)]
    public async Task MultiModalPolicy_OptionsConstructor_ReadsComponentsAndInheritedSeed()
    {
        await Task.Yield();
        const int components = 2;
        var network = CreateNetwork(StateSize, components * (1 + 2 * ContinuousActionSize));
        MultiModalPolicy<double> Build() => new(network, WithSeed(new MultiModalPolicyOptions<double>
        {
            ActionSize = ContinuousActionSize,
            NumComponents = components,
            ExplorationStrategy = new NoExploration<double>(),
        }));

        AssertSameSampledActions(Build, ContinuousActionSize);
    }

    private static TOptions WithSeed<TOptions>(TOptions options) where TOptions : AiDotNet.Models.Options.ModelOptions
    {
        AiDotNet.Models.Options.ModelOptions inherited = options;
        inherited.Seed = OptionsSeed;
        return options;
    }

    private static List<Vector<double>> AssertSameSampledActions<TPolicy>(Func<TPolicy> build, int expectedLength)
        where TPolicy : IDisposable, IPolicy<double>
    {
        var state = CreateState(StateSize, 0.25);
        using var first = build();
        using var second = build();
        var actions = new List<Vector<double>>();
        for (int step = 0; step < 6; step++)
        {
            var a = first.SelectAction(state, training: true);
            var b = second.SelectAction(state, training: true);
            Assert.Equal(expectedLength, a.Length);
            Assert.Equal(a.Length, b.Length);
            for (int i = 0; i < a.Length; i++)
                Assert.Equal(a[i], b[i]);
            actions.Add(a);
        }

        return actions;
    }

    private static NeuralNetwork<double> CreateNetwork(int inputSize, int outputSize)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.ReinforcementLearning,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize);

        return new NeuralNetwork<double>(architecture);
    }

    private static Vector<double> CreateState(int size, double start)
    {
        var state = new Vector<double>(size);
        for (int i = 0; i < size; i++)
        {
            state[i] = start + i * 0.1;
        }

        return state;
    }

    private static void AssertOneHot(Vector<double> action, int offset, int actionSize)
    {
        Assert.True(action.Length >= offset + actionSize);

        int nonZeroCount = 0;
        for (int i = offset; i < offset + actionSize; i++)
        {
            if (Math.Abs(action[i]) > 1e-6)
            {
                nonZeroCount++;
            }
        }

        Assert.Equal(1, nonZeroCount);
    }

    private static void AssertOneHot(Vector<double> action, int actionSize)
    {
        AssertOneHot(action, 0, actionSize);
    }

    private static void AssertFinite(double value)
    {
        Assert.False(double.IsNaN(value));
        Assert.False(double.IsInfinity(value));
    }

    private static void AssertFinite(Vector<double> vector, int offset, int length)
    {
        Assert.True(vector.Length >= offset + length);
        for (int i = offset; i < offset + length; i++)
        {
            AssertFinite(vector[i]);
        }
    }

    private static void AssertVectorInRange(Vector<double> vector, int offset, int length, double min, double max)
    {
        Assert.True(vector.Length >= offset + length);
        for (int i = offset; i < offset + length; i++)
        {
            Assert.InRange(vector[i], min, max);
        }
    }
}
