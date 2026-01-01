using System;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Policies;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class PoliciesIntegrationTests
{
    private const int StateSize = 3;
    private const int DiscreteActionSize = 3;
    private const int ContinuousActionSize = 2;

    [Fact]
    public void DiscretePolicy_SelectActionAndLogProb_ReturnValidOutputs()
    {
        var policy = new DiscretePolicy<double>(
            CreateNetwork(StateSize, DiscreteActionSize),
            DiscreteActionSize,
            new NoExploration<double>(),
            random: new Random(7));

        var state = CreateState(StateSize, 0.1);
        var action = policy.SelectAction(state, training: true);

        AssertOneHot(action, DiscreteActionSize);
        AssertFinite(policy.ComputeLogProb(state, action));
        Assert.Single(policy.GetNetworks());

        policy.Reset();
        policy.Dispose();
    }

    [Fact]
    public void ContinuousPolicy_SelectActionAndLogProb_ReturnValidOutputs()
    {
        var policy = new ContinuousPolicy<double>(
            CreateNetwork(StateSize, ContinuousActionSize * 2),
            ContinuousActionSize,
            new GaussianNoiseExploration<double>(initialStdDev: 0.0, noiseDecay: 1.0, minNoise: 0.0),
            useTanhSquashing: true,
            random: new Random(11));

        var state = CreateState(StateSize, 0.2);
        var action = policy.SelectAction(state, training: false);

        AssertVectorInRange(action, 0, ContinuousActionSize, -1.0, 1.0);
        AssertFinite(policy.ComputeLogProb(state, action));
        Assert.Single(policy.GetNetworks());

        policy.Reset();
        policy.Dispose();
    }

    [Fact]
    public void DeterministicPolicy_SelectActionAndLogProb_ReturnValidOutputs()
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

    [Fact]
    public void DeterministicPolicy_InvalidActionSize_Throws()
    {
        var policy = new DeterministicPolicy<double>(
            CreateNetwork(StateSize, outputSize: 1),
            actionSize: 2,
            new NoExploration<double>());

        var state = CreateState(StateSize, 0.1);

        Assert.Throws<ArgumentException>(() => policy.SelectAction(state, training: false));
    }

    [Fact]
    public void BetaPolicy_SelectActionAndLogProb_ReturnValidOutputs()
    {
        var policy = new BetaPolicy<double>(
            CreateNetwork(StateSize, ContinuousActionSize * 2),
            ContinuousActionSize,
            new NoExploration<double>(),
            actionMin: -1.0,
            actionMax: 1.0,
            random: new Random(17));

        var state = CreateState(StateSize, 0.4);
        var action = policy.SelectAction(state, training: false);

        AssertVectorInRange(action, 0, ContinuousActionSize, -1.0, 1.0);
        AssertFinite(policy.ComputeLogProb(state, action));
        Assert.Single(policy.GetNetworks());

        policy.Reset();
        policy.Dispose();
    }

    [Fact]
    public void MixedPolicy_SelectActionAndLogProb_ReturnValidOutputs()
    {
        var policy = new MixedPolicy<double>(
            CreateNetwork(StateSize, DiscreteActionSize),
            CreateNetwork(StateSize, ContinuousActionSize * 2),
            DiscreteActionSize,
            ContinuousActionSize,
            new NoExploration<double>(),
            new NoExploration<double>(),
            sharedFeatures: false,
            random: new Random(23));

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

    [Fact]
    public void MultiModalPolicy_SelectActionAndLogProb_ReturnValidOutputs()
    {
        int components = 2;
        int outputSize = components * (1 + 2 * ContinuousActionSize);

        var policy = new MultiModalPolicy<double>(
            CreateNetwork(StateSize, outputSize),
            ContinuousActionSize,
            components,
            new NoExploration<double>(),
            random: new Random(29));

        var state = CreateState(StateSize, 0.6);
        var action = policy.SelectAction(state, training: false);

        Assert.Equal(ContinuousActionSize, action.Length);
        AssertFinite(action, 0, ContinuousActionSize);
        AssertFinite(policy.ComputeLogProb(state, action));
        Assert.Single(policy.GetNetworks());

        policy.Reset();
        policy.Dispose();
    }

    [Fact]
    public void PolicyOptions_CanStoreCustomValues()
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

    [Fact]
    public void ReinforcementLearningOptions_InitValues_AreStored()
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
