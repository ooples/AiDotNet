using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Policies;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Each policy's options class builds that policy (#2393). The options classes existed with no constructor that read
/// them, so every option was inert. These pin what each option now changes: the network's input and output widths, its
/// hidden layers, the action sampling seed, Deterministic's exploration default and Mixed's shared trunk.
/// </summary>
public class PolicyOptionsConstructionTests
{
    private const int State = 5;

    private static Vector<double> StateVector()
    {
        var s = new Vector<double>(State);
        for (int i = 0; i < State; i++) s[i] = 0.1 * (i + 1);
        return s;
    }

    [Fact]
    public void Discrete_BuildsANetworkOfTheOptionsWidths_AndSamplesOneHot()
    {
        var policy = new DiscretePolicy<double>(new DiscretePolicyOptions<double>
        {
            StateSize = State, ActionSize = 4, HiddenLayers = new[] { 16, 8 }, Seed = 3,
        });
        var network = Assert.Single(policy.GetNetworks());
        Assert.Equal(4, network.Predict(AiDotNet.Tensors.LinearAlgebra.Tensor<double>.FromVector(StateVector())).Length);
        var action = policy.SelectAction(StateVector(), training: false);
        Assert.Equal(4, action.Length);
        Assert.Equal(1.0, action.Sum());
    }

    [Fact]
    public void Continuous_Beta_And_MultiModal_ReadTheirOptionsActionSize()
    {
        var continuous = new ContinuousPolicy<double>(new ContinuousPolicyOptions<double>
        {
            StateSize = State, ActionSize = 3, HiddenLayers = new[] { 8 }, Seed = 1,
        });
        var beta = new BetaPolicy<double>(new BetaPolicyOptions<double>
        {
            StateSize = State, ActionSize = 2, HiddenLayers = new[] { 8 }, ActionMin = -2, ActionMax = 2, Seed = 1,
        });
        var mixture = new MultiModalPolicy<double>(new MultiModalPolicyOptions<double>
        {
            StateSize = State, ActionSize = 2, NumComponents = 4, HiddenLayers = new[] { 8 }, Seed = 1,
        });
        Assert.Equal(3, continuous.SelectAction(StateVector()).Length);
        var betaAction = beta.SelectAction(StateVector());
        Assert.Equal(2, betaAction.Length);
        Assert.All(betaAction.ToArray(), a => Assert.InRange(a, -2.0, 2.0));
        Assert.Equal(2, mixture.SelectAction(StateVector()).Length);
    }

    // The default exploration was Ornstein-Uhlenbeck noise built for one action dimension: with three it threw on the
    // first exploratory step. The default is now sized to ActionSize.
    [Fact]
    public void Deterministic_DefaultExploration_IsSizedToTheActionSize()
    {
        var options = new DeterministicPolicyOptions<double> { StateSize = State, ActionSize = 3, HiddenLayers = new[] { 8 }, Seed = 2 };
        Assert.Null(options.ExplorationStrategy);
        var policy = new DeterministicPolicy<double>(options);
        var action = policy.SelectAction(StateVector(), training: true);
        Assert.Equal(3, action.Length);
        Assert.All(action.ToArray(), a => Assert.InRange(a, -1.0, 1.0));

        // Control: the former default, noise for one action dimension, fails on this three-dimensional policy.
        var formerDefault = new DeterministicPolicy<double>(new DeterministicPolicyOptions<double>
        {
            StateSize = State, ActionSize = 3, HiddenLayers = new[] { 8 }, Seed = 2,
            ExplorationStrategy = new AiDotNet.ReinforcementLearning.Policies.Exploration.OrnsteinUhlenbeckNoise<double>(actionSize: 1),
        });
        Assert.ThrowsAny<System.ArgumentException>(() => formerDefault.SelectAction(StateVector(), training: true));
    }

    [Fact]
    public void Seed_MakesSampledActionsRepeat()
    {
        ContinuousPolicyOptions<double> Options() => new() { StateSize = State, ActionSize = 2, HiddenLayers = new[] { 8 }, Seed = 11 };
        var first = new ContinuousPolicy<double>(Options());
        var second = new ContinuousPolicy<double>(Options());
        // Same seed for the sampling; copy the weights so only the sampler can differ.
        second.GetNetworks()[0].SetParameters(first.GetNetworks()[0].GetParameters());
        Assert.Equal(first.SelectAction(StateVector()).ToArray(), second.SelectAction(StateVector()).ToArray());
    }

    // SharedFeatures used to be stored and never read. It now builds one network whose output carries both heads.
    [Fact]
    public void Mixed_SharedFeatures_UsesOneNetworkForBothHeads()
    {
        MixedPolicyOptions<double> Options(bool shared) => new()
        {
            StateSize = State, DiscreteActionSize = 3, ContinuousActionSize = 2, HiddenLayers = new[] { 8 },
            SharedFeatures = shared, Seed = 4,
        };
        var shared = new MixedPolicy<double>(Options(shared: true));
        var separate = new MixedPolicy<double>(Options(shared: false));

        var trunk = Assert.Single(shared.GetNetworks());
        Assert.Equal(3 + 2 * 2, trunk.Predict(AiDotNet.Tensors.LinearAlgebra.Tensor<double>.FromVector(StateVector())).Length);
        Assert.Equal(2, separate.GetNetworks().Count);

        var action = shared.SelectAction(StateVector(), training: false);
        Assert.Equal(3 + 2, action.Length);
        Assert.Equal(1.0, action.ToArray().Take(3).Sum());
        Assert.False(double.IsNaN(shared.ComputeLogProb(StateVector(), action)));
    }

    [Fact]
    public void Mixed_SharedFeatures_RejectsTwoDifferentNetworks()
    {
        var first = new DiscretePolicy<double>(new DiscretePolicyOptions<double> { StateSize = State, ActionSize = 7 }).GetNetworks()[0];
        var second = new DiscretePolicy<double>(new DiscretePolicyOptions<double> { StateSize = State, ActionSize = 7 }).GetNetworks()[0];
        Assert.Throws<System.ArgumentException>(() => new MixedPolicy<double>(first, second, 3, 2,
            new AiDotNet.ReinforcementLearning.Policies.Exploration.NoExploration<double>(),
            new AiDotNet.ReinforcementLearning.Policies.Exploration.NoExploration<double>(), sharedFeatures: true));
    }

    [Fact]
    public void Options_WithoutAStateSize_AreRejected()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            new DiscretePolicy<double>(new DiscretePolicyOptions<double> { ActionSize = 2 }));
    }
}
