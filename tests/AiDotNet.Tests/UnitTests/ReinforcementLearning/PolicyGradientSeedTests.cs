using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.A2C;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Agents.A3C;
using AiDotNet.ReinforcementLearning.Agents.PPO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

public sealed class PolicyGradientSeedTests
{
    private static List<double[]> Parameters(string kind, int seed)
    {
        using DeepReinforcementLearningAgentBase<double> agent = kind switch
        {
            "A2C" => new A2CAgent<double>(new A2COptions<double> { StateSize = 4, ActionSize = 3, Seed = seed,
                PolicyHiddenLayers = new List<int> { 8 }, ValueHiddenLayers = new List<int> { 8 } }),
            "A3C" => new A3CAgent<double>(new A3COptions<double> { StateSize = 4, ActionSize = 3, Seed = seed,
                PolicyHiddenLayers = new List<int> { 8 }, ValueHiddenLayers = new List<int> { 8 } }),
            "PPO" => new PPOAgent<double>(new PPOOptions<double> { StateSize = 4, ActionSize = 3, Seed = seed,
                PolicyHiddenLayers = new List<int> { 8 }, ValueHiddenLayers = new List<int> { 8 } }),
            _ => throw new ArgumentException("Unknown policy-gradient agent.", nameof(kind)),
        };
        var result = new List<double[]>();
        foreach (var network in agent.GetNetworks())
        {
            network.Predict(new Tensor<double>(new[] { 0.1, 0.2, 0.3, 0.4 }, new[] { 1, 4 }));
            result.Add(network.GetParameters().ToArray());
        }
        return result;
    }

    [Theory]
    [InlineData("A2C", 3)]
    [InlineData("A2C", int.MaxValue)]
    [InlineData("A3C", 3)]
    [InlineData("A3C", int.MaxValue)]
    [InlineData("PPO", 3)]
    [InlineData("PPO", int.MaxValue)]
    public void SeedControlsBothNetworkInitializations(string kind, int seed)
    {
        var original = Parameters(kind, seed);
        var repeated = Parameters(kind, seed);
        // System.Random aliases the boundary seeds with zero; compare the boundary against an ordinary seed.
        var different = Parameters(kind, seed == int.MaxValue ? 3 : seed + 1);
        Assert.Equal(2, original.Count);
        for (int i = 0; i < original.Count; i++)
        {
            Assert.Equal(original[i], repeated[i]);
            Assert.False(original[i].SequenceEqual(different[i]), $"The contrasting seed must change network {i} weights.");
        }
    }
}
