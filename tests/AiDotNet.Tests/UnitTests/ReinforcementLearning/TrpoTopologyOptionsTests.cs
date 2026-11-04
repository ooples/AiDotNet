using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.TRPO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

public sealed class TrpoTopologyOptionsTests
{
    [Theory]
    [InlineData(false, 67)]
    [InlineData(true, 67)]
    public void EachHiddenWidthReachesThePolicyAndValueNetworks(bool continuous, int policyParameters)
    {
        using var agent = new TRPOAgent<double>(new TRPOOptions<double>
        {
            StateSize = 4, ActionSize = 2, IsContinuous = continuous,
            PolicyHiddenLayers = new List<int> { 7, 3 },
            ValueHiddenLayers = new List<int> { 5, 2 }, Seed = 17,
        });
        var state = new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4 });
        Assert.Equal(2, agent.SelectAction(state, training: false).Length);
        Assert.Equal(policyParameters, Network(agent, "_policyNetwork").GetParameters().Length);
        // Appendix D uses a mean network and separate state-independent log deviations.
        Assert.Equal(continuous ? 2 : 0, agent.LogStandardDeviations.Length);
        Assert.Equal(policyParameters + (continuous ? 2 : 0), agent.PolicyParameterCount);
        var value = Network(agent, "_valueNetwork");
        value.Predict(new Tensor<double>(state.ToArray(), new[] { 1, 4 }));
        Assert.Equal(40, value.GetParameters().Length);
    }

    private static INeuralNetwork<double> Network(TRPOAgent<double> agent, string field) =>
        (INeuralNetwork<double>)(typeof(TRPOAgent<double>).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(agent)
            ?? throw new InvalidOperationException("TRPO network missing: " + field));
}
