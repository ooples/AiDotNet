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
        // A Gaussian policy predicts means; its state-independent log standard deviations are
        // separate trainable parameters, rather than a second state-dependent network output.
        Assert.Equal(continuous ? 2 : 0, agent.LogStandardDeviations.Length);
        Assert.Equal(policyParameters + (continuous ? 2 : 0), agent.PolicyParameterCount);
        var value = Network(agent, "_valueNetwork");
        value.Predict(new Tensor<double>(state.ToArray(), new[] { 1, 4 }));
        Assert.Equal(40, value.GetParameters().Length);
    }

    // GAE (Schulman et al. 2016, Eq. 16) sums (gamma lambda)^l delta_{t+l} only within an episode: a terminal step's
    // advantage is its own TD error. Once the backward pass carried the next episode's advantage across the boundary.
    [Fact]
    public void GaeAdvantagesStopAtEpisodeBoundaries()
    {
        const double Gamma = 0.9, Lambda = 0.8;
        using var agent = new TRPOAgent<double>(new TRPOOptions<double>
        {
            StateSize = 2, ActionSize = 2, IsContinuous = false, DiscountFactor = Gamma, GaeLambda = Lambda, Seed = 5,
        });
        var s0 = new Vector<double>(new[] { 0.1, -0.2 });
        var s1 = new Vector<double>(new[] { 0.4, 0.3 });
        var s2 = new Vector<double>(new[] { -0.5, 0.6 });
        var s3 = new Vector<double>(new[] { 0.2, 0.9 });
        var action = new Vector<double>(new[] { 1.0, 0.0 });
        agent.StoreExperience(s0, action, 1.0, s1, true);
        agent.StoreExperience(s1, action, 0.5, s2, false);
        agent.StoreExperience(s2, action, -1.0, s3, true);

        var computed = typeof(TRPOAgent<double>).GetMethod("ComputeAdvantages", BindingFlags.Instance | BindingFlags.NonPublic)
            ?.Invoke(agent, null) ?? throw new InvalidOperationException("ComputeAdvantages missing.");
        var advantages = (List<double>)(computed.GetType().GetField("Item3")?.GetValue(computed)
            ?? throw new InvalidOperationException("advantages missing."));

        var valueNetwork = Network(agent, "_valueNetwork");
        double V(Vector<double> s) => valueNetwork.Predict(Tensor<double>.FromVector(s)).ToVector()[0];
        double a2 = -1.0 - V(s2);
        double a1 = 0.5 + Gamma * V(s2) - V(s1) + Gamma * Lambda * a2;
        double a0 = 1.0 - V(s0);
        var raw = new[] { a0, a1, a2 };
        double mean = AiDotNet.Helpers.StatisticsHelper<double>.CalculateMean(raw);
        double std = AiDotNet.Helpers.StatisticsHelper<double>.CalculateStandardDeviation(raw);

        Assert.Equal(3, advantages.Count);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal((raw[i] - mean) / std, advantages[i], 9);
        }
    }

    private static INeuralNetwork<double> Network(TRPOAgent<double> agent, string field) =>
        (INeuralNetwork<double>)(typeof(TRPOAgent<double>).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(agent)
            ?? throw new InvalidOperationException("TRPO network missing: " + field));
}
