using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.Policies;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>Parts for RL policy fixtures: a seeded network over a 4-value state, and a seeded generator (#2139).</summary>
public static class PolicyFixtureParts
{
    /// <summary>A small seeded network mapping a 4-value state to <paramref name="outputSize"/> values.</summary>
    public static NeuralNetwork<double> Network(int outputSize)
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(8, activationFunction: new TanhActivation<double>()),
            new DenseLayer<double>(outputSize, activationFunction: (IActivationFunction<double>?)null),
        };
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 4, outputSize: outputSize, layers: layers)
        { RandomSeed = 3 };
        var network = new NeuralNetwork<double>(architecture);
        _ = network.Predict(new Tensor<double>(new[] { 1, 4 }));
        return network;
    }

    /// <summary>The same seeded generator for every fixture, so two policies built alike sample alike.</summary>
    public static Random Random() => RandomHelper.CreateSeededRandom(5);
}

/// <summary>
/// Family invariants for RL policies, the state-to-action half of an agent (#2139).
/// </summary>
/// <remarks>
/// <para>
/// Policies were excluded because the RL agent family asserts agent semantics a policy does not have (it does not train
/// itself). This base asserts what a policy does promise. Inference (Predict, training: false) is the distribution's mode,
/// so it is deterministic; writing this family found that four policies sampled there, so Predict changed on every call.
/// Sampled actions are finite and as long as inference ones, sampling from a seeded generator is reproducible, and
/// log-probabilities are finite (and, for a discrete policy, normalized).
/// </para>
/// </remarks>
public abstract class PolicyTestBase
{
    /// <summary>Subclasses build their policy from <see cref="PolicyFixtureParts"/>.</summary>
    protected abstract PolicyBase<double> CreatePolicy();

    private static readonly Vector<double> State = new(new[] { 0.3, -0.2, 0.5, 0.1 });

    private static void AssertFinite(Vector<double> v, string what)
    {
        for (int i = 0; i < v.Length; i++)
            Assert.False(double.IsNaN(v[i]) || double.IsInfinity(v[i]), $"{what}[{i}] is {v[i]}.");
    }

    [Fact(Timeout = 60000)]
    public async Task Predict_IsTheInferenceActionAndDeterministic()
    {
        await Task.Yield();
        var policy = CreatePolicy();

        var first = policy.Predict(State);
        var second = policy.Predict(State);

        AssertFinite(first, "inference action");
        Assert.Equal(first.ToArray(), second.ToArray());
        // Predict is the inference path, whose action PolicyModeTests checks is each distribution's mode.
        Assert.Equal(policy.SelectAction(State, training: false).ToArray(), first.ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task SampledActions_AreFiniteAndAsLongAsInferenceActions()
    {
        await Task.Yield();
        var policy = CreatePolicy();

        var sampled = policy.SelectAction(State, training: true);

        AssertFinite(sampled, "sampled action");
        Assert.Equal(policy.Predict(State).Length, sampled.Length);
    }

    [Fact(Timeout = 60000)]
    public async Task Sampling_FromASeededGenerator_IsReproducible()
    {
        await Task.Yield();
        var a = CreatePolicy();
        var b = CreatePolicy();

        for (int i = 0; i < 3; i++)
            Assert.Equal(a.SelectAction(State, training: true).ToArray(), b.SelectAction(State, training: true).ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task LogProbability_IsFinite_AndNormalizedForADiscretePolicy()
    {
        await Task.Yield();
        var policy = CreatePolicy();
        var action = policy.SelectAction(State, training: true);

        double logProb = Convert.ToDouble(policy.ComputeLogProb(State, action));
        Assert.False(double.IsNaN(logProb) || double.IsInfinity(logProb), $"log-probability is {logProb}.");

        if (policy is DiscretePolicy<double>)
        {
            double total = 0;
            for (int a = 0; a < action.Length; a++)
            {
                var oneHot = new Vector<double>(action.Length);
                oneHot[a] = 1.0;
                total += Math.Exp(Convert.ToDouble(policy.ComputeLogProb(State, oneHot)));
            }
            Assert.True(Math.Abs(total - 1.0) < 1e-9, $"The action probabilities sum to {total}, not 1.");
        }
    }
}
