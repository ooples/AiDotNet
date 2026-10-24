using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.Policies;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

/// <summary>
/// The inference action of each stochastic policy is its distribution's mode, including the cases where a shortcut gets
/// it wrong: a uniform Beta, a mixture whose heaviest component is not where the density peaks, and discrete logits too
/// close to tell apart after the softmax rounds them.
/// </summary>
public class PolicyModeTests
{
    [Fact]
    public void The_Beta_mode_is_alpha_minus_one_over_alpha_plus_beta_minus_two()
    {
        Assert.Equal(2.0 / 3.0, BetaPolicy<double>.BetaMode(3.0, 2.0), 12);
        Assert.Equal(1.0, BetaPolicy<double>.BetaMode(2.0, 1.0), 12);
    }

    [Fact]
    public void The_uniform_Beta_takes_the_midpoint_rather_than_zero_over_zero()
    {
        // Softplus(-40) + 1 rounds to 1 for both parameters: Beta(1, 1), the uniform distribution.
        double mode = BetaPolicy<double>.BetaMode(1.0, 1.0);

        Assert.Equal(0.5, mode);
    }

    [Fact]
    public void A_mixture_mode_is_where_the_density_peaks_not_the_heaviest_component()
    {
        // Weight 0.6 at 0 with standard deviation 100 peaks at 0.6 / (100 sqrt(2 pi)); weight 0.4 at 5 with standard
        // deviation 0.01 peaks 40 times higher. The heaviest component's mean is not the mode.
        var mode = MultiModalPolicy<double>.MixtureMode(
            new[] { 0.6, 0.4 },
            new[] { new[] { 0.0 }, new[] { 5.0 } },
            new[] { new[] { Math.Log(100.0) }, new[] { Math.Log(0.01) } });

        Assert.Equal(5.0, mode[0], 6);
    }

    [Fact]
    public void Two_overlapping_equal_components_have_one_mode_between_them()
    {
        // Means at -0.5 and 0.5 with unit variance merge into one peak at 0 (they separate only beyond 2 sigma).
        var mode = MultiModalPolicy<double>.MixtureMode(
            new[] { 0.5, 0.5 },
            new[] { new[] { -0.5, 1.0 }, new[] { 0.5, 1.0 } },
            new[] { new[] { 0.0, 0.0 }, new[] { 0.0, 0.0 } });

        Assert.Equal(0.0, mode[0], 9);
        Assert.Equal(1.0, mode[1], 9);
    }

    [Fact]
    public void A_single_component_mixture_mode_is_its_mean()
    {
        var mode = MultiModalPolicy<double>.MixtureMode(
            new[] { 1.0 }, new[] { new[] { 0.25, -3.0 } }, new[] { new[] { 0.3, -1.2 } });

        Assert.Equal(new[] { 0.25, -3.0 }, mode);
    }

    [Fact]
    public void The_discrete_inference_action_compares_logits_that_round_to_equal_probabilities()
    {
        // A network whose output is its bias, [0, 1e-8]: softmax in float rounds both probabilities to 0.5.
        var layers = new List<ILayer<float>> { new DenseLayer<float>(2, activationFunction: new IdentityActivation<float>()) };
        var network = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 2, outputSize: 2, layers: layers));
        var state = new Vector<float>(new[] { 1f, 1f });
        _ = network.Predict(Tensor<float>.FromVector(state));
        var parameters = new Vector<float>(network.GetParameters().Length);
        parameters[parameters.Length - 1] = 1e-8f;
        network.UpdateParameters(parameters);
        var logits = network.Predict(Tensor<float>.FromVector(state)).ToVector();
        Assert.Equal(0f, logits[0]);
        Assert.Equal(1e-8f, logits[1]);

        var policy = new DiscretePolicy<float>(network, 2, new EpsilonGreedyExploration<float>());
        var action = policy.SelectAction(state, training: false);

        Assert.Equal(new[] { 0f, 1f }, action.ToArray());
    }
}
