using System.Linq;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks;
using AiDotNet.UncertaintyQuantification.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.UncertaintyQuantification;

/// <summary>
/// A Bayesian dense layer draws its initial weight means and every weight sample from its own stream. That stream
/// read only the constructor's randomSeed argument, so the layers a model built for itself were unseeded: two
/// BayesianNeuralNetworks with the same Options.Seed started from different means and drew different samples, and
/// the generated memorization test passed or failed by chance.
/// </summary>
public class BayesianDenseLayerSeedTests
{
    private static BayesianNeuralNetwork<double> Build(int seed)
        => new(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 2),
            new BayesianNeuralNetworkOptions { Seed = seed, NumSamples = 5 });

    private static Tensor<double> Input()
    {
        var input = new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < input.Length; i++)
            input[i] = 0.1 * (i % 7) - 0.3;
        return input;
    }

    [Fact]
    public void TheSameOptionsSeed_GivesTheSameInitialMeansAndTheSameSamples()
    {
        var first = Build(7);
        var second = Build(7);

        Assert.Equal(first.GetParameters().ToArray(), second.GetParameters().ToArray());
        Assert.Equal(
            first.PredictWithUncertainty(Input()).Mean.ToVector().ToArray(),
            second.PredictWithUncertainty(Input()).Mean.ToVector().ToArray());
    }

    [Fact]
    public void DifferentOptionsSeeds_GiveDifferentInitialMeans()
    {
        Assert.NotEqual(Build(7).GetParameters().ToArray(), Build(8).GetParameters().ToArray());
    }

    [Fact]
    public void ALayerBuiltBeforeItsSeed_RedrawsItsMeansWhenTheSeedArrives()
    {
        var first = new BayesianDenseLayer<double>(4, 3);
        var second = new BayesianDenseLayer<double>(4, 3);

        first.RandomSeed = 9;
        second.RandomSeed = 9;

        Assert.Equal(first.GetParameters().ToArray(), second.GetParameters().ToArray());
    }

    [Fact]
    public void AnExplicitSeed_TakesPrecedenceOverTheModelsSeed()
    {
        var explicitOnly = new BayesianDenseLayer<double>(4, 3, randomSeed: 5);
        var insideASeededModel = Build(7);
        var explicitAfterTheModel = new BayesianDenseLayer<double>(4, 3, randomSeed: 5);

        Assert.Equal(5, explicitAfterTheModel.RandomSeed);
        Assert.Equal(explicitOnly.GetParameters().ToArray(), explicitAfterTheModel.GetParameters().ToArray());
        Assert.NotNull(insideASeededModel);
    }
}
