using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class CycleGANTests : GANModelTestBase
{
    // CycleGAN's parameterless ctor (src/NeuralNetworks/CycleGAN.cs:246)
    // instantiates four 784-unit generator/discriminator architectures
    // (28x28 flattened MNIST) per Zhu et al. 2017's unpaired image-to-image
    // translation setting. The GANModelTestBase default [1, 4] latent shape
    // tripped TensorShapeMismatchException at FeedForwardNeuralNetwork
    // .Predict ("Expected shape [784], but got [1, 4]").
    protected override int[] InputShape => [784];
    protected override int[] OutputShape => [784];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new CycleGAN<double>();
}
