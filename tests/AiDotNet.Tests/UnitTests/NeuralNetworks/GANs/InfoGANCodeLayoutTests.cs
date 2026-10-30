using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.GANs;

/// <summary>
/// InfoGAN's latent code is categorical codes plus continuous codes, each scored its own way
/// (Chen et al. 2016, sections 5 and 7) (#2396).
/// </summary>
public class InfoGANCodeLayoutTests
{
    private const int NoiseSize = 4;
    private const int ImageSize = 6;

    private static InfoGAN<double> Create(InfoGANOptions? options, int latentCodeSize)
    {
        NeuralNetworkArchitecture<double> Arch(int inputs, int outputs, NeuralNetworkTaskType task) =>
            new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional, taskType: task,
                inputSize: inputs, outputSize: outputs);

        return new InfoGAN<double>(
            Arch(NoiseSize + latentCodeSize, ImageSize, NeuralNetworkTaskType.Regression),
            Arch(ImageSize, 1, NeuralNetworkTaskType.BinaryClassification),
            Arch(ImageSize, latentCodeSize, NeuralNetworkTaskType.Regression),
            latentCodeSize,
            InputType.OneDimensional,
            options);
    }

    [Fact]
    public void Defaults_AreThePapersMnistLayout()
    {
        var options = new InfoGANOptions();
        Assert.Equal(1, options.NumCategoricalCodes);
        Assert.Equal(10, options.CategoricalCodeDimension);
        Assert.Equal(2, options.NumContinuousCodes);
        Assert.Equal(12, options.LatentCodeSize);
    }

    [Fact]
    public void SampledCodes_AreOneHotThenUniform()
    {
        var model = Create(new InfoGANOptions(), 12);
        var codes = model.GenerateRandomLatentCodes(64);

        for (int b = 0; b < 64; b++)
        {
            int ones = 0;
            for (int c = 0; c < 10; c++)
            {
                double v = codes[b, c];
                Assert.True(v == 0.0 || v == 1.0, $"Categorical entry [{b}, {c}] = {v} is not one-hot.");
                if (v == 1.0) ones++;
            }

            Assert.Equal(1, ones);
            for (int c = 10; c < 12; c++)
            {
                Assert.InRange(codes[b, c], -1.0, 1.0);
            }
        }
    }

    [Fact]
    public void CodeLoss_IsCrossEntropyForCategoricalCodesPlusSquaredErrorForContinuousOnes()
    {
        var model = Create(new InfoGANOptions(), 12);
        var codes = new Tensor<double>(new[] { 2, 12 });
        codes[0, 3] = 1.0;
        codes[1, 7] = 1.0;
        codes[0, 10] = 0.5; codes[0, 11] = -0.5;
        codes[1, 10] = 1.0; codes[1, 11] = 0.0;
        // Zero logits give a uniform softmax, so each categorical cross-entropy is ln 10; zero predicted
        // means leave the continuous squared error at the mean of the targets' squares.
        var predicted = new Tensor<double>(new[] { 2, 12 });

        double loss = model.CodeLoss(predicted, codes).ToVector()[0];

        double expected = Math.Log(10.0) + (0.25 + 0.25 + 1.0 + 0.0) / 4.0;
        Assert.Equal(expected, loss, 9);
    }

    [Fact]
    public void ExplicitWidth_MustMatchTheOptionsLayout()
    {
        Assert.Throws<ArgumentException>(() => Create(new InfoGANOptions(), 10));
    }

    [Fact]
    public void ExplicitWidth_WithoutOptions_KeepsEveryCodeContinuous()
    {
        var model = Create(null, 3);
        var codes = model.GenerateRandomLatentCodes(16);
        var predicted = new Tensor<double>(new[] { 16, 3 });

        double meanSquare = 0;
        for (int b = 0; b < 16; b++)
            for (int c = 0; c < 3; c++)
                meanSquare += codes[b, c] * codes[b, c];
        meanSquare /= 48;

        Assert.Equal(meanSquare, model.CodeLoss(predicted, codes).ToVector()[0], 9);
    }
}
