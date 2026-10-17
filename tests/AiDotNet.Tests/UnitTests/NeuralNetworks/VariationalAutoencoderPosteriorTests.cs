using System;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// The VAE's encoder ends in a dense layer with 2 * latentSize outputs: the mean of q(z|x) and its log variance
/// (Kingma &amp; Welling 2014, appendix C.2). The model used to run a MeanLayer and a LogVarianceLayer in sequence
/// after it, reductions over axis 0 that collapsed the posterior to two scalars and decoded from a scalar.
/// </summary>
public sealed class VariationalAutoencoderPosteriorTests
{
    private static VariationalAutoencoder<double> NewModel() => new(
        new NeuralNetworkArchitecture<double>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 24,
            outputSize: 24),
        latentSize: 4);

    private static Vector<double> Pattern(int length)
    {
        var values = new double[length];
        for (int i = 0; i < length; i++) values[i] = 0.5 + 0.4 * Math.Sin(0.7 * i);
        return new Vector<double>(values);
    }

    [Fact(Timeout = 60000)]
    public async Task Encode_ReturnsAPosteriorPerLatentDimension()
    {
        await Task.Yield();
        var model = NewModel();

        var (mean, logVariance) = model.Encode(Pattern(24));

        Assert.Equal(4, mean.Length);
        Assert.Equal(4, logVariance.Length);
        Assert.Equal(24, model.Decode(mean).Length);
    }

    [Fact(Timeout = 60000)]
    public async Task TrainingForward_AddsTheKlTermToTheObjective()
    {
        await Task.Yield();
        var model = NewModel();
        var input = new Tensor<double>(new[] { 2, 24 });
        for (int i = 0; i < input.Length; i++) input[i] = 0.5 + 0.4 * Math.Sin(0.3 * i);

        var reconstructed = model.ForwardForTraining(input);
        var kl = model.ComputeAuxiliaryLoss();

        Assert.Equal(input.Shape.ToArray(), reconstructed.Shape.ToArray());
        Assert.True(kl > 0.0, $"KL(q(z|x) || N(0, I)) of an untrained encoder should be positive, got {kl}.");
    }
}
