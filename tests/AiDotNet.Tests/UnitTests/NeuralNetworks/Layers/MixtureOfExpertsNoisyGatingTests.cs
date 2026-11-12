using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Reference tests for the noisy top-K gating of <see cref="MixtureOfExpertsLayer{T}"/>
/// (Shazeer et al. 2017, §3.2): the gating noise is a fresh standard-normal draw per training step.
/// </summary>
public sealed class MixtureOfExpertsNoisyGatingTests
{
    [Fact]
    public void TrainingForward_DrawsFreshGatingNoiseEachStep()
    {
        // A Random re-seeded with 42 inside every forward produced the same "noise" each step, so two
        // training forwards over the same input returned identical outputs.
        var experts = new List<ILayer<double>>
        {
            new DenseLayer<double>(3),
            new DenseLayer<double>(3),
            new DenseLayer<double>(3),
        };
        var router = new DenseLayer<double>(3);
        var layer = new MixtureOfExpertsLayer<double>(experts, router, [4], [3]);
        layer.SetTrainingMode(true);
        var input = new Tensor<double>([0.3, -1.2, 0.8, 2.0, 1.1, 0.4, -0.6, 0.9], [2, 4]);

        var first = layer.Forward(input);
        var second = layer.Forward(input);

        double maxDifference = 0;
        for (int i = 0; i < first.Length; i++)
        {
            maxDifference = Math.Max(maxDifference, Math.Abs(first.GetFlat(i) - second.GetFlat(i)));
        }

        Assert.True(maxDifference > 1e-9, "Two training forwards drew identical gating noise.");
    }

    [Fact]
    public void EvaluationForward_IsDeterministic()
    {
        // Paper §3.4 / inference: clean logits, no noise.
        var experts = new List<ILayer<double>> { new DenseLayer<double>(3), new DenseLayer<double>(3) };
        var router = new DenseLayer<double>(2);
        var layer = new MixtureOfExpertsLayer<double>(experts, router, [4], [3], topK: 1);
        layer.SetTrainingMode(false);
        var input = new Tensor<double>([0.3, -1.2, 0.8, 2.0], [1, 4]);

        var first = layer.Forward(input);
        var second = layer.Forward(input);

        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(first.GetFlat(i), second.GetFlat(i), precision: 12);
        }
    }
}
