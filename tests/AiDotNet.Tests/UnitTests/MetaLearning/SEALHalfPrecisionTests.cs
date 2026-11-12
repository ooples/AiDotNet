using System;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// <see cref="SEALAlgorithm{T, TInput, TOutput}"/> in Half: the 1e-8 floor and epsilon of the adaptive inner rates,
/// and the epsilon shift of the entropy term, round to 0, and squares of moderate gradients overflow.
/// </summary>
public class SEALHalfPrecisionTests
{
    private static SEALAlgorithm<Half, Tensor<Half>, Tensor<Half>> Algorithm(SEALAdaptiveLearningRateMode mode)
        => new(new SEALOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>())
        {
            InnerLearningRate = 0.01,
            AdaptiveLearningRateMode = mode,
        });

    private static Vector<Half> Gradients() => new(new[] { (Half)0.0f, (Half)1.0f, (Half)300.0f });

    [Fact]
    public void GradientNormRates_AreFiniteAndScaleInverselyWithTheGradient()
    {
        // lr_i = 0.01 / (|g_i| + eps). The old sqrt(g^2) overflowed 300^2 = 90000 to inf (rate 0), and the zero
        // gradient's rate was 0.01 / 0 = inf, so its update inf * 0 was NaN.
        Vector<Half>? running = null;
        var rates = Algorithm(SEALAdaptiveLearningRateMode.GradientNorm).ComputeAdaptiveLearningRates(Gradients(), ref running);

        float zeroGradientRate = (float)rates[0];
        Assert.False(float.IsInfinity(zeroGradientRate) || float.IsNaN(zeroGradientRate));
        Assert.Equal(0.01, (double)(float)rates[1], 4);
        Assert.Equal(0.01 / 300.0, (double)(float)rates[2], 6);
    }

    [Fact]
    public void PerLayerRate_UsesTheRootMeanSquareWithoutOverflow()
    {
        // rms = sqrt((0 + 1 + 90000) / 3) = 173.2; the old Half dot product overflowed to inf and the rate became 0.
        Vector<Half>? running = null;
        var rates = Algorithm(SEALAdaptiveLearningRateMode.PerLayer).ComputeAdaptiveLearningRates(Gradients(), ref running);

        double expected = 0.01 / Math.Sqrt(90001.0 / 3.0);
        for (int i = 0; i < 3; i++) Assert.Equal(expected, (double)(float)rates[i], 6);
    }

    [Fact]
    public void Entropy_CountsAnUnderflowedProbabilityAsZero()
    {
        // softmax(0, -100) = (1, e^-100): the second probability is 0 in Half and 0 log 0 was NaN; H is 0.
        var predictions = Tensor<Half>.FromVector(new Vector<Half>(new[] { (Half)0.0f, (Half)(-100.0f) }));

        double entropy = (double)(float)Algorithm(SEALAdaptiveLearningRateMode.GradientNorm)
            .ComputeEntropyRegularization(predictions);

        Assert.Equal(0.0, entropy, 3);
    }
}
