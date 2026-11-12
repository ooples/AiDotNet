using AiDotNet;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.UncertaintyQuantification;

/// <summary>
/// Platt scaling (Platt 1999) clips p1 before logit(p1). The clip margin must be representable in T: 1 - 1e-12
/// rounds to 1 in float, which made logit(1) = +infinity and the Newton fit return NaN.
/// </summary>
public class PlattScalingPrecisionTests
{
    [Fact]
    public void ProbabilityClipBounds_Float_AreStrictlyInsideTheUnitInterval()
    {
        var (low, high) = AiModelBuilder<float, Matrix<float>, Vector<float>>.ProbabilityClipBounds();

        Assert.True(low > 0f);
        Assert.True(high < 1f);
    }

    [Fact]
    public void ProbabilityClipBounds_Double_KeepsTheOneEMinusTwelveMargin()
    {
        var (low, high) = AiModelBuilder<double, Matrix<double>, Vector<double>>.ProbabilityClipBounds();

        Assert.Equal(1e-12, low, 20);
        Assert.Equal(1.0 - 1e-12, high, 20);
    }

    [Fact]
    public void FitPlattScalingBinary_Float_ConfidentProbabilitiesGiveFiniteParameters()
    {
        // p1 = 1.0f exactly (a confident softmax) on half the calibration samples.
        var probabilities = new Tensor<float>(new[] { 6, 2 }, new Vector<float>(new[]
        {
            0f, 1f,
            0f, 1f,
            0.8f, 0.2f,
            0.3f, 0.7f,
            0f, 1f,
            0.9f, 0.1f,
        }));
        var labels = new Vector<int>(new[] { 1, 1, 0, 1, 0, 0 });

        var (a, b) = AiModelBuilder<float, Matrix<float>, Vector<float>>.FitPlattScalingBinary(probabilities, labels, 6);

        Assert.False(float.IsNaN(a) || float.IsInfinity(a));
        Assert.False(float.IsNaN(b) || float.IsInfinity(b));

        var logit = AiModelBuilder<float, Matrix<float>, Vector<float>>.PositiveClassLogit(probabilities, 6);
        for (int i = 0; i < 6; i++)
        {
            Assert.False(float.IsNaN(logit[i]) || float.IsInfinity(logit[i]));
        }
    }
}
