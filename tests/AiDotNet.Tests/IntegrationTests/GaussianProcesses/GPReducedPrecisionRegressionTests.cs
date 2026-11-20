using System;
using AiDotNet.GaussianProcesses;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.GaussianProcesses;

/// <summary>
/// Regression tests for the Gaussian-process review fixes: jitter tried on a copy (no diagonal drift),
/// likelihood epsilons that resolve in half precision, and the empty-feature guard of LinearMean.FromData.
/// </summary>
public class GPReducedPrecisionRegressionTests
{
    [Fact]
    public void AdaptiveJitter_Float_RejectedJittersDoNotDriftTheDiagonal()
    {
        // Indefinite (eigenvalue -0.05) until the last jitter 0.1. Adding and subtracting each rejected jitter in
        // place leaves the float diagonal a few ulps off, so the result would not be exactly diag + 0.1.
        const float diag = 0.990445197f;
        const float offDiag = 1.04044521f;
        var k = new Matrix<float>(new float[,] { { diag, offDiag }, { offDiag, diag } });

        GaussianProcessEngineOps.AddAdaptiveJitter(k);

        float expected = diag;
        expected += 0.1f;
        Assert.Equal(expected, k[0, 0]);
        Assert.Equal(expected, k[1, 1]);
        Assert.Equal(offDiag, k[0, 1]);
        Assert.Equal(offDiag, k[1, 0]);
    }

    [Fact]
    public void BetaLikelihood_Half_BoundaryObservationIsFinite()
    {
        // The clamp epsilon 1e-10 is 0 in half precision, so y = 1 would give log(1 - y) = -inf.
        var ops = MathHelper.GetNumericOperations<Half>();
        var likelihood = new BetaLikelihood<Half>();

        double value = ops.ToDouble(likelihood.LogLikelihood(
            new Vector<Half>(new[] { ops.One }), new Vector<Half>(new[] { ops.Zero })));

        Assert.False(double.IsNaN(value) || double.IsInfinity(value), $"Log likelihood was {value}.");
    }

    [Fact]
    public void LinearMean_FromData_NoFeatureColumns_ThrowsForX()
    {
        var x = new Matrix<double>(3, 0);
        var y = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        var ex = Assert.Throws<ArgumentException>(() => LinearMean<double>.FromData(x, y));
        Assert.Equal("X", ex.ParamName);
    }
}
