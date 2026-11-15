using AiDotNet.GaussianProcesses;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.GaussianProcesses;

/// <summary>
/// The Gaussian-process likelihoods and mean functions used to evaluate in double and convert the result
/// to T; they now evaluate in T. These tests pin the single-precision cases where a constant that is
/// harmless in double (1 - 1e-10, a 1e-6 ridge) rounds away in float and used to produce -inf or NaN.
/// </summary>
public class GPSinglePrecisionRegressionTests
{
    private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

    [Fact]
    public void BetaLikelihood_Float_SaturatedLatent_IsFiniteAndMatchesDouble()
    {
        // sigmoid(20f) rounds to exactly 1, so 1 - mu cancelled to 0, beta = 0 and lgamma(0) = +inf.
        var single = new BetaLikelihood<float>(10.0);
        var reference = new BetaLikelihood<double>(10.0);
        var y = new Vector<float>(new[] { 0.5f, 0.9f });
        var f = new Vector<float>(new[] { 20f, 1f });
        var yDouble = new Vector<double>(new[] { 0.5, 0.9 });
        var fDouble = new Vector<double>(new[] { 20.0, 1.0 });

        float logLikelihood = single.LogLikelihood(y, f);
        var gradient = single.LogLikelihoodGradient(y, f);
        var hessian = single.LogLikelihoodHessianDiag(y, f);
        double expectedLogLikelihood = reference.LogLikelihood(yDouble, fDouble);
        var expectedGradient = reference.LogLikelihoodGradient(yDouble, fDouble);

        Assert.True(IsFinite(logLikelihood), $"log-likelihood = {logLikelihood}");
        Assert.Equal(expectedLogLikelihood, logLikelihood, 1);
        for (int i = 0; i < 2; i++)
        {
            Assert.True(IsFinite(gradient[i]), $"gradient[{i}] = {gradient[i]}");
            Assert.True(IsFinite(hessian[i]), $"hessian[{i}] = {hessian[i]}");
            Assert.Equal(expectedGradient[i], gradient[i], 1);
        }
    }

    [Fact]
    public void BetaLikelihood_Float_ObservationAtBoundary_IsFiniteAndMatchesDouble()
    {
        // 1 - 1e-10 rounds to exactly 1 in single precision, so clamping y left log(1 - y) = log(0).
        var single = new BetaLikelihood<float>(10.0);
        var reference = new BetaLikelihood<double>(10.0);
        var y = new Vector<float>(new[] { 1.0f, 0.0f, 0.9f });
        var f = new Vector<float>(new[] { 0.3f, -0.4f, 1f });
        var yDouble = new Vector<double>(new[] { 1.0, 0.0, 0.9 });
        var fDouble = new Vector<double>(new[] { (double)0.3f, (double)-0.4f, 1.0 });

        float logLikelihood = single.LogLikelihood(y, f);
        var gradient = single.LogLikelihoodGradient(y, f);
        double expected = reference.LogLikelihood(yDouble, fDouble);

        Assert.True(IsFinite(logLikelihood), $"log-likelihood = {logLikelihood}");
        Assert.InRange(logLikelihood, expected - 0.01 * Math.Abs(expected), expected + 0.01 * Math.Abs(expected));
        for (int i = 0; i < 3; i++)
        {
            Assert.True(IsFinite(gradient[i]), $"gradient[{i}] = {gradient[i]}");
        }
    }

    [Fact]
    public void LinearMean_FromData_Float_CollinearColumns_ReturnsFiniteFit()
    {
        // Two identical features of magnitude ~100: the Gram diagonal is ~3e6, where the absolute 1e-6
        // ridge is below single-precision resolution, so the normal equations are exactly singular.
        const int n = 200;
        var x = new Matrix<float>(n, 2);
        var y = new Vector<float>(n);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
        for (int i = 0; i < n; i++)
        {
            float value = (float)(100 + rng.NextDouble() * 50);
            x[i, 0] = value;
            x[i, 1] = value;
            y[i] = 2 * value + 1;
        }

        var mean = LinearMean<float>.FromData(x, y);
        float prediction = mean.Evaluate(new Vector<float>(new[] { 120f, 120f }));

        Assert.True(IsFinite(prediction), $"prediction = {prediction}");
        Assert.InRange(prediction, 236f, 246f);
    }
}
