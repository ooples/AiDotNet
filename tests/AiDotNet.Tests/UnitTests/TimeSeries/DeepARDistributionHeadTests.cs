using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Tests for DeepAR's pluggable predictive-distribution heads (Gaussian / Student-t / spline-quantile).
/// Training is fully deterministic here (fixed seeds + deterministic data + Adam), so the learning and
/// calibration assertions are stable, not statistical.
/// </summary>
public sealed class DeepARDistributionHeadTests
{
    private static DeepAROptions<double> Options(string likelihood) => new()
    {
        LookbackWindow = 12,
        ForecastHorizon = 3,
        HiddenSize = 16,
        NumLayers = 1,
        Epochs = 40,
        BatchSize = 8,
        NumSamples = 200,
        LikelihoodType = likelihood,
        StudentTDegreesOfFreedom = 4.0,
    };

    // A smooth, learnable series: the model should be able to reduce every head's training loss on it.
    private static (Matrix<double> x, Vector<double> y) SmoothSeries(int n, int lookback)
    {
        var x = new Matrix<double>(n, lookback);
        var y = new Vector<double>(n);
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < lookback; j++)
                x[i, j] = Math.Sin((i + j) * 0.25) + (i + j) * 0.01;
            y[i] = Math.Sin((i + lookback) * 0.25) + (i + lookback) * 0.01;
        }

        return (x, y);
    }

    [Theory]
    [InlineData("Gaussian")]
    [InlineData("StudentT")]
    [InlineData("Spline")]
    [Trait("category", "unit")]
    public void Each_head_reduces_its_training_loss_and_predicts_finite(string likelihood)
    {
        var options = Options(likelihood);
        var model = new DeepARModel<double>(options);
        var (x, y) = SmoothSeries(140, options.LookbackWindow);

        model.Train(x, y);

        var history = model.TrainingLossHistory;
        Assert.NotEmpty(history);
        Assert.All(history, v => Assert.True(!double.IsNaN(v) && !double.IsInfinity(v), $"non-finite loss {v}"));
        // The head actually learned: its best epoch loss is a real improvement on its first-epoch loss.
        Assert.True(history.Min() < history[0] - 1e-9,
            $"{likelihood} head did not reduce loss (first={history[0]}, best={history.Min()})");

        var window = new Vector<double>(options.LookbackWindow);
        for (var j = 0; j < options.LookbackWindow; j++)
            window[j] = Math.Sin(j * 0.25) + j * 0.01;
        var p = model.PredictSingle(window);
        Assert.True(!double.IsNaN(p) && !double.IsInfinity(p), $"{likelihood} produced non-finite prediction {p}");
    }

    [Theory]
    [InlineData("Gaussian")]
    [InlineData("StudentT")]
    [InlineData("Spline")]
    [Trait("category", "unit")]
    public void Quantile_bands_are_monotone_and_finite(string likelihood)
    {
        var options = Options(likelihood);
        var model = new DeepARModel<double>(options);
        var (x, y) = SmoothSeries(140, options.LookbackWindow);
        model.Train(x, y);

        var history = new Vector<double>(options.LookbackWindow);
        for (var j = 0; j < options.LookbackWindow; j++)
            history[j] = Math.Sin(j * 0.25) + j * 0.01;

        var bands = model.ForecastWithQuantiles(history, new[] { 0.1, 0.5, 0.9 });
        for (var h = 0; h < options.ForecastHorizon; h++)
        {
            double lo = bands[0.1][h], mid = bands[0.5][h], hi = bands[0.9][h];
            Assert.True(!double.IsNaN(lo) && !double.IsNaN(mid) && !double.IsNaN(hi), $"{likelihood} NaN band at h={h}");
            Assert.True(lo <= mid + 1e-9 && mid <= hi + 1e-9, $"{likelihood} non-monotone band [{lo},{mid},{hi}] at h={h}");
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void StudentT_tail_is_heavier_than_the_normal_tail()
    {
        // The standardized Student-t 95th/99th percentiles must exceed the normal's — that heavier tail is the
        // whole point of the Student-t head (it stops under-pricing large moves). Deterministic math, no training.
        double zNormal95 = DeepARDistMath.Probit(0.95);
        double zNormal99 = DeepARDistMath.Probit(0.99);
        double t95 = DeepARDistMath.StudentTQuantile(0.95, 4.0);
        double t99 = DeepARDistMath.StudentTQuantile(0.99, 4.0);

        Assert.True(t95 > zNormal95, $"t95 {t95} should exceed normal {zNormal95}");
        Assert.True(t99 > zNormal99, $"t99 {t99} should exceed normal {zNormal99}");
        // Tail excess grows further out and shrinks as ν→∞ (t approaches normal).
        Assert.True((t99 - zNormal99) > (t95 - zNormal95), "tail excess should grow toward the extreme");
        double t99HighNu = DeepARDistMath.StudentTQuantile(0.99, 100.0);
        Assert.True(Math.Abs(t99HighNu - zNormal99) < Math.Abs(t99 - zNormal99), "large ν should approach the normal");
    }

    [Fact]
    [Trait("category", "unit")]
    public void DistMath_loggamma_and_probit_match_known_values()
    {
        Assert.Equal(Math.Log(24.0), DeepARDistMath.LogGamma(5.0), 6);      // Γ(5) = 4! = 24
        Assert.Equal(0.5 * Math.Log(Math.PI), DeepARDistMath.LogGamma(0.5), 6); // Γ(1/2) = √π
        Assert.Equal(1.959963985, DeepARDistMath.Probit(0.975), 5);        // standard 97.5% z
        Assert.Equal(0.0, DeepARDistMath.Probit(0.5), 6);                  // median
    }

    [Fact]
    [Trait("category", "unit")]
    public void StudentT_sampler_is_finite_and_symmetric_on_average()
    {
        var rng = new Random(7);
        int n = 20000;
        double sum = 0;
        for (var i = 0; i < n; i++)
        {
            double s = DeepARDistMath.SampleStudentT(4.0, rng);
            Assert.True(!double.IsNaN(s) && !double.IsInfinity(s), $"non-finite t sample {s}");
            sum += s;
        }

        // Mean of a symmetric t is 0; a large sample should sit near it.
        Assert.True(Math.Abs(sum / n) < 0.1, $"t sample mean {sum / n} not near zero");
    }
}
