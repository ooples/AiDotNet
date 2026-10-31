using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Neural;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// The DeepAR models over the shared network; the network itself is checked against a PyTorch transcription of the
/// paper in <see cref="DeepARReferenceTests"/>.
/// </summary>
public class DeepARNetworkModelTests
{
    private static DeepAROptions<double> Small(string likelihood = "Gaussian") => new()
    {
        LookbackWindow = 12, ForecastHorizon = 4, HiddenSize = 8, NumLayers = 2, DropoutRate = 0.0, Epochs = 3,
        BatchSize = 8, NumSamples = 60, LikelihoodType = likelihood, UseAgeFeature = false, Seed = 5,
    };

    [Fact(Timeout = 60000)]
    public async Task UnknownLikelihoods_AreRejected()
    {
        await Task.Yield();
        Assert.Throws<ArgumentException>(() => new DeepARModel<double>(Small("Spline")));
        Assert.Throws<ArgumentException>(() => new DeepAR<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), Small("Poisson")));
    }

    [Fact(Timeout = 60000)]
    public async Task TimeSeriesAndFinanceDeepARs_AreOneNetwork()
    {
        await Task.Yield();
        // Same options and seed, no age feature: the same network, initialized and sampled identically.
        var series = new DeepARModel<double>(Small());
        var finance = new DeepAR<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), Small());
        var history = new Vector<double>(12);
        var window = new Tensor<double>(new[] { 1, 12, 1 });
        for (int t = 0; t < 12; t++) window[0, t, 0] = history[t] = 2 + Math.Sin(t * 0.6);
        var a = series.ForecastWithQuantiles(history, new[] { 0.1, 0.5, 0.9 });
        var b = finance.Forecast(window, new[] { 0.1, 0.5, 0.9 });
        Assert.Equal(new[] { 1, 4, 3 }, b.Shape.ToArray());
        for (int h = 0; h < 4; h++)
        {
            Assert.Equal(a[0.1][h], b[0, h, 0], 12);
            Assert.Equal(a[0.5][h], b[0, h, 1], 12);
            Assert.Equal(a[0.9][h], b[0, h, 2], 12);
            Assert.True(a[0.1][h] <= a[0.5][h] && a[0.5][h] <= a[0.9][h]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task NegativeBinomial_ForecastsNonNegativeCounts()
    {
        await Task.Yield();
        var rng = new Random(2);
        var y = new Vector<double>(80);
        for (int i = 0; i < y.Length; i++) y[i] = Math.Round(5 + 3 * Math.Sin(i * 0.5) + rng.NextDouble());
        var model = new DeepARModel<double>(Small("NegativeBinomial"));
        model.Train(new Matrix<double>(y.Length, 1), y);
        var quantiles = model.ForecastWithQuantiles(y, new[] { 0.05, 0.5, 0.95 });
        foreach (var q in quantiles.Values)
            for (int h = 0; h < 4; h++)
            {
                Assert.True(q[h] >= 0, $"negative count {q[h]}");
                Assert.Equal(Math.Round(q[h]), q[h]);
            }
    }

    [Fact(Timeout = 120000)]
    public async Task Serialization_RoundTripsExactly()
    {
        await Task.Yield();
        var y = new Vector<double>(60);
        for (int i = 0; i < y.Length; i++) y[i] = 3 + Math.Sin(i * 0.4) + 0.02 * i;
        var options = Small();
        options.UseAgeFeature = true;
        var model = new DeepARModel<double>(options);
        model.Train(new Matrix<double>(y.Length, 1), y);
        // A fresh model of the same options: untrained weights, the same sampling seed.
        var restoredOptions = Small();
        restoredOptions.UseAgeFeature = true;
        var restored = new DeepARModel<double>(restoredOptions);
        Assert.NotEqual(model.GetParameters().ToArray(), restored.GetParameters().ToArray());
        restored.Deserialize(model.Serialize());
        var a = model.Predict(new Matrix<double>(6, 1));
        var b = restored.Predict(new Matrix<double>(6, 1));
        for (int h = 0; h < 6; h++) Assert.Equal(a[h], b[h]);
    }
}
