using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Transformers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// The Informer models' behaviour; the network itself is checked against the official code in
/// <see cref="InformerReferenceTests"/>.
/// </summary>
public class InformerModelTests
{
    private static InformerOptions<double> Small(int seed = 3) => new()
    {
        LookbackWindow = 16, LabelLength = 8, ForecastHorizon = 4, EmbeddingDim = 16, FeedForwardDim = 32,
        NumEncoderLayers = 2, NumDecoderLayers = 1, NumAttentionHeads = 2, SamplingFactor = 2, DropoutRate = 0.0,
        Epochs = 8, BatchSize = 8, LearningRate = 1e-3, Seed = seed,
    };

    private static Vector<double> Sine(int length)
    {
        var y = new Vector<double>(length);
        for (int i = 0; i < length; i++) y[i] = 3.0 + Math.Sin(2 * Math.PI * i / 12.0) + 0.02 * i;
        return y;
    }

    private static double InSampleMse(InformerModel<double> model, Vector<double> y)
    {
        var p = model.Predict(new Matrix<double>(y.Length, 1));
        double s = 0;
        for (int i = 16; i < y.Length; i++) s += (p[i] - y[i]) * (p[i] - y[i]);
        return s / (y.Length - 16);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_RejectsInvalidSizes()
    {
        await Task.Yield();
        Assert.Throws<ArgumentException>(() => new InformerModel<double>(new InformerOptions<double> { EmbeddingDim = 0 }));
        Assert.Throws<ArgumentException>(() => new InformerModel<double>(new InformerOptions<double> { NumEncoderLayers = 0 }));
        Assert.Throws<ArgumentException>(() => new InformerModel<double>(new InformerOptions<double> { NumAttentionHeads = 0 }));
        Assert.Throws<ArgumentException>(() => new InformerModel<double>(new InformerOptions<double> { LookbackWindow = 8, LabelLength = 9 }));
        // The width must split evenly over the heads.
        Assert.Throws<ArgumentException>(() => new InformerModel<double>(new InformerOptions<double> { EmbeddingDim = 10, NumAttentionHeads = 4, LookbackWindow = 8, LabelLength = 4 }));
    }

    [Fact(Timeout = 120000)]
    public async Task Training_LowersTheInSampleError()
    {
        await Task.Yield();
        var y = Sine(120);
        var options = Small();
        options.Epochs = 1;
        var oneEpoch = new InformerModel<double>(options);
        oneEpoch.Train(new Matrix<double>(y.Length, 1), y);
        var trained = new InformerModel<double>(Small());
        trained.Train(new Matrix<double>(y.Length, 1), y);
        double early = InSampleMse(oneEpoch, y), late = InSampleMse(trained, y);
        // The paper's schedule halves the rate every epoch; measured 0.97 -> 0.66.
        Assert.True(late < 0.8 * early, $"MSE after 1 epoch {early}, after 8 {late}");
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_InSample_AndAfterTheSeries()
    {
        await Task.Yield();
        var y = Sine(80);
        var model = new InformerModel<double>(Small());
        model.Train(new Matrix<double>(y.Length, 1), y);
        var inSample = model.Predict(new Matrix<double>(y.Length, 1));
        Assert.Equal(y.Length, inSample.Length);
        Assert.All(inSample, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));

        // Ten steps after the series: the first four are one generative forecast, the rest continue from it.
        var future = model.Predict(new Matrix<double>(10, 1));
        var direct = model.ForecastHorizon(y);
        for (int h = 0; h < 4; h++) Assert.Equal(direct[h], future[h], 12);
        var extended = y.ToList();
        extended.AddRange(future.Take(4));
        var next = model.ForecastHorizon(new Vector<double>(extended.ToArray()));
        for (int h = 0; h < 4; h++) Assert.Equal(next[h], future[4 + h], 12);
    }

    [Fact(Timeout = 120000)]
    public async Task Serialization_RoundTripsExactly()
    {
        await Task.Yield();
        var y = Sine(80);
        var model = new InformerModel<double>(Small());
        model.Train(new Matrix<double>(y.Length, 1), y);
        var restored = new InformerModel<double>(Small(seed: 99));
        restored.Deserialize(model.Serialize());
        var a = model.ForecastHorizon(y);
        var b = restored.ForecastHorizon(y);
        for (int h = 0; h < a.Length; h++) Assert.Equal(a[h], b[h]);
    }

    [Fact(Timeout = 60000)]
    public async Task TimeSeriesAndFinanceInformers_AreOneNetwork()
    {
        await Task.Yield();
        // Same options and seed: the same layers are built and initialized identically, so on the same (standardized)
        // history the two classes forecast the same values.
        var series = new InformerModel<double>(Small());
        var finance = new Informer<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), Small());
        var history = new Vector<double>(16);
        var window = new Tensor<double>(new[] { 1, 16, 1 });
        for (int t = 0; t < 16; t++) window[0, t, 0] = history[t] = Math.Sin(t * 0.7);
        var a = series.ForecastHorizon(history);
        var b = finance.Forecast(window);
        Assert.Equal(new[] { 1, 4, 1 }, b.Shape.ToArray());
        for (int h = 0; h < 4; h++) Assert.Equal(a[h], b[0, h, 0], 12);
    }
}
