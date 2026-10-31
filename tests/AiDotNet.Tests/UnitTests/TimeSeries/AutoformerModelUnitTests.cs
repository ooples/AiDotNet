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
/// The Autoformer models' behaviour; the network itself is checked against the official code in
/// <see cref="AutoformerReferenceTests"/>.
/// </summary>
public class AutoformerModelUnitTests
{
    private static AutoformerOptions<double> Small(int seed = 3) => new()
    {
        LookbackWindow = 24, LabelLength = 12, ForecastHorizon = 6, EmbeddingDim = 16, FeedForwardDim = 32,
        NumEncoderLayers = 1, NumDecoderLayers = 1, NumAttentionHeads = 2, MovingAverageKernel = 5, DropoutRate = 0.0,
        Epochs = 6, BatchSize = 8, LearningRate = 1e-3, Seed = seed,
    };

    private static Vector<double> Seasonal(int length)
    {
        var y = new Vector<double>(length);
        for (int i = 0; i < length; i++) y[i] = 2.0 + Math.Sin(2 * Math.PI * i / 8.0) + 0.03 * i;
        return y;
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_RejectsInvalidSizes()
    {
        await Task.Yield();
        Assert.Throws<ArgumentException>(() => new AutoformerModel<double>(new AutoformerOptions<double> { MovingAverageKernel = 4 }));
        Assert.Throws<ArgumentException>(() => new AutoformerModel<double>(new AutoformerOptions<double> { LookbackWindow = 8, LabelLength = 9 }));
        Assert.Throws<ArgumentException>(() => new AutoformerModel<double>(new AutoformerOptions<double> { AutoCorrelationFactor = 0 }));
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_OutOfSample_ForecastsAheadOrPerWindow()
    {
        await Task.Yield();
        var y = Seasonal(96);
        var model = new AutoformerModel<double>(Small());
        model.Train(new Matrix<double>(y.Length, 1), y);
        // One column is time steps after the series: the direct forecast, then continued from it.
        var ahead = model.Predict(new Matrix<double>(9, 1));
        var direct = model.PredictMultiple(y);
        for (int h = 0; h < 6; h++) Assert.Equal(direct[h], ahead[h], 12);
        var extended = y.ToList();
        extended.AddRange(ahead.Take(6));
        var next = model.PredictMultiple(new Vector<double>(extended.ToArray()));
        for (int h = 0; h < 3; h++) Assert.Equal(next[h], ahead[6 + h], 12);

        // Wider rows are history windows: each prediction is the first step of that window's direct forecast.
        var windows = new Matrix<double>(9, 40);
        for (int r = 0; r < windows.Rows; r++)
            for (int c = 0; c < windows.Columns; c++) windows[r, c] = y[r + c];
        var future = model.Predict(windows);
        for (int r = 0; r < windows.Rows; r++) Assert.Equal(model.PredictMultiple(windows.GetRow(r))[0], future[r], 12);
    }

    [Fact(Timeout = 120000)]
    public async Task Serialization_RoundTripsExactly()
    {
        await Task.Yield();
        var y = Seasonal(96);
        var model = new AutoformerModel<double>(Small());
        model.Train(new Matrix<double>(y.Length, 1), y);
        var restored = new AutoformerModel<double>(Small(seed: 99));
        restored.Deserialize(model.Serialize());
        var a = model.PredictMultiple(y);
        var b = restored.PredictMultiple(y);
        for (int h = 0; h < a.Length; h++) Assert.Equal(a[h], b[h]);
    }

    [Fact(Timeout = 60000)]
    public async Task TimeSeriesAndFinanceAutoformers_AreOneNetwork()
    {
        await Task.Yield();
        var series = new AutoformerModel<double>(Small());
        var finance = new Autoformer<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), Small());
        var history = new Vector<double>(24);
        var window = new Tensor<double>(new[] { 1, 24, 1 });
        for (int t = 0; t < 24; t++) window[0, t, 0] = history[t] = Math.Sin(t * 0.8) + 0.05 * t;
        var a = series.PredictMultiple(history);
        var b = finance.Forecast(window);
        Assert.Equal(new[] { 1, 6, 1 }, b.Shape.ToArray());
        for (int h = 0; h < 6; h++) Assert.Equal(a[h], b[0, h, 0], 12);
    }
}
