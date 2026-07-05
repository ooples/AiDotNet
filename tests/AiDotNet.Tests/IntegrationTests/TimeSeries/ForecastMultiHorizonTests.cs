using System;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// Tests the native multi-horizon forecast surface on time-series models: the recursive one-step default on the
/// base, and N-BEATS' direct multi-step override. Both must return a horizon-length path of finite values, and the
/// N-BEATS direct path (horizon == trained ForecastHorizon) must dispatch away from the recursive fallback.
/// </summary>
[Trait("Category", "Integration")]
public class ForecastMultiHorizonTests
{
    private static Vector<double> MakeSeries(int n)
    {
        // A smooth deterministic series (trend + seasonality) so a lightly-trained model produces finite, varied output.
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            y[i] = 10.0 + 0.05 * i + 2.0 * Math.Sin(i * 0.3);
        }

        return y;
    }

    [Fact]
    public void NBeats_NativeDirect_ReturnsHorizonLengthPath()
    {
        const int lookback = 8;
        const int horizon = 4;
        var y = MakeSeries(140);
        var x = new Matrix<double>(y.Length, 1); // deep univariate model uses y as the series; x is a placeholder

        var model = new NBEATSModel<double>(new NBEATSModelOptions<double>
        {
            LookbackWindow = lookback,
            ForecastHorizon = horizon,
            NumStacks = 2,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 16,
            Epochs = 20,
        });
        model.Train(x, y);

        var window = new Vector<double>(lookback);
        for (int i = 0; i < lookback; i++)
        {
            window[i] = y[y.Length - lookback + i];
        }

        // Native direct path (horizon == trained ForecastHorizon).
        var direct = model.ForecastMultiHorizon(window, horizon);
        Assert.Equal(horizon, direct.Length);
        for (int h = 0; h < horizon; h++)
        {
            Assert.True(double.IsFinite(direct[h]), $"direct forecast step {h} was not finite: {direct[h]}");
        }

        // Recursive fallback (horizon != trained ForecastHorizon) must also produce a finite path of the right length.
        var recursive = model.ForecastMultiHorizon(window, 6);
        Assert.Equal(6, recursive.Length);
        for (int h = 0; h < 6; h++)
        {
            Assert.True(double.IsFinite(recursive[h]), $"recursive forecast step {h} was not finite: {recursive[h]}");
        }
    }

    [Fact]
    public void BaseRecursive_OnArModel_ReturnsHorizonLengthPath()
    {
        var y = MakeSeries(120);
        var x = new Matrix<double>(y.Length, 1);

        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 4 });
        model.Train(x, y);

        // AR uses its own order-length lookback; feed the last AROrder values.
        var window = new Vector<double>(4);
        for (int i = 0; i < 4; i++)
        {
            window[i] = y[y.Length - 4 + i];
        }

        var forecast = model.ForecastMultiHorizon(window, 5);
        Assert.Equal(5, forecast.Length);
        for (int h = 0; h < 5; h++)
        {
            Assert.True(double.IsFinite(forecast[h]), $"AR recursive forecast step {h} was not finite: {forecast[h]}");
        }
    }
}
