using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Hand-derived reference for an AR model with a constant: the coefficients are fitted on the de-meaned
/// series and the mean is added back in forecasts (statsmodels AutoReg / ARIMA with trend 'c').
/// </summary>
public class ARModelMeanReferenceTests
{
    // y = [1, 3, 2, 4, 3, 5], mean μ = 3, centered c = [-2, 0, -1, 1, 0, 2].
    // AR(1) least squares on the centered series: φ = Σ c[t]·c[t-1] / Σ c[t-1]² = (-1) / 6.
    // One step:  ŷ₆ = μ + φ·(y₅ - μ) = 3 - (1/6)·2 = 8/3.
    // Two steps: ŷ₇ = μ + φ·(ŷ₆ - μ) = 3 - (1/6)·(-1/3) = 55/18.
    private static readonly double[] Series = { 1, 3, 2, 4, 3, 5 };

    private static ARModel<double> TrainAR1()
    {
        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 1 });
        var x = new Matrix<double>(Series.Length, 1);
        for (int i = 0; i < Series.Length; i++)
        {
            x[i, 0] = i;
        }

        model.Train(x, new Vector<double>(Series));
        return model;
    }

    [Fact]
    public void Forecast_AddsTheSeriesMeanBack()
    {
        var model = TrainAR1();

        var forecast = model.Forecast(new Vector<double>(Series), 2);

        Assert.Equal(2, forecast.Length);
        Assert.Equal(8.0 / 3.0, forecast[0], 10);
        Assert.Equal(55.0 / 18.0, forecast[1], 10);
    }

    [Fact]
    public void PredictSingle_AddsTheSeriesMeanBack()
    {
        var model = TrainAR1();

        Assert.Equal(8.0 / 3.0, model.PredictSingle(new Vector<double>(Series)), 10);
    }
}
