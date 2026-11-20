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

    [Fact]
    public void ConstantNonzeroSeries_PreservesMeanAndParameterLayoutAcrossSerializeAndClone()
    {
        var untrained = new ARModel<double>(new ARModelOptions<double> { AROrder = 1 });
        Assert.Equal(0, untrained.ParameterCount);

        var constantSeries = new[] { 7.5, 7.5, 7.5, 7.5, 7.5, 7.5 };
        var x = new Matrix<double>(constantSeries.Length, 1);
        for (int i = 0; i < constantSeries.Length; i++)
        {
            x[i, 0] = i;
        }

        var trained = new ARModel<double>(new ARModelOptions<double> { AROrder = 1 });
        trained.Train(x, new Vector<double>(constantSeries));

        // Parameter surface is _arCoefficients (AROrder = 1) + _trainedSeries buffer (6);
        // _seriesMean is [FittedParameter(InputSized = true)] so it adds 0 flat parameter slots.
        int expectedParameterCount = 1 + constantSeries.Length;
        Assert.Equal(expectedParameterCount, trained.ParameterCount);
        Assert.Equal(expectedParameterCount, trained.GetParameters().Length);
        var arCoeffs = (Vector<double>)trained.GetModelMetadata().AdditionalInfo["ARCoefficients"];
        Assert.Single(arCoeffs);
        Assert.Equal(0.0, arCoeffs[0], 10);

        var query = new Vector<double>(new[] { 7.5, 7.5, 7.5 });
        var zeroQuery = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        AssertConstantForecast(trained, query, zeroQuery, 7.5);

        byte[] payload = trained.Serialize();
        var reloaded = new ARModel<double>(new ARModelOptions<double> { AROrder = 1 });
        reloaded.Deserialize(payload);
        Assert.Equal(expectedParameterCount, reloaded.ParameterCount);
        AssertConstantForecast(reloaded, query, zeroQuery, 7.5);

        var cloned = (ARModel<double>)trained.Clone();
        Assert.Equal(expectedParameterCount, cloned.ParameterCount);
        AssertConstantForecast(cloned, query, zeroQuery, 7.5);

        var deepCopied = (ARModel<double>)trained.DeepCopy();
        Assert.Equal(expectedParameterCount, deepCopied.ParameterCount);
        AssertConstantForecast(deepCopied, query, zeroQuery, 7.5);
    }

    [Fact]
    public void NonConstantSeries_PreservesMeanAcrossSerializeAndClone()
    {
        var trained = TrainAR1();
        int expectedParameterCount = 1 + Series.Length;
        Assert.Equal(expectedParameterCount, trained.ParameterCount);

        var reloaded = new ARModel<double>(new ARModelOptions<double> { AROrder = 1 });
        reloaded.Deserialize(trained.Serialize());

        var cloned = (ARModel<double>)trained.Clone();

        foreach (var candidate in new[] { reloaded, cloned })
        {
            Assert.Equal(expectedParameterCount, candidate.ParameterCount);
            var forecast = candidate.Forecast(new Vector<double>(Series), 2);
            Assert.Equal(8.0 / 3.0, forecast[0], 10);
            Assert.Equal(55.0 / 18.0, forecast[1], 10);
            Assert.Equal(8.0 / 3.0, candidate.PredictSingle(new Vector<double>(Series)), 10);
        }
    }

    private static void AssertConstantForecast(
        ARModel<double> model,
        Vector<double> constantHistory,
        Vector<double> zeroHistory,
        double expectedMean)
    {
        var forecast = model.Forecast(constantHistory, 2);
        Assert.Equal(2, forecast.Length);
        Assert.Equal(expectedMean, forecast[0], 10);
        Assert.Equal(expectedMean, forecast[1], 10);
        Assert.Equal(expectedMean, model.PredictSingle(constantHistory), 10);
        Assert.Equal(expectedMean, model.PredictSingle(zeroHistory), 10);
    }
}
