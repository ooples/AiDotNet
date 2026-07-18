using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Regression test for the DeepAR constant-prediction bug: when a caller passes prediction rows SHORTER than
/// LookbackWindow (e.g. a cross-sectional feature matrix), Predict used to replace every row with the same fixed
/// slice of the training series, so the whole test block came out identical. It must instead use each caller
/// row (left-padded) and produce row-specific, varying predictions.
/// </summary>
public sealed class DeepARShortWindowRegressionTests
{
    [Fact]
    [Trait("category", "unit")]
    public void Predict_on_short_rows_is_not_a_constant_column()
    {
        const int lookback = 16;
        const int train = 120;

        // Train on proper [N, lookback] windows of a smooth series (the model's documented contract).
        var xTrain = new Matrix<double>(train, lookback);
        var yTrain = new Vector<double>(train);
        for (var i = 0; i < train; i++)
        {
            for (var j = 0; j < lookback; j++)
            {
                xTrain[i, j] = Math.Sin((i + j) * 0.3) + (i + j) * 0.01;
            }

            yTrain[i] = Math.Sin((i + lookback) * 0.3) + (i + lookback) * 0.01;
        }

        var model = new DeepARModel<double>(new DeepAROptions<double>
        {
            LookbackWindow = lookback, ForecastHorizon = 1, HiddenSize = 16, NumLayers = 1, Epochs = 15,
        });
        model.Train(xTrain, yTrain);

        // Predict on SHORT rows (2 columns < lookback) whose values differ per row — the cross-sectional case
        // that used to collapse to one constant.
        var xTest = new Matrix<double>(24, 2);
        for (var i = 0; i < 24; i++)
        {
            xTest[i, 0] = Math.Sin(i * 0.5);
            xTest[i, 1] = Math.Cos(i * 0.5);
        }

        var predictions = model.Predict(xTest);

        Assert.Equal(24, predictions.Length);
        Assert.All(Enumerable.Range(0, 24), i => Assert.True(
            !double.IsNaN(predictions[i]) && !double.IsInfinity(predictions[i]), $"non-finite at {i}"));
        Assert.True(predictions.Distinct().Count() > 1, "DeepAR produced a constant prediction column on short rows");
    }
}
