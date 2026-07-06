using System;
using System.IO;
using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Regression.MultiOutput;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests for <see cref="MultiOutputRegressor{T}"/> — the multi-horizon substrate that turns a
/// single-output regressor into an n×H model (one head per target column). Verifies the round-trip on synthetic
/// data where each output column is a KNOWN independent linear function of the inputs, so a correct multi-output
/// wrapper must recover each column near-exactly.
/// </summary>
[Trait("Category", "Integration")]
public class MultiOutputRegressorTests
{
    private static (Matrix<double> x, Matrix<double> y) CreateMultiHorizonData(int n, int seed)
    {
        var random = new Random(seed);
        // Two features; three "horizons", each a distinct linear combination of the features.
        var x = new Matrix<double>(n, 2);
        var y = new Matrix<double>(n, 3);
        for (var i = 0; i < n; i++)
        {
            var a = random.NextDouble() * 10.0;
            var b = random.NextDouble() * 10.0;
            x[i, 0] = a;
            x[i, 1] = b;
            y[i, 0] = 2.0 * a + 1.0 * b + 3.0;   // horizon 1
            y[i, 1] = -1.0 * a + 4.0 * b - 2.0;  // horizon 2
            y[i, 2] = 0.5 * a - 0.5 * b + 1.0;   // horizon 3
        }

        return (x, y);
    }

    private static MultiOutputRegressor<double> NewModel()
        => new(() => new MultipleRegression<double>());

    [Fact]
    public void TrainThenPredict_RecoversEachHorizon()
    {
        var (x, y) = CreateMultiHorizonData(200, seed: 42);
        var model = NewModel();

        model.Train(x, y);
        var pred = model.Predict(x);

        Assert.Equal(3, model.OutputCount);
        Assert.Equal(x.Rows, pred.Rows);
        Assert.Equal(3, pred.Columns);

        // Each column is a clean linear function → wrapper should recover it to high precision.
        for (var i = 0; i < x.Rows; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(pred[i, j] - y[i, j]) < 1e-6,
                    $"row {i} col {j}: predicted {pred[i, j]} vs expected {y[i, j]}");
            }
        }
    }

    [Fact]
    public void Predict_ColumnsAreIndependent_NotBroadcastOfOne()
    {
        // Guards against the crippling defect this substrate replaces: a single scalar repeated across horizons.
        var (x, y) = CreateMultiHorizonData(100, seed: 7);
        var model = NewModel();
        model.Train(x, y);
        var pred = model.Predict(x);

        var col0DiffersFromCol1 = false;
        var col1DiffersFromCol2 = false;
        for (var i = 0; i < x.Rows; i++)
        {
            if (Math.Abs(pred[i, 0] - pred[i, 1]) > 1e-3) col0DiffersFromCol1 = true;
            if (Math.Abs(pred[i, 1] - pred[i, 2]) > 1e-3) col1DiffersFromCol2 = true;
        }

        Assert.True(col0DiffersFromCol1, "horizon 0 and 1 predictions must differ (independent heads)");
        Assert.True(col1DiffersFromCol2, "horizon 1 and 2 predictions must differ (independent heads)");
    }

    [Fact]
    public void Serialize_RoundTrips_PreservingPredictions()
    {
        var (x, y) = CreateMultiHorizonData(120, seed: 11);
        var model = NewModel();
        model.Train(x, y);
        var before = model.Predict(x);

        var bytes = model.Serialize();
        var restored = NewModel();
        restored.Deserialize(bytes);
        var after = restored.Predict(x);

        Assert.Equal(before.Rows, after.Rows);
        Assert.Equal(before.Columns, after.Columns);
        for (var i = 0; i < before.Rows; i++)
        {
            for (var j = 0; j < before.Columns; j++)
            {
                Assert.True(Math.Abs(before[i, j] - after[i, j]) < 1e-9,
                    $"serialize round-trip mismatch at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void SaveLoadState_RoundTrips_PreservingPredictions()
    {
        var (x, y) = CreateMultiHorizonData(80, seed: 13);
        var model = NewModel();
        model.Train(x, y);
        var before = model.Predict(x);

        using var ms = new MemoryStream();
        model.SaveState(ms);
        ms.Position = 0;
        var restored = NewModel();
        restored.LoadState(ms);
        var after = restored.Predict(x);

        for (var i = 0; i < before.Rows; i++)
        {
            for (var j = 0; j < before.Columns; j++)
            {
                Assert.True(Math.Abs(before[i, j] - after[i, j]) < 1e-9,
                    $"state round-trip mismatch at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void Clone_IsIndependentOfOriginal()
    {
        var (x, y) = CreateMultiHorizonData(60, seed: 17);
        var model = NewModel();
        model.Train(x, y);

        var clone = (MultiOutputRegressor<double>)model.Clone();
        var a = model.Predict(x);
        var b = clone.Predict(x);

        for (var i = 0; i < a.Rows; i++)
        {
            for (var j = 0; j < a.Columns; j++)
            {
                Assert.True(Math.Abs(a[i, j] - b[i, j]) < 1e-9, $"clone mismatch at [{i},{j}]");
            }
        }

        // Mutating the clone's parameters must not affect the original.
        var p = clone.GetParameters();
        for (var k = 0; k < p.Length; k++)
        {
            p[k] = p[k] + 100.0;
        }

        clone.SetParameters(p);
        var aAfter = model.Predict(x);
        for (var i = 0; i < a.Rows; i++)
        {
            for (var j = 0; j < a.Columns; j++)
            {
                Assert.True(Math.Abs(a[i, j] - aAfter[i, j]) < 1e-9, "mutating clone changed original");
            }
        }
    }
}
