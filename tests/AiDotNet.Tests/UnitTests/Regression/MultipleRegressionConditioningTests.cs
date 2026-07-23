using System;
using System.Linq;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Regression;

/// <summary>
/// Regression test for the numerical-conditioning fix: near-collinear features make the normal-equations matrix
/// X'X severely ill-conditioned (normal equations square the condition number). Cholesky still succeeds on it and
/// used to return absurd, blown-up coefficients — huge but finite, so no NaN/Inf guard tripped — producing
/// ~1e10 predictions. A tiny Tikhonov diagonal-loading must keep predictions bounded.
/// </summary>
public sealed class MultipleRegressionConditioningTests
{
    [Fact]
    [Trait("category", "unit")]
    public void Near_collinear_features_do_not_blow_up_predictions()
    {
        var rng = new Random(3);
        const int n = 60;
        var x = new Matrix<double>(n, 3);
        var y = new Vector<double>(n);
        for (var i = 0; i < n; i++)
        {
            var a = rng.NextDouble();
            var b = rng.NextDouble();
            x[i, 0] = a;
            x[i, 1] = a + (rng.NextDouble() - 0.5) * 1e-7;   // near-duplicate of column 0
            x[i, 2] = b;
            y[i] = 3.0 * a - 1.0 * b + (rng.NextDouble() - 0.5) * 0.01;   // bounded target ~[-1, 3]
        }

        var model = new MultipleRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(n, predictions.Length);
        Assert.All(Enumerable.Range(0, n), i =>
        {
            var p = predictions[i];
            Assert.True(!double.IsNaN(p) && !double.IsInfinity(p), $"non-finite prediction {p}");
            // The target lives in ~[-1, 3]; a well-conditioned fit predicts in that range. Anything beyond a
            // generous bound means the ill-conditioned normal equations blew the coefficients up.
            Assert.True(Math.Abs(p) < 50.0, $"blown-up prediction {p} at row {i}");
        });
    }
}
