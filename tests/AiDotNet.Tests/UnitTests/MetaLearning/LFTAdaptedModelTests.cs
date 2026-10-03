// Nullable is off for this file: Constructor_RejectsANullHead passes null on purpose to check the guard.
#nullable disable
using System;
using System.Linq;
using AiDotNet.MetaLearning.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Behavioural checks of <see cref="LFTAdaptedModel{T, TInput, TOutput}"/>. It is the model LFT (Tseng et al.
/// 2020) returns after adaptation: a frozen encoder followed by a metric head that scores each example as
/// <c>softmax(W f + b)</c>. Here the encoder is a <see cref="LinearEmbeddingModel"/> with known weights, so
/// every score can be computed in closed form.
/// </summary>
public class LFTAdaptedModelTests
{
    private const int InputDim = 3;
    private const int FeatureDim = 2;
    private const int Classes = 3;

    private static LinearEmbeddingModel Encoder()
    {
        var encoder = new LinearEmbeddingModel(InputDim, FeatureDim);
        // f = [[1, -1, 0.5], [0.2, 0.3, -0.4]] x + [0.1, -0.2]
        encoder.SetParameters(Vec(1, -1, 0.5, 0.2, 0.3, -0.4, 0.1, -0.2));
        return encoder;
    }

    // W [classes, features] row-major, then b [classes].
    private static Vector<double> Head() => Vec(0.5, -1.0, 1.5, 0.25, -0.75, 2.0, 0.1, 0.0, -0.3);

    private static Vector<double> Vec(params double[] values)
    {
        var v = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++) v[i] = values[i];
        return v;
    }

    private static Matrix<double> Inputs()
    {
        var x = new Matrix<double>(3, InputDim);
        double[,] values = { { 0.5, -1.0, 2.0 }, { 1.5, 0.25, -0.5 }, { -2.0, 1.0, 0.0 } };
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < InputDim; c++) x[r, c] = values[r, c];
        return x;
    }

    private static double[] Expected(Matrix<double> x, int row, Vector<double> head)
    {
        var f = new[]
        {
            (1 * x[row, 0]) - x[row, 1] + (0.5 * x[row, 2]) + 0.1,
            (0.2 * x[row, 0]) + (0.3 * x[row, 1]) - (0.4 * x[row, 2]) - 0.2
        };
        var logits = Enumerable.Range(0, Classes)
            .Select(o => head[(Classes * FeatureDim) + o] + (head[o * FeatureDim] * f[0]) + (head[(o * FeatureDim) + 1] * f[1]))
            .ToArray();
        double max = logits.Max();
        var exp = logits.Select(l => Math.Exp(l - max)).ToArray();
        return exp.Select(e => e / exp.Sum()).ToArray();
    }

    private static LFTAdaptedModel<double, Matrix<double>, Tensor<double>> Model(Vector<double> head) =>
        new(Encoder(), head, FeatureDim, Classes);

    [Fact]
    public void Predict_IsTheSoftmaxOfTheMetricHeadOverTheEncoderFeatures()
    {
        var model = Model(Head());
        var x = Inputs();
        var scores = model.Predict(x);
        Assert.Equal(new[] { 3, Classes }, scores.Shape.ToArray());
        for (int r = 0; r < 3; r++)
        {
            var expected = Expected(x, r, Head());
            for (int c = 0; c < Classes; c++) Assert.Equal(expected[c], scores[r, c], 10);
        }
    }

    [Fact]
    public void Predict_ScoresEachExampleOnItsOwn()
    {
        var model = Model(Head());
        var x = Inputs();
        var altered = Inputs();
        for (int c = 0; c < InputDim; c++) altered[1, c] = -altered[1, c] + 3;
        var a = model.Predict(x);
        var b = model.Predict(altered);
        for (int c = 0; c < Classes; c++)
        {
            Assert.Equal(a[0, c], b[0, c], 12);
            Assert.Equal(a[2, c], b[2, c], 12);
        }
        Assert.True(Enumerable.Range(0, Classes).Sum(c => Math.Abs(a[1, c] - b[1, c])) > 1e-6,
            "Changing example 1 did not change its own scores.");
    }

    [Fact]
    public void Constructor_CopiesTheMetricHead()
    {
        var head = Head();
        var model = Model(head);
        var before = model.Predict(Inputs());
        for (int i = 0; i < head.Length; i++) head[i] = 0;
        var after = model.Predict(Inputs());
        Assert.Equal(before.ToVector().ToArray(), after.ToVector().ToArray());
        Assert.Equal(Head().ToArray(), model.MetricHead.ToArray());
    }

    [Fact]
    public void WithParameters_ReplacesTheMetricHead()
    {
        var model = Model(Head());
        var other = Vec(-0.5, 1.0, 0.0, 0.3, 0.9, -1.2, 0.0, 0.4, 0.2);
        var replaced = (LFTAdaptedModel<double, Matrix<double>, Tensor<double>>)model.WithParameters(other);
        Assert.Equal(other.ToArray(), replaced.MetricHead.ToArray());
        var x = Inputs();
        var scores = replaced.Predict(x);
        var expected = Expected(x, 0, other);
        for (int c = 0; c < Classes; c++) Assert.Equal(expected[c], scores[0, c], 10);
    }

    [Fact]
    public void Constructor_RejectsANullHead()
    {
        Assert.Throws<ArgumentNullException>(() => Model(null));
    }
}
