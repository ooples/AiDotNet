using System;
using AiDotNet.MetaLearning.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Behavioural checks of <see cref="LinearEmbeddingModel"/>, the affine embedding <c>e = W x + b</c> that
/// metric-based meta-learners use as an encoder. The parameter layout is W row-major <c>[embedding, input]</c>
/// followed by b.
/// </summary>
public class LinearEmbeddingModelTests
{
    private const int InputDim = 3;
    private const int EmbeddingDim = 2;

    private static Matrix<double> Inputs()
    {
        var x = new Matrix<double>(4, InputDim);
        double[,] values = { { 0.5, -1.0, 2.0 }, { 1.5, 0.25, -0.5 }, { -2.0, 1.0, 0.0 }, { 0.1, 0.2, 0.3 } };
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < InputDim; c++) x[r, c] = values[r, c];
        return x;
    }

    private static Tensor<double> Targets()
    {
        var y = new Tensor<double>(new[] { 4, EmbeddingDim });
        double[] values = { 1.0, -0.5, 0.25, 2.0, -1.5, 0.0, 0.75, 0.5 };
        for (int i = 0; i < values.Length; i++) y[i] = values[i];
        return y;
    }

    private static Vector<double> Parameters(params double[] values)
    {
        var p = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++) p[i] = values[i];
        return p;
    }

    private static double Loss(LinearEmbeddingModel model, Matrix<double> x, Tensor<double> y) =>
        model.DefaultLossFunction.CalculateLoss(model.Predict(x).ToVector(), y.ToVector());

    [Fact]
    public void Predict_IsTheAffineMapOfItsParameters()
    {
        var model = new LinearEmbeddingModel(InputDim, EmbeddingDim);
        // W = [[1, 2, 3], [-1, 0, 0.5]], b = [0.25, -2]
        model.SetParameters(Parameters(1, 2, 3, -1, 0, 0.5, 0.25, -2));
        var x = Inputs();
        var e = model.Predict(x);
        Assert.Equal(new[] { 4, EmbeddingDim }, e.Shape.ToArray());
        for (int r = 0; r < 4; r++)
        {
            double e0 = (1 * x[r, 0]) + (2 * x[r, 1]) + (3 * x[r, 2]) + 0.25;
            double e1 = (-1 * x[r, 0]) + (0 * x[r, 1]) + (0.5 * x[r, 2]) - 2;
            Assert.Equal(e0, e[r, 0], 12);
            Assert.Equal(e1, e[r, 1], 12);
        }
    }

    [Fact]
    public void ComputeGradients_MatchesCentralDifferencesOfItsLoss()
    {
        var model = new LinearEmbeddingModel(InputDim, EmbeddingDim);
        model.SetParameters(Parameters(0.3, -0.2, 0.1, 0.4, 0.05, -0.6, 0.2, -0.1));
        var x = Inputs();
        var y = Targets();
        var analytic = model.ComputeGradients(x, y);
        var theta = model.GetParameters();
        Assert.Equal(theta.Length, analytic.Length);
        const double h = 1e-6;
        for (int i = 0; i < theta.Length; i++)
        {
            var plus = theta.Clone(); plus[i] += h;
            var minus = theta.Clone(); minus[i] -= h;
            var up = new LinearEmbeddingModel(InputDim, EmbeddingDim); up.SetParameters(plus);
            var down = new LinearEmbeddingModel(InputDim, EmbeddingDim); down.SetParameters(minus);
            double numeric = (Loss(up, x, y) - Loss(down, x, y)) / (2 * h);
            Assert.True(Math.Abs(numeric - analytic[i]) <= 1e-5 * Math.Max(1, Math.Abs(numeric)),
                $"parameter {i}: analytic {analytic[i]} vs central difference {numeric}");
        }
    }

    [Fact]
    public void Train_LowersTheLoss()
    {
        var model = new LinearEmbeddingModel(InputDim, EmbeddingDim, learningRate: 0.05);
        var x = Inputs();
        var y = Targets();
        double first = Loss(model, x, y);
        for (int step = 0; step < 20; step++) model.Train(x, y);
        double last = Loss(model, x, y);
        Assert.True(last < first, $"20 gradient steps did not lower the loss: {first} -> {last}");
    }

    [Fact]
    public void WithParameters_BuildsAnIndependentModel()
    {
        var model = new LinearEmbeddingModel(InputDim, EmbeddingDim);
        var original = model.GetParameters().Clone();
        var replacement = Parameters(1, 1, 1, 1, 1, 1, 1, 1);
        var copy = model.WithParameters(replacement);
        Assert.Equal(replacement.ToArray(), copy.GetParameters().ToArray());
        Assert.Equal(original.ToArray(), model.GetParameters().ToArray());
    }

    [Fact]
    public void Predict_RejectsInputNarrowerThanTheModel()
    {
        var model = new LinearEmbeddingModel(InputDim, EmbeddingDim);
        Assert.Throws<ArgumentException>(() => model.Predict(new Matrix<double>(2, InputDim - 1)));
    }
}
