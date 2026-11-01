using System.IO;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Model = AiDotNet.MetaLearning.Models.LinearVectorModel;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// The library's linear meta-learning base model, y = X w + b. Its gradient must be the exact gradient of the
/// loss it is handed, since every meta-learner built on it steps on that gradient.
/// </summary>
public class LinearVectorModelTests
{
    private static Matrix<double> X() => new(new double[,] { { 1, 2 }, { -1, 0.5 }, { 3, -2 } });

    [Fact]
    public void Predict_IsTheAffineMap()
    {
        var model = new Model(2);
        model.SetParameters(new Vector<double>(new[] { 2.0, -1.0, 0.5 }));

        var y = model.Predict(X());

        Assert.Equal(new[] { 0.5, -2.0, 8.5 }, y.ToArray());
    }

    [Fact]
    public void Gradient_MatchesFiniteDifferencesOfTheGivenLoss()
    {
        var model = new Model(2);
        var target = new Vector<double>(new[] { 1.0, 0.0, -1.0 });
        var loss = new MeanSquaredErrorLoss<double>();
        var parameters = new Vector<double>(new[] { 0.3, -0.7, 0.2 });
        model.SetParameters(parameters);

        var gradient = model.ComputeGradients(X(), target, loss);

        const double h = 1e-6;
        for (int i = 0; i < parameters.Length; i++)
        {
            var plus = parameters.Clone();
            plus[i] += h;
            var minus = parameters.Clone();
            minus[i] -= h;
            model.SetParameters(plus);
            double lossPlus = loss.CalculateLoss(model.Predict(X()), target);
            model.SetParameters(minus);
            double lossMinus = loss.CalculateLoss(model.Predict(X()), target);
            Assert.Equal((lossPlus - lossMinus) / (2 * h), gradient[i], 6);
        }
    }

    [Fact]
    public void Training_RecoversAnExactlyLinearTarget()
    {
        var model = new Model(2, learningRate: 0.05);
        var x = X();
        var target = new Vector<double>(3);
        for (int r = 0; r < 3; r++)
            target[r] = 1.5 * x[r, 0] - 0.5 * x[r, 1] + 0.25;

        for (int step = 0; step < 4000; step++)
            model.Train(x, target);

        var learned = model.GetParameters();
        Assert.Equal(1.5, learned[0], 4);
        Assert.Equal(-0.5, learned[1], 4);
        Assert.Equal(0.25, learned[2], 4);
    }

    [Fact]
    public void State_RoundTripsThroughAStream()
    {
        var model = new Model(2);
        model.SetParameters(new Vector<double>(new[] { 4.0, 5.0, 6.0 }));
        using var stream = new MemoryStream();
        model.SaveState(stream);
        stream.Position = 0;

        var restored = new Model(2);
        restored.LoadState(stream);

        Assert.Equal(model.GetParameters().ToArray(), restored.GetParameters().ToArray());
        Assert.Equal(model.Predict(X()).ToArray(), restored.Predict(X()).ToArray());
    }
}
