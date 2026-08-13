using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LossFunctions;

public class MeanSquaredErrorTapeGradientTests
{
    [Fact(Timeout = 30000)]
    public async Task ComputeTapeLoss_GradientMatchesFiniteDifference()
    {
        await Task.Yield();

        var loss = new MeanSquaredErrorLoss<double>();
        var predicted = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new[] { 0.2, -0.4, 0.7, 1.1, -0.3, 0.5 }));
        var target = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new[] { 0.1, 0.3, -0.2, 0.6, -0.8, 0.9 }));

        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(predicted, target);
        var gradients = tape.ComputeGradients(objective, new[] { predicted });
        Assert.True(gradients.TryGetValue(predicted, out var analytical));

        const double epsilon = 1e-6;
        for (int i = 0; i < predicted.Length; i++)
        {
            double original = predicted[i];
            predicted[i] = original + epsilon;
            double plus = loss.ComputeTapeLoss(predicted, target)[0];
            predicted[i] = original - epsilon;
            double minus = loss.ComputeTapeLoss(predicted, target)[0];
            predicted[i] = original;

            double numerical = (plus - minus) / (2.0 * epsilon);
            Assert.True(
                Math.Abs(analytical[i] - numerical) < 1e-8,
                $"MSE tape gradient[{i}] differs: analytical={analytical[i]:G17}, numerical={numerical:G17}.");
        }
    }
}
