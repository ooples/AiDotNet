using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LossFunctions;

public class CrossEntropyWithLogitsGradientStabilityTests
{
    [Fact]
    public void RankThreeTiedExtremeLogits_HaveFiniteSoftmaxMinusTargetGradient()
    {
        var logits = new Tensor<double>([1, 2, 4], new Vector<double>(
        [
            1000.0, 1000.0, -1000.0, -1000.0,
            -1000.0, -1000.0, 1000.0, 1000.0
        ]));
        var target = new Tensor<double>([1, 2, 4], new Vector<double>(
        [
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]));
        var loss = new CrossEntropyWithLogitsLoss<double>();

        Tensor<double> gradient;
        using (var tape = new GradientTape<double>())
        {
            var objective = loss.ComputeTapeLoss(logits, target);
            Assert.False(double.IsNaN(objective[0]) || double.IsInfinity(objective[0]));
            var gradients = tape.ComputeGradients(objective, new[] { logits });
            Assert.True(gradients.TryGetValue(logits, out gradient));
        }

        // Mean reduction over two positions: (softmax - target) / 2. The tied maxima each
        // receive probability 0.5, and the underflowed classes receive zero.
        double[] expected = [0.25, -0.25, 0.0, 0.0, 0.0, 0.0, -0.25, 0.25];
        Assert.Equal(expected.Length, gradient.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(double.IsNaN(gradient[i]) || double.IsInfinity(gradient[i]),
                $"Gradient[{i}] is {gradient[i]}.");
            Assert.Equal(expected[i], gradient[i], 12);
        }
    }
}
