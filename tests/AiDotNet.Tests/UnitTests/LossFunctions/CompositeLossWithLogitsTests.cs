using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LossFunctions;

public class CompositeLossWithLogitsTests
{
    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void NonFiniteTermWeight_IsRejectedBeforeNumericConversion(double weight)
    {
        var exception = Assert.Throws<ArgumentException>(() =>
            new CompositeLossWithLogits<double>((new DiceLoss<double>(), weight)));

        Assert.Contains("term 0", exception.Message);
        Assert.Equal("terms", exception.ParamName);
    }

    [Fact]
    public void ExtremeLogits_ProduceFiniteNonNegativeLoss_InBothPublicPaths()
    {
        var loss = new CompositeLossWithLogits<double>();
        var logits = new Vector<double>([-1000.0, -10.0, 0.0, 10.0, 1000.0]);
        var target = new Vector<double>([0.0, 0.0, 1.0, 1.0, 0.0]);

        double vectorLoss = loss.CalculateLoss(logits, target);
        var tapeLoss = loss.ComputeTapeLoss(
            new Tensor<double>([5], logits),
            new Tensor<double>([5], target));

        Assert.True(!double.IsNaN(vectorLoss) && !double.IsInfinity(vectorLoss) && vectorLoss >= 0.0,
            $"Vector path returned invalid logits loss {vectorLoss:G17}.");
        Assert.Single(tapeLoss.ToArray());
        Assert.True(!double.IsNaN(tapeLoss[0]) && !double.IsInfinity(tapeLoss[0]) && tapeLoss[0] >= 0.0,
            $"Tape path returned invalid logits loss {tapeLoss[0]:G17}.");
    }

    [Fact]
    public void VectorAndTapePaths_AgreeForModerateLogits()
    {
        var loss = new CompositeLossWithLogits<double>();
        var logits = new Vector<double>([-2.0, -0.25, 0.5, 1.75]);
        var target = new Vector<double>([0.0, 1.0, 0.0, 1.0]);

        double vectorLoss = loss.CalculateLoss(logits, target);
        var tapeLoss = loss.ComputeTapeLoss(
            new Tensor<double>([4], logits),
            new Tensor<double>([4], target));

        Assert.Single(tapeLoss.ToArray());
        Assert.Equal(vectorLoss, tapeLoss[0], 12);
    }

    [Fact]
    public void TapeGradient_MatchesFiniteDifferenceThroughSigmoidBoundary()
    {
        var loss = new CompositeLossWithLogits<double>();
        var predicted = new Tensor<double>([3], new Vector<double>([-2.0, 0.25, 1.5]));
        var target = new Tensor<double>([3], new Vector<double>([0.0, 1.0, 1.0]));

        Tensor<double> analytical;
        using (var tape = new GradientTape<double>())
        {
            var objective = loss.ComputeTapeLoss(predicted, target);
            var gradients = tape.ComputeGradients(objective, new[] { predicted });
            Assert.True(gradients.TryGetValue(predicted, out analytical));
        }

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
            double scale = Math.Max(1e-8, Math.Abs(analytical[i]) + Math.Abs(numerical));
            double relativeError = Math.Abs(analytical[i] - numerical) / scale;
            Assert.True(relativeError < 1e-5,
                $"Gradient[{i}] differs across the logits boundary: analytical={analytical[i]:G17}, "
                + $"numerical={numerical:G17}, relativeError={relativeError:G6}.");
        }
    }
}
