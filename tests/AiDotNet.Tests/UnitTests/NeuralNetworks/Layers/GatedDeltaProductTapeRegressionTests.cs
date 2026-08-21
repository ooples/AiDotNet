using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Regression coverage for the GatedDeltaProduct recurrence's autodiff graph.
/// </summary>
public class GatedDeltaProductTapeRegressionTests
{
    [Fact]
    public void HouseholderWeightGradient_MatchesFiniteDifference()
    {
        using var layer = new GatedDeltaProductLayer<double>(
            sequenceLength: 2, modelDimension: 4, numHeads: 2, numHouseholders: 2);
        layer.SetTrainingMode(false);
        using var input = new Tensor<double>(new[] { 2, 4 }, new Vector<double>(new[]
        {
            0.2, -0.1, 0.4, 0.3,
            -0.3, 0.5, 0.1, -0.2,
        }));
        using var projection = new Tensor<double>(new[] { 2, 4 }, new Vector<double>(new[]
        {
            0.7, -0.4, 0.2, 0.9,
            -0.6, 0.3, 0.8, -0.5,
        }));
        var weights = layer.GetHouseholderWeights();

        Tensor<double>? analytical = null;
        using (var tape = new GradientTape<double>())
        {
            var output = layer.Forward(input);
            var projected = AiDotNetEngine.Current.TensorMultiply(output, projection);
            var loss = AiDotNetEngine.Current.ReduceSum(projected, new[] { 0, 1 }, keepDims: false);
            var gradients = tape.ComputeGradients(loss, new[] { weights });
            gradients.TryGetValue(weights, out analytical);
        }

        Assert.NotNull(analytical);

        const double step = 1e-5;
        int strongestIndex = -1;
        double strongestNumerical = 0.0;
        for (int i = 0; i < weights.Length; i++)
        {
            double original = weights[i];
            weights[i] = original + step;
            double plus = ProjectionLoss(layer, input, projection);
            weights[i] = original - step;
            double minus = ProjectionLoss(layer, input, projection);
            weights[i] = original;

            double numerical = (plus - minus) / (2.0 * step);
            if (Math.Abs(numerical) > Math.Abs(strongestNumerical))
            {
                strongestNumerical = numerical;
                strongestIndex = i;
            }
        }

        Assert.True(strongestIndex >= 0 && Math.Abs(strongestNumerical) > 1e-7,
            "The test input did not exercise a Householder-weight derivative.");
        double scale = Math.Max(1.0, Math.Max(Math.Abs(strongestNumerical), Math.Abs(analytical![strongestIndex])));
        double relativeError = Math.Abs(strongestNumerical - analytical[strongestIndex]) / scale;
        Assert.True(relativeError < 1e-3,
            $"Householder gradient mismatch at {strongestIndex}: numerical={strongestNumerical:G17}, " +
            $"analytical={analytical[strongestIndex]:G17}, relative error={relativeError:G6}.");
    }

    private static double ProjectionLoss(
        GatedDeltaProductLayer<double> layer,
        Tensor<double> input,
        Tensor<double> projection)
    {
        var output = layer.Forward(input);
        double sum = 0.0;
        for (int i = 0; i < output.Length; i++) sum += output[i] * projection[i];
        return sum;
    }
}
