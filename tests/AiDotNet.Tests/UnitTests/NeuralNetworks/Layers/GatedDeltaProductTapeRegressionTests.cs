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
        double strongestMagnitude = 0.0;
        var mismatches = new List<string>();
        for (int i = 0; i < weights.Length; i++)
        {
            double original = weights[i];
            weights[i] = original + step;
            double plus = ProjectionLoss(layer, input, projection);
            weights[i] = original - step;
            double minus = ProjectionLoss(layer, input, projection);
            weights[i] = original;

            double numerical = (plus - minus) / (2.0 * step);
            strongestMagnitude = Math.Max(strongestMagnitude, Math.Abs(numerical));

            double scale = Math.Max(1.0, Math.Max(Math.Abs(numerical), Math.Abs(analytical![i])));
            double relativeError = Math.Abs(numerical - analytical[i]) / scale;
            if (relativeError >= 1e-3)
                mismatches.Add(
                    $"[{i}] numerical={numerical:G17}, analytical={analytical[i]:G17}, rel={relativeError:G6}");
        }

        Assert.True(strongestMagnitude > 1e-7,
            "The test input did not exercise a Householder-weight derivative.");
        Assert.True(mismatches.Count == 0,
            "Householder gradient mismatch:" + Environment.NewLine + string.Join(Environment.NewLine, mismatches));
    }

    [Fact]
    public void BatchedHouseholderVectors_MatchIndependentSequenceForwards()
    {
        using var layer = new GatedDeltaProductLayer<double>(
            sequenceLength: 2, modelDimension: 4, numHeads: 2, numHouseholders: 2);
        layer.SetTrainingMode(false);
        using var input = new Tensor<double>([2, 2, 4]);
        for (int i = 0; i < input.Length; i++)
            input[i] = Math.Sin((i + 1) * 0.37);

        using var batchedOutput = layer.Forward(input);
        double strongestOutputMagnitude = 0.0;
        for (int i = 0; i < batchedOutput.Length; i++)
            strongestOutputMagnitude = Math.Max(strongestOutputMagnitude, Math.Abs(batchedOutput[i]));
        Assert.True(strongestOutputMagnitude > 1e-8,
            $"The batched recurrence produced only degenerate near-zero outputs (max |y|={strongestOutputMagnitude:G6}).");

        for (int batch = 0; batch < 2; batch++)
        {
            layer.ResetState();
            using var sequence = new Tensor<double>([2, 4]);
            for (int i = 0; i < sequence.Length; i++)
                sequence[i] = input[(batch * sequence.Length) + i];

            using var independentOutput = layer.Forward(sequence);
            for (int i = 0; i < independentOutput.Length; i++)
            {
                Assert.Equal(
                    independentOutput[i],
                    batchedOutput[(batch * independentOutput.Length) + i],
                    10);
            }
        }
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
