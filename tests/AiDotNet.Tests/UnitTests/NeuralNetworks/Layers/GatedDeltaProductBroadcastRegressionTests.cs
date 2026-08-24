using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Fixtures;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Regression coverage for broadcasted gate and projection biases in GatedDeltaProduct.
/// </summary>
[Collection(LayerSerializationCollection.Name)]
public sealed class GatedDeltaProductBroadcastRegressionTests
{
    [Fact(Timeout = 30000)]
    public async Task BroadcastedGateBiases_PreserveEveryInputVjpComponent()
    {
        await Task.Yield();
        var layer = new GatedDeltaProductLayer<double>(
            sequenceLength: 2,
            modelDimension: 4,
            numHeads: 2,
            numHouseholders: 1,
            activationFunction: new IdentityActivation<double>());

        // Remove random initialization from the proof. The pattern deliberately
        // gives every projection and bias a distinct non-zero contribution.
        var parameters = layer.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
            parameters[i] = 0.015 * (((i * 13) % 17) - 8);
        layer.SetParameters(parameters);
        layer.SetTrainingMode(true);

        var input = Pattern([1, 2, 4], 0.08);
        var projection = Pattern([1, 2, 4], 0.11);

        AssertInputVjp(layer, input, projection, Enumerable.Range(0, input.Length));
    }

    [Fact(Timeout = 60000)]
    public async Task GeneratedFixtureShape_BroadcastedGateBiases_PreserveSampledInputVjp()
    {
        await Task.Yield();
        var layer = new GatedDeltaProductLayer<double>(sequenceLength: 4);

        var parameters = layer.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
            parameters[i] = 0.002 * (((i * 13) % 17) - 8);
        layer.SetParameters(parameters);
        layer.SetTrainingMode(true);

        var input = Pattern([4, 256], 0.04);
        var projection = Pattern([4, 256], 0.03);
        var sampledIndices = Enumerable.Range(0, 12)
            .Select(sample => Math.Min(input.Length - 1, sample * input.Length / 12));

        AssertInputVjp(layer, input, projection, sampledIndices);
    }

    private static void AssertInputVjp(
        GatedDeltaProductLayer<double> layer,
        Tensor<double> input,
        Tensor<double> projection,
        IEnumerable<int> indices)
    {
        Tensor<double> gradient;
        using (var tape = new GradientTape<double>())
        {
            var output = layer.Forward(input);
            var objective = AiDotNetEngine.Current.ReduceSum(
                AiDotNetEngine.Current.TensorMultiply(output, projection),
                Enumerable.Range(0, output.Shape.Length).ToArray(),
                keepDims: false);
            var gradients = tape.ComputeGradients(objective, [input]);
            Assert.True(gradients.TryGetValue(input, out gradient),
                "GatedDeltaProduct must keep a tape path to every input component.");
        }

        const double step = 1e-5;
        foreach (int index in indices)
        {
            double original = input[index];
            input[index] = original + step;
            double plus = Project(layer, input, projection);
            input[index] = original - step;
            double minus = Project(layer, input, projection);
            input[index] = original;

            double numerical = (plus - minus) / (2.0 * step);
            double analytical = gradient![index];
            double tolerance = Math.Max(2e-5, Math.Abs(numerical) * 5e-3);
            Assert.True(Math.Abs(numerical - analytical) <= tolerance,
                $"Input VJP mismatch at {index}: numerical={numerical:R}, " +
                $"analytical={analytical:R}, tolerance={tolerance:R}.");
        }
    }

    private static double Project(
        GatedDeltaProductLayer<double> layer,
        Tensor<double> input,
        Tensor<double> projection)
    {
        using var noGrad = new NoGradScope<double>();
        var output = layer.Forward(input);
        double sum = 0.0;
        for (int i = 0; i < output.Length; i++)
            sum += output[i] * projection[i];
        return sum;
    }

    private static Tensor<double> Pattern(int[] shape, double scale)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = scale * (((i * 7) % 11) - 5) / 5.0;
        return tensor;
    }
}
