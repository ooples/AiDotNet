using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public sealed class GroupedQueryAttentionGradientTests
{
    [Fact(Timeout = 30000)]
    public async Task RoPE_GqaInputGradient_MatchesFiniteDifferencesAtVideoDecoderShape()
    {
        await Task.Yield();

        const int batch = 4;
        const int sequence = 16;
        const int embedding = 32;
        var layer = new GroupedQueryAttentionLayer<double>(
            sequenceLength: 64,
            embeddingDimension: embedding,
            numHeads: 4,
            numKVHeads: 2,
            activationFunction: new IdentityActivation<double>(),
            deferAllocation: true);
        layer.ConfigurePositionalEncoding(
            PositionalEncodingType.Rotary,
            ropeTheta: 10000.0,
            maxSequenceLength: 64);
        layer.SetTrainingMode(false);

        var input = CreatePattern([batch, sequence, embedding], 0.02);
        var projection = CreatePattern([batch, sequence, embedding], 0.03);

        Tensor<double> gradient;
        using (var tape = new GradientTape<double>())
        {
            var output = layer.Forward(input);
            var projected = AiDotNetEngine.Current.TensorMultiply(output, projection);
            var objective = AiDotNetEngine.Current.ReduceSum(
                projected, [0, 1, 2], keepDims: false);
            var gradients = tape.ComputeGradients(objective, [input]);
            Assert.True(gradients.TryGetValue(input, out gradient),
                "RoPE GQA must preserve the tape path to its input.");
        }

        const double step = 1e-6;
        int[] samples = [0, input.Length / 3, input.Length - 1];
        foreach (int index in samples)
        {
            double original = input[index];
            input[index] = original + step;
            double plus = Project(layer, input, projection);
            input[index] = original - step;
            double minus = Project(layer, input, projection);
            input[index] = original;

            double numerical = (plus - minus) / (2.0 * step);
            double analytical = gradient![index];
            double tolerance = Math.Max(1e-8, Math.Abs(numerical) * 1e-5);
            Assert.True(Math.Abs(numerical - analytical) <= tolerance,
                $"GQA input gradient mismatch at {index}: numerical={numerical:R}, " +
                $"analytical={analytical:R}, difference={Math.Abs(numerical - analytical):R}, " +
                $"tolerance={tolerance:R}.");
        }
    }

    private static double Project(
        GroupedQueryAttentionLayer<double> layer,
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

    private static Tensor<double> CreatePattern(int[] shape, double scale)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = scale * (((i * 17) % 31) - 15) / 15.0;
        return tensor;
    }
}
