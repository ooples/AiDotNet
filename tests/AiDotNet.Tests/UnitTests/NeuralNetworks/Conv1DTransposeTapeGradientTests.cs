using AiDotNet.ActivationFunctions;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class Conv1DTransposeTapeGradientTests
{
    [Fact(Timeout = 30000)]
    public async Task ResidualConvBlock_InputVjpMatchesAcrossSerialAndParallelTopologies()
    {
        await Task.Yield();

        var configurations = new[]
        {
            (Name: "single", Kernels: new[] { 3 }, Dilations: new[] { 1 }),
            (Name: "serial", Kernels: new[] { 3 }, Dilations: new[] { 1, 3, 5 }),
            (Name: "parallel", Kernels: new[] { 3, 7, 11 }, Dilations: new[] { 1 }),
            (Name: "serial+parallel", Kernels: new[] { 3, 7, 11 }, Dilations: new[] { 1, 3, 5 })
        };

        foreach (var configuration in configurations)
        {
            var layer = new HiFiGANResBlockLayer<double>(
                channels: 4,
                kernelSizes: configuration.Kernels,
                dilations: configuration.Dilations);
            layer.SetTrainingMode(false);
            var input = new Tensor<double>(new[] { 1, 4, 8 });
            for (int i = 0; i < input.Length; i++)
                input[i] = 0.15 * Math.Sin(i * 0.17) + 0.05 * Math.Cos(i * 0.07);

            Tensor<double> projection;
            Tensor<double> analytical;
            using (var tape = new GradientTape<double>())
            {
                var output = layer.Forward(input);
                projection = new Tensor<double>(output.Shape.ToArray());
                for (int i = 0; i < projection.Length; i++)
                    projection[i] = 0.1 * Math.Cos(i * 0.11);
                var product = AiDotNetEngine.Current.TensorMultiply(output, projection);
                var axes = Enumerable.Range(0, product.Shape.Length).ToArray();
                var objective = AiDotNetEngine.Current.ReduceSum(product, axes, keepDims: false);
                var gradients = tape.ComputeGradients(objective, new[] { input });
                Assert.True(gradients.TryGetValue(input, out analytical));
            }

            const double epsilon = 1e-5;
            const int index = 9;
            double original = input[index];
            input[index] = original + epsilon;
            double plus = ProjectionLoss(layer.Forward(input), projection);
            input[index] = original - epsilon;
            double minus = ProjectionLoss(layer.Forward(input), projection);
            input[index] = original;

            double numerical = (plus - minus) / (2.0 * epsilon);
            double expected = analytical[index];
            double relativeError = Math.Abs(expected - numerical) /
                Math.Max(1.0, Math.Max(Math.Abs(expected), Math.Abs(numerical)));
            Assert.True(relativeError < 1e-3,
                $"{configuration.Name} residual topology input VJP differs: " +
                $"analytical={expected:E6}, numerical={numerical:E6}, relative error={relativeError:F4}.");
        }
    }

    [Fact(Timeout = 30000)]
    public async Task VocoderParameterManifest_DoesNotDuplicateTensorReferences()
    {
        await Task.Yield();

        var architecture = new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 1);
        var options = new HiFiGANOptions
        {
            MelChannels = 4,
            UpsampleInitialChannels = 64,
            UpsampleRates = new[] { 8, 2 },
            UpsampleKernelSizes = new[] { 16, 4 },
            ResblockKernelSizes = new[] { 3 }
        };
        using var model = new HiFiGAN<double>(architecture, options);
        model.SetTrainingMode(false);
        _ = model.ForwardForTraining(new Tensor<double>(new[] { 1, 4, 4 }));

        var firstIds = new Dictionary<Tensor<double>, string>(ReferenceIdentityComparer<Tensor<double>>.Instance);
        var duplicates = new List<string>();
        foreach (var chunk in model.GetParameterStateChunks())
        {
            if (firstIds.TryGetValue(chunk.Tensor, out var firstId))
                duplicates.Add($"{chunk.StableId} aliases {firstId}");
            else
                firstIds.Add(chunk.Tensor, chunk.StableId);
        }

        Assert.True(duplicates.Count == 0,
            "The flat parameter manifest contains duplicate physical tensors: " +
            string.Join(", ", duplicates.Take(8)));
    }

    [Fact(Timeout = 30000)]
    public async Task KernelGradient_MatchesFiniteDifferenceAtVocoderGeometry()
    {
        await Task.Yield();

        var layer = new Conv1DTransposeLayer<double>(
            inputChannels: 64,
            outputChannels: 32,
            kernelSize: 16,
            stride: 8,
            padding: 4,
            outputPadding: 0,
            dilation: 1,
            activation: new LeakyReLUActivation<double>());
        layer.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 1, 64, 4 });
        for (int i = 0; i < input.Length; i++)
            input[i] = 0.15 * Math.Sin(i * 0.17) + 0.05 * Math.Cos(i * 0.07);

        var target = new Tensor<double>(new[] { 1, 32, 32 });
        for (int i = 0; i < target.Length; i++)
            target[i] = 0.1 * Math.Cos(i * 0.11);

        var loss = new MeanSquaredErrorLoss<double>();
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(layer.Forward(input), target);
        var parameters = layer.GetTrainableParameters();
        var gradients = tape.ComputeGradients(objective, parameters);

        Assert.True(parameters.Count >= 2, "Transposed convolution must expose its kernel and bias.");
        var kernel = parameters[0];
        Assert.True(gradients.TryGetValue(kernel, out var analytical));

        const int index = 8744;
        const double epsilon = 1e-6;
        double original = kernel[index];
        kernel[index] = original + epsilon;
        double plus = loss.ComputeTapeLoss(layer.Forward(input), target)[0];
        kernel[index] = original - epsilon;
        double minus = loss.ComputeTapeLoss(layer.Forward(input), target)[0];
        kernel[index] = original;

        double numerical = (plus - minus) / (2.0 * epsilon);
        double expected = analytical[index];
        double relativeError = Math.Abs(expected - numerical) /
            Math.Max(1e-7, Math.Abs(expected) + Math.Abs(numerical));
        Assert.True(relativeError < 1e-3,
            $"Vocoder transposed-convolution gradient differs at index {index}: " +
            $"analytical={expected:E6}, numerical={numerical:E6}, relative error={relativeError:F4}.");
    }

    private static double ProjectionLoss(Tensor<double> output, Tensor<double> projection)
    {
        double sum = 0;
        for (int i = 0; i < output.Length; i++) sum += output[i] * projection[i];
        return sum;
    }
}
