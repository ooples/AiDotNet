using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class ConvolutionalLayerFastKernelRegressionTests
{
    [Fact(Timeout = 120000)]
    public async Task ExactInference_ThreeByThreeElu_PromotesFastKernelAfterParityCheck()
    {
        await Task.Yield();

        var input = CreateInput([1, 3, 8, 8]);
        var layer = CreateInitializedLayer(input, inputDepth: 3, outputDepth: 4);
        layer.SetTrainingMode(false);

        var first = layer.Forward(input);
        var firstSnapshot = first.AsSpan().ToArray();
        var second = layer.Forward(input);

        var matchField = typeof(ConvolutionalLayer<float>).GetField(
            "_optimizedMatchesReference",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        Assert.NotNull(matchField);
        Assert.True((bool)matchField.GetValue(layer)!,
            "Exact inference must keep the platform-fast canonical route after its first-shape " +
            "parity check; falling back to the platform-slow kernel reintroduces the MiDaS regression.");
        Assert.Equal(firstSnapshot, second.AsSpan().ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task TapeTraining_ThreeByThreeElu_KernelAndBiasGradientsMatchFiniteDifferences()
    {
        await Task.Yield();

        var input = CreateInput([1, 1, 5, 5]);
        var layer = CreateInitializedLayer(input, inputDepth: 1, outputDepth: 1);
        layer.SetTrainingMode(true);
        var parameters = layer.GetTrainableParameters();
        Assert.Equal(2, parameters.Count);

        IReadOnlyDictionary<Tensor<float>, Tensor<float>> gradients;
        using (var tape = new GradientTape<float>())
        {
            var objective = SquaredLoss(layer.Forward(input));
            gradients = tape.ComputeGradients(objective, parameters);
        }

        AssertFiniteDifferenceGradient(layer, input, parameters[0], gradients, index: 4);
        AssertFiniteDifferenceGradient(layer, input, parameters[1], gradients, index: 0);
    }

    [Fact(Timeout = 120000)]
    public async Task CompiledTraining_ThreeByThreeElu_MatchesEagerAcrossTraceAndReplay()
    {
        await Task.Yield();

        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();

            var input = CreateInput([1, 1, 5, 5]);
            var target = new Tensor<float>([1, 1, 5, 5]);
            var eagerLayer = CreateInitializedLayer(input, inputDepth: 1, outputDepth: 1);
            var compiledLayer = CreateInitializedLayer(input, inputDepth: 1, outputDepth: 1);
            CopyParameters(eagerLayer, compiledLayer);
            eagerLayer.SetTrainingMode(true);
            compiledLayer.SetTrainingMode(true);

            IReadOnlyList<ITrainableLayer<float>> eagerLayers = [eagerLayer];
            IReadOnlyList<ITrainableLayer<float>> compiledLayers = [compiledLayer];
            Func<Tensor<float>, Tensor<float>, Tensor<float>> loss =
                (prediction, expected) => SquaredLoss(AiDotNetEngine.Current.TensorSubtract(prediction, expected));

            for (int step = 0; step < 2; step++)
            {
                float eagerLoss = TapeTrainingStep<float>.Step(
                    eagerLayers, input, target, 0.002f, eagerLayer.Forward, loss);
                float compiledLoss = CompiledTapeTrainingStep<float>.Step(
                    compiledLayers, input, target, 0.002f, compiledLayer.Forward, loss);

                Assert.InRange(Math.Abs(eagerLoss - compiledLoss), 0f, 2e-5f);
                AssertParametersClose(eagerLayer, compiledLayer, tolerance: 2e-5f);
            }

            AssertCompiledTrainingPlanCached();
        }
        finally
        {
            CompiledTapeTrainingStep<float>.Invalidate();
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }

    private static ConvolutionalLayer<float> CreateInitializedLayer(
        Tensor<float> input,
        int inputDepth,
        int outputDepth)
    {
        var layer = ConvolutionalLayer<float>.WithInputDepth(
            inputDepth,
            outputDepth,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new ELUActivation<float>());
        layer.SetTrainingMode(false);
        _ = layer.Forward(input);

        var parameters = layer.GetTrainableParameters();
        Assert.Equal(2, parameters.Count);
        for (int parameterIndex = 0; parameterIndex < parameters.Count; parameterIndex++)
        {
            var parameter = parameters[parameterIndex];
            for (int i = 0; i < parameter.Length; i++)
                parameter[i] = (float)(((i + parameterIndex * 3) % 11) - 5) * 0.025f;
        }

        return layer;
    }

    private static Tensor<float> CreateInput(int[] shape)
    {
        var input = new Tensor<float>(shape);
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(0.35 * Math.Sin(i * 0.19) - 0.2 * Math.Cos(i * 0.07));
        return input;
    }

    private static Tensor<float> SquaredLoss(Tensor<float> value)
    {
        var squared = AiDotNetEngine.Current.TensorMultiply(value, value);
        return AiDotNetEngine.Current.ReduceSum(squared, null);
    }

    private static void AssertFiniteDifferenceGradient(
        ConvolutionalLayer<float> layer,
        Tensor<float> input,
        Tensor<float> parameter,
        IReadOnlyDictionary<Tensor<float>, Tensor<float>> gradients,
        int index)
    {
        Assert.True(gradients.TryGetValue(parameter, out var analytical),
            "The optimized convolution must remain connected to every trainable parameter.");
        Assert.All(analytical.AsSpan().ToArray(), value => Assert.True(float.IsFinite(value)));

        const float epsilon = 1e-3f;
        float original = parameter[index];
        parameter[index] = original + epsilon;
        float plus = SquaredLoss(layer.Forward(input))[0];
        parameter[index] = original - epsilon;
        float minus = SquaredLoss(layer.Forward(input))[0];
        parameter[index] = original;

        double numerical = (plus - minus) / (2.0 * epsilon);
        double expected = analytical[index];
        double relativeError = Math.Abs(expected - numerical) /
            Math.Max(1e-3, Math.Abs(expected) + Math.Abs(numerical));
        Assert.True(relativeError < 0.03,
            $"Gradient differs at parameter index {index}: analytical={expected:E6}, " +
            $"numerical={numerical:E6}, relative error={relativeError:F4}.");
    }

    private static void CopyParameters(
        ConvolutionalLayer<float> source,
        ConvolutionalLayer<float> destination)
    {
        var sourceParameters = source.GetTrainableParameters();
        var destinationParameters = destination.GetTrainableParameters();
        Assert.Equal(sourceParameters.Count, destinationParameters.Count);
        for (int i = 0; i < sourceParameters.Count; i++)
            sourceParameters[i].AsSpan().CopyTo(destinationParameters[i].Data.Span);
    }

    private static void AssertCompiledTrainingPlanCached()
    {
        const System.Reflection.BindingFlags flags =
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic;
        var cacheField = typeof(CompiledTapeTrainingStep<float>).GetField("_cache", flags);
        Assert.NotNull(cacheField);
        var cache = cacheField.GetValue(null);
        Assert.NotNull(cache);

        var plansField = cache.GetType().GetField(
            "_trainingPlans",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        Assert.NotNull(plansField);
        var plans = plansField.GetValue(cache);
        Assert.NotNull(plans);
        var countProperty = plans.GetType().GetProperty("Count");
        Assert.NotNull(countProperty);
        Assert.True((int)countProperty.GetValue(plans)! > 0,
            "Compiled training must cache a real graph plan; eager fallback does not prove graph replay.");
    }

    private static void AssertParametersClose(
        ConvolutionalLayer<float> expected,
        ConvolutionalLayer<float> actual,
        float tolerance)
    {
        var expectedParameters = expected.GetTrainableParameters();
        var actualParameters = actual.GetTrainableParameters();
        Assert.Equal(expectedParameters.Count, actualParameters.Count);
        for (int parameterIndex = 0; parameterIndex < expectedParameters.Count; parameterIndex++)
        {
            for (int i = 0; i < expectedParameters[parameterIndex].Length; i++)
            {
                Assert.InRange(
                    Math.Abs(expectedParameters[parameterIndex][i] - actualParameters[parameterIndex][i]),
                    0f,
                    tolerance);
            }
        }
    }
}
