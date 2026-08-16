using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class APNet2NetworkGradientTests
{
    [Fact(Timeout = 30000)]
    public async Task ComputeGradients_MatchesFullTapeAndFiniteDifference()
    {
        await Task.Yield();

        using var model = new APNet2<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 640,
                outputSize: 1539),
            new APNet2Options
            {
                ConvNeXtChannels = 32,
                ConvNeXtIntermediateChannels = 96,
                NumConvNeXtBlocks = 2,
            });

        var input = CreateTensor(new[] { 1, 80, 2 }, 0.02);
        var target = CreateTensor(new[] { 1, 80, 1539 }, -0.03);
        var loss = Assert.IsType<APNet2GeneratorLoss<double>>(model.DefaultLossFunction);

        // Materialize any genuinely lazy parameters before capturing their tensor identities.
        // The generated conformance path does the same shape-resolution work before it builds
        // its manifest; a diagnostic that holds pre-materialization tensors tests stale handles,
        // not the network's gradient.
        using (new NoGradScope<double>())
        {
            _ = model.ForwardForTraining(input);
        }

        var parameters = TapeTrainingStep<double>.CollectParameters(model.Layers, structureVersion: -1);

        IReadOnlyDictionary<Tensor<double>, Tensor<double>> full;
        using (var tape = new GradientTape<double>())
        {
            var prediction = model.ForwardForTraining(input);
            var objective = loss.ComputeTapeLoss(prediction, target);
            full = tape.ComputeGradients(objective, sources: null);
        }

        IReadOnlyDictionary<Tensor<double>, Tensor<double>> selective;
        using (var tape = new GradientTape<double>())
        {
            var prediction = model.ForwardForTraining(input);
            var objective = loss.ComputeTapeLoss(prediction, target);
            selective = tape.ComputeGradients(objective, parameters);
        }

        const int targetFlatIndex = 19560;
        int parameterStart = 0;
        Tensor<double>? targetParameter = null;
        int targetCoordinate = -1;
        foreach (var parameter in parameters)
        {
            if (targetFlatIndex < parameterStart + parameter.Length)
            {
                targetParameter = parameter;
                targetCoordinate = targetFlatIndex - parameterStart;
                break;
            }
            parameterStart += parameter.Length;
        }

        Assert.NotNull(targetParameter);
        Assert.True(targetCoordinate >= 0);
        double original = targetParameter![targetCoordinate];
        const double epsilon = 1e-6;
        targetParameter[targetCoordinate] = original + epsilon;
        double plus = loss.ComputeTapeLoss(model.ForwardForTraining(input), target)[0];
        targetParameter[targetCoordinate] = original - epsilon;
        double minus = loss.ComputeTapeLoss(model.ForwardForTraining(input), target)[0];
        targetParameter[targetCoordinate] = original;
        double numerical = (plus - minus) / (2.0 * epsilon);

        Assert.Equal(parameters.Count, selective.Count);
        Assert.Equal(parameters.Count, parameters.Count(p => full.ContainsKey(p)));
        foreach (var parameter in parameters)
        {
            Assert.True(selective.TryGetValue(parameter, out var selectiveGradient));
            Assert.True(full.TryGetValue(parameter, out var fullGradient));
            Assert.Equal(fullGradient.Length, selectiveGradient.Length);
            for (int i = 0; i < fullGradient.Length; i++)
            {
                Assert.True(
                    Math.Abs(fullGradient[i] - selectiveGradient[i]) < 1e-12,
                    $"Selective gradient differs at parameter length {parameter.Length}, index {i}: " +
                    $"selective={selectiveGradient[i]:G17}, full={fullGradient[i]:G17}, " +
                    $"target-coordinate numerical={numerical:G17}.");
            }
        }

        double analytical = full[targetParameter][targetCoordinate];
        double scale = Math.Max(Math.Abs(analytical), Math.Abs(numerical));
        Assert.True(
            Math.Abs(analytical - numerical) <= Math.Max(1e-8, scale * 1e-4),
            $"APNet2 flat coordinate {targetFlatIndex} (parameter coordinate {targetCoordinate}) " +
            $"has analytical={analytical:G17}, numerical={numerical:G17}.");
    }

    private static Tensor<double> CreateTensor(int[] shape, double offset)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = offset + (0.001 * ((i % 97) - 48));
        return tensor;
    }
}
