using System.Reflection;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Guards the shared RG-LRU training route used by Griffin, Hawk, and RecurrentGemma. RG-LRU has
/// a data-dependent recurrent graph, so its layer metadata—not three model overrides—must route
/// every containing network through a freshly-recorded eager tape.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public sealed class RgLruFamilyFusedCompiledTrainingTests
{
    [Fact(Timeout = 120000)]
    public async Task RecurrentGemma_LayerCapabilityRoutesFiniteEagerUpdates()
    {
        await Task.Yield();
        AssertAutomaticallyRoutedEagerUpdate(
            "RecurrentGemma",
            () => new RecurrentGemmaLanguageModel<double>(
                CreateArchitecture(), vocabSize: 128, modelDimension: 32,
                numLayers: 1, maxSeqLength: 32));
    }

    [Fact(Timeout = 120000)]
    public async Task Griffin_LayerCapabilityRoutesFiniteEagerUpdates()
    {
        await Task.Yield();
        AssertAutomaticallyRoutedEagerUpdate(
            "Griffin",
            () => new GriffinLanguageModel<double>(
                CreateArchitecture(), vocabSize: 128, modelDimension: 32,
                numLayers: 1, maxSeqLength: 32,
                options: new GriffinOptions { RecurrenceDimension = 40 }));
    }

    [Fact(Timeout = 120000)]
    public async Task Hawk_LayerCapabilityRoutesFiniteEagerUpdates()
    {
        await Task.Yield();
        AssertAutomaticallyRoutedEagerUpdate(
            "Hawk",
            () => new HawkLanguageModel<double>(
                CreateArchitecture(), vocabSize: 128, modelDimension: 32,
                numLayers: 1, maxSeqLength: 32,
                options: new HawkOptions { RecurrenceDimension = 40 }));
    }

    private static NeuralNetworkArchitecture<double> CreateArchitecture()
        => new(
            InputType.OneDimensional,
            NeuralNetworkTaskType.TextGeneration,
            inputSize: 32,
            outputSize: 128);

    private static void AssertAutomaticallyRoutedEagerUpdate(
        string modelName,
        Func<INeuralNetworkModel<double>> createModel)
    {
        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<double>.Invalidate();
            CompiledTapeTrainingStep<double>.ResetFusedStepCount();

            using var model = createModel();
            var recurrentLayers = ((ILayeredModel<double>)model).Layers
                .OfType<RealGatedLinearRecurrenceLayer<double>>()
                .ToArray();
            Assert.NotEmpty(recurrentLayers);
            Assert.All(recurrentLayers, layer =>
            {
                var property = layer.GetType().GetCustomAttribute<LayerPropertyAttribute>(inherit: false);
                Assert.NotNull(property);
                Assert.False(property.SupportsFusedCompiledTraining);
            });

            var input = new Tensor<double>([32]);
            for (int i = 0; i < input.Length; i++) input[i] = i % 128;

            var prediction = model.Predict(input);
            var target = new Tensor<double>(prediction.Shape.ToArray());
            int rows = target.Length / 128;
            for (int row = 0; row < rows; row++)
                target[(row * 128) + ((row + 1) % 128)] = 1.0;

            var parametersBefore = model.GetParameters().ToArray();
            model.Train(input, target);
            var parametersAfter = model.GetParameters().ToArray();
            var gradients = model.GetParameterGradients().ToArray();

            Assert.Equal(0, CompiledTapeTrainingStep<double>.GetFusedStepCount());
            Assert.True(
                parametersBefore.Where((value, index) => value != parametersAfter[index]).Any(),
                $"{modelName}'s automatically selected eager step did not update a live parameter.");
            Assert.True(double.IsFinite(model.GetLastLoss()),
                $"{modelName}'s automatically selected eager loss was not finite.");
            Assert.All(parametersAfter, value => Assert.True(double.IsFinite(value),
                $"{modelName} produced a non-finite parameter."));
            Assert.All(gradients, value => Assert.True(double.IsFinite(value),
                $"{modelName} published a non-finite gradient."));
            Assert.Contains(gradients, value => value != 0.0);

            // A second step proves this is stable routing, not a first-step fallback whose
            // next call silently re-enters the compiled plan with stale recurrent state.
            model.Train(input, target);
            Assert.Equal(0, CompiledTapeTrainingStep<double>.GetFusedStepCount());
            Assert.True(double.IsFinite(model.GetLastLoss()));
        }
        finally
        {
            CompiledTapeTrainingStep<double>.Invalidate();
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }
}
