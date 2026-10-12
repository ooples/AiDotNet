using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// Issue #2147: a factory that takes the model's architecture and never reads it promises a stack
/// sized from that architecture while sizing it purely from option defaults.
/// </summary>
/// <remarks>
/// Re-derived with one instrument (a C# syntax tree, which bounds each method body exactly rather
/// than guessing where a doc comment stops): 194 factories in <c>LayerHelper</c> take a
/// <c>NeuralNetworkArchitecture&lt;T&gt;</c> and 92 never read it. The five covered here are the
/// time-series transformers whose final projection IS the model's flat output width AND whose
/// model agrees with it. TFT looked like a fifth and is not: its fixture passes an architecture
/// whose <c>OutputSize</c> disagrees with <c>predictionHorizon * numFeatures</c>, and honouring it
/// failed all 25 of TFT's own tests, so TFT needs the per-model reconciliation the issue's risk
/// section describes rather than this mechanical change.
/// </remarks>
public sealed class LayerFactoryArchitectureTests
{
    private static NeuralNetworkArchitecture<double> WithOutputSize(int outputSize) =>
        new(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 32,
            outputSize: outputSize);

    /// <summary>
    /// An architecture built for a different horizon than the options default must size the
    /// output projection, otherwise the model silently emits the option's width.
    /// </summary>
    [Theory(Timeout = 60000)]
    [InlineData("PatchTST")]
    [InlineData("ITransformer")]
    [InlineData("ETSformer")]
    public async Task TheArchitecturesOutputSizeSizesTheOutputProjection(string factory)
    {
        await Task.Yield();

        const int declared = 5;
        var layers = Build(factory, WithOutputSize(declared));
        var last = layers[layers.Count - 1] as DenseLayer<double>;

        Assert.True(last is not null,
            $"{factory}'s stack does not end in a projection; it ends in "
                + layers[layers.Count - 1].GetType().Name + ".");
        Assert.Equal(declared, last.GetOutputShape()[last.GetOutputShape().Length - 1]);
    }

    /// <summary>
    /// With the architecture's output size left unset the option default still decides, so the
    /// change cannot disturb a caller that passes no architecture sizing.
    /// </summary>
    [Theory(Timeout = 60000)]
    [InlineData("PatchTST")]
    [InlineData("ITransformer")]
    [InlineData("ETSformer")]
    public async Task AnUnsetOutputSizeLeavesTheOptionDefaultInCharge(string factory)
    {
        await Task.Yield();

        var layers = Build(factory, WithOutputSize(0));
        var last = layers[layers.Count - 1] as DenseLayer<double>;

        Assert.True(last is not null, $"{factory}'s stack does not end in a projection.");
        int width = last.GetOutputShape()[last.GetOutputShape().Length - 1];
        Assert.True(width > 0, $"{factory} produced a zero-width projection from an unset architecture.");
        Assert.NotEqual(5, width);
    }

    private static List<ILayer<double>> Build(
        string factory, NeuralNetworkArchitecture<double> architecture) => factory switch
    {
        "PatchTST" => LayerHelper<double>.CreateDefaultPatchTSTLayers(
            architecture, sequenceLength: 32, predictionHorizon: 8, numFeatures: 4,
            patchSize: 8, stride: 4, numLayers: 1, numHeads: 2, modelDimension: 8,
            feedForwardDimension: 16).ToList(),
        "ITransformer" => LayerHelper<double>.CreateDefaultITransformerLayers(
            architecture, sequenceLength: 32, predictionHorizon: 8, numFeatures: 4,
            numLayers: 1, numHeads: 2, modelDimension: 8, feedForwardDimension: 16).ToList(),
        "ETSformer" => LayerHelper<double>.CreateDefaultETSformerLayers(
            architecture, sequenceLength: 32, predictionHorizon: 8, numFeatures: 4,
            modelDimension: 8, feedForwardDim: 16, numEncoderLayers: 1, numDecoderLayers: 1,
            numHeads: 2).ToList(),
        _ => throw new ArgumentOutOfRangeException(nameof(factory), factory, "Unknown factory."),
    };
}
