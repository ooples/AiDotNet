using System.Reflection;
using AiDotNet.Audio.SourceSeparation;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>
/// The RoFormer separators' WeightDecay option reaches their default AdamW (#2393). It was documented as the training weight
/// decay but never passed on, so AdamW ran with its own 0.01 whatever the option said.
/// </summary>
public class SourceSeparationWeightDecayTests
{
    private const double Decay = 0.123;

    private static NeuralNetworkArchitecture<double> Architecture() => new(
        inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression, inputSize: 16, outputSize: 4);

    private static object Create(string model) => model switch
    {
        nameof(BSRoFormer<double>) => new BSRoFormer<double>(Architecture(), new BSRoFormerOptions
        { NumBands = 2, BandEmbeddingDim = 8, TransformerDim = 16, NumTransformerLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 32, WeightDecay = Decay }),
        nameof(MelBandRoFormer<double>) => new MelBandRoFormer<double>(Architecture(), new MelBandRoFormerOptions
        { NumMelBands = 2, BandEmbeddingDim = 8, TransformerDim = 16, NumTransformerLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 32, WeightDecay = Decay }),
        _ => throw new System.ArgumentOutOfRangeException(nameof(model)),
    };

    [Theory]
    [InlineData("BSRoFormer")]
    [InlineData("MelBandRoFormer")]
    public void DefaultOptimizer_UsesTheOptionsWeightDecay(string model)
    {
        var instance = Create(model);
        var field = instance.GetType().GetField("_optimizer", BindingFlags.Instance | BindingFlags.NonPublic);
        var optimizer = Assert.IsType<AdamWOptimizer<double, Tensor<double>, Tensor<double>>>(field?.GetValue(instance));
        var options = Assert.IsType<AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        Assert.Equal(Decay, options.WeightDecay, 12);
    }
}
