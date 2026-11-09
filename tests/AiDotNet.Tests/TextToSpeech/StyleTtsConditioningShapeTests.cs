using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.StyleEmotion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class StyleTtsConditioningShapeTests
{
    [Fact]
    public void AdaptiveInstanceNormKeepsItsStyleWidthDuringAcousticShapeDiscovery()
    {
        var module = new StyleTtsAdaIn<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), 3, 8);
        // A flat layer walk sees the eight-channel acoustic branch immediately before this affine.
        // Its real input is the independent three-value style branch.
        module.Style.ResolveShapesOnly(new[] { 1, 8 });
        Assert.Equal(new[] { 3 }, module.Style.GetInputShape());
        Assert.Equal(64, module.Style.ParameterCount);
        Assert.Equal(new[] { 1, 8, 4 }, module.Forward(new Tensor<double>(new[] { 1, 8, 4 }),
            new Tensor<double>(new[] { 1, 3 })).Shape);
        Assert.Equal(64, module.Style.ParameterCount);
    }

    [Fact]
    public void AdaptiveLayerNormKeepsItsStyleWidthDuringAcousticShapeDiscovery()
    {
        var module = new StyleTtsAdaLayerNorm<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), 3, 8);
        module.Style.ResolveShapesOnly(new[] { 1, 8 });
        Assert.Equal(new[] { 3 }, module.Style.GetInputShape());
        Assert.Equal(64, module.Style.ParameterCount);
        Assert.Equal(new[] { 4, 8 }, module.Forward(new Tensor<double>(new[] { 4, 8 }),
            new Tensor<double>(new[] { 1, 3 })).Shape);
        Assert.Equal(64, module.Style.ParameterCount);
    }

    [Fact]
    public void HarmonicMergeKeepsItsNineSourceChannelsDuringAcousticShapeDiscovery()
    {
        var module = new StyleTts2Source<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), 24000, 300);
        module.Merge.ResolveShapesOnly(new[] { 1, 64 });
        Assert.Equal(new[] { 9 }, module.Merge.GetInputShape());
        Assert.Equal(10, module.Merge.ParameterCount);
        Assert.Equal(new[] { 1, 1, 300 }, module.Forward(new Tensor<double>(new[] { 1, 1 }),
            new Random(2294)).Shape);
        Assert.Equal(10, module.Merge.ParameterCount);
    }
}
