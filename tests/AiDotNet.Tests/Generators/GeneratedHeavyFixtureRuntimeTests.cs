using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NER.Options;
using AiDotNet.NER.TransformerBased;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Video.Motion;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Verifies that the compiled, generated fixtures construct the intended real models.</summary>
public sealed class GeneratedHeavyFixtureRuntimeTests
{
    [Fact]
    public void MemFlow_GeneratedFactoryAndShapesUseTheBoundedTwoFrameArchitecture()
    {
        var fixture = new MemFlowProbe();
        using var model = fixture.CreateModel();

        Assert.Equal(new[] { 1, 6, 64, 64 }, fixture.DeclaredInputShape);
        Assert.Equal(new[] { 1, 2, 64, 64 }, fixture.DeclaredOutputShape);
        Assert.Equal(64, model.Architecture.InputHeight);
        Assert.Equal(64, model.Architecture.InputWidth);
        Assert.Equal(6, model.Architecture.InputDepth);
        Assert.Equal(2, model.Architecture.OutputSize);
        Assert.Equal(4, model.Layers.Count);
        for (int layer = 0; layer < 3; layer++)
        {
            Assert.Equal(8, Assert.IsType<ConvolutionalLayer<float>>(model.Layers[layer]).OutputDepth);
        }
        Assert.Equal(2, Assert.IsType<ConvolutionalLayer<float>>(model.Layers[3]).OutputDepth);
    }

    [Fact]
    public void MemFlow_ProductionDefaultRetainsItsOriginalScale()
    {
        using var model = new MemFlow<float>();

        Assert.Equal(256, model.Architecture.InputHeight);
        Assert.Equal(256, model.Architecture.InputWidth);
        Assert.Equal(6, model.Architecture.InputDepth);
        Assert.Equal(2, model.Architecture.OutputSize);
        Assert.Equal(10, model.Layers.Count);
        for (int layer = 0; layer < 9; layer++)
        {
            Assert.Equal(64, Assert.IsType<ConvolutionalLayer<float>>(model.Layers[layer]).OutputDepth);
        }
        Assert.Equal(2, Assert.IsType<ConvolutionalLayer<float>>(model.Layers[9]).OutputDepth);
    }

    [Fact]
    public void MelGAN_GeneratedFactoryOnlyReducesWidthAndRetainsItsOneFrameContract()
    {
        var fixture = new MelGANProbe();
        using var model = fixture.CreateModel();
        var options = Assert.IsType<MelGANOptions>(model.GetOptions());
        var defaults = new MelGANOptions();

        Assert.Equal(new[] { 1, 80, 1 }, fixture.DeclaredInputShape);
        Assert.Equal(new[] { 1, 1, 256 }, fixture.DeclaredOutputShape);
        Assert.Equal(32, options.NgfBase);
        Assert.Equal(512, defaults.NgfBase);
        Assert.Equal(defaults.NumResStacks, options.NumResStacks);
        Assert.Equal(defaults.MelChannels, options.MelChannels);
        Assert.Equal(defaults.HopSize, options.HopSize);
        Assert.Equal(defaults.SampleRate, options.SampleRate);
    }

    [Fact]
    public void TemplateNER_GeneratedFactoryBoundsTheRealEncoderAndRetainsDropout()
    {
        var fixture = new TemplateNERProbe();
        using var model = fixture.CreateModel();

        Assert.Equal(new[] { 8, 32 }, fixture.DeclaredInputShape);
        AssertBoundedTransformerTopology(model);
    }

    [Fact]
    public void XLMRoBERTaNER_GeneratedFactoryBoundsTheRealEncoderAndRetainsDropout()
    {
        var fixture = new XLMRoBERTaNERProbe();
        using var model = fixture.CreateModel();

        Assert.Equal(new[] { 8, 32 }, fixture.DeclaredInputShape);
        AssertBoundedTransformerTopology(model);
    }

    private static void AssertBoundedTransformerTopology(TransformerNERBase<float> model)
    {
        var options = Assert.IsType<TransformerNEROptions>(model.GetOptions());
        Assert.Equal(32, model.Architecture.InputSize);
        Assert.Equal(9, model.Architecture.OutputSize);
        Assert.Equal(new[] { 16, 32 }, model.ExpectedInputShape);
        Assert.Equal(32, options.HiddenDimension);
        Assert.Equal(4, options.NumAttentionHeads);
        Assert.Equal(2, options.NumTransformerLayers);
        Assert.Equal(64, options.IntermediateDimension);
        Assert.Equal(16, options.MaxSequenceLength);
        Assert.Equal(9, options.NumLabels);
        Assert.Equal(5e-6, options.LearningRate);
        Assert.Equal(options.LearningRate / options.WarmupSteps, options.WarmupInitialLearningRate);
        Assert.Equal(0.1, options.DropoutRate);
        Assert.Collection(model.Layers,
            layer => Assert.IsType<TransformerEncoderLayer<float>>(layer),
            layer => Assert.IsType<DropoutLayer<float>>(layer),
            layer => Assert.IsType<TransformerEncoderLayer<float>>(layer),
            layer => Assert.IsType<DropoutLayer<float>>(layer),
            layer => Assert.IsType<DenseLayer<float>>(layer));

        var defaults = new TransformerNEROptions();
        Assert.Equal(768, defaults.HiddenDimension);
        Assert.Equal(12, defaults.NumTransformerLayers);
        Assert.Equal(12, defaults.NumAttentionHeads);
        Assert.Equal(3072, defaults.IntermediateDimension);
        Assert.Equal(256, defaults.MaxSequenceLength);
        Assert.Equal(5e-5, defaults.LearningRate);
        Assert.Equal(0.0, defaults.WarmupInitialLearningRate);
    }

    private sealed class MemFlowProbe : ModelFamilyTests.Generated.MemFlowTests
    {
        public int[] DeclaredInputShape => InputShape;
        public int[] DeclaredOutputShape => OutputShape;
        public MemFlow<float> CreateModel() => Assert.IsType<MemFlow<float>>(CreateNetwork());
    }

    private sealed class MelGANProbe : ModelFamilyTests.Generated.MelGANTests
    {
        public int[] DeclaredInputShape => InputShape;
        public int[] DeclaredOutputShape => OutputShape;
        public MelGAN<float> CreateModel() => Assert.IsType<MelGAN<float>>(CreateNetwork());
    }

    private sealed class TemplateNERProbe : ModelFamilyTests.Generated.TemplateNERTests
    {
        public int[] DeclaredInputShape => InputShape;
        public TemplateNER<float> CreateModel() => Assert.IsType<TemplateNER<float>>(CreateNetwork());
    }

    private sealed class XLMRoBERTaNERProbe : ModelFamilyTests.Generated.XLMRoBERTaNERTests
    {
        public int[] DeclaredInputShape => InputShape;
        public XLMRoBERTaNER<float> CreateModel() => Assert.IsType<XLMRoBERTaNER<float>>(CreateNetwork());
    }
}
