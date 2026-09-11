using AiDotNet.NeuralNetworks.Layers;
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
}
