using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Regularization;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class DeepVoice3OptionAliasesTests
{
    private static DeepVoice3Options SmallOptions() => new()
    {
        SampleRate = 16000, HopSize = 200, FftSize = 64, WindowSize = 64, MelChannels = 8, OutputsPerStep = 2,
        EmbeddingDim = 16, EncoderChannels = 8, NumEncoderLayers = 2, DecoderPrenetSizes = [8, 16],
        NumDecoderLayers = 2, AttentionDim = 8, NumConverterLayers = 1, ConverterChannels = 16,
        DropoutRate = 0, MaxDecoderSteps = 6, MinDecoderSteps = 2,
    };

    private static DeepVoice3<double> Create(DeepVoice3Options options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: options.MelChannels) { RandomSeed = 2294 }, options);

    [Fact]
    public void DefaultsDescribeThePublishedEncoderDecoderAndAlignmentWidths()
    {
        var options = new DeepVoice3Options();
        Assert.Equal((256, 256, 80, 64, 1),
            (options.HiddenDim, options.EncoderDim, options.DecoderDim, options.EncoderChannels, options.NumHeads));
        Assert.Equal(new[] { 128, 256 }, options.DecoderPrenetSizes);
        Assert.Equal(0, options.WeightDecay);
    }

    [Theory]
    [InlineData("HiddenDim", "DecoderPrenetSizes", 18)]
    [InlineData("EncoderDim", "EmbeddingDim", 18)]
    [InlineData("DecoderDim", "MelChannels", 9)]
    public void BaseTypedWidthsReachTheNativeGraphAndIndependentCopies(string property, string nativeProperty, int value)
    {
        var options = SmallOptions();
        using var original = Create(new DeepVoice3Options(options));
        var declaringType = property == "HiddenDim" ? typeof(TtsModelOptions) : typeof(AcousticModelOptions);
        declaringType.GetProperty(property)!.SetValue(options, value);
        if (nativeProperty == "DecoderPrenetSizes") Assert.Equal(value, options.DecoderPrenetSizes[^1]);
        else Assert.Equal(value, options.GetType().GetProperty(nativeProperty)!.GetValue(options));
        var copy = (DeepVoice3Options)AiDotNet.Models.CloneEngine.CopyConfiguration(options);
        using var changed = Create(copy);
        Assert.NotEqual(original.GetParameters().Length, changed.GetParameters().Length);
        Assert.NotSame(options.DecoderPrenetSizes, copy.DecoderPrenetSizes);
        Assert.Equal(value, declaringType.GetProperty(property)!.GetValue(copy));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LegacySharedMetadataDoesNotOverwriteSerializedNativeDimensions(bool reverse)
    {
        var options = SmallOptions();
        var json = JObject.FromObject(options);
        json["HiddenDim"] = 256;
        json["EncoderDim"] = 256;
        json["DecoderDim"] = 80;
        json["NumHeads"] = 8;
        if (reverse) json = new JObject(json.Properties().Reverse().Select(property => new JProperty(property.Name, property.Value.DeepClone())));
        var restored = JsonConvert.DeserializeObject<DeepVoice3Options>(json.ToString())!;
        Assert.Equal((16, 16, 8, 1), (restored.HiddenDim, restored.EncoderDim, restored.DecoderDim, restored.NumHeads));
        Assert.Equal(options.DecoderPrenetSizes, restored.DecoderPrenetSizes);
        using var original = Create(options);
        using var copy = Create(restored);
        Assert.Equal(original.GetParameters().Length, copy.GetParameters().Length);
    }

    [Fact]
    public void PartialSharedWidthsConfigureNativeDimensions()
    {
        var options = JsonConvert.DeserializeObject<DeepVoice3Options>(
            "{\"HiddenDim\":12,\"EncoderDim\":16,\"DecoderDim\":8}")!;
        Assert.Equal((12, 16, 8), (options.DecoderPrenetSizes[^1], options.EmbeddingDim, options.MelChannels));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.2)]
    public void OptionalDecayConfiguresCoupledL2WhileRetainingAdam(double decay)
    {
        var options = SmallOptions();
        options.WeightDecay = decay;
        using var model = Create(options);
        var optimizer = Assert.IsAssignableFrom<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>(
            typeof(DeepVoice3<double>).GetField("_optimizer", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model));
        var configured = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        Assert.Equal(options.LearningRate, configured.InitialLearningRate);
        Assert.False(configured.UseAdaptiveBetas);
        Assert.Equal(decay > 0, configured.RegularizationExplicitlySet);
        if (decay > 0)
            Assert.Equal(decay, Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(configured.Regularization).GetOptions().Strength);
    }

    [Fact]
    public void NativeAttentionRejectsAnUnsupportedHeadCount()
    {
        var options = SmallOptions();
        options.NumHeads = 2;
        Assert.Contains("NumHeads must be one", Assert.Throws<ArgumentException>(() => Create(options)).Message);
    }
}
