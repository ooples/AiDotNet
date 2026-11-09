using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class E3TTSOptionsTests
{
    [Theory]
    [InlineData("NumEncoderLayers", "BertLayers", "Layers", 2)]
    [InlineData("NumHeads", "BertHeads", "Heads", 1)]
    [InlineData("DropoutRate", "BertDropout", "Dropout", .25)]
    public void InheritedTextEncoderSettingsReachTheNativeBertAndIndependentCopies(
        string inheritedName, string nativeName, string configurationName, double value)
    {
        var original = E3TTSPaperTests.TinyOptions();
        var inherited = typeof(AiDotNet.TextToSpeech.TtsModelOptions).GetProperty(inheritedName)!;
        var native = typeof(E3TTSOptions).GetProperty(nativeName)!;
        var expected = Convert.ChangeType(value, inherited.PropertyType);
        inherited.SetValue(original, expected);
        Assert.Equal(expected, native.GetValue(original));
        var copy = (E3TTSOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(original);
        using var model = new E3TTS<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), copy);
        var configuration = model.TextEncoder!.Network.Configuration;
        Assert.Equal(expected, configuration.GetType().GetProperty(configurationName)!.GetValue(configuration));
        var changed = Convert.ChangeType(value == 1 ? 2 : value / 2, inherited.PropertyType);
        native.SetValue(copy, changed);
        Assert.Equal(changed, inherited.GetValue(copy));
        Assert.Equal(expected, inherited.GetValue(original));
    }

    [Fact]
    public void InheritedTextEncoderDefaultsDescribeThePublishedBert()
    {
        var options = new E3TTSOptions();
        Assert.Equal((12, 12, .1), (options.NumEncoderLayers, options.NumHeads, options.DropoutRate));
    }

    [Fact]
    public void CopyPreservesInheritedTrainingAndSynthesisSettings()
    {
        var original = new E3TTSOptions
        {
            ConditionDim = 48, TimeEmbeddingDim = 32, NumDiffusionSteps = 37, NumFlowSteps = 7,
            InterChannels = 40, FilterChannels = 160, EncoderDim = 80, DecoderDim = 96,
            SampleRate = 24000, HiddenDim = 64, NumEncoderLayers = 3, NumDecoderLayers = 4,
            NumHeads = 2, VocabSize = 1234, MaxTextLength = 127, MaxMelLength = 513,
            MelChannels = 64, HopSize = 128, FftSize = 512, DropoutRate = 0.25,
            LearningRate = 0.003, WeightDecay = 0.02, ModelPath = "custom-e3tts.onnx",
        };

        var copy = new E3TTSOptions(original);

        Assert.Equal(Settings(original), Settings(copy));
        Assert.NotSame(original.OnnxOptions, copy.OnnxOptions);
    }

    private static object?[] Settings(E3TTSOptions options) =>
    [
        options.ConditionDim, options.TimeEmbeddingDim, options.NumDiffusionSteps, options.NumFlowSteps,
        options.InterChannels, options.FilterChannels, options.EncoderDim, options.DecoderDim,
        options.SampleRate, options.HiddenDim, options.NumEncoderLayers, options.NumDecoderLayers,
        options.NumHeads, options.VocabSize, options.MaxTextLength, options.MaxMelLength,
        options.MelChannels, options.HopSize, options.FftSize, options.DropoutRate,
        options.LearningRate, options.WeightDecay, options.ModelPath,
    ];
}
