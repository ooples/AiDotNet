using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class E3TTSOptionsTests
{
    [Fact]
    public void CopyPreservesNonuniformChannelsWhosePeakCannotScaleTheDefaultLayout()
    {
        var original = E3TTSPaperTests.TinyOptions();
        original.BlockChannels = [3, 3, 4, 4, 5, 6, 6, 6, 6, 6];
        var copy = (E3TTSOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(original);
        Assert.Equal(6, copy.HiddenDim);
        Assert.Equal(original.BlockChannels, copy.BlockChannels);
        Assert.NotSame(original.BlockChannels, copy.BlockChannels);
        using var model = new E3TTS<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), copy);
        Assert.Equal(original.BlockChannels,
            model.WaveformUnet!.DownBlocks.Select(block => block.FirstConvolutions[0].GetOutputShape()[0]));
        copy.BlockChannels[0] = 2;
        Assert.Equal(3, original.BlockChannels[0]);
    }

    [Theory]
    [InlineData("HiddenDim")]
    [InlineData("EncoderDim")]
    [InlineData("DecoderDim")]
    public void SharedWidthSettingsResizeTheActualUnetAndIndependentCopies(string property)
    {
        var options = E3TTSPaperTests.TinyOptions();
        var originalChannels = (int[])options.BlockChannels.Clone();
        Assert.Equal(16, options.HiddenDim);
        typeof(E3TTSOptions).GetProperty(property)!.SetValue(options, 32);
        Assert.Equal(originalChannels.Select(width => width * 2), options.BlockChannels);
        var copy = new E3TTSOptions(options);
        Assert.Equal(32, copy.HiddenDim);
        Assert.NotSame(options.BlockChannels, copy.BlockChannels);
        using var model = new E3TTS<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), copy);
        var nativeChannels = model.WaveformUnet!.DownBlocks.Select(block => block.FirstConvolutions[0].GetOutputShape()[0]);
        Assert.Equal(options.BlockChannels, nativeChannels);
        ((AiDotNet.TextToSpeech.TtsModelOptions)copy).HiddenDim = 16;
        Assert.Equal(originalChannels, copy.BlockChannels);
        Assert.Equal(32, options.HiddenDim);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(15)]
    public void SharedWidthRejectsInvalidStageScalingWithoutChangingTheLayout(int width)
    {
        var options = E3TTSPaperTests.TinyOptions();
        var before = (int[])options.BlockChannels.Clone();
        Assert.ThrowsAny<ArgumentException>(() => options.HiddenDim = width);
        Assert.Equal(before, options.BlockChannels);
    }

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
