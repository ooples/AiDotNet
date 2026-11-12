using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class CosyVoiceOptionConstraintTests
{
    [Fact]
    public void StandardSamplerOwnershipPreservesCopiesAndNativeConfiguration()
    {
        var options = SmallOptions(2);
        options.TopP = 0.6;
        options.TopK = 3;
        options.RepetitionWindow = 7;
        options.RepetitionRatio = 0.2;
        var copy = new CosyVoiceOptions(options);
        using var model = new CosyVoice<double>(Architecture(), copy);
        var configuration = model.LanguageModel!.Configuration;
        Assert.Equal(options.TextEncoderDim, configuration.TextEncoderInputSize);
        Assert.Equal(options.NumEncoderLayers, copy.NumEncoderLayers);
        Assert.Equal(options.TopP, configuration.TopP);
        Assert.Equal(options.TopK, configuration.TopK);
        Assert.Equal(options.RepetitionWindow, configuration.RepetitionWindow);
        Assert.Equal(options.RepetitionRatio, configuration.RepetitionRatio);
        var second = new CosyVoice2Options { TopP = 0.7, TopK = 4, RepetitionWindow = 8, RepetitionRatio = 0.3 };
        var third = new CosyVoice3Options { TopP = 0.9, TopK = 5, RepetitionWindow = 9, RepetitionRatio = 0.4 };
        AssertSamplerCopy(second, new CosyVoice2Options(second));
        var thirdCopy = new CosyVoice3Options(third);
        Assert.Equal(third.TopP, thirdCopy.TopP);
        Assert.Equal(third.TopK, thirdCopy.TopK);
        Assert.Equal(third.RepetitionWindow, thirdCopy.RepetitionWindow);
        Assert.Equal(third.RepetitionRatio, thirdCopy.RepetitionRatio);
        foreach (string name in new[] { "TopP", "TopK", "RepetitionWindow", "RepetitionRatio" })
            Assert.Null(typeof(AiDotNet.TextToSpeech.VoiceCloning.ChatterboxOptions).GetProperty(name));
        foreach (string name in new[] { "TextEncoderDim", "NumEncoderLayers", "HiddenDim", "NumDecoderLayers",
            "MaxMelLength", "MaxCodecFrames", "LanguageModelName", "ModelPath" })
            Assert.Null(typeof(AiDotNet.TextToSpeech.VoiceCloning.ChatterboxOptions).GetProperty(name));
    }

    [Theory]
    [InlineData("TextEncoderDim")]
    [InlineData("NumEncoderLayers")]
    public void V1RetainsItsConsumedTextEncoderControls(string option)
    {
        var options = SmallOptions(2);
        using var control = new CosyVoice<double>(Architecture(), options);
        var changedOptions = new CosyVoiceOptions(options);
        if (option == "TextEncoderDim") changedOptions.TextEncoderDim = 12;
        else changedOptions.NumEncoderLayers = 2;
        using var changed = new CosyVoice<double>(Architecture(), changedOptions);
        Assert.NotEqual(control.GetParameters().Length, changed.GetParameters().Length);
    }

    private static void AssertSamplerCopy(CosyVoiceAutoregressiveOptions source, CosyVoiceAutoregressiveOptions copy)
    {
        Assert.Equal(source.TopP, copy.TopP);
        Assert.Equal(source.TopK, copy.TopK);
        Assert.Equal(source.RepetitionWindow, copy.RepetitionWindow);
        Assert.Equal(source.RepetitionRatio, copy.RepetitionRatio);
    }

    [Theory]
    [InlineData(4, 8)]
    [InlineData(8, 4)]
    public void HopSize_RejectsAnUnpairedUpsamplingChange(int rate, int hop)
    {
        var options = SmallOptions(rate);
        options.HopSize = hop;
        var error = Assert.Throws<ArgumentException>(() => new CosyVoice<double>(Architecture(), options));
        Assert.Contains("samples per mel frame", error.Message);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void PairedHopAndUpsampling_ChangeTheActualWaveformLength(int rate)
    {
        var options = SmallOptions(rate);
        using var model = new CosyVoice<double>(Architecture(), options);
        model.SetTrainingMode(false);
        var mel = new Tensor<double>(new[] { 1, options.MelChannels, 4 });
        var waveform = model.Vocoder!.Forward(mel, new Random(1337));
        Assert.Equal(4 * options.HopSize, waveform.Length);
    }

    private static NeuralNetworkArchitecture<double> Architecture() => new(inputFeatures: 1, outputSize: 4);

    private static CosyVoiceOptions SmallOptions(int rate) => new()
    {
        TextEncoderDim = 8, LLMDim = 8, NumEncoderLayers = 1, NumLLMLayers = 1,
        NumHeads = 2, LmFeedForwardDim = 16, CodebookSize = 8,
        FlowDim = 8, FlowHeads = 2, FlowFeedForwardDim = 16, FlowEncoderLayers = 1,
        DecoderChannels = new[] { 8 }, DecoderHeads = 2, DecoderHeadDim = 4,
        DecoderBlocks = 1, DecoderMidBlocks = 1, FlowTimesteps = 2,
        HiftBaseChannels = 8, HiftHarmonics = 1, HiftUpsampleRates = new[] { rate },
        HiftUpsampleKernels = new[] { 2 * rate }, HiftSourceKernels = new[] { 3 }, HopSize = 4 * rate,
        SpeechTokenizerState = 8, SpeechTokenizerHeads = 2, SpeechTokenizerLayers = 1,
        SpeakerEncoderChannels = 8, SpeakerEncoderGrowthRate = 2, SpeakerEmbeddingDim = 4,
        MelChannels = 4, DropoutRate = 0, IncludeDiscriminators = false
    };

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(8)]
    public void SpeechTokenStream_RejectsRvqCodebookCounts(int codebooks)
    {
        var exception = Assert.Throws<ArgumentException>(() => new CosyVoice<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: 1, outputSize: 80),
            new CosyVoiceOptions { NumCodebooks = codebooks }));
        Assert.Equal("NumCodebooks", exception.ParamName);
    }
}
