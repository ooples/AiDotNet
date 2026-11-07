using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class WhisperSpeechOptionContextTests
{
    [Fact]
    public void OptionsExposeTheConsumedStageControls()
    {
        foreach (var retired in new[] { "MelChannels", "FftSize", "HiddenDim", "NumEncoderLayers",
            "NumDecoderLayers", "NumHeads", "VocabSize", "MaxMelLength", "DropoutRate", "LearningRate", "WeightDecay" })
            Assert.Null(typeof(WhisperSpeechOptions).GetProperty(retired));
        var options = new WhisperSpeechOptions();
        Assert.Equal(24000, options.SampleRate);
        Assert.Equal(320, options.HopSize);
        Assert.Equal(550, options.MaxTextLength);
        Assert.Equal(1.5e-3, options.T2SLearningRate);
        Assert.Equal(3e-3, options.S2ALearningRate);
        Assert.Equal(9e-4, options.VqLearningRate);
        Assert.Equal(16, options.WhisperHeads);
        Assert.Equal(16, options.VqHeads);
    }

    [Theory]
    [InlineData(24000, 2560)]
    [InlineData(16000, 1280)]
    public void PairedCodecRateAndHopRunNativeEncodingAndVocosDecoding(int rate, int hop)
    {
        var options = WhisperSpeechFixture.Options();
        options.SampleRate = options.Codec.SampleRate = rate;
        options.HopSize = hop;
        options.Codec.Ratios = new[] { 8, 5, 4, 4, hop / 640 };
        options.Codec.TargetBandwidthKbps = 4.0 * rate / hop * Math.Log(options.Codec.CodebookSize, 2) / 1000;
        using var model = WhisperSpeechFixture.Tts<double>(options);
        var audio = new Tensor<double>(new[] { 3 * hop });
        for (int i = 0; i < audio.Length; i++) audio[i] = .2 * Math.Sin(2 * Math.PI * 220 * i / rate);
        var codes = model.EncodeToTokens(audio);
        Assert.Equal(new[] { 4, 3 }, codes.Shape.ToArray());
        var wave = model.DecodeFromTokens(codes);
        Assert.Equal(3 * hop, wave.Length);
        Assert.All(wave, value => Assert.True(double.IsFinite(value)));
        Assert.Contains(wave, value => value != 0);
    }

    [Theory]
    [InlineData("SampleRate")]
    [InlineData("HopSize")]
    public void UnpairedCodecGeometryIsRejected(string option)
    {
        var options = WhisperSpeechFixture.Options();
        if (option == "SampleRate") options.SampleRate = 16000;
        else options.HopSize = 1280;
        Assert.Throws<ArgumentException>(() => WhisperSpeechFixture.Tts<double>(options));
    }
}
