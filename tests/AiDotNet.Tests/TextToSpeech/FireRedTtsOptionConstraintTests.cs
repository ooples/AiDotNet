using System;
using System.Linq;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class FireRedTtsOptionConstraintTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ValidCodecStrideFactorizationsEncodeAndDecodeTheSameFrameGeometry(bool decomposed)
    {
        var options = FireRedTtsFixture.Options();
        options.CodecStrides = decomposed ? new[] { 1, 2 } : new[] { 2 };
        using var model = Create(options);
        var codes = model.MelCodec.Tokenize(Signal(8, options.MelChannels));
        Assert.Equal(4, codes.GetLength(0));
        Assert.Equal(options.CodecStreams, codes.GetLength(1));
        var mel = model.MelCodec.Detokenize(codes, Signal(options.SpeakerDim));
        Assert.Equal(new[] { 8, options.MelChannels }, mel.Shape.ToArray());
        Finite(mel);
    }

    private static FireRedTTS<double> Create(FireRedTTSOptions options) =>
        new(new NeuralNetworkArchitecture<double>(inputFeatures: 1, outputSize: 1) { RandomSeed = 1337 }, options);

    private static Tensor<double> Signal(params int[] shape)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = 0.2 * Math.Sin(i * 0.17);
        return tensor;
    }

    private static void Finite(Tensor<double> tensor) =>
        Assert.All(tensor, value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    public void PairedSpeakerTokenizerAndCodecWidthsRunBothDecoders(int width)
    {
        var options = FireRedTtsFixture.Options();
        options.SpeakerDim = options.TokenizerDim = options.CodecDim = width;
        using var model = Create(options);
        var speaker = Signal(width);
        var (ssl, mel) = model.SpeechTokenizer.Reconstruct(Signal(2, width), speaker);
        Assert.Equal(options.SslDim, ssl.Shape[1]);
        Assert.Equal(80, mel.Shape[1]); // The tokenizer reconstructs the speaker frontend; the flow/codec use 100 mel bins.
        Finite(ssl); Finite(mel);
        var encoded = model.MelCodec.Encode(Signal(8, options.MelChannels));
        var decoded = model.MelCodec.Decode(encoded, speaker);
        Assert.Equal(new[] { 8, options.MelChannels }, decoded.Shape.ToArray());
        Finite(decoded);
    }

    [Theory]
    [InlineData(16)]
    [InlineData(32)]
    public void PairedHubertAndSslWidthsRunTheNativeFeatureEncoderAndTokenizer(int width)
    {
        var options = FireRedTtsFixture.Options();
        options.HubertDim = options.SslDim = width;
        using var model = Create(options);
        var features = model.SpeechTokenizer.SslFeatures(Signal(1280).ToArray());
        Assert.Equal(width, features.Shape[1]);
        var encoded = model.SpeechTokenizer.Encode(features);
        Assert.Equal(options.TokenizerDim, encoded.Shape[1]);
        Finite(features); Finite(encoded);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public void PairedCodebookAndAudioVocabulariesRunQuantizerAndGptHead(int size)
    {
        var options = FireRedTtsFixture.Options();
        options.CodebookSize = size;
        options.AudioTokens = size * size + 2;
        using var model = Create(options);
        var quantizer = model.SpeechTokenizer.Quantizer;
        var tokens = quantizer.Tokens(quantizer.Project(Signal(3, options.TokenizerDim)));
        Assert.Equal(3, tokens.Length);
        Assert.All(tokens, token => Assert.InRange(token, 0, size * size - 1));
        var logits = model.Gpt.AudioHead.Forward(Signal(3, options.HiddenDim));
        Assert.Equal(new[] { 3, options.AudioTokens }, logits.Shape.ToArray());
        Finite(logits);
    }

    [Theory]
    [InlineData(FireRedTtsVocoder.Release24k, 24000, 960, 480)]
    [InlineData(FireRedTtsVocoder.Paper48k, 48000, 1920, 960)]
    public void VocoderPresetsPairRateHopAndActualWaveform(FireRedTtsVocoder preset, int rate, int hop, int samples)
    {
        var options = FireRedTtsFixture.Options();
        options.Vocoder = preset;
        Assert.Equal(rate, options.SampleRate);
        Assert.Equal(hop, options.HopSize);
        using var model = Create(options);
        var waveform = model.Vocoder.Forward(Signal(1, options.MelChannels, 2));
        Assert.Equal(samples, waveform.Length);
        Finite(waveform);
    }

    [Theory]
    [InlineData("AudioTokens")]
    [InlineData("CodebookSize")]
    [InlineData("SpeakerDim")]
    [InlineData("TokenizerDim")]
    [InlineData("CodecDim")]
    [InlineData("SslDim")]
    [InlineData("HubertDim")]
    [InlineData("SampleRate")]
    [InlineData("HopSize")]
    public void UnpairedNativeWidthsAndVocabulariesAreRejected(string option)
    {
        var options = FireRedTtsFixture.Options();
        var property = typeof(FireRedTTSOptions).GetProperty(option)!;
        property.SetValue(options, (int)property.GetValue(options)! + 2);
        Assert.Throws<ArgumentException>(() => Create(options));
    }

    [Fact]
    public void InvalidCodecStrideProductIsRejected()
    {
        var options = FireRedTtsFixture.Options();
        options.CodecStrides = new[] { 2, 2 };
        Assert.Throws<ArgumentException>(() => Create(options));
    }
}
