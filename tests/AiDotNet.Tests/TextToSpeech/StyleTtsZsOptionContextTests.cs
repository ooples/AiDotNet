using AiDotNet.TextToSpeech.StyleEmotion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class StyleTtsZsOptionContextTests
{
    [Fact]
    public void NativeTextFrontendClipsAtTheConfiguredLimit()
    {
        foreach (int limit in new[] { 4, 8 })
        {
            var options = StyleTtsZsFixture.Options();
            options.MaxTextLength = limit;
            using var model = StyleTtsZsFixture.Create<double>(options);
            Assert.Equal(limit, model.TextToTokens("hello world, this is a longer sentence").Length);
        }
    }

    [Fact]
    public void WaveformConditionsRepeatEachPhonemeDurationOverItsFrames()
    {
        Assert.Equal(new[] { 2.0, 2.0, 2.0, 2.0, 1.0, 1.0 },
            StyleTTSZS<double>.FrameDurationCondition(new[] { 0, 2, 0, 1 }, 6).ToArray());
        Assert.Equal(Enumerable.Repeat(3.0, 6),
            StyleTTSZS<double>.FrameDurationCondition(new[] { 3 }, 6).ToArray());
        Assert.Throws<ArgumentException>(() => StyleTTSZS<double>.FrameDurationCondition(new[] { 1, 1 }, 6));
        Assert.Throws<ArgumentOutOfRangeException>(() => StyleTTSZS<double>.FrameDurationCondition(new[] { 1, -1, 2 }, 6));
    }

    [Theory]
    [InlineData(16, 2, StyleTts2DecoderType.HiFiGan, 2, 4, 20, 5)]
    [InlineData(32, 4, StyleTts2DecoderType.HiFiGan, 2, 4, 20, 5)]
    [InlineData(16, 2, StyleTts2DecoderType.HiFiGan, 3, 2, 20, 5)]
    [InlineData(16, 2, StyleTts2DecoderType.IstftNet, 2, 4, 20, 5)]
    [InlineData(16, 2, StyleTts2DecoderType.IstftNet, 2, 4, 4, 1)]
    public void PairedWidthsAndDecoderGeometryRunTheNativeWaveform(int width, int heads,
        StyleTts2DecoderType decoder, int firstRate, int secondRate, int fft, int istftHop)
    {
        var options = StyleTtsZsFixture.Options();
        options.ModelDim = options.StyleDim = width;
        options.Heads = heads;
        options.DecoderType = decoder;
        options.UpsampleRates = new[] { firstRate, secondRate };
        options.UpsampleKernels = new[] { firstRate * 2, secondRate * 2 };
        options.IstftFftSize = fft;
        options.IstftHopSize = istftHop;
        options.HopSize = firstRate * secondRate * (decoder == StyleTts2DecoderType.IstftNet ? istftHop : 1);
        using var model = StyleTtsZsFixture.Create<double>(options);
        var text = new Tensor<double>(new[] { 1, width, 3 });
        for (int i = 0; i < text.Length; i++) text[i] = .1 * Math.Sin(i * .17);
        var pitch = new Tensor<double>(new[] { 6 });
        var energy = new Tensor<double>(new[] { 6 });
        for (int i = 0; i < 6; i++) { pitch[i] = 180; energy[i] = .3; }
        var style = new Tensor<double>(new[] { 1, width });
        var wave = model.DecoderNetwork.Forward(text, pitch, energy, style, false, new Random(7));
        Assert.Equal(6 * options.HopSize, wave.Length);
        Assert.All(wave, value => Assert.True(double.IsFinite(value)));
        Assert.Contains(wave, value => Math.Abs(value) > 1e-12);
    }

    [Theory]
    [InlineData("StyleDim")]
    [InlineData("ModelDim")]
    [InlineData("Heads")]
    [InlineData("HopSize")]
    [InlineData("UpsampleRates")]
    [InlineData("ExtraKernel")]
    [InlineData("UpsampleKernels")]
    public void UnpairedNativeGeometryIsRejectedBeforeAllocation(string option)
    {
        var options = StyleTtsZsFixture.Options();
        switch (option)
        {
            case "StyleDim": options.StyleDim = 32; break;
            case "ModelDim": options.ModelDim = 32; break;
            case "Heads": options.Heads = 0; break;
            case "HopSize": options.HopSize++; break;
            case "UpsampleRates": options.UpsampleRates = new[] { 4, 4 }; break;
            case "ExtraKernel": options.UpsampleKernels = new[] { 4, 8, 8 }; break;
            case "UpsampleKernels": options.UpsampleKernels = new[] { 5, 8 }; break;
            default: throw new ArgumentOutOfRangeException(nameof(option));
        }
        Assert.Throws<ArgumentException>(() => StyleTtsZsFixture.Create<double>(options));
    }
}
