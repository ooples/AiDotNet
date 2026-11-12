using System;
using System.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Chatterbox's HiFT sine source against the release (<c>tools/reference-data/chatterbox_sinegen_reference.py</c>): S3Gen
/// keeps the original SineGen at 24 kHz, where CosyVoice 2 uses SineGen2. The released-checkpoint audit feeds HiFT a
/// recorded excitation, so it cannot see which source synthesis uses; this test pins both the selection and the arithmetic.
/// </summary>
public class ChatterboxSineGenReferenceTests
{
    [Fact]
    public void Chatterbox_UsesTheOriginalSineGenAt24kHz()
    {
        var f = WhisperSpeechT2SReferenceTests.Load("chatterbox_sinegen_reference.json");
        using var model = ChatterboxFixture.Create<double>();
        var vocoder = model.Vocoder!;
        Assert.Equal(24000, vocoder.Configuration.SampleRate);
        Assert.True(vocoder.Configuration.OriginalSineGen);

        var f0 = f["f0"]!.Select(v => (double)v).ToArray();
        int scale = (int)f["scale"]!, harmonics = (int)f["harmonics"]!, samples = f0.Length * scale;
        var reference = vocoder.Configuration with
        {
            Harmonics = harmonics - 1, UpsampleRates = new[] { 8, 5, 3 }, UpsampleKernels = new[] { 16, 11, 7 }, SourceKernels = new[] { 7, 7, 11 },
            SourceDilations = new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } }, HopSize = 4,
        };
        Assert.Equal(scale, reference.UpsampleScale);
        var generator = new AiDotNet.TextToSpeech.Vocoders.HiftGenerator<double>(AiDotNet.Tensors.Engines.AiDotNetEngine.Current,
            new System.Collections.Generic.List<AiDotNet.NeuralNetworks.Layers.LayerBase<double>>(), reference, new Random(0));
        var flat = f["noise"]!.Select(v => (double)v).ToArray();
        var noise = new double[harmonics, samples];
        for (int h = 0; h < harmonics; h++)
            for (int s = 0; s < samples; s++) noise[h, s] = flat[h * samples + s];
        var sines = generator.Harmonics(new Tensor<double>(new[] { f0.Length }, new Vector<double>(f0)), f["phases"]!.Select(v => (double)v).ToArray(), noise);
        var expected = f["sines"]!.Select(v => (double)v).ToArray();
        Assert.Equal(expected.Length, sines.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - sines[i]) <= 1e-9, $"sines[{i}]: {expected[i]:R} vs {sines[i]:R}");
    }
}
