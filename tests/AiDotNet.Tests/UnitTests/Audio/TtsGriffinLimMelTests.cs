using System;
using System.Reflection;
using AiDotNet.Audio.TextToSpeech;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>
/// TtsModel's Griffin-Lim fallback reconstructs audio from the acoustic model's mel spectrogram (#2393 review). It
/// handed the NumMels log-mel bins to Griffin-Lim as if they were the FftSize / 2 + 1 linear-magnitude bins, so the
/// waveform it produced did not have the spectrum it was asked for. The mel is now inverted through the filterbank first.
/// </summary>
public class TtsGriffinLimMelTests
{
    private const int SampleRate = 22050, NumMels = 80, FftSize = 1024, HopLength = 256;

    [Fact]
    public void GriffinLimFallback_ReproducesTheMelSpectrumItWasGiven()
    {
        var options = new TtsOptions
        {
            SampleRate = SampleRate, NumMels = NumMels, FftSize = FftSize, HopLength = HopLength, GriffinLimIterations = 32,
            HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, NumDecoderLayers = 1, MaxPhonemeLength = 16, Seed = 1,
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression, inputSize: 16, outputSize: NumMels);
        var model = new TtsModel<double>(architecture, options);

        // A 440 Hz tone with a 1 kHz partial: a mel target whose energy sits in a few known bands.
        var mels = new MelSpectrogram<double>(SampleRate, NumMels, FftSize, HopLength);
        var tone = new Tensor<double>([SampleRate / 2]);
        for (int i = 0; i < tone.Length; i++)
        {
            double t = (double)i / SampleRate;
            tone[i] = 0.6 * Math.Sin(2 * Math.PI * 440 * t) + 0.3 * Math.Sin(2 * Math.PI * 1000 * t);
        }
        var target = mels.Forward(tone);

        var synthesize = typeof(TtsModel<double>).GetMethod("GriffinLimSynthesize", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(synthesize);
        var audio = Assert.IsType<Tensor<double>>(synthesize?.Invoke(model, new object[] { target }));
        Assert.True(audio.Length > FftSize, "Griffin-Lim returned no usable audio.");

        // The reconstruction's own mel spectrum must follow the target's: mean per-frame Pearson correlation of the dB
        // values over the frames both have, so a constant offset in level does not count either way.
        var produced = mels.Forward(audio);
        int frames = Math.Min(target.Shape[0], produced.Shape[0]);
        double total = 0;
        for (int f = 0; f < frames; f++)
        {
            double ma = 0, mb = 0;
            for (int m = 0; m < NumMels; m++) { ma += target[f, m]; mb += produced[f, m]; }
            ma /= NumMels; mb /= NumMels;
            double cov = 0, va = 0, vb = 0;
            for (int m = 0; m < NumMels; m++)
            {
                double a = target[f, m] - ma, b = produced[f, m] - mb;
                cov += a * b; va += a * a; vb += b * b;
            }
            total += cov / Math.Sqrt(va * vb + 1e-12);
        }
        double correlation = total / frames;
        Assert.True(correlation > 0.8, $"Reconstructed mel spectrum correlates {correlation:F4} with the requested one.");
    }
}
