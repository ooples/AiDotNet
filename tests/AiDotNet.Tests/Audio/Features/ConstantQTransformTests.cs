using System;
using System.Threading.Tasks;
using AiDotNet.Audio.Features;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.Features;

/// <summary>
/// Invariants of the Constant-Q Transform (Brown 1991): geometric bin centres, one bin per pitch, a
/// linear transform before the magnitude, and a calibrated magnitude.
/// </summary>
/// <remarks>
/// <para>
/// ConstantQTransform is a fixed signal transform: Train does nothing and Predict is Transform. No
/// generated model family describes it, so the scaffold generator excludes it and points here. A
/// construction smoke test was its only coverage.
/// </para>
/// <para>
/// The configuration is small enough to reason about exactly. At 8 kHz with fMin 200 Hz, 12 bins per
/// octave and 2 octaves, Q = 1 / (2^(1/12) - 1), about 16.8, so the longest window, at bin 0, is
/// ceil(Q * 8000 / 200) = 673 samples. A one-second signal therefore gives frame 0 a complete window in
/// every bin.
/// </para>
/// </remarks>
public class ConstantQTransformTests
{
    private const int SampleRate = 8000;
    private const double FMin = 200.0;
    private const int BinsPerOctave = 12;
    private const int Octaves = 2;
    private const int Hop = 128;
    private const int Samples = 8000;

    private static ConstantQTransform<double> Create()
        => new(sampleRate: SampleRate, fMin: FMin, binsPerOctave: BinsPerOctave, numOctaves: Octaves, hopLength: Hop);

    private static Tensor<double> Tone(double frequency, double amplitude)
    {
        var audio = new Tensor<double>(new[] { Samples });
        for (int i = 0; i < Samples; i++)
        {
            audio[i] = amplitude * Math.Sin(2.0 * Math.PI * frequency * i / SampleRate);
        }

        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Output_HasOneRowPerFrameAndOneColumnPerBin()
    {
        await Task.Yield();
        var cqt = Create();

        var spectrum = cqt.Transform(Tone(440.0, 1.0));

        // Frames advance by the hop until the shortest window -- the highest bin's -- no longer fits.
        int shortestWindow = (int)Math.Ceiling(cqt.QFactor * SampleRate / cqt.Frequencies[cqt.NumBins - 1]);
        int expectedFrames = Math.Max(1, (Samples - shortestWindow) / Hop + 1);
        Assert.Equal(BinsPerOctave * Octaves, cqt.NumBins);
        Assert.Equal(new[] { expectedFrames, cqt.NumBins }, spectrum.Shape.ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task BinCentres_AreGeometricFromFMin()
    {
        await Task.Yield();
        var cqt = Create();

        Assert.Equal(FMin, cqt.Frequencies[0], 9);
        for (int k = 0; k + BinsPerOctave < cqt.NumBins; k++)
        {
            // One octave up doubles the frequency: that is what constant Q means.
            Assert.Equal(2.0 * cqt.Frequencies[k], cqt.Frequencies[k + BinsPerOctave], 9);
        }
    }

    [Theory(Timeout = 60000)]
    [InlineData(0)]
    [InlineData(5)]
    [InlineData(12)]
    [InlineData(23)]
    public async Task ToneAtABinCentre_PeaksInThatBin(int bin)
    {
        await Task.Yield();
        var cqt = Create();

        var spectrum = cqt.Transform(Tone(cqt.Frequencies[bin], 1.0));

        int loudest = 0;
        for (int k = 1; k < cqt.NumBins; k++)
        {
            if (spectrum[0, k] > spectrum[0, loudest]) loudest = k;
        }

        Assert.Equal(bin, loudest);
    }

    [Theory(Timeout = 60000)]
    [InlineData(0)]
    [InlineData(12)]
    [InlineData(23)]
    public async Task ToneAtABinCentre_HasTheCalibratedMagnitude(int bin)
    {
        await Task.Yield();
        var cqt = Create();
        const double amplitude = 0.8;

        var spectrum = cqt.Transform(Tone(cqt.Frequencies[bin], amplitude));

        // Each kernel is a Hann-windowed complex exponential normalised by its length, so a sine of
        // amplitude A at the bin's own frequency gives |sum| = A * mean(window) / 2 = A / 4, up to the
        // negligible image of the negative frequency. A normalisation error would miss by a factor.
        Assert.InRange(spectrum[0, bin], 0.95 * amplitude / 4.0, 1.05 * amplitude / 4.0);
    }

    [Fact(Timeout = 60000)]
    public async Task ScalingTheSignal_ScalesTheMagnitudeByTheSameFactor()
    {
        await Task.Yield();
        var cqt = Create();
        var signal = Tone(311.0, 0.5);
        var scaled = Tone(311.0, 1.5);

        var baseline = cqt.Transform(signal);
        var tripled = cqt.Transform(scaled);

        // The transform is linear before the magnitude, and |c z| = |c| |z|.
        for (int i = 0; i < baseline.Length; i++)
        {
            Assert.Equal(3.0 * baseline[i], tripled[i], 9);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Silence_GivesZero()
    {
        await Task.Yield();
        var cqt = Create();

        var spectrum = cqt.Transform(new Tensor<double>(new[] { Samples }));

        for (int i = 0; i < spectrum.Length; i++)
        {
            Assert.Equal(0.0, spectrum[i]);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Transform_IsDeterministic()
    {
        await Task.Yield();
        var cqt = Create();
        var signal = Tone(523.25, 0.7);

        var first = cqt.Transform(signal);
        var second = cqt.Transform(signal);

        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(first[i], second[i]);
        }
    }
}
