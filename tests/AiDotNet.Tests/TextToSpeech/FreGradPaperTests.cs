using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// FreGrad follows its paper (Nguyen et al. 2024): diffusion over the two Haar sub-bands, a separate energy prior per
/// band from the lower and upper mel bands, the zero-terminal-SNR schedule (Eq. 8), the loss Σ_{l,h} L_diff + λ L_mag
/// (Eq. 10) and a network of frequency-aware dilated convolutions.
/// </summary>
/// <remarks>
/// Before this change FreGrad's synthesis was a hand-written loop with no network in it and no wavelet transform;
/// training regressed a generic layer stack.
/// </remarks>
public class FreGradPaperTests
{
    private static FreGradOptions Options(double magnitudeWeight = 0.1) => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, UpsampleStrides = [4, 2],
        MelMinFrequency = 0, MelMaxFrequency = 2000,
        ResChannels = 4, NumResLayers = 4, DilationCycle = 2,
        NoiseSchedule = FreGradOptions.ZeroTerminalSnr(DiffWaveOptions.Linear(1e-4, 0.05, 20), 1e-4), CropFrames = 8,
        StftFftSizes = [32, 64], StftHopSizes = [8, 16], StftWindowSizes = [16, 32], MagnitudeLossWeight = magnitudeWeight,
    };

    // The constant Adam rate the probes measure learning at; built outside the model type, which carries the paper recipe.
    private static Probe CreateProbe(FreGradOptions options)
        => new(options, new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private sealed class Probe : FreGrad<double>
    {
        public Probe(FreGradOptions options, AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
            : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: 8, outputSize: 16) { RandomSeed = 8 }, options, optimizer)
        {
        }

        public Tensor<double> Bands(Tensor<double> waveform) => ToDiffusionSpace(waveform);

        public Tensor<double> Waveform(Tensor<double> bands) => FromDiffusionSpace(bands);

        public Tensor<double> Loss(Tensor<double> noise, Tensor<double> predicted, Tensor<double> mel) => NoiseLoss(noise, predicted, mel);
    }

    private static Tensor<double> Audio(int length = 512)
    {
        var audio = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 250 * i / 4000.0) + 0.1 * Math.Sin(2 * Math.PI * 1700 * i / 4000.0);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task HaarSubBands_HalveTheLength_AndReconstructTheWaveformExactly()
    {
        await Task.Yield();
        var model = CreateProbe(Options());
        var wave = new Tensor<double>(new[] { 1, 1, 64 });
        for (int i = 0; i < 64; i++) wave[0, 0, i] = Math.Sin(i * 0.7) + (i % 3) * 0.1;
        var bands = model.Bands(wave);
        Assert.Equal(new[] { 1, 2, 32 }, bands.Shape.ToArray());
        for (int i = 0; i < 32; i++)
        {
            Assert.Equal((wave[0, 0, 2 * i] + wave[0, 0, 2 * i + 1]) / Math.Sqrt(2), bands[0, 0, i], 12);
            Assert.Equal((wave[0, 0, 2 * i] - wave[0, 0, 2 * i + 1]) / Math.Sqrt(2), bands[0, 1, i], 12);
        }
        var back = model.Waveform(bands);
        for (int i = 0; i < 64; i++) Assert.Equal(wave[0, 0, i], back[0, 0, i], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task ZeroTerminalSnr_EndsAtNearlyPureNoise_AndKeepsTheFirstStep()
    {
        await Task.Yield();
        var betas = DiffWaveOptions.Linear(1e-4, 0.05, 50);
        var shifted = FreGradOptions.ZeroTerminalSnr(betas, 1e-4);
        double before = 1, after = 1;
        var rootBefore = new double[50];
        var rootAfter = new double[50];
        for (int i = 0; i < 50; i++)
        {
            before *= 1 - betas[i];
            after *= 1 - shifted[i];
            rootBefore[i] = Math.Sqrt(before);
            rootAfter[i] = Math.Sqrt(after);
        }
        Assert.Equal(betas[0], shifted[0], 12);
        // Eq. 8: √ᾱ' = √ᾱ_1 (√ᾱ − √ᾱ_T + γ) / (√ᾱ_1 − √ᾱ_T + γ); at the last step √ᾱ' = √ᾱ_1 γ / (√ᾱ_1 − √ᾱ_T + γ).
        double scale = rootBefore[0] / (rootBefore[0] - rootBefore[49] + 1e-4);
        for (int i = 0; i < 50; i++) Assert.Equal((rootBefore[i] - rootBefore[49] + 1e-4) * scale, rootAfter[i], 9);
        // √ᾱ_T' = √ᾱ_1 γ / (√ᾱ_1 − √ᾱ_T + γ) ≈ 2.1e-4 here: (nearly) zero against the unshifted 0.53.
        Assert.True(rootAfter[49] < 5e-4, $"The last step keeps signal (√ᾱ_T = {rootAfter[49]}).");
        Assert.True(rootBefore[49] > 0.2, "The unshifted schedule should keep signal at its last step.");
    }

    [Fact(Timeout = 60000)]
    public async Task Priors_ComeFromTheLowerAndUpperMelBandsSeparately()
    {
        await Task.Yield();
        var model = CreateProbe(Options());
        var mel = new Tensor<double>(new[] { 1, 8, 2 });
        for (int m = 0; m < 8; m++)
            for (int f = 0; f < 2; f++) mel[0, m, f] = m < 4 ? Math.Log(1.0) : Math.Log(1e-5);
        var std = model.PriorStd(mel);
        Assert.Equal(new[] { 1, 2, 16 }, std.Shape.ToArray());
        double floor = Math.Sqrt(8e-5);
        for (int i = 0; i < 16; i++)
        {
            Assert.Equal((2.0 - floor) / (4.0 - floor), std[0, 0, i], 9);                                 // √4 = 2 in the low half
            Assert.Equal(0.1, std[0, 1, i], 9);                                                            // silence in the high half
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Loss_AddsTheWeightedStftMagnitudeTerm_AndIsZeroForAPerfectPrediction()
    {
        await Task.Yield();
        var mel = CreateProbe(Options()).ComputeMel(Audio(128));
        int n = mel.Shape[2] * 8;
        var random = new Random(6);
        var noise = new Tensor<double>(new[] { 1, 2, n });
        var predicted = new Tensor<double>(new[] { 1, 2, n });
        for (int i = 0; i < noise.Length; i++)
        {
            noise[i] = random.NextDouble() - 0.5;
            predicted[i] = noise[i] + 0.3 * (random.NextDouble() - 0.5);
        }
        double perfect = CreateProbe(Options()).Loss(noise, noise, mel)[0];
        double without = CreateProbe(Options(0.0)).Loss(noise, predicted, mel)[0];
        double with = CreateProbe(Options(0.1)).Loss(noise, predicted, mel)[0];
        Assert.Equal(0.0, perfect, 9);
        Assert.True(with > without, $"The magnitude term added nothing ({without} vs {with}).");
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheNoiseLoss()
    {
        await Task.Yield();
        var model = CreateProbe(Options());
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Audio();
        var full = model.ComputeMel(audio);
        var mel = new Tensor<double>(new[] { 1, 8, 32 });
        for (int c = 0; c < 8; c++)
            for (int f = 0; f < 32; f++) mel[0, c, f] = full[0, c, f];
        double before = provider.EvaluateTrainingObjective(mel, audio);
        for (int i = 0; i < 30; i++) model.Train(mel, audio);
        double after = provider.EvaluateTrainingObjective(mel, audio);
        Assert.True(after < before, $"The noise loss did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Sampling_ReturnsTheInverseTransformOfBothBands_Repeatably()
    {
        await Task.Yield();
        var model = CreateProbe(Options());
        var mel = model.ComputeMel(Audio());
        var first = model.MelToWaveform(mel);
        Assert.Equal(mel.Shape[2] * 16, first.Length);
        // Each band is clamped to [−1, 1], so a reconstructed sample is at most √2 in magnitude.
        for (int i = 0; i < first.Length; i++) Assert.InRange(first[i], -Math.Sqrt(2) - 1e-9, Math.Sqrt(2) + 1e-9);
        Assert.Equal(first.ToVector().ToArray(), model.MelToWaveform(mel).ToVector().ToArray());
    }
}
