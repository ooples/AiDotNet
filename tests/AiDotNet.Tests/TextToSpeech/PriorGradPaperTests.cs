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
/// The PriorGrad vocoder follows its paper (Lee et al. 2022): the prior is N(0, Σ) with σ the mel spectrogram's frame
/// energy normalized to (0, 1] and clipped at 0.1, training and sampling draw their noise from it, and the loss is the
/// Mahalanobis distance under Σ.
/// </summary>
/// <remarks>
/// Before this change PriorGrad's synthesis was a hand-written loop with no network in it and no prior computed from
/// the spectrogram; training regressed a generic layer stack.
/// </remarks>
public class PriorGradPaperTests
{
    private static PriorGradOptions Options(bool fast = false) => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, UpsampleStrides = [4, 4],
        MelMinFrequency = 0, MelMaxFrequency = 2000,
        ResChannels = 8, NumResLayers = 4, DilationCycle = 2, NoiseSchedule = DiffWaveOptions.Linear(1e-4, 0.05, 20),
        InferenceNoiseSchedule = [1e-4, 0.01, 0.2], UseFastSampling = fast, CropFrames = 8,
    };

    // The constant Adam rate the probes measure learning at; built outside the model type, which carries the paper recipe.
    private static Probe CreateProbe(PriorGradOptions options)
        => new(options, new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private sealed class Probe : PriorGrad<double>
    {
        public Probe(PriorGradOptions options, AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
            : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: 8, outputSize: 16) { RandomSeed = 8 }, options, optimizer)
        {
        }

        public Tensor<double> Prior(Tensor<double> mel, int samples, Random random) => PriorNoise(mel, samples, random);

        public Tensor<double> Loss(Tensor<double> noise, Tensor<double> predicted, Tensor<double> mel) => NoiseLoss(noise, predicted, mel);
    }

    private static Tensor<double> Audio(double amplitude = 0.4, int length = 512)
    {
        var audio = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++) audio[i] = amplitude * Math.Sin(2 * Math.PI * 250 * i / 4000.0);
        return audio;
    }

    // A mel spectrogram [1, 8, 2] whose first frame has every band at ln(e²/8) (energy e) and second at the silence floor.
    private static Tensor<double> TwoFrames(double energy)
    {
        var mel = new Tensor<double>(new[] { 1, 8, 2 });
        for (int m = 0; m < 8; m++)
        {
            mel[0, m, 0] = Math.Log(energy * energy / 8);
            mel[0, m, 1] = Math.Log(1e-5);
        }
        return mel;
    }

    [Fact(Timeout = 60000)]
    public async Task PriorStd_IsTheNormalizedFrameEnergy_ClippedAtTheMinimum_AndRepeatedOverTheHop()
    {
        await Task.Yield();
        var model = CreateProbe(Options());
        double floor = Math.Sqrt(8 * 1e-5);
        var std = model.PriorStd(TwoFrames(2.0));
        Assert.Equal(new[] { 1, 1, 32 }, std.Shape.ToArray());
        double expected = (2.0 - floor) / (4.0 - floor);
        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(expected, std[0, 0, i], 9);
            Assert.Equal(0.1, std[0, 0, 16 + i], 9);
        }
        // Energies above the maximum map to 1, so σ stays in (0, 1].
        Assert.Equal(1.0, model.PriorStd(TwoFrames(9.0))[0, 0, 0], 9);
    }

    [Fact(Timeout = 60000)]
    public async Task PriorNoise_HasThePriorsStandardDeviationPerFrame()
    {
        await Task.Yield();
        var model = CreateProbe(Options());
        var mel = new Tensor<double>(new[] { 1, 8, 2 });
        var two = TwoFrames(2.0);
        for (int i = 0; i < mel.Length; i++) mel[i] = two[i];
        var random = new Random(4);
        double loud = 0, quiet = 0;
        const int draws = 400;
        for (int d = 0; d < draws; d++)
        {
            var noise = model.Prior(mel, 32, random);
            for (int i = 0; i < 16; i++)
            {
                loud += noise[0, 0, i] * noise[0, 0, i];
                quiet += noise[0, 0, 16 + i] * noise[0, 0, 16 + i];
            }
        }
        double expected = (2.0 - Math.Sqrt(8e-5)) / (4.0 - Math.Sqrt(8e-5));
        Assert.Equal(expected, Math.Sqrt(loud / (16 * draws)), 1);
        Assert.Equal(0.1, Math.Sqrt(quiet / (16 * draws)), 2);
    }

    [Fact(Timeout = 60000)]
    public async Task NoiseLoss_IsTheMahalanobisDistanceUnderThePrior()
    {
        await Task.Yield();
        var model = CreateProbe(Options());
        var mel = TwoFrames(2.0);
        var std = model.PriorStd(mel);
        // An error of exactly one standard deviation at every sample costs 1, however loud the frame.
        var loss = model.Loss(new Tensor<double>(new[] { 1, 1, 32 }), std, mel);
        Assert.Equal(1.0, loss[0], 9);
    }

    [Fact(Timeout = 120000)]
    public async Task Sampling_StartsFromThePrior_SoQuietFramesStayQuiet()
    {
        await Task.Yield();
        var model = CreateProbe(Options(fast: true));
        // The untrained network predicts no noise (its output projection is zero-initialized), so the samples are the
        // prior's noise carried through the reverse process: their spread follows σ frame by frame.
        var wave = model.MelToWaveform(TwoFrames(4.0));
        double loud = Math.Sqrt(Enumerable.Range(0, 16).Average(i => wave[i] * wave[i]));
        double quiet = Math.Sqrt(Enumerable.Range(16, 16).Average(i => wave[i] * wave[i]));
        Assert.True(quiet < loud / 3, $"The quiet frame was not quieter ({quiet} vs {loud}).");
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheMahalanobisNoiseError()
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

    [Fact(Timeout = 60000)]
    public async Task FitEnergyStatistics_TakesTheTrainingSetsLowestFrameEnergy()
    {
        await Task.Yield();
        var options = Options();
        var model = CreateProbe(options);
        model.FitEnergyStatistics(new[] { Audio(0.4), Audio(0.05) });
        var lowest = new[] { Audio(0.4), Audio(0.05) }
            .SelectMany(a => { var m = model.ComputeMel(a); return Enumerable.Range(0, m.Shape[2]).Select(f => Math.Sqrt(Enumerable.Range(0, 8).Sum(b => Math.Exp(m[0, b, f])))); })
            .Min();
        Assert.Equal(lowest, options.EnergyMin!.Value, 9);
        Assert.Equal(4.0, options.EnergyMax);
    }
}
