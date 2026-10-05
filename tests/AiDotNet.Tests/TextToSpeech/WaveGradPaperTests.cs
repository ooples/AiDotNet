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
/// WaveGrad trains and samples as its paper specifies (Chen et al. 2021): a continuous noise level drawn
/// hierarchically between neighbouring levels of the schedule, the L1 noise loss, a FiLM-modulated UBlock/DBlock network
/// conditioned on the scaled level, and a reverse process that runs over any schedule with one trained model.
/// </summary>
/// <remarks>
/// Before this change WaveGrad's synthesis was a hand-written loop with no network in it, and training regressed a
/// generic layer stack; there was no diffusion process at all.
/// </remarks>
public class WaveGradPaperTests
{
    private static WaveGradOptions Options(double[]? inference = null) => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, MelMaxFrequency = 2000,
        UpsampleFactors = [2, 2, 2, 2], UpsampleChannels = [8, 8, 8, 8],
        UpsampleDilations = [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 1, 2], [1, 2, 1, 2]],
        MelProjectionChannels = 16, WaveformChannels = 4, CropFrames = 8,
        InferenceNoiseSchedule = inference ?? [1e-4, 1e-2, 0.2],
    };

    // The constant Adam rate the probes measure learning at; built outside the model type, which carries the paper recipe.
    private static Probe CreateProbe(WaveGradOptions options)
        => new(options, new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private sealed class Probe : WaveGrad<double>
    {
        public Probe(WaveGradOptions options, AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
            : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: 8, outputSize: 16) { RandomSeed = 8 }, options, optimizer)
        {
        }

        public (double Level, double SqrtAlphaBar) Draw(Random random) => DrawTrainingLevel(random);

        public Tensor<double> Epsilon(Tensor<double> noisy, double level, Tensor<double> mel) => Denoise(noisy, level, mel);
    }

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 250 * i / 4000.0);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task OrthogonalInitialization_HasOrthonormalRowsOrColumns()
    {
        await Task.Yield();
        var random = new Random(3);
        foreach (var (rows, cols) in new[] { (4, 12), (12, 4), (6, 6) })
        {
            var w = WaveGradNetwork<double>.OrthogonalMatrix(random, rows, cols);
            bool byRows = rows <= cols;
            int vectors = byRows ? rows : cols, length = byRows ? cols : rows;
            double At(int v, int i) => byRows ? w[v * cols + i] : w[i * cols + v];
            for (int a = 0; a < vectors; a++)
                for (int b = 0; b < vectors; b++)
                {
                    double dot = 0;
                    for (int i = 0; i < length; i++) dot += At(a, i) * At(b, i);
                    Assert.Equal(a == b ? 1.0 : 0.0, dot, 9);
                }
        }
    }

    [Fact(Timeout = 60000)]
    public async Task TrainingLevels_AreDrawnBetweenNeighbouringScheduleLevels()
    {
        await Task.Yield();
        var model = CreateProbe(Options());
        var betas = WaveGradOptions.Linear(1e-6, 0.01, 1000);
        double product = 1;
        foreach (var b in betas) product *= 1 - b;
        double lowest = Math.Sqrt(product);
        var random = new Random(5);
        var draws = Enumerable.Range(0, 4000).Select(_ => model.Draw(random)).ToArray();
        foreach (var (level, sqrtAlphaBar) in draws)
        {
            Assert.Equal(level, sqrtAlphaBar);
            Assert.InRange(level, lowest, 1.0);
        }
        // The segment is uniform over 1..S, so the levels follow the schedule's spacing, not U(0, 1): half the draws
        // fall at or above the level halfway through the schedule.
        double prefix = 1;
        for (int s = 0; s < 500; s++) prefix *= 1 - betas[s];
        double fraction = draws.Count(d => d.Level >= Math.Sqrt(prefix)) / (double)draws.Length;
        Assert.InRange(fraction, 0.46, 0.54);
    }

    [Fact(Timeout = 60000)]
    public async Task NoiseLevel_ConditionsThePrediction()
    {
        await Task.Yield();
        var model = CreateProbe(Options());
        var mel = model.ComputeMel(Audio());
        var slice = new Tensor<double>(new[] { 1, 8, 4 });
        for (int c = 0; c < 8; c++)
            for (int f = 0; f < 4; f++) slice[0, c, f] = mel[0, c, f];
        var noisy = new Tensor<double>(new[] { 1, 1, 64 });
        var random = new Random(2);
        for (int i = 0; i < 64; i++) noisy[0, 0, i] = random.NextDouble() - 0.5;
        var a = model.Epsilon(noisy, 0.5, slice);
        var b = model.Epsilon(noisy, 0.5001, slice);
        Assert.Equal(new[] { 1, 1, 64 }, a.Shape.ToArray());
        // C = 5000 makes a 1e-4 change of √ᾱ a half-radian change of the fastest encoding frequency.
        double difference = Enumerable.Range(0, 64).Sum(i => Math.Abs(a[0, 0, i] - b[0, 0, i]));
        Assert.True(difference > 1e-6, $"The prediction ignored the noise level ({difference}).");
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheL1NoiseError()
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
    public async Task Sampling_RunsAnyScheduleFromOneModel_Clamped_AndRepeatable()
    {
        await Task.Yield();
        var six = CreateProbe(Options([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]));
        var mel = six.ComputeMel(Audio());
        var first = six.MelToWaveform(mel);
        Assert.Equal(mel.Shape[2] * 16, first.Length);
        for (int i = 0; i < first.Length; i++) Assert.InRange(first[i], -1.0, 1.0);
        Assert.Equal(first.ToVector().ToArray(), six.MelToWaveform(mel).ToVector().ToArray());
        var three = CreateProbe(Options([1e-4, 1e-2, 0.2]));
        Assert.Equal(first.Length, three.MelToWaveform(mel).Length);
    }
}
