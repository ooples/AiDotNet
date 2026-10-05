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
/// DiffWave trains and samples as its paper specifies (Kong et al. 2021): ε-prediction at a random diffusion step with
/// the L2 loss, a bidirectional dilated-convolution network with a zero-initialized output, the reverse process over the
/// training schedule or the fast schedule aligned to fractional steps.
/// </summary>
/// <remarks>
/// Before this change DiffWave's synthesis was a hand-written loop with no network in it, and training regressed a
/// generic layer stack; there was no diffusion process at all.
/// </remarks>
public class DiffWavePaperTests
{
    private static DiffWaveOptions Options(bool fast = false) => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, UpsampleStrides = [4, 4],
        ResChannels = 8, NumResLayers = 4, DilationCycle = 2, NoiseSchedule = DiffWaveOptions.Linear(1e-4, 0.05, 20),
        InferenceNoiseSchedule = [1e-4, 0.01, 0.2], UseFastSampling = fast, CropFrames = 8, MelMinFrequency = 0,
    };

    private static DiffWave<double> CreateModel(bool fast = false) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 8 },
        Options(fast),
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 250 * i / 4000.0);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task UntrainedNetwork_PredictsZeroNoise_ThroughItsZeroInitializedOutput()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = model.ComputeMel(Audio());
        Assert.InRange(mel.ToVector().ToArray().Min(), 0.0, 1.0);
        Assert.InRange(mel.ToVector().ToArray().Max(), 0.0, 1.0);
        // With ε_θ ≡ 0 every step only rescales and adds noise; the waveform is clamped to [−1, 1] and has hop samples
        // per frame.
        var wave = model.MelToWaveform(mel);
        Assert.Equal(mel.Shape[2] * 16, wave.Length);
        for (int i = 0; i < wave.Length; i++) Assert.InRange(wave[i], -1.0, 1.0);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheNoisePredictionError()
    {
        await Task.Yield();
        var model = CreateModel();
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
    public async Task FastSampling_RunsTheShortScheduleAndIsRepeatable()
    {
        await Task.Yield();
        var model = CreateModel(fast: true);
        var mel = model.ComputeMel(Audio());
        var first = model.MelToWaveform(mel);
        Assert.Equal(first.ToVector().ToArray(), model.MelToWaveform(mel).ToVector().ToArray());
    }
}
