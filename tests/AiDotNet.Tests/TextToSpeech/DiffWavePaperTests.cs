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

    [Fact(Timeout = 30000)]
    public async Task GenerationPresets_AreThePapersSection52And53Settings()
    {
        await Task.Yield();
        var o = DiffWaveOptions.Unconditional();
        Assert.Equal(DiffWaveConditioner.Unconditional, o.Conditioner);
        Assert.Equal((16000, 16000, 36, 256), (o.SampleRate, o.UtteranceSamples, o.NumResLayers, o.ResChannels));
        // Dilation cycle [1, 2, ..., 2048] is 12 doublings; T = 200 with beta linear from 1e-4 to 0.02.
        Assert.Equal(2048, 1 << (o.DilationCycle - 1));
        Assert.Equal(200, o.NoiseSchedule.Length);
        Assert.Equal(1e-4, o.NoiseSchedule[0], 15);
        Assert.Equal(0.02, o.NoiseSchedule[^1], 15);
        Assert.False(o.UseFastSampling);
        Assert.Equal(2e-4, o.LearningRate);
        var c = DiffWaveOptions.ClassConditional(10);
        Assert.Equal((DiffWaveConditioner.ClassLabel, 10, 36, 256), (c.Conditioner, c.NumClasses, c.NumResLayers, c.ResChannels));
        Assert.Throws<ArgumentOutOfRangeException>(() => DiffWaveOptions.ClassConditional(0));
    }

    private static DiffWave<double> CreateGenerator(DiffWaveConditioner conditioner) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 256) { RandomSeed = 8 },
        new DiffWaveOptions
        {
            Conditioner = conditioner, NumClasses = conditioner == DiffWaveConditioner.ClassLabel ? 3 : 0, UtteranceSamples = 256,
            SampleRate = 4000, ResChannels = 8, NumResLayers = 4, DilationCycle = 2, NoiseSchedule = DiffWaveOptions.Linear(1e-4, 0.02, 20),
        },
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Tone(double hz)
    {
        var audio = new Tensor<double>(new[] { 256 });
        for (int i = 0; i < 256; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * hz * i / 4000.0);
        return audio;
    }

    private static Tensor<double> Label(int label) => new(new[] { 1 }, new Vector<double>(new double[] { label }));

    [Fact(Timeout = 300000)]
    public async Task Unconditional_TrainsOnWholeUtterances_AndGeneratesTheUtteranceLength()
    {
        await Task.Yield();
        var model = CreateGenerator(DiffWaveConditioner.Unconditional);
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Tone(250);
        double before = provider.EvaluateTrainingObjective(Label(0), audio);
        for (int i = 0; i < 30; i++) model.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = audio });
        double after = provider.EvaluateTrainingObjective(Label(0), audio);
        Assert.True(after < before, $"The noise loss did not fall ({before} -> {after}).");
        var wave = model.Generate();
        Assert.Equal(256, wave.Length);
        for (int i = 0; i < wave.Length; i++) Assert.InRange(wave[i], -1.0, 1.0);
        Assert.Equal(wave.ToVector().ToArray(), model.Generate().ToVector().ToArray());
        Assert.Throws<ArgumentException>(() => model.Generate(1));
    }

    [Fact(Timeout = 300000)]
    public async Task ClassConditional_TheLabelSteersTheNoisePrediction()
    {
        await Task.Yield();
        var model = CreateGenerator(DiffWaveConditioner.ClassLabel);
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var low = Tone(250);
        var high = Tone(900);
        for (int i = 0; i < 60; i++)
        {
            model.Train(Label(0), low);
            model.Train(Label(1), high);
        }
        // Each recording is predicted better under its own label than under the other's.
        Assert.True(provider.EvaluateTrainingObjective(Label(0), low) < provider.EvaluateTrainingObjective(Label(1), low));
        Assert.True(provider.EvaluateTrainingObjective(Label(1), high) < provider.EvaluateTrainingObjective(Label(0), high));
        Assert.NotEqual(model.Generate(0).ToVector().ToArray(), model.Generate(1).ToVector().ToArray());
        Assert.Throws<ArgumentException>(() => model.Generate());
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Generate(3));
        Assert.Throws<ArgumentException>(() => model.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 1 }), Audio = low,
        }));
    }
}
