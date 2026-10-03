using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// FastSpeech trains and synthesizes as the paper specifies (Ren et al. 2019): an FFT phoneme encoder, a duration
/// predictor driving the length regulator, and an FFT mel decoder, trained on externally extracted durations.
/// </summary>
/// <remarks>
/// Before this change FastSpeech's synthesis computed durations from hidden-state magnitudes with a fixed formula,
/// clamped them to 15 frames, slowed speech by a default factor of 2.5, and threw on every call because its encoder
/// boundary pointed past the end of the layer stack.
/// </remarks>
public class FastSpeechPaperTests
{
    private static FastSpeechOptions SmallOptions() => new()
    {
        EncoderDim = 32,
        HiddenDim = 32,
        NumHeads = 2,
        NumEncoderLayers = 1,
        NumDecoderLayers = 1,
        FftFilterSize = 64,
        DurationPredictorFilterSize = 32,
        DurationPredictorDropout = 0.0,
        DropoutRate = 0.0,
    };

    private static FastSpeech<double> CreateModel(FastSpeechOptions? options = null)
    {
        var arch = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 16, outputSize: 80);
        // Plain Adam at 1e-3: the paper's Noam warmup barely moves the weights in a few dozen steps.
        return new FastSpeech<double>(arch, options ?? SmallOptions(),
            new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));
    }

    private static Tensor<double> Tokens(int count)
    {
        var tokens = new Tensor<double>(new[] { count });
        for (int i = 0; i < count; i++) tokens[i] = 10 + (i * 7) % 60;
        return tokens;
    }

    private static Tensor<double> Mel(int frames)
    {
        var mel = new Tensor<double>(new[] { frames, 80 });
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < 80; m++) mel[f, m] = Math.Sin(0.1 * f + 0.05 * m) - 2.0;
        return mel;
    }

    [Fact(Timeout = 120000)]
    public async Task Training_ReducesThePaperObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = new TtsTrainingSample<double>
        {
            Tokens = Tokens(8),
            Mel = Mel(24),
            Durations = new[] { 3, 3, 3, 3, 3, 3, 3, 3 },
        };

        double first = model.Train(sample);
        double last = first;
        for (int i = 0; i < 30; i++) last = model.Train(sample);

        Assert.True(double.IsFinite(first) && double.IsFinite(last), $"Loss went non-finite: {first} -> {last}.");
        Assert.True(last < first * 0.5, $"The FastSpeech objective did not halve: {first} -> {last} over 30 steps.");
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_IsRefusedForWantOfDurations()
    {
        await Task.Yield();
        var model = CreateModel();
        var ex = Assert.Throws<NotSupportedException>(() => model.Train(Tokens(4), Mel(8)));
        Assert.Contains("Durations", ex.Message);
    }

    [Fact(Timeout = 60000)]
    public async Task LengthRegulator_HasNoPitchOrEnergyBranch()
    {
        await Task.Yield();
        var model = CreateModel();
        Assert.False(model.VarianceAdaptor.UsePitch);
        Assert.False(model.VarianceAdaptor.UseEnergy);

        var adapted = model.VarianceAdaptor.Adapt(new Tensor<double>(new[] { 4, 32 }), targets: null);
        Assert.Null(adapted.PitchSpectrogram);
        Assert.Null(adapted.Energy);
    }

    [Fact(Timeout = 60000)]
    public async Task SpeedControl_ScalesThePredictedDurations()
    {
        await Task.Yield();
        var model = CreateModel();
        var hidden = new Tensor<double>(new[] { 6, 32 });
        for (int i = 0; i < hidden.Length; i++) hidden[i] = 0.3 * Math.Cos(i);

        // Train the duration predictor toward 4 frames per phoneme so the scaled predictions are non-trivial.
        var sample = new TtsTrainingSample<double> { Tokens = Tokens(6), Mel = Mel(24), Durations = Enumerable.Repeat(4, 6).ToArray() };
        for (int i = 0; i < 40; i++) model.Train(sample);

        int normal = model.VarianceAdaptor.Adapt(hidden, targets: null, durationScale: 1.0).Durations.Sum();
        int slow = model.VarianceAdaptor.Adapt(hidden, targets: null, durationScale: 2.0).Durations.Sum();
        Assert.True(normal > 0);
        Assert.InRange(slow, 2 * normal - 6, 2 * normal + 6); // per-phoneme rounding: at most 1 frame each
    }
}
