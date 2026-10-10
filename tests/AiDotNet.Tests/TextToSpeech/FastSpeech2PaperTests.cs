using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// FastSpeech 2 trains and synthesizes as the paper specifies (Ren et al. 2021): ground-truth duration, pitch and
/// energy drive the variance adaptor in training, its predictions drive it at inference.
/// </summary>
/// <remarks>
/// Before this change FastSpeech 2's synthesis path computed durations with <c>log(1 + |h| * 3)</c>, pitch as a fixed
/// sine and energy as <c>|h| * 0.05</c>, none of it trained, and threw on every call because its encoder boundary
/// pointed past the end of the layer stack.
/// </remarks>
public class FastSpeech2PaperTests
{
    private static FastSpeech2Options SmallOptions() => new()
    {
        EncoderDim = 32,
        HiddenDim = 32,
        NumHeads = 2,
        NumEncoderLayers = 1,
        NumDecoderLayers = 1,
        FftFilterSize = 64,
        VariancePredictorFilterSize = 32,
        VariancePredictorDropout = 0.0,
        DropoutRate = 0.0,
    };

    private static FastSpeech2<double> CreateModel(FastSpeech2Options? options = null,
        AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? optimizer = null)
    {
        var arch = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 16, outputSize: 80);
        return new FastSpeech2<double>(arch, options ?? SmallOptions(), optimizer);
    }

    /// <summary>A gliding harmonic voice with an unvoiced gap (the WORLD reference signal), 0.6 s at 22050 Hz.</summary>
    private static Tensor<double> Recording()
    {
        const int fs = 22050;
        const double seconds = 0.6;
        int n = (int)(fs * seconds);
        var x = new Tensor<double>(new[] { n });
        double phase = 0;
        for (int i = 0; i < n; i++)
        {
            double t = (double)i / fs;
            phase += 2 * Math.PI * (120.0 + 100.0 * t / seconds) / fs;
            double v = 0;
            if (!(0.55 * seconds <= t && t < 0.70 * seconds))
                for (int h = 1; h < 6; h++) v += 0.3 * Math.Sin(h * phase) / h;
            x[i] = v;
        }
        return x;
    }

    private static Tensor<double> Tokens(int count)
    {
        var tokens = new Tensor<double>(new[] { count });
        for (int i = 0; i < count; i++) tokens[i] = 10 + (i * 7) % 60;
        return tokens;
    }

    private static int[] SpreadDurations(int frames, int tokens)
        => Enumerable.Range(0, tokens).Select(i => frames / tokens + (i < frames % tokens ? 1 : 0)).ToArray();

    [Fact(Timeout = 120000)]
    public async Task TrainingOnARecording_ReducesThePaperObjective()
    {
        await Task.Yield();
        // Plain Adam at 1e-3: the paper's Noam schedule warms up over 4000 steps, so 30 steps at its rate barely move
        // the weights. This test is about the objective and its gradients, not the schedule. The energy branch is
        // ablated (the paper's own ablation): its target is the raw frame energy, up to ~115 here, whose MSE starts
        // near 1e4 and needs thousands of steps to fit at this rate, which would hide every other term.
        var options = SmallOptions();
        options.UseEnergyPredictor = false;
        var model = CreateModel(options, new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));
        var audio = Recording();
        int frames = new TacotronSpectrogram().FrameCount(audio.Length);
        var sample = new TtsTrainingSample<double>
        {
            Tokens = Tokens(12),
            Audio = audio,
            Durations = SpreadDurations(frames, 12),
        };

        double first = model.Train(sample);
        double last = first;
        for (int i = 0; i < 30; i++) last = model.Train(sample);

        Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(first) && AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(last), $"Loss went non-finite: {first} -> {last}.");
        Assert.True(last < first * 0.5, $"The FastSpeech 2 objective did not halve: {first} -> {last} over 30 steps.");
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_IsRefusedForWantOfDurations()
    {
        await Task.Yield();
        var model = CreateModel();
        var ex = Assert.Throws<NotSupportedException>(() => model.Train(Tokens(4), new Tensor<double>(new[] { 8, 80 })));
        Assert.Contains("Durations", ex.Message);

        var noDurations = new TtsTrainingSample<double> { Tokens = Tokens(4), Mel = new Tensor<double>(new[] { 8, 80 }) };
        Assert.Throws<ArgumentException>(() => model.Train(noDurations));
    }

    [Fact(Timeout = 60000)]
    public async Task LengthRegulator_RepeatsEachPhonemeForItsDuration()
    {
        await Task.Yield();
        var adaptor = new VarianceAdaptorLayer<double>(4, 8, dropoutRate: 0.0);
        var phonemes = new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++) phonemes[i, j] = 10 * i + j;

        var frames = adaptor.LengthRegulate(phonemes, new[] { 2, 0, 3 });

        Assert.Equal(new[] { 5, 4 }, frames.Shape.ToArray());
        int[] source = { 0, 0, 2, 2, 2 };
        for (int f = 0; f < 5; f++)
            for (int j = 0; j < 4; j++) Assert.Equal(10 * source[f] + j, frames[f, j]);
    }

    [Fact(Timeout = 60000)]
    public async Task Synthesis_ProducesAFiniteMelSpectrogram()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = model.Synthesize("hello world");

        Assert.Equal(2, mel.Rank);
        Assert.Equal(80, mel.Shape[1]);
        Assert.True(mel.Shape[0] > 0);
        for (int i = 0; i < mel.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(mel[i]), $"mel[{i}] = {mel[i]}");
    }

    [Fact(Timeout = 60000)]
    public async Task InferenceExpansion_LastsAsLongAsThePredictedDurations()
    {
        await Task.Yield();
        var adaptor = new VarianceAdaptorLayer<double>(8, 8, dropoutRate: 0.0);
        var hidden = new Tensor<double>(new[] { 5, 8 });
        for (int i = 0; i < hidden.Length; i++) hidden[i] = Math.Sin(i);

        var adapted = adaptor.Adapt(hidden, targets: null);

        Assert.Equal(5, adapted.Durations.Length);
        Assert.Equal(adapted.Durations.Sum(), adapted.Expanded.Shape[0]);
    }
}
