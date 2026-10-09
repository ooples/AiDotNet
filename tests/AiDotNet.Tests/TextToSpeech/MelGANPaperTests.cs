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
/// MelGAN trains and generates as its paper specifies (Kumar et al. 2019): a transposed-convolution generator with
/// dilated residual stacks, three multi-scale window discriminators, the hinge loss with feature matching (λ = 10) and
/// no spectrogram loss.
/// </summary>
/// <remarks>
/// Before this change MelGAN built a HiFi-GAN generator and trained it by regression; it had no discriminators.
/// </remarks>
public class MelGANPaperTests
{
    private static MelGANOptions Options() => new()
    {
        Ngf = 4, UpsampleRates = [4, 4], DiscriminatorWidthDivisor = 16, MelChannels = 8, FftSize = 64, WindowSize = 64,
        HopSize = 16, SampleRate = 4000, SegmentSize = 512,
    };

    private static MelGAN<double> CreateModel(MelGANOptions? options = null) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 7 },
        options ?? Options(),
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 220 * i / 4000.0) + 0.2 * Math.Sin(2 * Math.PI * 610 * i / 4000.0);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Generator_UpsamplesEachFrameByTheProductOfItsRatios_IntoTanhRange()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = model.ComputeMel(Audio());
        var wave = model.MelToWaveform(mel);
        Assert.Equal(mel.Shape[2] * 16, wave.Length);
        for (int i = 0; i < wave.Length; i++) Assert.InRange(wave[i], -1.0, 1.0);
    }

    [Fact(Timeout = 60000)]
    public async Task InputFeatures_AreLog10Mel_FlooredAt1e5()
    {
        await Task.Yield();
        var model = CreateModel();
        var silence = model.ComputeMel(new Tensor<double>(new[] { 512 }));
        for (int i = 0; i < silence.Length; i++) Assert.Equal(-5.0, silence[i], 9);
    }

    [Fact(Timeout = 60000)]
    public async Task OddUpsamplingRatios_UseOutputPadding_AndStillGiveRatioTimesFrames()
    {
        await Task.Yield();
        var options = Options();
        options.UpsampleRates = [3, 5];
        options.HopSize = 15;
        var model = CreateModel(options);
        var mel = new Tensor<double>(new[] { 1, 8, 4 });
        Assert.Equal(4 * 15, model.MelToWaveform(mel).Length);
    }

    [Fact(Timeout = 300000)]
    public async Task AdversarialTraining_RunsBothSteps_AndChangesTheGeneratedAudio()
    {
        await Task.Yield();
        var model = CreateModel();
        var audio = Audio();
        var mel = model.ComputeMel(audio);
        var before = model.MelToWaveform(mel).ToVector().ToArray();
        var parameters = model.GetParameters().ToArray();
        for (int i = 0; i < 3; i++) model.Train(mel, audio);
        var after = model.MelToWaveform(mel).ToVector().ToArray();
        Assert.Contains(Enumerable.Range(0, before.Length), i => Math.Abs(before[i] - after[i]) > 1e-9);
        var trained = model.GetParameters().ToArray();
        Assert.Contains(Enumerable.Range(0, parameters.Length), i => parameters[i] != trained[i]);
        Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(Convert.ToDouble(model.GetLastLoss())));
    }
}
