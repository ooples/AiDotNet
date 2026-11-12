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
/// iSTFTNet generates and trains as its paper specifies (Kaneko et al. 2022): HiFi-GAN's generator up to two ×8
/// upsamplings, an exponential magnitude and sine phase head, iSTFT(16, 4, 16), and HiFi-GAN's adversarial, feature
/// matching and mel losses with Adam (0.5, 0.9).
/// </summary>
/// <remarks>
/// Before this change iSTFTNet emitted a 513-channel spectrogram and had no inverse STFT, discriminators or adversarial
/// training.
/// </remarks>
public class ISTFTNetPaperTests
{
    private static ISTFTNetOptions Options() => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, MelMaxFrequency = 2000,
        UpsampleRates = [2, 2], UpsampleKernelSizes = [4, 4], UpsampleInitialChannels = 16, ResblockKernelSizes = [3],
        ResblockDilationSizes = [[1, 3]], DiscriminatorPeriods = [2, 3], DiscriminatorWidthDivisor = 32, SegmentSize = 512,
    };

    private static ISTFTNet<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 9 },
        Options(),
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    [Fact(Timeout = 60000)]
    public async Task InverseStft_TurnsAConstantBinIntoItsCosine()
    {
        await Task.Yield();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var istft = new InverseStft<double>(engine, 16, 4, 16);
        int frames = 12;
        var magnitude = new Tensor<double>(new[] { 1, 9, frames });
        var phase = new Tensor<double>(new[] { 1, 9, frames });
        // Bin 4 advances by 2π·4·4/16 = 2π per hop, so a constant zero phase gives every frame the tone of amplitude
        // 2·8/16 = 1. torch.istft windows each frame, overlap-adds and divides by the summed squared window, so the
        // steady tone comes out scaled by Σw / Σw² = 2 / 1.5 = 4/3 for a Hann window at a quarter-window hop.
        for (int f = 0; f < frames; f++) magnitude[0, 4, f] = 8.0;
        var wave = istft.Forward(magnitude, phase);
        Assert.Equal((frames - 1) * 4, wave.Length);
        for (int t = 8; t < wave.Length - 8; t++) Assert.Equal(4.0 / 3 * Math.Cos(Math.PI * t / 2), wave[t], 9);
    }

    [Fact(Timeout = 60000)]
    public async Task Generator_GivesExactlyHopSamplesPerFrame()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = new Tensor<double>(new[] { 1, 8, 5 });
        for (int i = 0; i < mel.Length; i++) mel[i] = Math.Sin(0.7 * i);
        Assert.Equal(5 * 16, model.MelToWaveform(mel).Length);
    }

    [Fact(Timeout = 300000)]
    public async Task AdversarialTraining_ReducesTheMelReconstructionError()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 220 * i / 4000.0);
        var mel = model.ComputeMel(audio);
        double before = provider.EvaluateTrainingObjective(mel, audio);
        for (int i = 0; i < 20; i++) model.Train(mel, audio);
        double after = provider.EvaluateTrainingObjective(mel, audio);
        Assert.True(after < before, $"The mel loss did not fall ({before} -> {after}).");
    }
}
