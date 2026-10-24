using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// APNet2 follows its paper (Du et al. 2023): ConvNeXt v2 amplitude and phase predictors with global response
/// normalization, the linear anti-wrapping phase error, the centred inverse STFT and hinge adversarial training against
/// the MPD and the MRD.
/// </summary>
/// <remarks>
/// Before this change APNet2 emitted its three spectra as the model output and regressed them; it never reconstructed a
/// waveform and had no discriminators.
/// </remarks>
public class APNet2PaperTests
{
    private static APNet2Options Options() => new()
    {
        MelChannels = 8, FftSize = 64, HopSize = 16, WindowSize = 64, SampleRate = 4000, MelMaxFrequency = 2000,
        ConvNeXtChannels = 8, ConvNeXtIntermediateChannels = 16, NumConvNeXtBlocks = 2, DiscriminatorPeriods = [2, 3],
        DiscriminatorWidthDivisor = 32, ResolutionFftSizes = [64, 128], ResolutionHopSizes = [16, 32], ResolutionWindowSizes = [64, 128],
        SegmentSize = 512,
    };

    private sealed class Probe : APNet2<double>
    {
        public Probe(APNet2Options options, AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
            : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: 8, outputSize: 16) { RandomSeed = 6 }, options, optimizer)
        {
        }

        public Tensor<double> AntiWrapping(Tensor<double> difference) => PhaseError(difference);
    }

    // The constant Adam rate the probes measure learning at; built outside the model type, which carries the paper recipe.
    private static Probe CreateProbe() => new(Options(), new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 375 * i / 4000.0) + 0.1 * Math.Sin(2 * Math.PI * 1250 * i / 4000.0);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task AntiWrappingError_IsTheDistanceToTheNearestWholeTurn()
    {
        await Task.Yield();
        var model = CreateProbe();
        var d = new Tensor<double>(new[] { 5 });
        double[] x = [0.3, 2 * Math.PI + 0.3, -3.5, 3.0, 4 * Math.PI - 0.2];
        for (int i = 0; i < 5; i++) d[i] = x[i];
        var e = model.AntiWrapping(d);
        double[] expected = [0.3, 0.3, 2 * Math.PI - 3.5, 3.0, 0.2];
        for (int i = 0; i < 5; i++) Assert.Equal(expected[i], e[i], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task GlobalResponseNorm_StartsAsTheIdentity()
    {
        await Task.Yield();
        var grn = new GlobalResponseNormLayer<double>(4);
        var x = new Tensor<double>(new[] { 1, 4, 6 });
        var random = new Random(2);
        for (int i = 0; i < x.Length; i++) x[i] = random.NextDouble() - 0.5;
        var y = grn.Forward(x);
        for (int i = 0; i < x.Length; i++) Assert.Equal(x[i], y[i], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task CentredFeatures_RoundTripToTheRecordingsLength_WithAWrappedPhase()
    {
        await Task.Yield();
        var model = CreateProbe();
        var mel = model.ComputeMel(Audio());
        Assert.Equal(512 / 16 + 1, mel.Shape[2]);
        Assert.Equal(512, model.MelToWaveform(mel).Length);
        var (_, phase) = model.PredictSpectra(mel);
        for (int i = 0; i < phase.Length; i++) Assert.InRange(phase[i], -Math.PI - 1e-12, Math.PI + 1e-12);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheMultilevelReconstructionLoss()
    {
        await Task.Yield();
        var model = CreateProbe();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Audio();
        var mel = model.ComputeMel(audio);
        double before = provider.EvaluateTrainingObjective(mel, audio);
        for (int i = 0; i < 15; i++) model.Train(mel, audio);
        double after = provider.EvaluateTrainingObjective(mel, audio);
        Assert.True(after < before, $"The amplitude, phase, spectrum and mel losses did not fall ({before} -> {after}).");
    }
}
