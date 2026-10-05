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
/// APNet follows its paper (Ai and Ling 2023): frame-level amplitude and phase predictors whose spectra the centred
/// inverse STFT turns into the waveform, phase losses that ignore wrapping, the STFT consistency loss and HiFi-GAN's
/// adversarial training.
/// </summary>
/// <remarks>
/// Before this change APNet regressed a generic layer stack: no amplitude or phase spectra, no inverse STFT and none
/// of the paper's losses.
/// </remarks>
public class APNetPaperTests
{
    private static APNetOptions Options() => new()
    {
        MelChannels = 8, FftSize = 64, HopSize = 16, WindowSize = 32, SampleRate = 4000, MelMaxFrequency = 2000, Channels = 8,
        ResblockKernelSizes = [3, 5], ResblockDilationSizes = [[1, 3], [1, 3]], DiscriminatorPeriods = [2, 3],
        DiscriminatorWidthDivisor = 32, SegmentSize = 512,
    };

    private static APNet<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 6 },
        Options(),
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 375 * i / 4000.0) + 0.1 * Math.Sin(2 * Math.PI * 1250 * i / 4000.0);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task ComplexStft_MatchesTheDirectTransformOfAWindowedFrame()
    {
        await Task.Yield();
        var stft = new CenteredComplexStft<double>(AiDotNetEngine.Current, 64, 16, 32);
        var audio = Audio();
        var (re, im) = stft.Forward(audio);
        Assert.Equal(new[] { 1, 33, 33 }, re.Shape.ToArray());
        // Frame 10 is centred on sample 160: samples 160 − 32 + n under the 32-sample Hann window centred in 64 points.
        for (int k = 0; k < 33; k++)
        {
            double r = 0, i = 0;
            for (int n = 0; n < 32; n++)
            {
                double w = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / 32);
                double x = audio[160 - 32 + 16 + n] * w;
                double a = 2 * Math.PI * (16 + n) * k / 64;
                r += x * Math.Cos(a);
                i -= x * Math.Sin(a);
            }
            Assert.Equal(r, re[0, k, 10], 9);
            Assert.Equal(i, im[0, k, 10], 9);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task PhaseDifferences_FollowTheReferenceDifferenceMatrices()
    {
        await Task.Yield();
        var p = new Tensor<double>(new[] { 1, 3, 4 });
        for (int i = 0; i < p.Length; i++) p[i] = i * 0.7 - 2;
        var frequency = PhaseDifferences.Along(AiDotNetEngine.Current, p, 1);
        var time = PhaseDifferences.Along(AiDotNetEngine.Current, p, 2);
        for (int f = 0; f < 3; f++)
            for (int t = 0; t < 4; t++)
            {
                Assert.Equal((f > 0 ? p[0, f - 1, t] : 0) - p[0, f, t], frequency[0, f, t], 12);
                Assert.Equal((t > 0 ? p[0, f, t - 1] : 0) - p[0, f, t], time[0, f, t], 12);
            }
    }

    [Fact(Timeout = 60000)]
    public async Task Spectra_InvertToTheRecording_AndThePredictedPhaseIsWrapped()
    {
        await Task.Yield();
        var model = CreateModel();
        var audio = Audio();
        var mel = model.ComputeMel(audio);
        Assert.Equal(512 / 16 + 1, mel.Shape[2]);
        // The analysed amplitude and phase spectra resynthesize the recording through the inverse STFT.
        var (logAmplitude, phase, _, _) = model.AnalyzeSpectra(audio);
        var istft = new InverseStft<double>(AiDotNetEngine.Current, 64, 16, 32);
        var amplitude = new Tensor<double>(logAmplitude._shape);
        for (int i = 0; i < amplitude.Length; i++) amplitude[i] = Math.Exp(logAmplitude[i]) - 1e-5;
        var back = istft.Forward(amplitude, phase);
        Assert.Equal(512, back.Length);
        for (int i = 32; i < 480; i++) Assert.Equal(audio[i], back[i], 6);
        // The PSP's phase comes from the two-argument arctangent, so it is wrapped to (−π, π].
        var (_, predicted) = model.PredictSpectra(mel);
        for (int i = 0; i < predicted.Length; i++) Assert.InRange(predicted[i], -Math.PI - 1e-12, Math.PI + 1e-12);
        Assert.Equal(512, model.MelToWaveform(mel).Length);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheMultilevelReconstructionLoss()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Audio();
        var mel = model.ComputeMel(audio);
        double before = provider.EvaluateTrainingObjective(mel, audio);
        for (int i = 0; i < 15; i++) model.Train(mel, audio);
        double after = provider.EvaluateTrainingObjective(mel, audio);
        Assert.True(after < before, $"The amplitude, phase, spectrum and mel losses did not fall ({before} -> {after}).");
    }
}
