using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// HiFi-GAN generates and trains as its paper specifies (Kong et al. 2020): transposed-convolution upsampling with
/// multi-receptive-field residual blocks to a tanh waveform, and alternating LSGAN updates against multi-period and
/// multi-scale discriminators with feature matching and a 45-weighted mel loss.
/// </summary>
/// <remarks>
/// Before this change HiFi-GAN was a stack of generic layers trained by regression on the waveform, with no
/// discriminator, no residual blocks and no adversarial or mel loss.
/// </remarks>
public class HiFiGANPaperTests
{
    private static HiFiGAN<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 64) { RandomSeed = 11 },
        new HiFiGANOptions
        {
            MelChannels = 8, UpsampleRates = [4, 4], UpsampleKernelSizes = [8, 8], UpsampleInitialChannels = 16,
            ResblockKernelSizes = [3], ResblockDilationSizes = [[1, 3]], DiscriminatorPeriods = [2, 3], HopSize = 16,
            FftSize = 64, WindowSize = 64, SampleRate = 4000, MelMaxFrequency = 2000, SegmentSize = 256,
            DiscriminatorWidthDivisor = 16,
        },
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.5 * Math.Sin(2 * Math.PI * 220 * i / 4000.0) + 0.1 * Math.Sin(2 * Math.PI * 660 * i / 4000.0);
        return audio;
    }

    [Fact(Timeout = 120000)]
    public async Task Generator_UpsamplesEachFrameByTheProductOfItsRates_ToATanhWaveform()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = model.ComputeMel(Audio());
        var wave = model.MelToWaveform(mel);
        Assert.Equal(new[] { 1, 1, mel.Shape[2] * 16 }, wave.Shape.ToArray());
        for (int i = 0; i < wave.Length; i++) Assert.InRange(wave[i], -1.0, 1.0);
    }

    [Fact(Timeout = 240000)]
    public async Task AdversarialTraining_ReducesTheMelReconstructionError()
    {
        await Task.Yield();
        var model = CreateModel();
        var audio = Audio();
        var segment = new Tensor<double>(new[] { 256 }, new Vector<double>(audio.ToVector().Take(256).ToArray()));
        var mel = model.ComputeMel(segment);
        double Error()
        {
            var generated = model.MelToWaveform(mel);
            var regenerated = model.ComputeMel(new Tensor<double>(new[] { generated.Length }, generated.ToVector()));
            return Enumerable.Range(0, mel.Length).Average(i => Math.Abs(mel[i] - regenerated[i]));
        }
        double before = Error();
        for (int i = 0; i < 25; i++) model.Train(mel, segment);
        double after = Error();
        Assert.True(after < before, $"Mel error did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 60000)]
    public async Task SpectralNormalization_KeepsTheLargestSingularValueAtOne()
    {
        await Task.Yield();
        var conv = new NormedConv1DLayer<double>(3, 4, 3, 1, 1, 1, 1, false, ConvolutionNormalization.Spectral);
        conv.SetTrainingMode(true);
        var x = new Tensor<double>(new[] { 1, 3, 9 });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(i);
        for (int i = 0; i < 30; i++) conv.Forward(x);   // the power iterations converge
        conv.SetTrainingMode(false);
        var w = conv.Kernel();                           // [4, 3, 1, 3] -> a 4 x 9 matrix
        var m = new double[4, 9];
        for (int i = 0; i < 36; i++) m[i / 9, i % 9] = w[i];
        // Largest singular value by power iteration on WᵀW.
        var v = Enumerable.Repeat(1.0, 9).ToArray();
        double sigma = 0;
        for (int it = 0; it < 200; it++)
        {
            var u = new double[4];
            for (int r = 0; r < 4; r++) for (int c = 0; c < 9; c++) u[r] += m[r, c] * v[c];
            var next = new double[9];
            for (int c = 0; c < 9; c++) for (int r = 0; r < 4; r++) next[c] += m[r, c] * u[r];
            double norm = Math.Sqrt(next.Sum(a => a * a));
            sigma = Math.Sqrt(norm);
            v = next.Select(a => a / norm).ToArray();
        }
        Assert.Equal(1.0, sigma, 3);
    }
}
