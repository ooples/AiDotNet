using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SpeedySpeech's student network trains and synthesizes as its paper and its authors' implementation specify
/// (Vainer &amp; Dušek 2020; github.com/janvainer/speedyspeech): dilated residual convolution blocks with temporal batch
/// normalization, a duration predictor on detached encodings, phoneme-local positional encoding, and MAE + SSIM + Huber.
/// </summary>
/// <remarks>
/// Before this change SpeedySpeech had none of this: its synthesis derived durations from a fixed formula on encoder
/// values, and threw on every call because its encoder boundary pointed past the layer stack.
/// </remarks>
public class SpeedySpeechPaperTests
{
    private const int MelBins = 8;

    private static SpeedySpeech<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins),
        new SpeedySpeechOptions
        {
            HiddenDim = 16, EncoderDim = 16, MelChannels = MelBins, EncoderDilations = new[] { 1, 2 },
            DecoderDilations = new[] { 1, 2, 4 }, MelMean = 0.0, MelStd = 1.0, LearningRate = 2e-3,
        });

    private static Tensor<double> Tokens(int count)
    {
        var tokens = new Tensor<double>(new[] { count });
        for (int i = 0; i < count; i++) tokens[i] = 10 + (i * 7) % 60;
        return tokens;
    }

    private static TtsTrainingSample<double> Sample()
    {
        var mel = new Tensor<double>(new[] { 14, MelBins });
        for (int f = 0; f < 14; f++)
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
        return new TtsTrainingSample<double> { Tokens = Tokens(5), Mel = mel, Durations = new[] { 3, 3, 3, 3, 2 } };
    }

    [Fact(Timeout = 60000)]
    public async Task Ssim_MatchesTheReferenceImplementation()
    {
        await Task.Yield();
        // Reference: SSIM with an 11x11 Gaussian window (sigma 1.5), zero padding, C1 = 0.01^2, C2 = 0.03^2 -- the
        // formula of pytorch_ssim in the SpeedySpeech repository -- evaluated with scipy.signal.convolve2d.
        var a = new Tensor<double>(new[] { 9, 10 });
        var b = new Tensor<double>(new[] { 9, 10 });
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 10; j++)
            {
                a[i, j] = Math.Sin(0.3 * i + 0.7 * j);
                b[i, j] = Math.Cos(0.2 * i - 0.4 * j) * 0.8;
            }
        var engine = AiDotNetEngine.Current;
        Assert.Equal(0.23034436442142095, SpectrogramSsim.Ssim(engine, a, b)[0], 10);
        Assert.Equal(1.0, SpectrogramSsim.Ssim(engine, a, a)[0], 10);
    }

    [Fact(Timeout = 60000)]
    public async Task ResidualBlock_IsValidConvolution_ZeroPaddedAfterwards()
    {
        await Task.Yield();
        // Kernel 4, dilation 2: the unpadded convolution loses 6 frames, put back as 3 zeros in front and 3 behind before
        // ReLU and batch normalization; the residual adds the input. With every weight and bias zero the block's inner
        // path is zero everywhere, so the output must equal the input exactly.
        var block = new DilatedResidualConvBlock<double>(channels: 4, kernelSize: 4, dilation: 2, convolutions: 2);
        block.SetParameters(new Vector<double>((int)block.ParameterCount));
        block.SetTrainingMode(false);
        var x = new Tensor<double>(new[] { 10, 4 });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.7 * i);
        var y = block.Forward(x);
        Assert.Equal(x.Shape, y.Shape);
        for (int i = 0; i < x.Length; i++) Assert.Equal(x[i], y[i], 12);
    }

    [Fact(Timeout = 120000)]
    public async Task Training_ReducesThePaperObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample();
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 30; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 60000)]
    public async Task Synthesis_ProducesAFiniteMelSpectrogram_OfThePredictedLength()
    {
        await Task.Yield();
        var mel = CreateModel().Synthesize("hello");
        Assert.Equal(2, mel.Rank);
        Assert.Equal(MelBins, mel.Shape[1]);
        Assert.True(mel.Shape[0] >= 5, $"Every phoneme lasts at least one frame; got {mel.Shape[0]} frames for 5 phonemes.");
        for (int i = 0; i < mel.Length; i++) Assert.True(double.IsFinite(mel[i]), $"mel[{i}] = {mel[i]}");
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_Throws_BecauseTheStudentTrainsOnTeacherDurations()
    {
        await Task.Yield();
        var model = CreateModel();
        Assert.Throws<NotSupportedException>(() => model.Train(Tokens(5), Sample().Mel!));
    }
}
