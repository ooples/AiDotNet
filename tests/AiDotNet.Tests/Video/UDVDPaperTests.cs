using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Denoising;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Sheth et al. 2021, UDVD, against the authors' release (github.com/sreyas-mohan/udvd: models/udvd.py
/// "blind-video-net-4", utils/loss_funtions.py "loglike", utils/post_processing.py): the blind spot, the four
/// rotations, the Gaussian log-likelihood and the posterior-mean output (#2397).
/// </summary>
public class UDVDPaperTests
{
    private const int Size = 16;

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: 5, inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: 5);

    private static UDVD<double> Small(double learningRate = 1e-4) => new(Architecture(), new UDVDOptions
    {
        Seed = 1, EncoderFeatures = 8, DecoderFeatures = 8, FirstStageFeatures = 8, LearningRate = learningRate,
    });

    private static Tensor<double> Clip(int frames, double phase = 0)
    {
        var clip = new Tensor<double>(new[] { frames, 3, Size, Size });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37 + phase);
        return clip;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreBlindVideoNet4()
    {
        await Task.Yield();
        var o = new UDVDOptions();
        Assert.Equal((48, 96, 32), (o.EncoderFeatures, o.DecoderFeatures, o.FirstStageFeatures));
        Assert.Equal(30.0 / 255.0, o.NoiseLevel);
        Assert.Equal(1e-4, o.LearningRate);
    }

    [Fact(Timeout = 120000)]
    public async Task Network_NeverSeesThePixelItPredicts()
    {
        await Task.Yield();
        var model = Small();
        var clip = Clip(5);
        var baseline = model.NetworkOutputForTest(clip);
        int outChannels = baseline.Shape[1];
        Assert.Equal(3 + 6, outChannels); // RGB mean and the 3x3 upper-triangular factor

        const int y = 7, x = 9;
        // The same pixel of every frame: the blind spot is the pixel's whole temporal column.
        var spot = clip.Clone();
        for (int f = 0; f < 5; f++)
            for (int c = 0; c < 3; c++) spot[((f * 3 + c) * Size + y) * Size + x] += 0.4;
        var blind = model.NetworkOutputForTest(spot);
        for (int f = 0; f < 5; f++)
            for (int c = 0; c < outChannels; c++)
            {
                int at = ((f * outChannels + c) * Size + y) * Size + x;
                Assert.Equal(baseline[at], blind[at], 10);
            }

        // A neighbour in the window is seen.
        var neighbour = clip.Clone();
        for (int c = 0; c < 3; c++) neighbour[((2 * 3 + c) * Size + y) * Size + x + 1] += 0.4;
        var seen = model.NetworkOutputForTest(neighbour);
        double change = Enumerable.Range(0, outChannels)
            .Max(c => Math.Abs(seen[((2 * outChannels + c) * Size + y) * Size + x] - baseline[((2 * outChannels + c) * Size + y) * Size + x]));
        Assert.True(change > 1e-9, "the pixel to the right must be in the receptive field");
    }

    [Fact(Timeout = 60000)]
    public async Task Unrotate_UndoesEachOfTheFourRotations()
    {
        await Task.Yield();
        var model = Small();
        var x = new Tensor<double>(new[] { 2, 3, 4, 4 });
        for (int i = 0; i < x.Length; i++) x[i] = i;
        var rotated = model.Rotate(x);
        Assert.Equal(new[] { 8, 3, 4, 4 }, rotated.Shape.ToArray());
        // The 90 degree copy of image 0, x.transpose(2, 3).flip(3), puts x[.., W-1-j, i] at (i, j).
        Assert.Equal(x[(0 * 4 + 3) * 4 + 0], rotated[((2 * 3 + 0) * 4 + 0) * 4 + 0]);
        var back = model.Unrotate(rotated);
        Assert.Equal(new[] { 2, 12, 4, 4 }, back.Shape.ToArray());
        for (int n = 0; n < 2; n++)
            for (int q = 0; q < 4; q++)
                for (int i = 0; i < 48; i++)
                    Assert.Equal(x[n * 48 + i], back[(n * 4 + q) * 48 + i]);
    }

    [Fact(Timeout = 60000)]
    public async Task Loss_IsTheGaussianNegativeLogLikelihood()
    {
        await Task.Yield();
        var model = Small();
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(5);
        var output = new Tensor<double>(new[] { 1, 9, 2, 2 });
        var target = new Tensor<double>(new[] { 1, 3, 2, 2 });
        for (int i = 0; i < output.Length; i++) output[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();
        const double sigma = 0.1;
        double loss = model.NegativeLogLikelihood(output, target, sigma).ToArray()[0];

        double expected = 0;
        for (int p = 0; p < 4; p++)
        {
            // Row i of A holds columns 0 .. 2-i, from channel 3 on.
            var a = new double[3, 3];
            int k = 3;
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3 - i; j++) a[i, j] = output[k++ * 4 + p];
            var cov = new double[3, 3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    double s = i == j ? sigma * sigma + 1e-5 : 0;
                    for (int m = 0; m < 3; m++) s += a[m, i] * a[m, j];
                    cov[i, j] = s;
                }
            double det = cov[0, 0] * (cov[1, 1] * cov[2, 2] - cov[1, 2] * cov[2, 1])
                       - cov[0, 1] * (cov[1, 0] * cov[2, 2] - cov[1, 2] * cov[2, 0])
                       + cov[0, 2] * (cov[1, 0] * cov[2, 1] - cov[1, 1] * cov[2, 0]);
            var inv = new double[3, 3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    int r0 = (j + 1) % 3, r1 = (j + 2) % 3, c0 = (i + 1) % 3, c1 = (i + 2) % 3;
                    inv[i, j] = (cov[r0, c0] * cov[r1, c1] - cov[r0, c1] * cov[r1, c0]) / det;
                }
            var d = Enumerable.Range(0, 3).Select(c => target[c * 4 + p] - output[c * 4 + p]).ToArray();
            double quad = 0;
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) quad += d[i] * inv[i, j] * d[j];
            expected += 0.5 * quad + 0.5 * Math.Log(det);
        }
        expected = expected / 4 - 0.1 * sigma;
        Assert.Equal(expected, loss, 9);
    }

    [Fact(Timeout = 60000)]
    public async Task PosteriorMean_WeighsThePriorAgainstTheNoisyPixel()
    {
        await Task.Yield();
        var model = Small();
        var output = new Tensor<double>(new[] { 1, 9, 1, 1 });
        var noisy = new Tensor<double>(new[] { 1, 3, 1, 1 });
        for (int c = 0; c < 3; c++) { output[c] = 0.2 + 0.1 * c; noisy[c] = 0.9; }

        // A = 0: the network is certain (Σ = 1e-5 I), so the estimate is its mean.
        var certain = model.PosteriorMean(output, noisy, 0.1);
        for (int c = 0; c < 3; c++) Assert.Equal(output[c], certain[c], 2);

        // Row i of A fills columns 0 .. 2-i (channels 3-5, 6-7, 8), so A's anti-diagonal is channels 5, 7 and 8;
        // an anti-diagonal A = a J gives AᵀA = a² I and Σ = (a² + 1e-5) I, a scalar posterior per channel.
        const double a = 0.2, sigma = 0.1;
        output[5] = a; output[7] = a; output[8] = a;
        var mixed = model.PosteriorMean(output, noisy, sigma);
        double prior = 1 / (a * a + 1e-5), likelihood = 1 / (sigma * sigma + 1e-5);
        for (int c = 0; c < 3; c++)
            Assert.Equal((prior * output[c] + likelihood * noisy[c]) / (prior + likelihood), mixed[c], 9);
    }

    [Fact(Timeout = 600000)]
    public async Task UnsupervisedTraining_OnNoisyFramesAlone_BeatsTheNoise()
    {
        await Task.Yield();
        var model = Small(1e-3);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(2);
        var clean = new Tensor<double>(new[] { 5, 3, Size, Size });
        for (int i = 0; i < clean.Length; i++) clean[i] = 128 + 60 * Math.Sin(i * 0.37);
        var noisy = clean.Clone();
        for (int i = 0; i < noisy.Length; i++) noisy[i] += 40 * (rng.NextDouble() - 0.5);

        double noise = Enumerable.Range(0, noisy.Length).Average(i => (noisy[i] - clean[i]) * (noisy[i] - clean[i]));
        for (int step = 0; step < 80; step++) model.Train(noisy, noisy);
        var output = model.Predict(noisy);
        double after = Enumerable.Range(0, output.Length).Average(i => (output[i] - clean[i]) * (output[i] - clean[i]));
        Assert.True(after < noise, $"MSE noisy {noise:F1} -> denoised {after:F1}");
    }
}
