using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Denoising;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Li et al. 2023, Shift-Net, against the authors' denoising release (gshift_denoise1.py, options/davis_denoise.yml):
/// the temporal shift (torch.roll of the frame-channel axis), the grouped spatial shift by multiples of s, the
/// refinement U-Nets and shift encoder, and the PSNR-loss recipe (#2397).
/// </summary>
public class ShiftNetPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 16, inputWidth: 16, inputDepth: 3, outputSize: 3);

    private static ShiftNetOptions Small(double learningRate = 4e-4) => new()
    {
        Seed = 1, NumFeatures = 16, RefinementFeatures = 8, UNetGrowth = 4, NumUNets = 1, LearningRate = learningRate,
    };

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheDavisDenoiseRecipe()
    {
        await Task.Yield();
        var o = new ShiftNetOptions();
        Assert.Equal((80, 24, 12, 5, 4), (o.NumFeatures, o.RefinementFeatures, o.UNetGrowth, o.NumUNets, o.ShiftStride));
        Assert.Equal((4e-4, 1e-7, 450000), (o.LearningRate, o.MinLearningRate, o.CosineIterations));
        Assert.Equal(0.5, o.PsnrLossWeight);
    }

    [Fact(Timeout = 60000)]
    public async Task TemporalShift_IsTheReleasesRollOfTheFrameChannelAxis()
    {
        await Task.Yield();
        var model = new ShiftNet<double>(Architecture(), Small());
        // Three frames of four channels, each value frame*10 + channel.
        var x = new Tensor<double>(new[] { 3, 4, 1, 1 });
        for (int f = 0; f < 3; f++) for (int c = 0; c < 4; c++) x[f * 4 + c] = f * 10 + c;

        // Forward: a frame takes the previous frame's last two channels, then its own first two; frame 0 is kept.
        var forward = model.TemporalShiftForTest(x, reverse: false).ToArray();
        Assert.Equal(new double[] { 0, 1, 2, 3, 2, 3, 10, 11, 12, 13, 20, 21 }, forward);
        // Reverse: its own last two channels, then the next frame's first two; the last frame is kept.
        var reverse = model.TemporalShiftForTest(x, reverse: true).ToArray();
        Assert.Equal(new double[] { 2, 3, 10, 11, 12, 13, 20, 21, 20, 21, 22, 23 }, reverse);
    }

    [Fact(Timeout = 60000)]
    public async Task SpatialShift_MovesEachGroupByItsMultipleOfS()
    {
        await Task.Yield();
        var model = new ShiftNet<double>(Architecture(), Small());
        // number = 3: n2 = 1 channel per outer-ring group (distance 2s), n1 = 1 per inner-ring group (distance s).
        const int size = 20;
        var hw = new Tensor<double>(new[] { 1, 24, size, size });
        for (int c = 0; c < 24; c++) hw[(c * size + 10) * size + 10] = 1.0;
        var shifted = model.SpatialShiftForTest(hw);
        (int Dy, int Dx) Location(int channel)
        {
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    if (shifted[(channel * size + y) * size + x] == 1.0) return (y - 10, x - 10);
            throw new InvalidOperationException($"channel {channel} lost its pixel");
        }

        // spatial_shift2 with s = 4: group 0 moves by (+8, +8), group 3 by (+8, -4), group 12 by (0, +8), the first
        // inner group by (+4, +4) and the last by (-4, -4).
        Assert.Equal((8, 8), Location(0));
        Assert.Equal((8, -4), Location(3));
        Assert.Equal((0, 8), Location(12));
        Assert.Equal((4, 4), Location(16));
        Assert.Equal((-4, -4), Location(23));
        // Pushed off the frame is gone (zero), not wrapped.
        var edge = new Tensor<double>(new[] { 1, 24, size, size });
        edge[(0 * size + 15) * size + 15] = 1.0;
        Assert.Equal(0.0, model.SpatialShiftForTest(edge).ToArray().Sum());
    }

    [Fact(Timeout = 120000)]
    public async Task Frames_ExchangeInformationThroughTheShifts()
    {
        await Task.Yield();
        var model = new ShiftNet<double>(Architecture(), Small());
        var clip = new Tensor<double>(new[] { 3, 3, 16, 16 });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37);
        var baseline = model.DenoiseNormalizedForTest(clip, 0.05);
        var moved = clip.Clone();
        for (int i = 0; i < 3 * 16 * 16; i++) moved[i] += 0.3; // frame 0 only
        var output = model.DenoiseNormalizedForTest(moved, 0.05);
        int frame = 3 * 16 * 16;
        double change = Enumerable.Range(frame, frame).Max(i => Math.Abs(output[i] - baseline[i]));
        Assert.True(change > 1e-9, "frame 1 must read frame 0 through the temporal shift");
        // The noise map is an input, and an output frame is the noisy frame plus the predicted residual.
        var louder = model.DenoiseNormalizedForTest(clip, 0.2);
        Assert.True(Enumerable.Range(0, louder.Length).Max(i => Math.Abs(louder[i] - baseline[i])) > 1e-9);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheError()
    {
        await Task.Yield();
        var model = new ShiftNet<double>(Architecture(), Small(1e-3));
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(2);
        var clean = new Tensor<double>(new[] { 3, 3, 16, 16 });
        for (int i = 0; i < clean.Length; i++) clean[i] = 128 + 60 * Math.Sin(i * 0.37);
        var noisy = clean.Clone();
        for (int i = 0; i < noisy.Length; i++) noisy[i] += 40 * (rng.NextDouble() - 0.5);
        double Error()
        {
            var output = model.Predict(noisy);
            return Enumerable.Range(0, output.Length).Average(i => (output[i] - clean[i]) * (output[i] - clean[i]));
        }

        // The denoised clip must beat the noisy one: a bar independent of how far the untrained residual starts
        // from zero (the release's PyTorch-default init starts near the input, at about 1.6x the noise).
        double noise = Enumerable.Range(0, noisy.Length).Average(i => (noisy[i] - clean[i]) * (noisy[i] - clean[i]));
        double before = Error();
        for (int step = 0; step < 20; step++) model.Train(noisy, clean);
        double after = Error();
        Assert.True(after < before && after < noise, $"MSE {before:F1} -> {after:F1}, noise {noise:F1}");
    }

    [Fact(Timeout = 60000)]
    public async Task PsnrLoss_IsBasicSrsWeightedLogMse()
    {
        await Task.Yield();
        // 0.5 * 10/ln 10 * mean_i log(MSE_i + 1e-8), one MSE per sample.
        var predicted = new Tensor<double>(new[] { 2, 2 });
        var target = new Tensor<double>(new[] { 2, 2 });
        predicted[0] = 1; predicted[1] = 3; predicted[2] = 0.5; predicted[3] = 0.5;
        var loss = new PSNRLoss<double>(0.5).ComputeTapeLoss(predicted, target);
        double expected = 0.5 * 10 / Math.Log(10) * (Math.Log(5 + 1e-8) + Math.Log(0.25 + 1e-8)) / 2;
        Assert.Equal(expected, loss.ToArray()[0], 10);
    }
}
