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
/// Li et al. 2022, FloRNN, against the authors' release (models/flornn.py, train_models/sRGB_train.py): a forward
/// recurrent module plus a look-ahead module carried forward by softmax splatting and warped back, a frozen flow
/// estimator, and the training recipe (#2397).
/// </summary>
public class FloRNNPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture(int size = 16) =>
        new(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: size, inputWidth: size, inputDepth: 3, outputSize: 3);

    private static FloRNNOptions Small(int forwardCount = 2) => new()
    {
        Seed = 1, NumFeatures = 8, NumResBlocks = 1, ForwardCount = forwardCount, FlowPyramidLevels = 3,
    };

    private static Tensor<double> Clip(int frames, int seed, double noise)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var clip = new Tensor<double>(new[] { frames, 3, 16, 16 });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37) + noise * (rng.NextDouble() - 0.5);
        return clip;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheAuthorsSrgbRecipe()
    {
        await Task.Yield();
        var o = new FloRNNOptions();
        Assert.Equal((64, 15, 3), (o.NumFeatures, o.NumResBlocks, o.ForwardCount));
        Assert.Equal(0.1, o.BorderRatio);
        Assert.Equal(1e-4, o.LearningRate);
        Assert.Equal(20.0 / 255.0, o.NoiseLevel);
    }

    [Fact(Timeout = 60000)]
    public async Task Sequence_IsReflectedPastItsEnd()
    {
        await Task.Yield();
        // seqn[:, T+i] = seqn_not_pad[:, T-2-i].
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 3, 2, 1 }, Enumerable.Range(0, 8).Select(i => FloRNN<double>.ReflectFrame(i, 5)));
        Assert.Equal(new[] { 0, 0, 0 }, Enumerable.Range(0, 3).Select(i => FloRNN<double>.ReflectFrame(i, 1)));
    }

    [Fact(Timeout = 60000)]
    public async Task SoftSplatting_AveragesWhatLandsOnEachPixelAndWarpingTakesItBack()
    {
        await Task.Yield();
        var model = new FloRNN<double>(Architecture(), Small());
        var features = new Tensor<double>(new[] { 1, 1, 4, 4 });
        for (int i = 0; i < 16; i++) features[i] = i + 1;
        var flow = new Tensor<double>(new[] { 1, 2, 4, 4 });

        // A one-pixel shift right: each value lands one column over, the first column receives nothing (0), and the
        // last column's values fall off the frame.
        for (int i = 0; i < 16; i++) flow[i] = 1.0;
        var shifted = model.SoftSplatForTest(features, flow);
        for (int y = 0; y < 4; y++)
        {
            Assert.Equal(0.0, shifted[y * 4], 12);
            for (int x = 1; x < 4; x++) Assert.Equal(features[y * 4 + x - 1], shifted[y * 4 + x], 12);
        }

        // Two half-weight contributions on one pixel average to their mean ("average" divides by the weight).
        var half = new Tensor<double>(new[] { 1, 2, 4, 4 });
        half[0] = 0.5; // (0, 0) moves half a pixel right: half to (0, 0), half to (0, 1)
        half[1] = -0.5; // (0, 1) moves half a pixel left: half to (0, 0), half to (0, 1)
        var averaged = model.SoftSplatForTest(features, half);
        Assert.Equal(1.5, averaged[0], 12);
        Assert.Equal(1.5, averaged[1], 12);

        // utils/warp.py samples at p + flow and zeroes samples that leave the frame.
        var back = model.WarpMaskedForTest(shifted, flow);
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 3; x++) Assert.Equal(features[y * 4 + x], back[y * 4 + x], 9);
            Assert.Equal(0.0, back[y * 4 + 3], 12);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Output_LooksAheadExactlyForwardCountFrames()
    {
        await Task.Yield();
        // A unidirectional model with a look-ahead of k: frame t depends on frames up to t + k and on none after.
        const int k = 2;
        var model = new FloRNN<double>(Architecture(), Small(k));
        var clip = Clip(7, 3, 0.1);
        var baseline = model.Denoise(clip, 0.05);
        double Change(int perturbedFrame, int outputFrame)
        {
            var moved = clip.Clone();
            for (int i = 0; i < 3 * 16 * 16; i++) moved[perturbedFrame * 3 * 16 * 16 + i] += 0.2;
            var output = model.Denoise(moved, 0.05);
            double max = 0;
            for (int i = 0; i < 3 * 16 * 16; i++)
                max = Math.Max(max, Math.Abs(output[outputFrame * 3 * 16 * 16 + i] - baseline[outputFrame * 3 * 16 * 16 + i]));
            return max;
        }

        Assert.True(Change(1 + k, 1) > 1e-9, "frame t + k must reach frame t");
        Assert.Equal(0.0, Change(2 + k, 1), 12);
        // The noise-level map is an input.
        var louder = model.Denoise(clip, 0.2);
        Assert.True(Enumerable.Range(0, louder.Length).Max(i => Math.Abs(louder[i] - baseline[i])) > 1e-9);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LeavesTheFlowEstimatorFrozenAndReducesTheError()
    {
        await Task.Yield();
        var model = new FloRNN<double>(Architecture(), new FloRNNOptions
        {
            Seed = 2, NumFeatures = 8, NumResBlocks = 1, ForwardCount = 2, FlowPyramidLevels = 3, LearningRate = 1e-3,
        });
        var clean = Clip(6, 4, 0.0);
        var noisy = clean.Clone();
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(5);
        for (int i = 0; i < noisy.Length; i++) noisy[i] += 0.1 * (rng.NextDouble() - 0.5);
        double Error()
        {
            var output = model.Denoise(noisy, 0.1 / Math.Sqrt(12));
            double sum = 0;
            for (int i = 0; i < output.Length; i++) sum += (output[i] - clean[i]) * (output[i] - clean[i]);
            return sum / output.Length;
        }

        // The first forward materializes the lazily shaped layers.
        double before = Error();
        var flowBefore = model.FlowEstimatorParameters.Select(p => p.ToArray()).ToList();
        var allBefore = model.GetParameters().ToArray();
        for (int step = 0; step < 20; step++) model.Train(noisy, clean);
        double after = Error();

        var flowAfter = model.FlowEstimatorParameters.Select(p => p.ToArray()).ToList();
        Assert.Equal(flowBefore.Count, flowAfter.Count);
        for (int i = 0; i < flowBefore.Count; i++) Assert.Equal(flowBefore[i], flowAfter[i]);
        var allAfter = model.GetParameters().ToArray();
        Assert.Contains(Enumerable.Range(0, allAfter.Length), i => allAfter[i] != allBefore[i]);
        Assert.True(after < 0.5 * before, $"MSE {before:E3} -> {after:E3}");
    }
}
