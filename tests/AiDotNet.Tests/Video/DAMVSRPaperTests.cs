using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Kong et al. 2025, DAM-VSR (arXiv 2507.01012), Sec. 3-4, on Stable Video Diffusion: the configuration, the Karras
/// schedule entered by SDEdit, the backward pass's rotated attention maps, the ControlNet's zero initialization, the three
/// training stages and training (#2397).
/// </summary>
public class DAMVSRPaperTests
{
    private const int Low = 4;

    private static DAMVSR<double> Small(int controlNetIterations = 60000, int backwardIterations = 45000, double learningRate = 8e-5) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputFrames: 2, inputDepth: 3, inputHeight: Low, inputWidth: Low, outputSize: 3 * 4 * Low * Low),
            new DAMVSROptions
            {
                Seed = 1, UNetChannels = new[] { 32, 32 }, UNetHeads = new[] { 1, 1 }, LayersPerBlock = 1, AddedTimeEmbedDim = 8,
                VaeChannels = new[] { 32, 32 }, VaeLayersPerBlock = 1, ImageEncoderSize = 28, ImageEncoderPatchSize = 14,
                ImageEncoderWidth = 32, ImageEncoderDepth = 1, ImageEncoderHeads = 2, ImageEncoderMlpDim = 64, ImageEmbedDim = 32,
                ScaleFactor = 2, NumFrames = 2, NumInferenceSteps = 2, DecoderSamplingSteps = 1, PerceptualWeight = 0,
                DiscriminatorChannels = 4, ControlNetIterations = controlNetIterations, BackwardIterations = backwardIterations,
                LearningRate = learningRate,
            });

    private static Tensor<double> Pixels(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = 255.0 * random.NextDouble();
        return t;
    }

    private static Tensor<double> Input(int seed = 3) => Pixels(new[] { 2, 3, Low, Low }, seed);

    private static Tensor<double> Target(int seed = 4) => Pixels(new[] { 2, 3, 2 * Low, 2 * Low }, seed);

    private static List<double[]> Snapshot(IReadOnlyList<Tensor<double>> parameters)
        => parameters.Select(p => Enumerable.Range(0, p.Length).Select(i => p[i]).ToArray()).ToList();

    private static bool Changed(List<double[]> before, IReadOnlyList<Tensor<double>> after)
        => before.Select((values, k) => values.Where((v, i) => v != after[k][i]).Any()).Any(changed => changed);

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePaperConfiguration()
    {
        await Task.Yield();
        var o = new DAMVSROptions();
        Assert.Equal(new[] { 320, 640, 1280, 1280 }, o.UNetChannels);
        Assert.Equal(new[] { 5, 10, 20, 20 }, o.UNetHeads);
        Assert.Equal((2, 1, 256), (o.LayersPerBlock, o.TransformerDepth, o.AddedTimeEmbedDim));
        Assert.Equal(new[] { 128, 256, 512, 512 }, o.VaeChannels);
        Assert.Equal((224, 14, 1280, 32, 16, 5120, 1024),
            (o.ImageEncoderSize, o.ImageEncoderPatchSize, o.ImageEncoderWidth, o.ImageEncoderDepth, o.ImageEncoderHeads, o.ImageEncoderMlpDim, o.ImageEmbedDim));
        Assert.Equal((4, 14, 7, 127, 0.02), (o.ScaleFactor, o.NumFrames, o.Fps, o.MotionBucketId, o.NoiseAugmentationStrength));
        Assert.Equal((30, 0.6, 0.002, 700.0, 7.0, true),
            (o.NumInferenceSteps, o.SdeditStrength, o.SigmaMin, o.SigmaMax, o.KarrasRho, o.BidirectionalSampling));
        Assert.Equal((8e-5, 60000, 45000, 8, 1.0, 0.025), (o.LearningRate, o.ControlNetIterations, o.BackwardIterations,
            o.DecoderSamplingSteps, o.PerceptualWeight, o.GANWeight));
        Assert.Equal(0.18215, o.LatentScale);
    }

    // Sec. 4.1: 30 Euler steps on SVD's Karras schedule, SDEdit entering with strength 0.6, so 18 steps run.
    [Fact(Timeout = 120000)]
    public async Task Schedule_IsKarrasEnteredBySdedit()
    {
        await Task.Yield();
        var model = Small();
        var sigmas = model.Sigmas(30);
        Assert.Equal(31, sigmas.Length);
        Assert.Equal(700.0, sigmas[0], 9);
        Assert.Equal(0.002, sigmas[29], 12);
        Assert.Equal(0.0, sigmas[30]);
        for (int i = 0; i < 30; i++) Assert.True(sigmas[i] > sigmas[i + 1]);
        // Interior steps are evenly spaced in σ^(1/ρ).
        double max = Math.Pow(700.0, 1 / 7.0), min = Math.Pow(0.002, 1 / 7.0);
        Assert.Equal(Math.Pow(max + 14.0 / 29 * (min - max), 7), sigmas[14], 9);
        Assert.Equal(12, model.SdeditStart(30));
    }

    // Sec. 3.3: the backward pass applies the forward pass's temporal attention map rotated 180 degrees,
    // A'[k − x, k − y] = A[x, y]. A map in which every frame attends to the first frame becomes one in which every
    // frame attends to the last.
    [Fact(Timeout = 120000)]
    public async Task BackwardAttention_UsesTheRotatedForwardMap()
    {
        await Task.Yield();
        var model = Small();
        model.Predict(Input());
        const int tokens = 2, frames = 3, channels = 32;
        var random = new System.Random(5);
        var x = new Tensor<double>(new[] { tokens, frames, channels });
        for (int i = 0; i < x.Length; i++) x[i] = random.NextDouble() * 2 - 1;
        var map = new Tensor<double>(new[] { tokens, frames, frames });
        for (int s = 0; s < tokens; s++)
            for (int f = 0; f < frames; f++) map[(s * frames + f) * frames] = 1.0;

        Tensor<double> Repeated(int frame)
        {
            var r = new Tensor<double>(x.Shape.ToArray());
            for (int s = 0; s < tokens; s++)
                for (int f = 0; f < frames; f++)
                    for (int c = 0; c < channels; c++) r[(s * frames + f) * channels + c] = x[(s * frames + frame) * channels + c];
            return r;
        }

        var output = model.BackwardTemporalAttention(0, x, map);
        var last = model.BackwardTemporalAttention(0, Repeated(frames - 1), map);
        var first = model.BackwardTemporalAttention(0, Repeated(0), map);
        double toLast = 0, toFirst = 0;
        for (int i = 0; i < output.Length; i++)
        {
            toLast = Math.Max(toLast, Math.Abs(output[i] - last[i]));
            toFirst = Math.Max(toFirst, Math.Abs(output[i] - first[i]));
        }
        Assert.True(toLast < 1e-12, $"every frame should read the last frame; max deviation {toLast}");
        Assert.True(toFirst > 1e-3, "the unrotated map would read the first frame");
    }

    // Sec. 3.2 and ControlNet (Zhang et al. 2023): the zero convolutions make the untrained ControlNet leave SVD's output
    // unchanged whatever its other weights are; once they are non-zero the ControlNet steers the output.
    [Fact(Timeout = 180000)]
    public async Task ControlNet_ZeroConvolutionsLeaveTheUNetUnchanged()
    {
        await Task.Yield();
        var model = Small();
        var input = Input();
        var reference = model.Predict(input);
        var parameters = model.ControlNetParameters;
        var zero = parameters.Where(p => Enumerable.Range(0, p.Length).All(i => p[i] == 0.0)).ToList();
        var other = parameters.Where(p => !zero.Contains(p)).ToList();
        Assert.NotEmpty(zero);
        Assert.NotEmpty(other);

        foreach (var p in other)
            for (int i = 0; i < p.Length; i++) p[i] = 1.5 * p[i] + 0.01;
        var perturbed = model.Predict(input);
        for (int i = 0; i < reference.Length; i++) Assert.Equal(reference[i], perturbed[i]);

        foreach (var p in zero)
            for (int i = 0; i < p.Length; i++) p[i] = 0.05;
        var steered = model.Predict(input);
        Assert.Contains(Enumerable.Range(0, reference.Length), i => Math.Abs(reference[i] - steered[i]) > 1e-6);
    }

    // Sec. 4.1: the ControlNet trains with the U-Net frozen, then the backward pass's temporal value and output
    // projections alone, then the VAE decoder with its discriminator.
    [Fact(Timeout = 300000)]
    public async Task Training_RunsThePapersThreeStagesInTurn()
    {
        await Task.Yield();
        var model = Small(controlNetIterations: 1, backwardIterations: 1);
        var (input, target) = (Input(), Target());
        model.Predict(input);

        Assert.Equal(0, model.TrainingStage);
        var (unet, control, backward, decoder, discriminator) = (Snapshot(model.UNetParameters), Snapshot(model.ControlNetParameters),
            Snapshot(model.BackwardParameters), Snapshot(model.DecoderParameters), Snapshot(model.DiscriminatorParameters));
        model.Train(input, target);
        Assert.True(Changed(control, model.ControlNetParameters));
        Assert.False(Changed(unet, model.UNetParameters));
        Assert.False(Changed(backward, model.BackwardParameters));
        Assert.False(Changed(decoder, model.DecoderParameters));

        Assert.Equal(1, model.TrainingStage);
        control = Snapshot(model.ControlNetParameters);
        model.Train(input, target);
        Assert.True(Changed(backward, model.BackwardParameters));
        Assert.False(Changed(control, model.ControlNetParameters));
        Assert.False(Changed(unet, model.UNetParameters));
        Assert.False(Changed(decoder, model.DecoderParameters));

        Assert.Equal(2, model.TrainingStage);
        backward = Snapshot(model.BackwardParameters);
        model.Train(input, target);
        Assert.True(Changed(decoder, model.DecoderParameters));
        Assert.True(Changed(discriminator, model.DiscriminatorParameters));
        Assert.False(Changed(backward, model.BackwardParameters));
        Assert.False(Changed(control, model.ControlNetParameters));
        Assert.False(Changed(unet, model.UNetParameters));
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheControlNetObjective()
    {
        await Task.Yield();
        var model = Small(learningRate: 3e-3);
        var (input, target) = (Input(), Target());
        var provider = (ITrainingObjectiveProvider<double>)model;
        double before = provider.EvaluateTrainingObjective(input, target);
        for (int step = 0; step < 30; step++) model.Train(input, target);
        double after = provider.EvaluateTrainingObjective(input, target);
        Assert.Equal(0, model.TrainingStage);
        Assert.True(after < before, $"objective {before} -> {after}");
    }
}
