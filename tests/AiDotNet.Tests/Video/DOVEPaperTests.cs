using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Chen et al. 2025, DOVE (arXiv 2505.16239), Sec. 3-4, and the release (zhengchen1999/DOVE) on CogVideoX1.5-5B: the
/// configuration, the zero-terminal-SNR schedule at the one step's timestep, the 3D rotary embedding, the two training
/// stages and training (#2397).
/// </summary>
public class DOVEPaperTests
{
    private const int Low = 4;

    private static DOVE<double> Small(int adaptationIterations = 10000, double learningRate = 2e-5, double distsWeight = 0) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputFrames: 2, inputDepth: 3, inputHeight: Low, inputWidth: Low, outputSize: 3 * 4 * Low * Low),
            new DOVEOptions
            {
                Seed = 1, NumAttentionHeads = 1, AttentionHeadDim = 16, NumLayers = 1, TimeEmbedDim = 8, TextEmbedDim = 8,
                MaxTextSeqLength = 4, VaeChannels = new[] { 32, 32 }, VaeLayersPerBlock = 1, LatentChannels = 4,
                TemporalCompressionRatio = 2, ScaleFactor = 2, NumFrames = 2, DistsWeight = distsWeight, WarmupSteps = 0,
                AdaptationIterations = adaptationIterations, LearningRate = learningRate, RefinementLearningRate = learningRate,
                ImageRatio = 0.0,
            });

    private static Tensor<double> Pixels(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = 255.0 * random.NextDouble();
        return t;
    }

    private static Tensor<double> Input() => Pixels(new[] { 2, 3, Low, Low }, 3);

    private static Tensor<double> Target() => Pixels(new[] { 2, 3, 2 * Low, 2 * Low }, 4);

    private static List<double[]> Snapshot(IReadOnlyList<Tensor<double>> parameters)
        => parameters.Select(p => Enumerable.Range(0, p.Length).Select(i => p[i]).ToArray()).ToList();

    private static bool Changed(List<double[]> before, IReadOnlyList<Tensor<double>> after)
        => before.Select((values, k) => values.Where((v, i) => v != after[k][i]).Any()).Any(changed => changed);

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePaperConfiguration()
    {
        await Task.Yield();
        var o = new DOVEOptions();
        Assert.Equal((48, 64, 42, 512, 4096, 226), (o.NumAttentionHeads, o.AttentionHeadDim, o.NumLayers, o.TimeEmbedDim, o.TextEmbedDim, o.MaxTextSeqLength));
        Assert.Equal((2, 2, 1e-5), (o.PatchSize, o.PatchSizeT, o.NormEpsilon));
        Assert.Equal(new[] { 128, 256, 256, 512 }, o.VaeChannels);
        Assert.Equal((3, 16, 4, 0.7), (o.VaeLayersPerBlock, o.LatentChannels, o.TemporalCompressionRatio, o.LatentScale));
        Assert.Equal((4, 399, 1000, 0.00085, 0.012), (o.ScaleFactor, o.Timestep, o.NumTrainTimesteps, o.BetaStart, o.BetaEnd));
        Assert.Equal((2e-5, 5e-6, 10000, 1e-4, 0.1, 100), (o.LearningRate, o.RefinementLearningRate, o.AdaptationIterations,
            o.WeightDecay, o.MaxGradientNorm, o.WarmupSteps));
        Assert.Equal((1.0, 1.0, 0.8), (o.DistsWeight, o.FrameDifferenceWeight, o.ImageRatio));
    }

    // CogVideoXDDIMScheduler with rescale_betas_zero_snr: ᾱ_0 is unchanged, ᾱ_999 is exactly 0, and the one step
    // at t = 399 keeps a fraction of the signal between the ends.
    [Fact(Timeout = 60000)]
    public async Task Schedule_IsScaledLinearRescaledToZeroTerminalSnr()
    {
        await Task.Yield();
        var o = new DOVEOptions();
        var alphas = DOVE<double>.AlphasCumprod(o);
        Assert.Equal(1000, alphas.Length);
        Assert.Equal(1 - 0.00085, alphas[0], 12);
        Assert.Equal(0.0, alphas[999], 15);
        for (int i = 0; i < 999; i++) Assert.True(alphas[i] > alphas[i + 1]);
        // The rescale is affine in sqrt(ᾱ), anchored at both ends.
        double product = 1, first = 0, last;
        var roots = new double[1000];
        for (int i = 0; i < 1000; i++)
        {
            double b = Math.Sqrt(0.00085) + i / 999.0 * (Math.Sqrt(0.012) - Math.Sqrt(0.00085));
            product *= 1 - b * b;
            roots[i] = Math.Sqrt(product);
            if (i == 0) first = roots[0];
        }
        last = roots[999];
        double expected = Math.Pow((roots[399] - last) * first / (first - last), 2);
        Assert.Equal(expected, alphas[399], 12);
        Assert.InRange(alphas[399], 0.1, 0.9);
    }

    // get_3d_rotary_pos_embed: a quarter of each head rotates with the frame, three eighths each with the row and the
    // column, adjacent pairs share a frequency, and the first pair of each part turns by exactly its position.
    [Fact(Timeout = 60000)]
    public async Task Rope_SplitsTheHeadOverTimeHeightAndWidth()
    {
        await Task.Yield();
        var (cos, sin) = CogVideoXRope.Build<double>(64, 3, 2, 5);
        Assert.Equal(new[] { 1, 30, 64 }, cos.Shape.ToArray());
        // token (t = 2, h = 1, w = 4): parts start at 0 (time, 16 wide), 16 (height, 24) and 40 (width, 24).
        int row = ((2 * 2 + 1) * 5 + 4) * 64;
        Assert.Equal(Math.Cos(2.0), cos[row + 0], 12);
        Assert.Equal(Math.Cos(2.0), cos[row + 1], 12);
        Assert.Equal(Math.Sin(1.0), sin[row + 16], 12);
        Assert.Equal(Math.Sin(4.0), sin[row + 40], 12);
        Assert.Equal(Math.Cos(2.0 / Math.Pow(10000.0, 2.0 / 16)), cos[row + 2], 12);
        Assert.Equal(Math.Cos(4.0 / Math.Pow(10000.0, 2.0 / 24)), cos[row + 42], 12);

        // apply_rotary_emb rotates each adjacent pair: (x0, x1) to (x0·cos − x1·sin, x1·cos + x0·sin).
        var x = new Tensor<double>(new[] { 1, 30, 64 });
        for (int i = 0; i < x.Length; i++) x[i] = (i % 7) - 3;
        var rotated = CogVideoXRope.Apply(AiDotNetEngine.Current, x, cos, sin);
        int k = row + 40;
        Assert.Equal(x[k] * cos[k] - x[k + 1] * sin[k], rotated[k], 12);
        Assert.Equal(x[k + 1] * cos[k + 1] + x[k] * sin[k + 1], rotated[k + 1], 12);
    }

    // Sec. 3.3: only the transformer trains; the latent-space stage comes first, then the pixel-space stage.
    [Fact(Timeout = 300000)]
    public async Task Training_TrainsOnlyTheTransformerThroughBothStages()
    {
        await Task.Yield();
        var model = Small(adaptationIterations: 1, learningRate: 1e-3, distsWeight: 1.0);
        var (input, target) = (Input(), Target());
        model.Predict(input);
        var transformer = Snapshot(model.TransformerParameters);
        var vae = Snapshot(model.VaeParameters);

        Assert.Equal(0, model.TrainingStage);
        model.Train(input, target);
        Assert.True(Changed(transformer, model.TransformerParameters));
        Assert.False(Changed(vae, model.VaeParameters));

        Assert.Equal(1, model.TrainingStage);
        transformer = Snapshot(model.TransformerParameters);
        model.Train(input, target);
        Assert.True(Changed(transformer, model.TransformerParameters));
        Assert.False(Changed(vae, model.VaeParameters));
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_UpscalesEveryFrame()
    {
        await Task.Yield();
        var model = Small();
        var output = model.Predict(Pixels(new[] { 3, 3, Low, Low + 2 }, 7));
        Assert.Equal(new[] { 3, 3, 2 * Low, 2 * (Low + 2) }, output.Shape.ToArray());
        for (int i = 0; i < output.Length; i++) Assert.InRange(output[i], 0.0, 255.0);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheAdaptationObjective()
    {
        await Task.Yield();
        var model = Small(learningRate: 1e-3);
        var (input, target) = (Input(), Target());
        var provider = (ITrainingObjectiveProvider<double>)model;
        double before = provider.EvaluateTrainingObjective(input, target);
        for (int step = 0; step < 20; step++) model.Train(input, target);
        double after = provider.EvaluateTrainingObjective(input, target);
        Assert.Equal(0, model.TrainingStage);
        Assert.True(after < before, $"objective {before} -> {after}");
    }
}
