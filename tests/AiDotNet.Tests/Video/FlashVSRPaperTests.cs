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
/// Zhuang et al. 2025, FlashVSR (arXiv 2510.12747), Sec. 3-4, and the release (OpenImagingLab/FlashVSR) on Wan 2.1-1.3B:
/// the configuration, the flow-matching schedule, the 3D rotary embedding, the streamed block-sparse attention, the four
/// training stages and the one-step pipeline (#2397).
/// </summary>
public class FlashVSRPaperTests
{
    private const int Low = 8;

    private static FlashVSR<double> Small(int decoderIterations = 20000, int teacherIterations = 20000, int sparseIterations = 10000) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputFrames: 2, inputDepth: 3, inputHeight: Low, inputWidth: Low, outputSize: 3 * 16 * Low * Low),
            new FlashVSROptions
            {
                Seed = 1, ModelDim = 24, FfnDim = 32, NumHeads = 2, NumLayers = 1, FrequencyDim = 8, TextDim = 8, TextLength = 4,
                LoraRank = 4, LqHiddenDim1 = 8, LqHiddenDim2 = 8, DecoderChannels = new[] { 4, 4, 4, 4 }, VaeDim = 4,
                VaeDimMultipliers = new[] { 1, 1, 1, 1 }, VaeResidualBlocks = 1, ScaleFactor = 4, NumFrames = 2, LpipsWeight = 0.0,
                DecoderIterations = decoderIterations, TeacherIterations = teacherIterations, SparseIterations = sparseIterations,
            });

    private static Tensor<double> Pixels(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = 255.0 * random.NextDouble();
        return t;
    }

    private static Tensor<double> Input() => Pixels(new[] { 2, 3, Low, Low }, 3);

    private static Tensor<double> Target() => Pixels(new[] { 2, 3, 4 * Low, 4 * Low }, 4);

    private static List<double[]> Snapshot(IReadOnlyList<Tensor<double>> parameters)
        => parameters.Select(p => Enumerable.Range(0, p.Length).Select(i => p[i]).ToArray()).ToList();

    private static bool Changed(List<double[]> before, IReadOnlyList<Tensor<double>> after)
        => before.Select((values, k) => values.Where((v, i) => v != after[k][i]).Any()).Any(changed => changed);

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePaperConfiguration()
    {
        await Task.Yield();
        var o = new FlashVSROptions();
        Assert.Equal((1536, 8960, 12, 30, 256), (o.ModelDim, o.FfnDim, o.NumHeads, o.NumLayers, o.FrequencyDim));
        Assert.Equal((4096, 512, 1e-6, 384), (o.TextDim, o.TextLength, o.NormEpsilon, o.LoraRank));
        Assert.Equal((2048, 3072, 1), (o.LqHiddenDim1, o.LqHiddenDim2, o.LqProjectionLayers));
        Assert.Equal((2.0, 3, 11), (o.SparseRatio, o.KvCacheChunks, o.LocalRange));
        Assert.Equal(new[] { 512, 256, 128, 128 }, o.DecoderChannels);
        Assert.Equal((96, 2), (o.VaeDim, o.VaeResidualBlocks));
        Assert.Equal(new[] { 1, 2, 4, 4 }, o.VaeDimMultipliers);
        Assert.Equal((4, 89, true), (o.ScaleFactor, o.NumFrames, o.ColorFix));
        Assert.Equal((1e-5, 0.01, 2.0, 5.0, 1000), (o.LearningRate, o.WeightDecay, o.LpipsWeight, o.SigmaShift, o.NumTrainTimesteps));
    }

    // DiffSynth's FlowMatchScheduler(shift = 5, extra_one_step = True): σ starts at 1 and each σ is shifted to 5σ / (1 + 4σ).
    [Fact(Timeout = 60000)]
    public async Task Sigmas_AreWansShiftedFlowSchedule()
    {
        await Task.Yield();
        var sigmas = FlashVSR<double>.Sigmas(new FlashVSROptions());
        Assert.Equal(1000, sigmas.Length);
        Assert.Equal(1.0, sigmas[0], 12);
        Assert.Equal(5 * 0.5 / (1 + 4 * 0.5), sigmas[500], 12);
        Assert.Equal(5 * 0.001 / (1 + 4 * 0.001), sigmas[999], 12);
    }

    // precompute_freqs_cis_3d: of a 12-wide head, 12 - 2·4 = 4 for time, then 4 for height and 4 for width; a token's
    // time part rotates by its frame index and leaves the spatial parts at their own positions.
    [Fact(Timeout = 60000)]
    public async Task Rope_SplitsTheHeadAcrossTimeHeightAndWidth()
    {
        await Task.Yield();
        var (cos, sin) = WanRope.Build<double>(12, 2, 1, 1);
        // Token 1 is frame 1: its first pair turns by 1 radian and its second by 10000^(-2/4) = 0.01 radians.
        Assert.Equal(Math.Cos(1.0), cos[12 + 0], 12);
        Assert.Equal(Math.Sin(1.0), sin[12 + 1], 12);
        Assert.Equal(Math.Cos(0.01), cos[12 + 2], 12);
        // Its height and width parts sit at position 0.
        for (int d = 4; d < 12; d++) Assert.Equal(1.0, cos[12 + d], 12);
    }

    // The streamed attention as one mask: with one 8x8-token block a frame pair, the first three chunks see each other but
    // no later chunk, chunk 4 sees chunks 1 to 4 (the three-chunk cache), and each chunk keeps at most all but one of its
    // visible blocks (topk ≤ n − 1).
    [Fact(Timeout = 60000)]
    public async Task SparseMask_FollowsTheStreamedCacheAndTopK()
    {
        await Task.Yield();
        int frames = 10, side = 8, tokens = frames * side * side, hd = 4;
        var random = new System.Random(5);
        var q = new Tensor<double>(new[] { 1, tokens, hd });
        var k = new Tensor<double>(new[] { 1, tokens, hd });
        for (int i = 0; i < q.Length; i++)
        {
            q[i] = random.NextDouble() - 0.5;
            k[i] = random.NextDouble() - 0.5;
        }
        var mask = WanSparseMask.Build(q, k, new WanSparsePlan(frames, side, side, 1000.0, 3, 11));
        int perChunk = 2 * side * side;
        bool Sees(int queryChunk, int keyChunk) => mask[queryChunk * perChunk * tokens + keyChunk * perChunk] > 0;
        int Kept(int queryChunk) => Enumerable.Range(0, frames / 2).Count(c => Sees(queryChunk, c));
        for (int i = 0; i < 3; i++)
        {
            Assert.False(Sees(i, 3));
            Assert.False(Sees(i, 4));
            Assert.Equal(2, Kept(i));
        }
        Assert.False(Sees(4, 0));
        Assert.Equal(3, Kept(4));
        Assert.Equal(3, Kept(3));
    }

    // Sec. 3-4: the tiny decoder alone, then full-attention flow matching, then sparse causal flow matching with the
    // previous network frozen as the teacher, then distillation with the critic starting from the teacher.
    [Fact(Timeout = 600000)]
    public async Task Training_RunsThePapersFourStagesInTurn()
    {
        await Task.Yield();
        var model = Small(decoderIterations: 1, teacherIterations: 1, sparseIterations: 1);
        var (input, target) = (Input(), Target());
        model.Predict(input);

        Assert.Equal(0, model.TrainingStage);
        var (decoder, generator, teacher, critic) = (Snapshot(model.DecoderParameters), Snapshot(model.GeneratorParameters),
            Snapshot(model.TeacherParameters), Snapshot(model.CriticParameters));
        model.Train(input, target);
        Assert.True(Changed(decoder, model.DecoderParameters));
        Assert.False(Changed(generator, model.GeneratorParameters));

        Assert.Equal(1, model.TrainingStage);
        decoder = Snapshot(model.DecoderParameters);
        model.Train(input, target);
        Assert.True(Changed(generator, model.GeneratorParameters));
        Assert.False(Changed(decoder, model.DecoderParameters));
        Assert.False(Changed(teacher, model.TeacherParameters));
        Assert.False(Changed(critic, model.CriticParameters));

        Assert.Equal(2, model.TrainingStage);
        var trainedTeacher = Snapshot(model.GeneratorParameters);
        model.Train(input, target);
        Assert.False(Changed(trainedTeacher, model.TeacherParameters));
        Assert.True(Changed(trainedTeacher, model.GeneratorParameters));
        Assert.False(Changed(critic, model.CriticParameters));

        Assert.Equal(3, model.TrainingStage);
        generator = Snapshot(model.GeneratorParameters);
        model.Train(input, target);
        Assert.True(Changed(generator, model.GeneratorParameters));
        Assert.True(Changed(trainedTeacher, model.CriticParameters));
        Assert.False(Changed(trainedTeacher, model.TeacherParameters));
        Assert.False(Changed(decoder, model.DecoderParameters));
    }

    [Fact(Timeout = 300000)]
    public async Task Predict_ReturnsTheUpscaledClip()
    {
        await Task.Yield();
        var output = Small().Predict(Input());
        Assert.Equal(new[] { 2, 3, 4 * Low, 4 * Low }, output.Shape.ToArray());
        for (int i = 0; i < output.Length; i++) Assert.True(output[i] >= 0 && output[i] <= 255, $"value {output[i]} at {i}");
    }

    [Fact(Timeout = 600000)]
    public async Task Training_ReducesTheDecoderObjective()
    {
        await Task.Yield();
        var model = Small();
        var (input, target) = (Input(), Target());
        var provider = (ITrainingObjectiveProvider<double>)model;
        double before = provider.EvaluateTrainingObjective(input, target);
        for (int step = 0; step < 10; step++) model.Train(input, target);
        double after = provider.EvaluateTrainingObjective(input, target);
        Assert.Equal(0, model.TrainingStage);
        Assert.True(after < before, $"objective {before} -> {after}");
    }
}
