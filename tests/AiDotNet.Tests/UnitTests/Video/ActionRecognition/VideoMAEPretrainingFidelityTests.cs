using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.ActionRecognition;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video.ActionRecognition;

/// <summary>
/// VideoMAE masked-autoencoder pretraining against the paper (Tong et al. 2022): it must actually train,
/// its encoder must not see or process masked patches, gradients must reach every layer on the
/// reconstruction path, and the target must be per-patch normalised pixels by default.
/// </summary>
/// <remarks>
/// Layer indices follow the default stack: 0 patch-embed, 1-12 encoder blocks, 13 feature-reduce conv,
/// 14 pool, 15 classifier, 16-19 decoder blocks, 20 reconstruction head.
/// </remarks>
public class VideoMAEPretrainingFidelityTests
{
    private const int PatchSize = 16;
    private const int TubeletSize = 2;
    private const int Channels = 3;
    private const int Frames = 4;
    private const int Size = 32;

    private static NeuralNetworkArchitecture<double> VideoArch() => new(
        inputType: InputType.FourDimensional,
        taskType: NeuralNetworkTaskType.MultiClassClassification,
        inputFrames: Frames,
        inputDepth: Channels,
        inputHeight: Size,
        inputWidth: Size,
        outputSize: 4);

    private static VideoMAE<double> Model(double maskRatio = 0.5, VideoMAEOptions? options = null)
        => new(VideoArch(), numClasses: 4, numFrames: Frames, numFeatures: 16, maskRatio: maskRatio, options: options);

    private static Tensor<double> RandomClip(int seed)
    {
        var rng = new Random(seed);
        var clip = new Tensor<double>([1, Frames, Channels, Size, Size]);
        var span = clip.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = rng.NextDouble();
        }
        return clip;
    }

    private static double[] LayerParameters(VideoMAE<double> model, int index)
        => model.Layers[index].GetParameters().ToArray();

    private static bool Changed(double[] before, double[] after)
        => before.Length != after.Length || before.Where((v, i) => v != after[i]).Any();

    /// <summary>
    /// A clip whose every 16x16 patch repeats one random pattern (per frame and channel), so every masked
    /// patch has the same reconstruction target whatever the mask - a fixed clip the model can
    /// demonstrably learn in a few dozen steps, independent of which patches each step masks.
    /// </summary>
    private static Tensor<double> TiledClip(int seed)
    {
        var rng = new Random(seed);
        var tile = new double[Frames, Channels, PatchSize, PatchSize];
        for (int f = 0; f < Frames; f++)
            for (int c = 0; c < Channels; c++)
                for (int y = 0; y < PatchSize; y++)
                    for (int x = 0; x < PatchSize; x++)
                        tile[f, c, y, x] = rng.NextDouble();

        var clip = new Tensor<double>([1, Frames, Channels, Size, Size]);
        for (int f = 0; f < Frames; f++)
            for (int c = 0; c < Channels; c++)
                for (int y = 0; y < Size; y++)
                    for (int x = 0; x < Size; x++)
                        clip[0, f, c, y, x] = tile[f, c, y % PatchSize, x % PatchSize];
        return clip;
    }

    [Fact]
    public void PretrainMAE_UpdatesTheWeights_AndTheLossFallsOverStepsOnAFixedClip()
    {
        var model = Model();
        var clip = TiledClip(seed: 21);

        // Materialise the lazy decoder layers so their parameters exist to compare.
        model.PretrainMAE(clip);
        var headBefore = LayerParameters(model, VideoMAELayerLayout.ReconstructionHeadIndex);

        var losses = new List<double>();
        for (int step = 0; step < 60; step++)
        {
            losses.Add(model.PretrainMAE(clip));
        }

        // PretrainMAE used to compute the loss and return it, without back-propagating or stepping an
        // optimizer: the weights never moved and the loss only varied with the random mask.
        Assert.True(Changed(headBefore, LayerParameters(model, VideoMAELayerLayout.ReconstructionHeadIndex)),
            "the reconstruction head did not change across 60 pretraining steps");

        // Default Adam (lr 1e-3) on per-patch normalised targets: the loss starts near 1 (the target's
        // variance) and falls steadily; measured 0.994 -> 0.893 over the first 40 steps.
        double early = losses.Take(5).Average();
        double late = losses.Skip(losses.Count - 5).Average();
        Assert.True(late < 0.85 * early,
            $"pretraining did not reduce the reconstruction loss on a fixed clip: first five steps averaged {early:G6}, "
            + $"last five {late:G6} (all: {string.Join(", ", losses.Select(l => l.ToString("G4")))})");
    }

    [Fact]
    public void PretrainMAE_ReachesEveryLayerOnTheReconstructionPath_AndNothingElse()
    {
        var model = Model();
        var clip = RandomClip(seed: 22);

        model.PretrainMAE(clip); // materialise lazy layers
        var before = Enumerable.Range(0, model.Layers.Count).Select(i => LayerParameters(model, i)).ToArray();

        model.PretrainMAE(clip);

        // Patch embedding, all 12 encoder blocks, all 4 decoder blocks and the head are on the
        // reconstruction path, so one step must move each of them. The decoder used to stack a
        // Transform-based GELU after every block, which records no autodiff node: the loss reached the
        // head and nothing before it.
        var onPath = new List<int> { VideoMAELayerLayout.PatchEmbedIndex };
        onPath.AddRange(Enumerable.Range(VideoMAELayerLayout.FirstEncoderBlockIndex, VideoMAELayerLayout.EncoderBlockCount));
        onPath.AddRange(Enumerable.Range(VideoMAELayerLayout.FirstDecoderBlockIndex, VideoMAELayerLayout.DecoderBlockCount));
        onPath.Add(VideoMAELayerLayout.ReconstructionHeadIndex);

        var unchanged = onPath.Where(i => !Changed(before[i], LayerParameters(model, i))).ToList();
        Assert.True(unchanged.Count == 0,
            $"layers on the reconstruction path received no update: [{string.Join(", ", unchanged)}]");

        // The classification head is not on the pretraining path and must be left alone.
        foreach (int i in new[] { VideoMAELayerLayout.FeatureReduceIndex, VideoMAELayerLayout.ClassifierIndex })
        {
            Assert.False(Changed(before[i], LayerParameters(model, i)), $"classification-head layer {i} changed during pretraining");
        }
    }

    [Fact]
    public void EncodeVisiblePatches_LeavesEveryMaskedPatchEmpty_ThroughTheWholeEncoder()
    {
        var model = Model();
        var clip = RandomClip(seed: 23);
        int patches = Size / PatchSize;
        int numTubelets = Frames / TubeletSize;

        // Patches (0, 1) and (1, 0) masked; (0, 0) and (1, 1) visible.
        var mask = new bool[1, patches, patches];
        mask[0, 0, 1] = true;
        mask[0, 1, 0] = true;

        var features = model.EncodeVisiblePatches(clip, mask);
        Assert.Equal(new[] { numTubelets, 16, patches, patches }, features.Shape.ToArray());

        // The paper's encoder never processes masked tokens. The masked positions used to be zeroed once,
        // at the embedding, and then refilled by every 3x3 block from their visible neighbours and the
        // block bias, so the encoder processed them like any other token.
        bool anyVisibleNonZero = false;
        for (int t = 0; t < numTubelets; t++)
        {
            for (int c = 0; c < 16; c++)
            {
                for (int ph = 0; ph < patches; ph++)
                {
                    for (int pw = 0; pw < patches; pw++)
                    {
                        double v = features[t, c, ph, pw];
                        if (mask[0, ph, pw])
                        {
                            Assert.True(v == 0.0, $"masked patch ({ph}, {pw}) of tubelet {t} channel {c} holds {v}");
                        }
                        else if (v != 0.0)
                        {
                            anyVisibleNonZero = true;
                        }
                    }
                }
            }
        }

        Assert.True(anyVisibleNonZero, "every visible feature was zero, so the check above proves nothing");
    }

    [Fact]
    public void PretrainMAE_WithEveryPatchMasked_TeachesTheEncoderNothing()
    {
        // With every patch masked the encoder sees no content at all, so no gradient may reach the patch
        // embedding or any encoder block. The masked embeddings used to be zeroed by an in-place write the
        // tape never saw, so backward still carried gradient through them into the patch-embedding
        // weights, weighted by the masked patches' own pixels - the content the model is meant to predict
        // without seeing. The head, which learns to predict from nothing, must still move.
        var model = Model(maskRatio: 1.0);
        var clip = RandomClip(seed: 24);

        model.PretrainMAE(clip); // materialise lazy layers
        var before = Enumerable.Range(0, model.Layers.Count).Select(i => LayerParameters(model, i)).ToArray();

        model.PretrainMAE(clip);

        for (int i = VideoMAELayerLayout.PatchEmbedIndex; i < VideoMAELayerLayout.EncoderEndIndex; i++)
        {
            Assert.False(Changed(before[i], LayerParameters(model, i)), $"encoder layer {i} learned from fully masked input");
        }

        Assert.True(Changed(before[VideoMAELayerLayout.ReconstructionHeadIndex],
            LayerParameters(model, VideoMAELayerLayout.ReconstructionHeadIndex)), "the reconstruction head did not train");
    }

    /// <summary>
    /// The paper's per-patch normalised target, computed independently of the model: for each channel of
    /// each tubelet patch, (x - mean) / (unbiased std + 1e-6) over its tubeletSize x P x P pixels, laid out
    /// like the head's output.
    /// </summary>
    private static Tensor<double> NormalisedTarget(Tensor<double> clip)
    {
        int numTubelets = Frames / TubeletSize;
        int patches = Size / PatchSize;
        int patchDim = Channels * TubeletSize * PatchSize * PatchSize;
        var target = new Tensor<double>([numTubelets, patchDim, patches, patches]);

        for (int t = 0; t < numTubelets; t++)
            for (int ph = 0; ph < patches; ph++)
                for (int pw = 0; pw < patches; pw++)
                    for (int c = 0; c < Channels; c++)
                    {
                        var values = new List<double>();
                        for (int ts = 0; ts < TubeletSize; ts++)
                            for (int y = 0; y < PatchSize; y++)
                                for (int x = 0; x < PatchSize; x++)
                                    values.Add(clip[0, t * TubeletSize + ts, c, ph * PatchSize + y, pw * PatchSize + x]);

                        double mean = values.Average();
                        double std = Math.Sqrt(values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1));

                        for (int ts = 0; ts < TubeletSize; ts++)
                            for (int y = 0; y < PatchSize; y++)
                                for (int x = 0; x < PatchSize; x++)
                                {
                                    int channel = (((ts * Channels) + c) * PatchSize + y) * PatchSize + x;
                                    double v = clip[0, t * TubeletSize + ts, c, ph * PatchSize + y, pw * PatchSize + x];
                                    target[t, channel, ph, pw] = (v - mean) / (std + 1e-6);
                                }
                    }

        return target;
    }

    [Fact]
    public void ComputeReconstructionLoss_ByDefault_TargetsPerPatchNormalisedPixels()
    {
        var model = Model();
        var clip = RandomClip(seed: 25);
        int patches = Size / PatchSize;

        var mask = new bool[1, patches, patches];
        mask[0, 1, 1] = true;

        // The paper (and its reference code's default normlize_target=True) reconstructs each patch's
        // pixels normalised by that patch's own mean and standard deviation. The target used to be the
        // raw pixels, with no option to normalise.
        var normalised = NormalisedTarget(clip);
        Assert.Equal(0.0, model.ComputeReconstructionLoss(normalised, clip, mask), 10);

        // Raw pixels are therefore NOT a perfect prediction under the default.
        var raw = new Tensor<double>(normalised.Shape.ToArray());
        int numTubelets = Frames / TubeletSize;
        for (int t = 0; t < numTubelets; t++)
            for (int ts = 0; ts < TubeletSize; ts++)
                for (int c = 0; c < Channels; c++)
                    for (int y = 0; y < PatchSize; y++)
                        for (int x = 0; x < PatchSize; x++)
                            for (int ph = 0; ph < patches; ph++)
                                for (int pw = 0; pw < patches; pw++)
                                {
                                    int channel = (((ts * Channels) + c) * PatchSize + y) * PatchSize + x;
                                    raw[t, channel, ph, pw] = clip[0, t * TubeletSize + ts, c, ph * PatchSize + y, pw * PatchSize + x];
                                }

        Assert.True(model.ComputeReconstructionLoss(raw, clip, mask) > 0.1,
            "raw pixels scored as a perfect prediction, so the default target is not normalised");

        // Opting out restores the raw-pixel target exactly.
        var rawModel = Model(options: new VideoMAEOptions { NormalizeTarget = false });
        Assert.Equal(0.0, rawModel.ComputeReconstructionLoss(raw, clip, mask), 12);
    }

    [Fact]
    public void Options_NormalizeTarget_DefaultsToThePapersChoice()
    {
        Assert.True(new VideoMAEOptions().NormalizeTarget);
    }
}
