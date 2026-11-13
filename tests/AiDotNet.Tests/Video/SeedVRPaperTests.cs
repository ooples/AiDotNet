using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Wang et al. 2025, SeedVR (arXiv 2501.01320), and the release (ByteDance-Seed/SeedVR, <c>configs_7b</c>, <c>models/dit</c>,
/// <c>models/video_vae_v3</c>, <c>projects/video_diffusion_sr/infer.py</c>): the configuration, the 720p-relative and shifted
/// windows, the window-local rotary embedding, the resolution-dependent timestep shift, the sampler's timesteps, the causal
/// VAE's shapes, the training scope and the pipeline (#2397).
/// </summary>
public class SeedVRPaperTests
{
    [Fact]
    public void TimestepShift_LargeFrameCount_DoesNotOverflowIntoImageBranch()
    {
        Assert.InRange(SeedVR<double>.ShiftTimestep(500, int.MaxValue, 32, 32), 999.99, 1000);
    }

    private const int Low = 8;

    private static SeedVR<double> Small() =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputFrames: 2, inputDepth: 3, inputHeight: Low, inputWidth: Low, outputSize: 3 * 16 * Low * Low),
            new SeedVROptions
            {
                Seed = 1, ModelDim = 24, NumHeads = 2, HeadDim = 12, ExpandRatio = 2, NumLayers = 2, TextDim = 8, TextLength = 3,
                NegativeTextLength = 4, VaeChannels = new[] { 8, 8, 8, 8 }, VaeLayersPerBlock = 1, VaeGroups = 4, ScaleFactor = 4,
                NumFrames = 2, SamplingSteps = 2,
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
    public async Task Defaults_AreTheReleasedConfiguration()
    {
        await Task.Yield();
        var o = new SeedVROptions();
        Assert.Equal((3072, 24, 128, 4, 36), (o.ModelDim, o.NumHeads, o.HeadDim, o.ExpandRatio, o.NumLayers));
        Assert.Equal((1e-5, false), (o.NormEpsilon, o.QkBias));
        Assert.Equal((4, 3, 3), (o.WindowFrames, o.WindowRows, o.WindowColumns));
        Assert.Equal((5120, 58, 64), (o.TextDim, o.TextLength, o.NegativeTextLength));
        Assert.Equal(new[] { 128, 256, 512, 512 }, o.VaeChannels);
        Assert.Equal((2, 32, 16, 0.9152), (o.VaeLayersPerBlock, o.VaeGroups, o.LatentChannels, o.LatentScaling));
        Assert.Equal((50, 6.5, 0.0), (o.SamplingSteps, o.GuidanceScale, o.GuidanceRescale));
        Assert.Equal((0.1, 0.25), (o.ConditionNoiseScale, o.TrainingConditionNoiseScale));
    }

    // make_720Pwindows_bysize on the 720p token grid (45 x 80 after the 2x2 patch), 21 frames, counts (4, 3, 3): windows of
    // 6 x 15 x 27; the shifted variant starts every axis half a window early.
    [Fact(Timeout = 60000)]
    public async Task Windows_AreThe720pCountsAndTheirHalfShift()
    {
        await Task.Yield();
        var regular = SeedVrWindows.Build(21, 45, 80, (4, 3, 3), shifted: false);
        Assert.Equal(new[] { (0, 6), (6, 6), (12, 6), (18, 3) }, regular.Time);
        Assert.Equal(new[] { (0, 15), (15, 15), (30, 15) }, regular.Height);
        Assert.Equal(new[] { (0, 27), (27, 27), (54, 26) }, regular.Width);

        var shifted = SeedVrWindows.Build(21, 45, 80, (4, 3, 3), shifted: true);
        Assert.Equal(new[] { (0, 3), (3, 6), (9, 6), (15, 6) }, shifted.Time);
        Assert.Equal(new[] { (0, 7), (7, 15), (22, 15), (37, 8) }, shifted.Height);
        Assert.Equal(new[] { (0, 13), (13, 27), (40, 27), (67, 13) }, shifted.Width);

        // An axis no longer than its window is not shifted (2x2 tokens against 20-token windows), while two frames against
        // one-frame windows shift into [0, 1) and [1, 2) after the empty leading window is dropped.
        var small = SeedVrWindows.Build(2, 2, 2, (4, 3, 3), shifted: true);
        Assert.Equal(new[] { (0, 1), (1, 1) }, small.Time);
        Assert.Equal(new[] { (0, 2) }, small.Height);
        Assert.Equal(new[] { (0, 2) }, small.Width);
    }

    // rotary_embedding_torch with freqs_for="pixel": head 128 gives 21 features an axis, 10 frequencies from π to 128π, so 60
    // rotated features; positions span [−1, 1] (a lone position is −1).
    [Fact(Timeout = 60000)]
    public async Task Rope_IsPixelFrequenciesAtWindowLocalPositions()
    {
        await Task.Yield();
        var (cos, sin) = SeedVrRope.Build<double>(128, 1, 1, 2);
        // Token 1 is at width position +1; the width axis's first pair sits after the time and height pairs (index 40).
        Assert.Equal(Math.Cos(Math.PI), cos[128 + 40], 12);
        Assert.Equal(Math.Cos(Math.PI), cos[128 + 41], 12);
        // Token 0's time position is −1 with the lowest frequency π.
        Assert.Equal(Math.Cos(-Math.PI), cos[0], 12);
        Assert.Equal(Math.Sin(-128.0 * Math.PI), sin[18], 9);
        // Beyond the 60 rotated features the embedding passes through.
        for (int d = 60; d < 128; d++)
        {
            Assert.Equal(1.0, cos[d]);
            Assert.Equal(0.0, sin[d]);
        }
    }

    // timestep_transform: the shift is 5 at 1280 x 720 x 145 frames (latents 37 x 90 x 160) and 1 at 256² x 37.
    [Fact(Timeout = 60000)]
    public async Task TimestepShift_FollowsThePixelCount()
    {
        await Task.Yield();
        Assert.Equal(1000.0 * 5 * 0.5 / (1 + 4 * 0.5), SeedVR<double>.ShiftTimestep(500, 37, 90, 160), 9);
        Assert.Equal(500.0, SeedVR<double>.ShiftTimestep(500, 10, 32, 32), 9);
        // One frame uses the image line: 3.2 at 1024².
        Assert.Equal(1000.0 * 3.2 * 0.5 / (1 + 2.2 * 0.5), SeedVR<double>.ShiftTimestep(500, 1, 128, 128), 9);
        var timesteps = SeedVR<double>.SamplingTimesteps(50);
        Assert.Equal(50, timesteps.Length);
        Assert.Equal(1000.0, timesteps[0]);
        Assert.Equal(20.0, timesteps[49], 9);
    }

    // video_vae_v3: 8x spatial and 4x temporal compression with the first frame alone, and back.
    [Fact(Timeout = 120000)]
    public async Task Vae_CompressesFourFramesAndEightPixels()
    {
        await Task.Yield();
        var vae = new SeedVrVideoVae<double>(new[] { 8, 8, 8, 8 }, 1, 4, 16, 0.9152);
        var video = new Tensor<double>(new[] { 1, 3, 9, 16, 16 });
        var random = new System.Random(7);
        for (int i = 0; i < video.Length; i++) video[i] = random.NextDouble() * 2 - 1;
        var latents = vae.Encode(video);
        Assert.Equal(new[] { 1, 16, 3, 2, 2 }, latents.Shape.ToArray());
        Assert.Equal(new[] { 1, 3, 9, 16, 16 }, vae.Decode(latents).Shape.ToArray());
    }

    // The release trains the DiT; the precomputed prompt embeddings stay frozen.
    [Fact(Timeout = 600000)]
    public async Task Training_UpdatesTheDiTOnly()
    {
        await Task.Yield();
        var model = Small();
        var (input, target) = (Input(), Target());
        model.Predict(input);
        var (dit, prompts) = (Snapshot(model.DiTParameters), Snapshot(model.PromptParameters));
        model.Train(input, target);
        Assert.True(Changed(dit, model.DiTParameters));
        Assert.False(Changed(prompts, model.PromptParameters));
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
    public async Task Training_ReducesTheObjective()
    {
        await Task.Yield();
        var model = Small();
        var (input, target) = (Input(), Target());
        var provider = (ITrainingObjectiveProvider<double>)model;
        double before = provider.EvaluateTrainingObjective(input, target);
        for (int step = 0; step < 10; step++) model.Train(input, target);
        double after = provider.EvaluateTrainingObjective(input, target);
        Assert.True(after < before, $"objective {before} -> {after}");
    }
}
