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
/// Zuo et al. 2025, 4KAgent (arXiv 2507.07105), and the release (taco-group/4KAgent: pipeline/the4kagent_pipeline.py,
/// pipeline/profiles, utils/color_fix.py) with pyiqa's NIQE (pyiqa/archs/niqe_arch.py): the profile defaults, the plan,
/// the reflection score against pyiqa, the color fix, the quality-driven expert choice and the toolbox training (#2397).
/// </summary>
public class Upscale4KAgentPaperTests
{
    private const int Low = 8;

    private static Upscale4KAgentOptions SmallOptions(FourKAgentRestorePreference preference) => new()
    {
        Seed = 1, Upscale4K = false, ScaleFactor = 4, RestorePreference = preference,
        BasicVsr = new BasicVSROptions { MidChannels = 8, NumResBlocks = 1 },
        IconVsr = new IconVSROptions { MidChannels = 8, NumResBlocks = 1, EdvrChannels = 8, DeformableGroups = 1, EdvrExtractionBlocks = 1 },
        RealBasicVsr = new RealBasicVSROptions { MidChannels = 8, NumPropagationBlocks = 1, NumCleaningBlocks = 1 },
        RealViformer = new RealViformerOptions
        {
            NumFeatures = 8, NumBlocks = new[] { 1, 1, 1, 1 }, Heads = new[] { 1, 1, 1 }, MergeHeads = 1, SqueezeFactor = 2,
        },
    };

    private static Upscale4KAgent<double> Small(FourKAgentRestorePreference preference = FourKAgentRestorePreference.Perception) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputFrames: 2, inputDepth: 3, inputHeight: Low, inputWidth: Low, outputSize: 3 * 16 * Low * Low),
            SmallOptions(preference));

    private static Tensor<double> Pixels(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = 255.0 * random.NextDouble();
        return t;
    }

    // The two images niqe_ref.py scored with pyiqa's calculate_niqe (pyiqa 0.1.16, its niqe_modelparameters.mat).
    internal static double[] ReferenceImage(int mode, int height, int width)
    {
        var img = new double[3 * height * width];
        for (int ch = 0; ch < 3; ch++)
            for (int r = 0; r < height; r++)
                for (int c = 0; c < width; c++)
                {
                    long hash = (((long)r * 73856093) ^ ((long)c * 19349663) ^ ((long)ch * 83492791)) % 256;
                    img[(ch * height + r) * width + c] = mode == 0
                        ? 0.6 * (r + 2 * c) / (height + 2 * width) + 0.4 * hash / 255.0
                        : 0.5 + 0.3 * Math.Sin(r / 7.0 + ch) * Math.Cos(c / 11.0) + 0.08 * hash / 255.0;
                }
        return img;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheGen4KPerceptionProfile()
    {
        await Task.Yield();
        var o = new Upscale4KAgentOptions();
        Assert.True(o.Upscale4K);
        Assert.Equal(FourKAgentRestorePreference.Perception, o.RestorePreference);
        Assert.Equal((4, 0.5, 1024), (o.ScaleFactor, o.RollbackThreshold, o.ColorFixMinSide));
    }

    // get_target_factor and add_sr_tasks: 4K mode picks the smallest factor reaching 4000 pixels.
    [Fact(Timeout = 60000)]
    public async Task Plan_FollowsGetTargetFactor()
    {
        await Task.Yield();
        Assert.Equal(2, Upscale4KAgent<double>.TargetFactor(2000, true, 4));
        Assert.Equal(4, Upscale4KAgent<double>.TargetFactor(1999, true, 4));
        Assert.Equal(8, Upscale4KAgent<double>.TargetFactor(500, true, 4));
        Assert.Equal(16, Upscale4KAgent<double>.TargetFactor(249, true, 4));
        Assert.Equal(1, Upscale4KAgent<double>.TargetFactor(4000, true, 4));
        Assert.Equal(2, Upscale4KAgent<double>.TargetFactor(4000, false, 2));
        Assert.Equal(new[] { 2, 4 }, Upscale4KAgent<double>.PlanSteps(8));
        Assert.Equal(new[] { 4, 4 }, Upscale4KAgent<double>.PlanSteps(16));
        Assert.Empty(Upscale4KAgent<double>.PlanSteps(1));
    }

    // The native NIQE reproduces pyiqa 0.1.16's calculate_niqe on two 200x300 images (values printed by pyiqa itself).
    // The luma, the float32 half-scale image and every block feature agree exactly except where an AGGD α lands on a
    // grid point torch's vectorized float32 arange places one ulp (2.4e-7) away; that moves the score by about 3e-7 of
    // itself (4.5e-5 on the first image), so the comparison is to 1e-3.
    [Fact(Timeout = 120000)]
    public async Task Niqe_MatchesPyiqa()
    {
        await Task.Yield();
        Assert.Equal(171.23922389229912, Niqe.Score(ReferenceImage(0, 200, 300), 200, 300), 3);
        Assert.Equal(21.32270123459339, Niqe.Score(ReferenceImage(1, 200, 300), 200, 300), 3);
        Assert.True(double.IsNaN(Niqe.Score(ReferenceImage(1, 64, 64), 64, 64)));
    }

    // adain_color_fix: the output takes its input's per-channel mean and (unbiased) standard deviation.
    [Fact(Timeout = 60000)]
    public async Task ColorFix_TransfersTheInputsStatistics()
    {
        await Task.Yield();
        var agent = Small();
        var target = new Tensor<double>(new[] { 1, 1, 2, 2 });
        var source = new Tensor<double>(new[] { 1, 1, 1, 2 });
        (target[0], target[1], target[2], target[3]) = (0.0, 0.2, 0.4, 0.6);
        (source[0], source[1]) = (0.4, 0.6);
        var fixedUp = agent.AdainColorFix(target, source);
        // target mean 0.3, var 1/15 + 1e-5; source mean 0.5, var 0.02 + 1e-5.
        double scale = Math.Sqrt(0.02 + 1e-5) / Math.Sqrt(0.2 / 3 + 1e-5);
        for (int i = 0; i < 4; i++) Assert.Equal((0.2 * i - 0.3) * scale + 0.5, fixedUp[i], 12);
    }

    // The preference decides the toolbox, and the agent keeps one of its experts per planned step.
    [Fact(Timeout = 600000)]
    public async Task Predict_RunsTheToolboxAndKeepsOneExpertPerStep()
    {
        await Task.Yield();
        Assert.Equal(new[] { "basicvsr", "iconvsr" }, Small(FourKAgentRestorePreference.Fidelity).ExpertNames);
        var agent = Small();
        Assert.Equal(new[] { "realbasicvsr", "realviformer" }, agent.ExpertNames);
        var output = agent.Predict(Pixels(new[] { 2, 3, Low, Low }, 3));
        Assert.Equal(new[] { 2, 3, 4 * Low, 4 * Low }, output.Shape.ToArray());
        for (int i = 0; i < output.Length; i++) Assert.True(output[i] >= 0 && output[i] <= 255, $"value {output[i]} at {i}");
        var step = Assert.Single(agent.LastSteps);
        Assert.Equal(4, step.Factor);
        Assert.Contains(step.Expert, agent.ExpertNames);
        // 32x32 frames hold no 96x96 block, so the result is unscorable and the step fails into the compromise.
        Assert.False(step.Success);
    }

    // Training trains every expert in the toolbox, each on its own loss; RealBasicVSR's SPyNet stays fixed.
    [Fact(Timeout = 900000)]
    public async Task Training_UpdatesEveryExpert()
    {
        await Task.Yield();
        var agent = Small();
        var (input, target) = (Pixels(new[] { 2, 3, Low, Low }, 3), Pixels(new[] { 2, 3, 4 * Low, 4 * Low }, 4));
        agent.Predict(input);
        var before = agent.ExpertNames.ToDictionary(n => n, n => agent.ExpertParameters(n).Select(p => p.ToArray()).ToList());
        agent.Train(input, target);
        foreach (var name in agent.ExpertNames)
        {
            var after = agent.ExpertParameters(name);
            Assert.True(before[name].Where((values, k) => values.Where((v, i) => v != after[k][i]).Any()).Any(), $"{name} did not train");
        }
    }

    [Fact(Timeout = 900000)]
    public async Task Training_ReducesTheObjective()
    {
        await Task.Yield();
        var agent = Small();
        var (input, target) = (Pixels(new[] { 2, 3, Low, Low }, 3), Pixels(new[] { 2, 3, 4 * Low, 4 * Low }, 4));
        var provider = (ITrainingObjectiveProvider<double>)agent;
        double before = provider.EvaluateTrainingObjective(input, target);
        for (int step = 0; step < 10; step++) agent.Train(input, target);
        double after = provider.EvaluateTrainingObjective(input, target);
        Assert.True(after < before, $"objective {before} -> {after}");
    }
}
