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
/// Yang et al. 2024, MGLD-VSR (arXiv 2312.00853), and the release (IanYeung/MGLD-VSR) on Stable Diffusion 2.1: the
/// configuration, the DDPM schedule and its respacing, the flow utilities of the motion guidance and the warping loss,
/// the two training stages and the sampling pipeline (#2397).
/// </summary>
public class MGLDVSRPaperTests
{
    private const int Low = 8;

    private static MGLDVSR<double> Small(int diffusionIterations = 100000, int discriminatorStart = 501, double learningRate = 5e-5) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputFrames: 2, inputDepth: 3, inputHeight: Low, inputWidth: Low, outputSize: 3 * 16 * Low * Low),
            new MGLDVSROptions
            {
                Seed = 1, UNetChannels = 32, UNetMultipliers = new[] { 1, 1 }, UNetResBlocks = 1, AttentionLevels = 1, HeadChannels = 32,
                ContextDim = 8, ContextLength = 4, NumFrames = 2, StructureChannels = 32, StructureMultipliers = new[] { 1, 1 },
                StructureResBlocks = 1, StructureHeads = 1, VaeChannels = 32, VaeMultipliers = new[] { 1, 1, 1 }, VaeResBlocks = 1,
                FuseBlocks = 1, SamplingSteps = 2, MinimumSide = 0, PerceptualWeight = 0.0, ScaleFactor = 4,
                DiffusionIterations = diffusionIterations, DiscriminatorStart = discriminatorStart, LearningRate = learningRate,
                DecoderLearningRate = learningRate,
            });

    private static Tensor<double> Pixels(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var random = new Random(seed);
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
    public async Task Defaults_AreTheReleaseConfiguration()
    {
        await Task.Yield();
        var o = new MGLDVSROptions();
        Assert.Equal((320, 2, 3, 64, 1024, 77, 5), (o.UNetChannels, o.UNetResBlocks, o.AttentionLevels, o.HeadChannels, o.ContextDim, o.ContextLength, o.NumFrames));
        Assert.Equal(new[] { 1, 2, 4, 4 }, o.UNetMultipliers);
        Assert.Equal((256, 2, 4), (o.StructureChannels, o.StructureResBlocks, o.StructureHeads));
        Assert.Equal(new[] { 1, 1, 2, 2 }, o.StructureMultipliers);
        Assert.Equal((128, 2, 2, 0.18215), (o.VaeChannels, o.VaeResBlocks, o.FuseBlocks, o.LatentScale));
        Assert.Equal(new[] { 1, 2, 4, 4 }, o.VaeMultipliers);
        Assert.Equal((1000, 0.00085, 0.012, 1000, -10.0, 10), (o.NumTrainTimesteps, o.BetaStart, o.BetaEnd, o.SamplingSteps, o.GuidanceScale, o.FlowIterations));
        Assert.Equal((0.5, 1.0, 4, 512, true), (o.FusionWeight, o.TrainingFusionWeight, o.ScaleFactor, o.MinimumSide, o.ColorFix));
        Assert.Equal((5e-5, 5e-5, 0.5, 0.5, 0.5, 3.0, 0.025, 501),
            (o.LearningRate, o.DecoderLearningRate, o.PerceptualWeight, o.DifferenceWeight, o.TemporalWeight, o.EdgeWeight, o.DiscriminatorWeight, o.DiscriminatorStart));
    }

    // LDM's "linear" schedule spaces √β linearly, so β runs 0.00085 to 0.012 and ᾱ is their running product of 1 − β.
    [Fact(Timeout = 60000)]
    public async Task Schedule_IsLdmsLinearScheduleRespacedEvenly()
    {
        await Task.Yield();
        var abar = MGLDVSR<double>.AlphasCumprod(new MGLDVSROptions());
        Assert.Equal(1000, abar.Length);
        Assert.Equal(1.0 - 0.00085, abar[0], 12);
        Assert.Equal(abar[998] * (1.0 - 0.012), abar[999], 12);
        Assert.Equal(new[] { 0, 333, 666, 999 }, MGLDVSR<double>.SpacedTimesteps(1000, 4));
        Assert.Equal(new[] { 0 }, MGLDVSR<double>.SpacedTimesteps(1000, 1));
        Assert.Equal(Enumerable.Range(0, 1000).ToArray(), MGLDVSR<double>.SpacedTimesteps(1000, 1000));
    }

    // forward_backward_consistency_check: a flow and its exact reverse cancel, so every pixel is visible but those whose
    // warped reverse falls outside the frame (zero padding leaves the flow uncancelled there).
    [Fact(Timeout = 60000)]
    public async Task Occlusion_MarksOnlyWherePixelsLeaveTheFrame()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var forward = new Tensor<double>(new[] { 1, 2, 4, 4 });
        var backward = new Tensor<double>(new[] { 1, 2, 4, 4 });
        for (int i = 0; i < 16; i++)
        {
            forward[i] = 1.0;
            backward[i] = -1.0;
        }
        var (occludedForward, occludedBackward) = MgldFlow.Occlusion(engine, forward, backward);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
            {
                Assert.Equal(x == 3 ? 1.0 : 0.0, occludedForward[0, 0, y, x]);
                Assert.Equal(x == 0 ? 1.0 : 0.0, occludedBackward[0, 0, y, x]);
            }
    }

    // Eq. 4: content moving one pixel right between frames, with the matching flows and nothing occluded inside, warps
    // onto itself, so only the frame border (zero padding) is left; static frames with zero flow cost nothing.
    [Fact(Timeout = 60000)]
    public async Task WarpingLoss_IsZeroForConsistentMotion()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var frames = new Tensor<double>(new[] { 2, 1, 4, 6 });
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 6; x++)
            {
                frames[0, 0, y, x] = x;
                frames[1, 0, y, x] = x - 1;
            }
        var toNext = new Tensor<double>(new[] { 1, 2, 4, 6 });
        var toPrevious = new Tensor<double>(new[] { 1, 2, 4, 6 });
        var visibleNext = new Tensor<double>(new[] { 1, 1, 4, 6 });
        var visiblePrevious = new Tensor<double>(new[] { 1, 1, 4, 6 });
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 6; x++)
            {
                toNext[0, 0, y, x] = 1.0;
                toPrevious[0, 0, y, x] = -1.0;
                visibleNext[0, 0, y, x] = x < 5 ? 1.0 : 0.0;
                visiblePrevious[0, 0, y, x] = x > 0 ? 1.0 : 0.0;
            }
        var loss = MgldFlow.WarpingLoss(engine, frames, new[] { toNext }, new[] { toPrevious }, new[] { visibleNext }, new[] { visiblePrevious });
        Assert.Equal(0.0, loss[0], 10);
        var still = new Tensor<double>(new[] { 1, 2, 4, 6 });
        var all = new Tensor<double>(new[] { 1, 1, 4, 6 });
        for (int i = 0; i < all.Length; i++) all[i] = 1.0;
        var none = MgldFlow.WarpingLoss(engine, MgldOps.PerFrame(engine, engine.TensorNarrow(frames, 0, 0, 1), 2),
            new[] { still }, new[] { still }, new[] { all }, new[] { all });
        Assert.Equal(0.0, none[0], 10);
    }

    // The motion guidance steps along the warping loss's gradient while sampling, where no tape records, so it uses a
    // closed form; it has to be the gradient the tape takes of the loss itself (fractional flows, partial masks).
    [Fact(Timeout = 60000)]
    public async Task WarpingLossGradient_MatchesTheTape()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var random = new Random(11);
        Tensor<double> Fill(int[] shape, Func<double> value)
        {
            var t = new Tensor<double>(shape);
            for (int i = 0; i < t.Length; i++) t[i] = value();
            return t;
        }
        var frames = Fill(new[] { 3, 2, 5, 6 }, () => random.NextDouble() * 2 - 1);
        var toNext = Enumerable.Range(0, 2).Select(_ => Fill(new[] { 1, 2, 5, 6 }, () => random.NextDouble() * 3 - 1.5)).ToList();
        var toPrevious = Enumerable.Range(0, 2).Select(_ => Fill(new[] { 1, 2, 5, 6 }, () => random.NextDouble() * 3 - 1.5)).ToList();
        var visibleNext = Enumerable.Range(0, 2).Select(_ => Fill(new[] { 1, 1, 5, 6 }, () => random.NextDouble() < 0.8 ? 1.0 : 0.0)).ToList();
        var visiblePrevious = Enumerable.Range(0, 2).Select(_ => Fill(new[] { 1, 1, 5, 6 }, () => random.NextDouble() < 0.8 ? 1.0 : 0.0)).ToList();

        Tensor<double> expected;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            var loss = MgldFlow.WarpingLoss(engine, frames, toNext, toPrevious, visibleNext, visiblePrevious);
            expected = tape.ComputeGradients(loss, new[] { frames })[frames];
        }
        var actual = MgldFlow.WarpingLossGradient(frames, toNext, toPrevious, visibleNext, visiblePrevious);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 10);
    }

    // The release trains SPADE, the temporal modules and the structure encoder; Stable Diffusion's UNet and the
    // decoder stay as they are until the decoder stage, which trains the CFW modules and the temporal convolutions.
    [Fact(Timeout = 900000)]
    public async Task Training_RunsTheDiffusionThenTheDecoderStage()
    {
        await Task.Yield();
        var model = Small(diffusionIterations: 1, discriminatorStart: 0);
        var (input, target) = (Input(), Target());
        model.Predict(input);
        Assert.Equal(0, model.TrainingStage);
        var (adapters, unet, decoder, discriminator) = (Snapshot(model.DiffusionParameters), Snapshot(model.FrozenUNetParameters),
            Snapshot(model.DecoderParameters), Snapshot(model.DiscriminatorParameters));
        model.Train(input, target);
        Assert.True(Changed(adapters, model.DiffusionParameters));
        Assert.False(Changed(unet, model.FrozenUNetParameters));
        Assert.False(Changed(decoder, model.DecoderParameters));

        Assert.Equal(1, model.TrainingStage);
        adapters = Snapshot(model.DiffusionParameters);
        model.Train(input, target);
        Assert.True(Changed(decoder, model.DecoderParameters));
        Assert.True(Changed(discriminator, model.DiscriminatorParameters));
        Assert.False(Changed(adapters, model.DiffusionParameters));
        Assert.False(Changed(unet, model.FrozenUNetParameters));
    }

    [Fact(Timeout = 600000)]
    public async Task Predict_ReturnsTheUpscaledClip()
    {
        await Task.Yield();
        var output = Small().Predict(Input());
        Assert.Equal(new[] { 2, 3, 4 * Low, 4 * Low }, output.Shape.ToArray());
        for (int i = 0; i < output.Length; i++) Assert.True(output[i] >= 0 && output[i] <= 255, $"value {output[i]} at {i}");
    }

    [Fact(Timeout = 900000)]
    public async Task Training_ReducesTheDiffusionObjective()
    {
        await Task.Yield();
        var model = Small(learningRate: 1e-3);
        var (input, target) = (Input(), Target());
        var provider = (ITrainingObjectiveProvider<double>)model;
        double before = provider.EvaluateTrainingObjective(input, target);
        for (int step = 0; step < 10; step++) model.Train(input, target);
        double after = provider.EvaluateTrainingObjective(input, target);
        Assert.Equal(0, model.TrainingStage);
        Assert.True(after < before, $"objective {before} -> {after}");
    }
}
