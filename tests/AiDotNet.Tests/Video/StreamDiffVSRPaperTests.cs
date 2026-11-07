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
/// Shiu et al. 2025, Stream-DiffVSR (arXiv 2512.23709), and the release (jamichss/Stream-DiffVSR: <c>inference.py</c>,
/// <c>pipeline/stream_diffvsr_pipeline.py</c>, <c>temporal_autoencoder</c>, <c>train/train_artg.py</c>; the Hub configs of
/// Jamichsu/Stream-DiffVSR): the configuration, the trailing DDIM schedule, the release's flow warp, the temporal tiny
/// autoencoder, the ControlNet's initialization, the ARTG stage's training scope and the pipeline (#2397).
/// </summary>
public class StreamDiffVSRPaperTests
{
    private const int Low = 8;

    private static StreamDiffVSROptions SmallOptions() => new()
    {
        Seed = 1, BlockChannels = new[] { 32, 32, 32, 32 }, LayersPerBlock = 1, AttentionHeads = 2, CrossAttentionDim = 8,
        PromptLength = 3, ConditioningChannels = new[] { 8, 8, 8 }, VaeChannels = 8, EncoderBlocks = new[] { 1, 1, 1 },
        DecoderBlocks = new[] { 1, 1, 1 }, FlowIterations = 1, WarmupSteps = 0,
    };

    private static StreamDiffVSR<double> Small() =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputFrames: 2, inputDepth: 3, inputHeight: Low, inputWidth: Low, outputSize: 3 * 16 * Low * Low),
            SmallOptions());

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
        var o = new StreamDiffVSROptions();
        Assert.Equal(new[] { 256, 512, 512, 1024 }, o.BlockChannels);
        Assert.Equal(new[] { true, true, true, false }, o.OnlyCrossAttention);
        Assert.Equal((2, 8, 1024, 77), (o.LayersPerBlock, o.AttentionHeads, o.CrossAttentionDim, o.PromptLength));
        Assert.Equal(new[] { 64, 128, 256 }, o.ConditioningChannels);
        Assert.Equal((64, 4, 4, 12), (o.VaeChannels, o.ScaleFactor, o.SamplingSteps, o.FlowIterations));
        Assert.Equal(new[] { 1, 3, 3 }, o.EncoderBlocks);
        Assert.Equal(new[] { 3, 3, 1 }, o.DecoderBlocks);
        Assert.Equal((1000, 1e-4, 0.02), (o.NumTrainTimesteps, o.BetaStart, o.BetaEnd));
        Assert.Equal((5e-5, 0.9, 0.999, 1e-8, 1e-2, 1000), (o.LearningRate, o.AdamBeta1, o.AdamBeta2, o.AdamEpsilon, o.WeightDecay, o.WarmupSteps));
        Assert.Equal((1.0, 0.1, 0.025, 5), (o.LatentMseWeight, o.LpipsWeight, o.GanWeight, o.DiscriminatorInterval));
    }

    // The scheduler config: scaled_linear β from 1e-4 to 0.02, trailing spacing ("[999, 749, 499, 249]" in train_artg.py).
    [Fact(Timeout = 60000)]
    public async Task Schedule_IsTrailingDdimOnScaledLinear()
    {
        await Task.Yield();
        Assert.Equal(new[] { 999, 749, 499, 249 }, StreamDiffVSR<double>.TrailingTimesteps(1000, 4));
        Assert.Equal(new[] { 999, 666, 332 }, StreamDiffVSR<double>.TrailingTimesteps(1000, 3));
        var abar = StreamDiffVSR<double>.AlphasCumprod(new StreamDiffVSROptions());
        Assert.Equal(0.9999, abar[0], 12);
        Assert.Equal(0.8215213423020223, abar[249], 10);
        Assert.Equal(0.0007334124595808159, abar[999], 12);
    }

    // util/flow_utils.flow_warp normalizes by W − 1 but samples with align_corners=False: under a zero flow the centre
    // pixel of a 3-wide frame reads itself, and the last pixel reads half a pixel outside (zero padding), giving half.
    [Fact(Timeout = 60000)]
    public async Task FlowWarp_KeepsTheReleasesAlignCornersMismatch()
    {
        await Task.Yield();
        var image = new Tensor<double>(new[] { 1, 1, 3, 3 });
        image.Fill(1.0);
        var warped = StreamDiffVsrOps.FlowWarp(AiDotNetEngine.Current, image, new Tensor<double>(new[] { 1, 2, 3, 3 }));
        Assert.Equal(1.0, warped[0, 0, 1, 1], 12);
        Assert.Equal(0.5, warped[0, 0, 1, 2], 12);
        Assert.Equal(0.25, warped[0, 0, 2, 2], 12);
    }

    // TAESD-x4 with the TPM: 4x compression to four channels; one feature map per encoder block; the decoder's temporal
    // blocks change the frame when the warped previous frame's features are given.
    [Fact(Timeout = 120000)]
    public async Task TemporalAutoencoder_CompressesFourTimesAndFusesThePreviousFrame()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var group = new LazyModuleGroup<double>(MgldInit.Default);
        var encoder = new StreamDiffVsrTinyEncoder<double>(group, 8, new[] { 1, 3, 3 }, 4);
        var decoder = new StreamDiffVsrTemporalDecoder<double>(group, 8, new[] { 3, 3, 1 }, 4);
        var frame = new Tensor<double>(new[] { 1, 3, 32, 32 });
        var random = new System.Random(5);
        for (int i = 0; i < frame.Length; i++) frame[i] = 2 * random.NextDouble() - 1;
        var features = new List<Tensor<double>>();
        var latents = encoder.Forward(engine, frame, features);
        Assert.Equal(new[] { 1, 4, 8, 8 }, latents.Shape.ToArray());
        Assert.Equal(7, features.Count);
        var plain = decoder.Forward(engine, latents);
        var fused = decoder.Forward(engine, latents, features);
        Assert.Equal(new[] { 1, 3, 32, 32 }, plain.Shape.ToArray());
        Assert.Contains(Enumerable.Range(0, plain.Length), i => Math.Abs(plain[i] - fused[i]) > 1e-9);
    }

    // ControlNetModel.from_unet: the encoder copied from the UNet and every zero convolution zero, so the untrained ARTG
    // module leaves the UNet's prediction unchanged.
    [Fact(Timeout = 120000)]
    public async Task ControlNet_StartsAsTheUNetsEncoderAndAddsNothing()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var widths = new[] { 32, 32 };
        var cross = new[] { true, false };
        var unetGroup = new LazyModuleGroup<double>(MgldInit.Default);
        var controlGroup = new LazyModuleGroup<double>(MgldInit.Default);
        var unet = new StreamDiffVsrUNet<double>(unetGroup, 7, 4, widths, 1, cross, 2, 8);
        var control = new StreamDiffVsrControlNet<double>(controlGroup, 7, widths, 1, cross, 2, 8, new[] { 8, 8, 8 });
        control.DownGroup.SetParameters(unet.Down.Group.GetParameters());
        Assert.Equal(unet.Down.Group.GetParameters().ToArray(), control.DownGroup.GetParameters().ToArray());
        var x = new Tensor<double>(new[] { 1, 7, 4, 4 });
        var condition = new Tensor<double>(new[] { 1, 3, 16, 16 });
        var context = new Tensor<double>(new[] { 1, 3, 8 });
        var random = new System.Random(9);
        for (int i = 0; i < x.Length; i++) x[i] = random.NextDouble();
        for (int i = 0; i < condition.Length; i++) condition[i] = random.NextDouble();
        for (int i = 0; i < context.Length; i++) context[i] = random.NextDouble();
        var (residuals, mid) = control.Forward(engine, x, new[] { 500.0 }, context, condition, 1.0);
        Assert.Equal(unet.Down.ResidualChannels.Count, residuals.Count);
        Assert.All(residuals.Append(mid), r => Assert.All(Enumerable.Range(0, r.Length), i => Assert.Equal(0.0, r[i])));
        var bare = unet.Forward(engine, x, new[] { 500.0 }, context);
        var guided = unet.Forward(engine, x, new[] { 500.0 }, context, residuals, mid);
        Assert.Equal(bare.ToArray(), guided.ToArray());
    }

    // train_artg.py trains the ControlNet (and the discriminator); the UNet, the autoencoder and RAFT stay frozen.
    [Fact(Timeout = 900000)]
    public async Task Training_UpdatesTheControlNetAndDiscriminatorOnly()
    {
        await Task.Yield();
        var model = Small();
        var (input, target) = (Input(), Target());
        model.Predict(input);
        var control = Snapshot(model.ControlNetParameters);
        var discriminator = Snapshot(model.DiscriminatorParameters);
        var (unet, vae) = (Snapshot(model.UNetParameters), Snapshot(model.AutoencoderParameters));
        model.Train(input, target);
        Assert.True(Changed(control, model.ControlNetParameters));
        Assert.True(Changed(discriminator, model.DiscriminatorParameters));
        Assert.False(Changed(unet, model.UNetParameters));
        Assert.False(Changed(vae, model.AutoencoderParameters));
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
