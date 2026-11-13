using System;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Denoising;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.FrameInterpolation;
using AiDotNet.Video.Inpainting;
using AiDotNet.Video.Matting;
using AiDotNet.Video.Motion;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video;

/// <summary>
/// Reference behaviour of the video models' flow warps, masks, noise estimate and activations against the
/// published implementations.
/// </summary>
/// <remarks>
/// <para>
/// Every warp is a backward warp, <c>out(x, y) = image(x + dx, y + dy)</c>, sampled bilinearly with the
/// <c>grid_sample(align_corners=True)</c> pixel mapping the reference code uses (RAFT and FlowFormer
/// <c>bilinear_sampler</c>, mmedit/E2FGVI/ProPainter <c>flow_warp</c>, RIFE <c>warplayer.warp</c>, FILM
/// <c>dense_image_warp</c>). The image is affine, which bilinear interpolation reproduces exactly, so an integer
/// shift has an exact expected value and the half-pixel error of a mismatched coordinate mapping cannot hide.
/// Out-of-image taps are zero for the <c>padding_mode="zeros"</c> references and the edge pixel for the
/// border ones.
/// </para>
/// <para>
/// Each test fails on the code before this change: the warps sampled with the align_corners=False shim or the
/// wrong padding, the noise estimate omitted the sqrt(20) Laplacian gain, E2FGVI added 1e-3 x mean(mask) to the
/// first output pixel, RVM had no recurrent decoder and blended its output into the caller's frame, GMFlow compared the round trip with a
/// fixed 1-pixel threshold, and the E2FGVI/ProPainter activations were element transforms the tape never saw.
/// </para>
/// </remarks>
public class VideoWarpAndMaskReferenceTests
{
    private const int H = 6;
    private const int W = 8;

    private static IEngine Engine => AiDotNetEngine.Current;

    private static NeuralNetworkArchitecture<double> Arch() =>
        new(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3);

    // image(x, y) = x + 10 y, shaped [1, 1, H, W]
    private static Tensor<double> AffineImage()
    {
        var values = new double[H * W];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                values[(y * W) + x] = x + (10.0 * y);
        return new Tensor<double>(values, new[] { 1, 1, H, W });
    }

    private static Tensor<double> ConstantFlow(double dx, double dy)
    {
        var values = new double[2 * H * W];
        for (int p = 0; p < H * W; p++)
        {
            values[p] = dx;
            values[(H * W) + p] = dy;
        }
        return new Tensor<double>(values, new[] { 1, 2, H, W });
    }

    private static double At(Tensor<double> t, int y, int x) => t.ToArray()[(y * W) + x];

    /// <summary>
    /// Shifts by (2, 1): inside the image every pixel is exactly image(x + 2, y + 1); the last two columns sample
    /// past the right edge and must be zero (zeros padding) or the clamped edge value (border padding).
    /// </summary>
    private static void AssertBackwardWarp(Func<Tensor<double>, Tensor<double>, Tensor<double>> warp, bool zerosPadding)
    {
        var warped = warp(AffineImage(), ConstantFlow(2.0, 1.0));
        Assert.Equal(new[] { 1, 1, H, W }, warped.Shape.ToArray());

        for (int y = 0; y < H - 1; y++)
        {
            for (int x = 0; x < W - 2; x++)
                Assert.Equal((x + 2) + (10.0 * (y + 1)), At(warped, y, x), 9);

            for (int x = W - 2; x < W; x++)
            {
                double border = Math.Min(x + 2, W - 1) + (10.0 * (y + 1));
                Assert.Equal(zerosPadding ? 0.0 : border, At(warped, y, x), 9);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task RaftWarpIsAlignedCornersWithZeroPadding()
    {
        await Task.Yield();
        var model = new RAFT<double>(Arch());
        AssertBackwardWarp(model.WarpByFlow, zerosPadding: true);
    }

    [Fact(Timeout = 120000)]
    public async Task FlowFormerWarpIsAlignedCornersWithZeroPadding()
    {
        await Task.Yield();
        var model = new FlowFormer<double>(Arch());
        AssertBackwardWarp(model.WarpWithFlow, zerosPadding: true);
    }

    [Fact(Timeout = 120000)]
    public async Task BasicVsrPlusPlusWarpIsAlignedCornersWithZeroPadding()
    {
        await Task.Yield();
        var model = new BasicVSRPlusPlus<double>(Arch());
        AssertBackwardWarp(model.WarpFeatureTape, zerosPadding: true);
    }

    [Fact(Timeout = 120000)]
    public async Task FilmWarpIsAlignedCornersWithBorderPadding()
    {
        await Task.Yield();
        var model = new FILM<double>(Arch(), options: new FILMOptions { NumScales = 3, NumFeatures = 32 });
        AssertBackwardWarp(model.WarpFeatures, zerosPadding: false);
    }

    [Fact(Timeout = 120000)]
    public async Task RifeWarpIsAlignedCornersWithBorderPadding()
    {
        await Task.Yield();
        var model = new RIFE<double>(Arch());
        AssertBackwardWarp(model.WarpImage, zerosPadding: false);
    }

    [Fact(Timeout = 120000)]
    public async Task E2fgviWarpIsAlignedCornersWithZeroPadding()
    {
        await Task.Yield();
        var model = new E2FGVI<double>(Arch());
        AssertBackwardWarp(model.WarpFeatures, zerosPadding: true);
    }

    [Fact(Timeout = 120000)]
    public async Task ProPainterWarpIsAlignedCornersWithZeroPadding()
    {
        await Task.Yield();
        var model = new ProPainter<double>(Arch());
        AssertBackwardWarp(model.WarpImage, zerosPadding: true);
    }

    [Fact(Timeout = 120000)]
    public async Task FastDvdNetNoiseEstimateRecoversTheGaussianSigma()
    {
        await Task.Yield();
        // White Gaussian noise of sigma 0.1: the 4-neighbour Laplacian multiplies sigma by sqrt(20), so
        // median(|L|) / 0.6745 must still be divided by sqrt(20). Without that the estimate was ~0.447.
        const int size = 96;
        const double sigma = 0.1;
        var random = RandomHelper.CreateSeededRandom(1234);
        var values = new double[size * size];
        for (int i = 0; i < values.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            values[i] = 0.5 + (sigma * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        var model = new FastDVDNet<double>(Arch());
        double estimate = model.EstimateNoiseLevel(new Tensor<double>(values, new[] { 1, 1, size, size }));

        Assert.InRange(estimate, 0.09, 0.11);
    }

    [Fact(Timeout = 120000)]
    public async Task GmFlowOcclusionUsesThePaperMotionScaledTolerance()
    {
        await Task.Yield();
        // gmflow/geometry.py forward_backward_consistency_check: occluded <=> |f + b(x + f)| > 0.01 (|f| + |b(x)|) + 0.5,
        // unsquared norms, with the backward magnitude taken at x (not at x + f).
        const int height = 4, width = 64;
        var model = new GMFlow<double>(Arch());

        Tensor<double> Flow(Func<int, double> dxAt)
        {
            var values = new double[2 * height * width];
            for (int p = 0; p < height * width; p++) values[p] = dxAt(p % width);
            return new Tensor<double>(values, new[] { 1, 2, height, width });
        }

        Tensor<double> Uniform(double dx) => Flow(_ => dx);
        double MaskAt(Tensor<double> mask, int x) => mask.ToArray()[(1 * width) + x];

        // Large motion, 2-pixel round-trip error: 2 > 0.01 (20 + 18) + 0.5 = 0.88, occluded. The squared form
        // (4 < 0.01 (400 + 324) + 0.5 = 7.74) called it consistent.
        Assert.Equal(1.0, MaskAt(model.ComputeOcclusionMask(Uniform(20.0), Uniform(-18.0)), 5));

        // Small motion, 0.7-pixel error: 0.7 > 0.01 (0.7 + 0) + 0.5 = 0.507, occluded. The squared form
        // (0.49 < 0.0049 + 0.5) called it consistent.
        Assert.Equal(1.0, MaskAt(model.ComputeOcclusionMask(Uniform(0.7), Uniform(0.0)), 5));

        // Sub-tolerance error: 0.3 < 0.01 (30 + 29.7) + 0.5 = 1.097, consistent.
        Assert.Equal(0.0, MaskAt(model.ComputeOcclusionMask(Uniform(30.0), Uniform(-29.7)), 5));

        // |b| is read at x: b(5) = -60 while b(15) = -9, so the error is 1 < 0.01 (10 + 60) + 0.5 = 1.2,
        // consistent; using the warped magnitude (0.01 (10 + 9) + 0.5 = 0.69) would call it occluded.
        var mixed = model.ComputeOcclusionMask(Uniform(10.0), Flow(x => x == 5 ? -60.0 : -9.0));
        Assert.Equal(0.0, MaskAt(mixed, 5));

        // A perfect round trip whose target leaves the image samples a zero backward flow: 20 > 0.9, occluded.
        var leaving = model.ComputeOcclusionMask(Uniform(20.0), Uniform(-20.0));
        Assert.Equal(0.0, MaskAt(leaving, 5));
        Assert.Equal(1.0, MaskAt(leaving, width - 2));
    }

    [Fact(Timeout = 120000)]
    public async Task E2fgviKnownPixelsPassThroughUnchanged()
    {
        await Task.Yield();
        // m * prediction + (1 - m) * frame: where the mask is 0 the output IS the frame, including pixel 0, which
        // used to carry an extra 1e-3 x mean(mask).
        var model = new E2FGVI<double>(Arch());
        var frames = Tensor<double>.CreateDefault(new[] { 1, 3, H, W }, 0.25);
        var prediction = Tensor<double>.CreateDefault(new[] { 1, 3, H, W }, 0.75);
        var maskValues = new double[H * W];
        for (int p = 1; p < maskValues.Length; p++) maskValues[p] = 1.0;
        var masks = new Tensor<double>(maskValues, new[] { 1, 1, H, W });

        var output = model.BlendKnownPixels(frames, masks, prediction).ToArray();

        for (int c = 0; c < 3; c++)
        {
            Assert.Equal(0.25, output[c * H * W]);
            for (int p = 1; p < H * W; p++) Assert.Equal(0.75, output[(c * H * W) + p]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ProPainterRejectsAWidthThatDoesNotSplitAcrossTheHeads()
    {
        // NumFeatures 5 gives 20 transformer channels: 8 heads of 2 would leave 4 channels unattended and zero.
        await Task.Yield();
        var options = new ProPainterOptions { NumFeatures = 5, NumTransformerBlocks = 1, NumHeads = 8 };
        Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Throws<ArgumentException>(() => new ProPainter<double>(Arch(), options));
        new ProPainterOptions { NumFeatures = 4, NumTransformerBlocks = 1, NumHeads = 8 }.Validate();
    }

    [Fact(Timeout = 120000)]
    public async Task WarpRejectsAFlowThatDoesNotMatchTheFeatureGrid()
    {
        // A [1, 2, 3, 2] flow (H and W swapped) against a [1, 1, 2, 3] feature has the right element count per plane
        // but the wrong grid; it used to be sampled as if it matched.
        await Task.Yield();
        var model = new RAFT<double>(Arch());
        var feature = new Tensor<double>(new double[6], new[] { 1, 1, 2, 3 });
        var flow = new Tensor<double>(new double[12], new[] { 1, 2, 3, 2 });

        Assert.Throws<ArgumentException>(() => model.WarpByFlow(feature, flow));
    }

    [Fact(Timeout = 120000)]
    public async Task VideoClipRejectsATextWidthThatDoesNotSplitAcrossTwelveHeads()
    {
        // BERT's 12 heads: width 32 would give heads of 2 with 8 channels unattended; 24 splits evenly.
        await Task.Yield();
        Assert.Throws<ArgumentException>(() => new VideoCLIPVideoOptions { HiddenDimension = 32 }.Validate());
        new VideoCLIPVideoOptions { HiddenDimension = 24, HiddenDim = 24 }.Validate();

        // Supplied options size the tower from HiddenDimension, so an unused HiddenDim of 64 is no error.
        new VideoCLIPVideoOptions { HiddenDimension = 24, HiddenDim = 64 }.Validate();
    }

    [Fact(Timeout = 120000)]
    public async Task MaskedPsnrIsComputedInDouble()
    {
        // The masked error 3e38 - (-3e38) = 6e38 overflows float; in double MSE = 3.6e77 over the one masked
        // pixel and PSNR = 10 log10(1 / 3.6e77). The unmasked pixel's error is ignored.
        await Task.Yield();
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3);
        var model = new E2FGVI<float>(arch);
        var inpainted = new Tensor<float>(new[] { 3e38f, 5f }, new[] { 1, 1, 1, 2 });
        var groundTruth = new Tensor<float>(new[] { -3e38f, 0f }, new[] { 1, 1, 1, 2 });
        var masks = new Tensor<float>(new[] { 1f, 0f }, new[] { 1, 1, 1, 2 });

        double psnr = model.ComputeMaskedPSNR(inpainted, groundTruth, masks);

        Assert.Equal(10.0 * Math.Log10(1.0 / 3.6e77), psnr, 6);
    }

    [Fact(Timeout = 120000)]
    public async Task RvmCompositesEverySampleAndChannel()
    {
        // comp = a * fg + (1 - a) * bg for every sample and channel: 0.5 * 0.8 + 0.5 * 0.2 = 0.5 throughout. The old
        // loop wrote only sample 0's first three channels and left the rest of the batch zero.
        await Task.Yield();
        var model = new RVM<double>(Arch());
        int[] shape = [2, 4, 1, 2];
        var foreground = Tensor<double>.CreateDefault(shape, 0.8);
        var background = Tensor<double>.CreateDefault(shape, 0.2);
        var alpha = Tensor<double>.CreateDefault(new[] { 2, 1, 1, 2 }, 0.5);

        var composite = model.CompositeWithBackground(foreground, alpha, background);

        Assert.Equal(shape, composite.Shape.ToArray());
        Assert.All(composite.ToArray(), value => Assert.Equal(0.5, value, 12));
    }

    [Fact(Timeout = 120000)]
    public async Task RvmAsppGateReadsTheLastEncoderStage()
    {
        // LR-ASPP's gate is conv1x1(avgpool(f4)) over the last encoder stage (the 1x1 conv to the 960-channel
        // features, scaled), not the BatchNorm before it in the layer list. Without a declared depth the gate's input
        // depth was left to a sequential shape walk that bound it to the ASPP width.
        await Task.Yield();
        const int features = 4;
        var layers = AiDotNet.Helpers.LayerHelper<double>.CreateDefaultRVMLayers(3, 32, 32, features).ToList();
        Assert.Equal(RvmRecurrentNetwork<double>.LayerCount, layers.Count);
        int encoderLayers = RvmRecurrentNetwork<double>.EncoderStageLayers.Sum();
        var lastEncoderConv = Assert.IsType<AiDotNet.NeuralNetworks.Layers.ConvolutionalLayer<double>>(
            layers[encoderLayers - 3]);
        var gate = Assert.IsType<AiDotNet.NeuralNetworks.Layers.ConvolutionalLayer<double>>(layers[encoderLayers + 2]);
        Assert.Equal(lastEncoderConv.OutputDepth, gate.InputDepth);
    }

    [Fact(Timeout = 120000)]
    public async Task RvmEncoderIsMobileNetV3LargeStages()
    {
        // RVM's backbone is MobileNetV3-Large: a hard-swish stem, then 15 inverted residual blocks (squeeze-excitation
        // in the 40-, 112- and 160-channel ones), then the 1x1 conv to 960 channels. At the default width the four
        // stages end in 16, 24, 40 and 960 channels, and the stride-2 160-channel block runs at stride 1 so the last
        // stage stays at 1/16 as the reference's dilated block does.
        await Task.Yield();
        var layers = AiDotNet.Helpers.LayerHelper<double>.CreateDefaultRVMLayers().ToList();
        Assert.Equal(RvmRecurrentNetwork<double>.LayerCount, layers.Count);
        var blocks = layers.OfType<AiDotNet.NeuralNetworks.Layers.InvertedResidualBlock<double>>().ToList();
        Assert.Equal(15, blocks.Count);
        Assert.Equal(new[] { 16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160 },
            blocks.Select(b => b.OutChannels).ToArray());
        Assert.Equal(new[] { 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1 }, blocks.Select(b => b.Stride).ToArray());

        var stem = Assert.IsType<AiDotNet.NeuralNetworks.Layers.ConvolutionalLayer<double>>(layers[0]);
        Assert.Equal(16, stem.OutputDepth);
        int encoderLayers = RvmRecurrentNetwork<double>.EncoderStageLayers.Sum();
        var head = Assert.IsType<AiDotNet.NeuralNetworks.Layers.ConvolutionalLayer<double>>(layers[encoderLayers - 3]);
        Assert.Equal(960, head.OutputDepth);
    }

    private sealed class FixedDraw : Random
    {
        private readonly double _value;
        public FixedDraw(double value) => _value = value;
        public override double NextDouble() => _value;
        protected override double Sample() => _value;
    }

    [Fact(Timeout = 120000)]
    public async Task MiaVsrLogisticNoiseIsComputedInDouble()
    {
        // logit(u) for u = 1 - 1e-9 is 20.7232658642 (python: math.log(u) - math.log(1 - u)). Cast to float first
        // (the old code), u rounds to 1.0f, the 1 - 1e-10 clamp rounds to 1.0f too, and the noise is +infinity.
        await Task.Yield();
        var block = typeof(MiaVsrNetwork<float>).GetNestedType("AttentionBlock", BindingFlags.NonPublic);
        Assert.NotNull(block);
        var noise = block.MakeGenericType(typeof(float)).GetMethod("LogisticNoise", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(noise);

        var result = Assert.IsType<Tensor<float>>(noise.Invoke(null, new object[] { new[] { 3 }, new FixedDraw(1 - 1e-9) }));

        double u = 1 - 1e-9;
        double expected = Math.Log(u) - Math.Log(1 - u);
        Assert.All(result.ToArray(), value => Assert.Equal(expected, value, 4));
    }

    [Fact(Timeout = 120000)]
    public async Task ProPainterLayerNormStandardizesHugeFloatActivations()
    {
        // A sample of +/-1e20 standardizes to +/-1. The old form multiplied the scaled variance back by peak^2 = 1e40,
        // which overflows float to infinity, so every output was 0.
        await Task.Yield();
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3);
        var model = new ProPainter<float>(arch);
        var layerNorm = typeof(ProPainter<float>).GetMethod("LayerNorm", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(layerNorm);
        var input = new Tensor<float>(new[] { 1e20f, -1e20f }, new[] { 1, 2, 1, 1 });

        var output = Assert.IsType<Tensor<float>>(layerNorm.Invoke(model, new object[] { input })).ToArray();

        Assert.Equal(1.0, output[0], 5);
        Assert.Equal(-1.0, output[1], 5);
    }

    [Fact(Timeout = 120000)]
    public async Task ConsistencyIndexesPixelsPastHalfPrecision()
    {
        // H = 3, W = 1025: pixel (y = 2, x = 1) is flat index 2051, which Half rounds to 2052 (python:
        // struct.unpack('e', struct.pack('e', 2051.0)) gives 2052.0). With zero forward flow the round-trip error at
        // (2, 1) is |b(2, 1)| = 4; forming y * W + x in Half (the old code) read b(2, 2) = 0 instead.
        await Task.Yield();
        const int height = 3, width = 1025;
        var arch = new NeuralNetworkArchitecture<Half>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3);
        var model = new GMFlow<Half>(arch);
        var forward = new Tensor<Half>(new[] { 2, height, width });
        var backward = new Tensor<Half>(new[] { 2, height, width });
        backward[0, 2, 1] = (Half)4.0;

        var consistency = model.ComputeForwardBackwardConsistency(forward, backward);

        Assert.Equal(4.0, (double)consistency[2, 1], 6);
        Assert.Equal(0.0, (double)consistency[2, 2], 6);
    }

    private static double[] Varied(int length, double phase)
    {
        var values = new double[length];
        for (int i = 0; i < length; i++) values[i] = 0.5 + (0.4 * Math.Sin((0.37 * i) + phase));
        return values;
    }

    [Fact(Timeout = 120000)]
    public async Task RvmCarriesConvGruStateAcrossFramesAndResetsIt()
    {
        // RVM's recurrent decoder (Lin et al. 2021): the ConvGRU hidden states from frame t condition frame t + 1,
        // so the same frame seen twice is matted differently the second time, and ResetState returns to the
        // first-frame result. The old network was a feed-forward conv stack whose "hidden state" (its own 4-channel
        // output) never matched the 3-channel input it was blended with, so a repeated frame gave the identical
        // matte and the first assertion fails on it.
        await Task.Yield();
        var model = new RVM<double>(Arch());
        var frame = new Tensor<double>(Varied(3 * 32 * 32, 0.0), new[] { 1, 3, 32, 32 });
        var before = frame.ToArray();

        var first = model.Predict(frame).ToArray();
        var second = model.Predict(frame).ToArray();
        Assert.Equal(first.Length, second.Length);
        Assert.Contains(Enumerable.Range(0, first.Length), i => Math.Abs(first[i] - second[i]) > 1e-12);

        model.ResetState();
        var afterReset = model.Predict(frame).ToArray();
        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i], afterReset[i], 12);

        // The caller's frame is never written to (it used to receive the temporal blend in place).
        Assert.Equal(before, frame.ToArray());

        // Alpha and foreground are clamped to [0, 1].
        Assert.All(first, value => Assert.InRange(value, 0.0, 1.0));
    }

    [Fact(Timeout = 120000)]
    public async Task E2fgviUpsamplesFlowBilinearlyAndRescalesIt()
    {
        // E2FGVI's SPyNet: F.interpolate(flow, mode="bilinear", align_corners=False), dx *= W'/W, dy *= H'/H.
        // dx = [0, 2] per row resized from width 2 to 4 samples at x' = -0.25, 0.25, 0.75, 1.25 (clamped), giving
        // [0, 0.5, 1.5, 2], times 2 = [0, 1, 3, 4]. Nearest-pixel resizing (the old code) gave [0, 0, 4, 4].
        await Task.Yield();
        var model = new E2FGVI<double>(Arch());
        var flow = new Tensor<double>(new[] { 0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0 }, new[] { 1, 2, 2, 2 });

        var upsampled = model.UpsampleFlow(flow, 2, 4).ToArray();

        Assert.Equal(16, upsampled.Length);
        var expectedDx = new[] { 0.0, 1.0, 3.0, 4.0 };
        for (int y = 0; y < 2; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                Assert.Equal(expectedDx[x], upsampled[(y * 4) + x], 9);
                Assert.Equal(1.0, upsampled[8 + (y * 4) + x], 9);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task VideoClipTextAttentionIsBidirectionalOverRealTokens()
    {
        // VideoCLIP's text tower is BERT (Xu et al., EMNLP 2021): every token attends to every real token, so the
        // first position responds to a change in the last one. The old causal mask made position 0 blind to it, so
        // the first assertion fails on that code. Padding keys stay excluded: with the last position marked as
        // padding, changing it leaves every real position unchanged.
        await Task.Yield();
        const int hidden = 48, sequence = 6;
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 4);
        var options = new VideoCLIPVideoOptions
        {
            HiddenDimension = hidden,
            NumSpatialBlocks = 1,
            NumTemporalBlocks = 1,
            NumTextBlocks = 1,
            NumFrames = 4,
            EmbeddingDim = 4,
            TextMaxLength = sequence,
            VocabSize = 64,
        };
        using var model = new AiDotNet.Video.Understanding.VideoCLIP<double>(arch, options: options);

        var values = Varied(hidden * sequence, 0.3);
        var changed = (double[])values.Clone();
        for (int c = 0; c < hidden; c++) changed[(c * sequence) + sequence - 1] += 1.0;
        var original = new Tensor<double>(values, new[] { 1, hidden, 1, sequence });
        var perturbed = new Tensor<double>(changed, new[] { 1, hidden, 1, sequence });

        double MaxChangeAt(Tensor<double> a, Tensor<double> b, int position)
        {
            var x = a.ToArray();
            var y = b.ToArray();
            double largest = 0;
            for (int c = 0; c < hidden; c++)
                largest = Math.Max(largest, Math.Abs(x[(c * sequence) + position] - y[(c * sequence) + position]));
            return largest;
        }

        var full = model.TextMultiHeadAttention(original, 0, sequence);
        var fullChanged = model.TextMultiHeadAttention(perturbed, 0, sequence);
        Assert.True(MaxChangeAt(full, fullChanged, 0) > 1e-9, "Position 0 did not attend to the last token.");

        var padded = model.TextMultiHeadAttention(original, 0, sequence - 1);
        var paddedChanged = model.TextMultiHeadAttention(perturbed, 0, sequence - 1);
        for (int position = 0; position < sequence - 1; position++)
            Assert.Equal(0.0, MaxChangeAt(padded, paddedChanged, position), 12);
    }

    private static double[] TapeGradient(Func<Tensor<double>, Tensor<double>> activation, double[] values)
    {
        var input = new Tensor<double>(values, new[] { values.Length });
        using var tape = new GradientTape<double>();
        var objective = Engine.ReduceSum(activation(input), new[] { 0 }, keepDims: false);
        var gradients = tape.ComputeGradients(objective, new[] { input });
        Assert.True(gradients.TryGetValue(input, out var gradient), "The activation was not recorded on the tape.");
        return gradient.ToArray();
    }

    [Fact(Timeout = 120000)]
    public async Task E2fgviReluIsDifferentiable()
    {
        await Task.Yield();
        var model = new E2FGVI<double>(Arch());
        Assert.Equal(new[] { 0.0, 1.0 }, TapeGradient(model.ApplyReLU, new[] { -1.5, 2.0 }));
    }

    [Fact(Timeout = 120000)]
    public async Task ProPainterActivationsAreDifferentiable()
    {
        await Task.Yield();
        var model = new ProPainter<double>(Arch());
        Assert.Equal(new[] { 0.0, 1.0 }, TapeGradient(model.ApplyReLU, new[] { -1.5, 2.0 }));

        // d/dx [x Phi(x)] at x = 1 is Phi(1) + phi(1) = 1.08331547 (nn.GELU); the tanh form gives 1.0830, which six
        // decimals tell apart (three rounded both to 1.083).
        Assert.Equal(1.0833154706, TapeGradient(model.ApplyGELU, new[] { 1.0 })[0], 6);
    }
}
