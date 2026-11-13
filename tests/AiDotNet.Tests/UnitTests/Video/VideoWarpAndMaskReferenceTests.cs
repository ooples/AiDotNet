using System;
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
/// first output pixel, RVM blended its hidden state into the caller's frame, GMFlow compared the round trip with a
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
        // occluded <=> |f + b(x + f)|^2 > 0.01 (|f|^2 + |b(x + f)|^2) + 0.5 (forward_backward_consistency_check)
        const int height = 4, width = 64;
        var model = new GMFlow<double>(Arch());

        Tensor<double> Flow(double dx)
        {
            var values = new double[2 * height * width];
            for (int p = 0; p < height * width; p++) values[p] = dx;
            return new Tensor<double>(values, new[] { 1, 2, height, width });
        }

        double MaskAt(Tensor<double> mask, int x) => mask.ToArray()[(1 * width) + x];

        // Large motion, 2-pixel round-trip error: 4 < 0.01 (400 + 324) + 0.5 = 7.74, consistent (the old fixed
        // 1-pixel threshold called it occluded).
        var largeMotion = model.ComputeOcclusionMask(Flow(20.0), Flow(-18.0));
        Assert.Equal(0.0, MaskAt(largeMotion, 5));

        // Small motion, 1-pixel error: 1 > 0.01 (1 + 0) + 0.5 = 0.51, occluded (the old strict 1-pixel test
        // was not exceeded).
        var smallMotion = model.ComputeOcclusionMask(Flow(1.0), Flow(0.0));
        Assert.Equal(1.0, MaskAt(smallMotion, 5));

        // A perfect round trip whose target leaves the image samples a zero backward flow: 400 > 4.5, occluded.
        var leaving = model.ComputeOcclusionMask(Flow(20.0), Flow(-20.0));
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
    public async Task RvmPredictDoesNotOverwriteTheCallersFrame()
    {
        await Task.Yield();
        var model = new RVM<double>(Arch());
        var frame = Tensor<double>.CreateDefault(new[] { 1, 3, 32, 32 }, 0.5);
        var hidden = typeof(RVM<double>).GetField("_hiddenState", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(hidden);
        hidden?.SetValue(model, Tensor<double>.CreateDefault(new[] { 1, 3, 32, 32 }, 1.0));

        model.Predict(frame);

        // The temporal blend 0.9 frame + 0.1 hidden used to be written into the frame itself (0.55 here).
        foreach (var value in frame.ToArray()) Assert.Equal(0.5, value);
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

        // d/dx [x Phi(x)] at x = 1 is Phi(1) + phi(1) = 1.0833 (nn.GELU); the tanh form gives 1.0830.
        Assert.Equal(1.0833, TapeGradient(model.ApplyGELU, new[] { 1.0 })[0], 3);
    }
}
