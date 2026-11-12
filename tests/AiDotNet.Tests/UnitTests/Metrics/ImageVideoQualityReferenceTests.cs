using System;
using System.Threading.Tasks;
using AiDotNet.Metrics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Metrics;

/// <summary>
/// Reference values for the image and video quality metrics. SSIM values come from scikit-image's
/// structural_similarity(gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1), re-run
/// with scipy.ndimage.gaussian_filter (truncate=3.5, the 11-tap window of Wang et al. 2004) and averaged over
/// the valid window positions.
/// </summary>
public class ImageVideoQualityReferenceTests
{
    private const int Height = 14;
    private const int Width = 13;

    // skimage reference values for the images below.
    private const double GraySsim = 0.9742849121170994;
    private const double SecondChannelSsim = -0.1436849705555623;

    private static double X(int i, int j) => (Math.Sin(0.7 * i) * Math.Cos(0.5 * j) + 1.0) / 2.0;

    private static double Y(int i, int j) =>
        Math.Min(1.0, Math.Max(0.0, 0.8 * X(i, j) + 0.1 + 0.05 * Math.Sin(1.3 * i + 0.9 * j)));

    private static double X2(int i, int j) => (Math.Cos(0.3 * i + 0.2 * j) + 1.0) / 2.0;

    private static double Y2(int i, int j) => 1.0 - X2(i, j) * 0.5;

    private static Tensor<double> Image(Func<int, int, double> f)
    {
        var data = new double[Height * Width];
        for (int i = 0; i < Height; i++)
        {
            for (int j = 0; j < Width; j++)
            {
                data[i * Width + j] = f(i, j);
            }
        }

        return new Tensor<double>(new[] { Height, Width }, new Vector<double>(data));
    }

    private static Tensor<double> Hwc(Func<int, int, double> c0, Func<int, int, double> c1)
    {
        var data = new double[Height * Width * 2];
        for (int i = 0; i < Height; i++)
        {
            for (int j = 0; j < Width; j++)
            {
                data[(i * Width + j) * 2] = c0(i, j);
                data[(i * Width + j) * 2 + 1] = c1(i, j);
            }
        }

        return new Tensor<double>(new[] { Height, Width, 2 }, new Vector<double>(data));
    }

    [Fact(Timeout = 60000)]
    public async Task Ssim_GaussianWindow_MatchesScikitImage()
    {
        await Task.Yield();
        var ssim = new StructuralSimilarity<double>();

        Assert.Equal(GraySsim, ssim.Compute(Image(X), Image(Y)), 1e-10);
        Assert.Equal(SecondChannelSsim, ssim.Compute(Image(X2), Image(Y2)), 1e-10);
    }

    [Fact(Timeout = 60000)]
    public async Task Ssim_MultiChannel_AveragesPerChannelScikitImageValues()
    {
        await Task.Yield();
        var ssim = new StructuralSimilarity<double>();

        double result = ssim.Compute(Hwc(X, X2), Hwc(Y, Y2));

        Assert.Equal((GraySsim + SecondChannelSsim) / 2.0, result, 1e-10);
    }

    [Fact(Timeout = 60000)]
    public async Task VideoSsim_IsMeanOfPerFrameSsim_InBothLayouts()
    {
        await Task.Yield();
        var videoSsim = new VideoSSIM<double>();

        // Two single-channel frames: (X, Y) and (X2, Y2).
        var pred = new double[2 * Height * Width];
        var gt = new double[2 * Height * Width];
        for (int i = 0; i < Height; i++)
        {
            for (int j = 0; j < Width; j++)
            {
                pred[i * Width + j] = X(i, j);
                gt[i * Width + j] = Y(i, j);
                pred[Height * Width + i * Width + j] = X2(i, j);
                gt[Height * Width + i * Width + j] = Y2(i, j);
            }
        }

        var hwcPred = new Tensor<double>(new[] { 2, Height, Width, 1 }, new Vector<double>(pred));
        var hwcGt = new Tensor<double>(new[] { 2, Height, Width, 1 }, new Vector<double>(gt));
        var chwPred = new Tensor<double>(new[] { 2, 1, Height, Width }, new Vector<double>(pred));
        var chwGt = new Tensor<double>(new[] { 2, 1, Height, Width }, new Vector<double>(gt));

        double expected = (GraySsim + SecondChannelSsim) / 2.0;
        Assert.Equal(expected, videoSsim.Compute(hwcPred, hwcGt), 1e-10);
        Assert.Equal(expected, videoSsim.Compute(chwPred, chwGt, isChannelsFirst: true), 1e-10);
    }

    [Fact(Timeout = 60000)]
    public async Task PsnrBatch_IsPerImageTenLog10MaxSquaredOverMse()
    {
        await Task.Yield();
        var psnr = new PeakSignalToNoiseRatio<double>();
        var pred = new Tensor<double>(new[] { 2, 1, 2, 1 }, new Vector<double>(new[] { 0.0, 0.5, 0.2, 0.2 }));
        var gt = new Tensor<double>(new[] { 2, 1, 2, 1 }, new Vector<double>(new[] { 0.1, 0.3, 0.2, 0.2 }));

        var result = psnr.ComputeBatch(pred, gt);

        // Image 0: MSE = (0.01 + 0.04) / 2; image 1 is identical, which reports 100 dB.
        Assert.Equal(10.0 * Math.Log10(1.0 / 0.025), result[0], 1e-9);
        Assert.Equal(100.0, result[1], 1e-12);
    }

    [Fact(Timeout = 60000)]
    public async Task MeanIoU_CountsTruncatedLabelsAndSkipsOutOfRangeLabels()
    {
        await Task.Yield();
        var pred = new Tensor<double>(new[] { 5 }, new Vector<double>(new[] { 0.0, 1.7, 1.0, 2.0, 5.0 }));
        var gt = new Tensor<double>(new[] { 5 }, new Vector<double>(new[] { 0.0, 1.0, 2.0, 2.0, 1.0 }));

        // Class 0: IoU 1; class 1: 1 / 3; class 2: 1 / 2 (label 5 is outside [0, 3)).
        Assert.Equal((1.0 + 1.0 / 3.0 + 0.5) / 3.0, new MeanIntersectionOverUnion<double>(3).Compute(pred, gt), 1e-12);
        Assert.Equal((1.0 / 3.0 + 0.5) / 2.0,
            new MeanIntersectionOverUnion<double>(3, ignoreBackground: true).Compute(pred, gt), 1e-12);
        Assert.Equal(new[] { 1.0, 1.0 / 3.0, 0.5 }, new MeanIntersectionOverUnion<double>(3).ComputePerClass(pred, gt));
    }

    [Fact(Timeout = 60000)]
    public async Task TemporalConsistency_FlowWarp_ClampsToTheBorderPixel()
    {
        await Task.Yield();
        var tc = new TemporalConsistency<double>();

        // Frame 0 = [0, 1, 2] (H = 1, W = 3); dx = +1 samples [1, 2, 2] with border clamping, which is frame 1.
        var video = new Tensor<double>(new[] { 2, 1, 3, 1 }, new Vector<double>(new[] { 0.0, 1.0, 2.0, 1.0, 2.0, 2.0 }));
        var flow = new Tensor<double>(new[] { 1, 1, 3, 2 }, new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 }));

        Assert.Equal(1.0, tc.ComputeWithFlow(video, flow), 1e-12);
    }

    [Fact(Timeout = 60000)]
    public async Task TemporalConsistency_FlowWarp_InterpolatesAlongRows()
    {
        await Task.Yield();
        var tc = new TemporalConsistency<double>();

        // Frame 0 = [[0], [0.2]] (H = 2, W = 1); dy = 0.5 samples [0.1, 0.2]; frame 1 = [0.1, 0.1].
        var video = new Tensor<double>(new[] { 2, 2, 1, 1 }, new Vector<double>(new[] { 0.0, 0.2, 0.1, 0.1 }));
        var flow = new Tensor<double>(new[] { 1, 2, 1, 2 }, new Vector<double>(new[] { 0.0, 0.5, 0.0, 0.5 }));

        // MSE = (0 + 0.01) / 2 = 0.005; consistency = 1 - min(1, 10 * MSE).
        Assert.Equal(1.0 - 0.05, tc.ComputeWithFlow(video, flow), 1e-12);
    }

    [Fact(Timeout = 60000)]
    public async Task TemporalConsistency_SimpleAndFlicker_MatchHandComputedValues()
    {
        await Task.Yield();
        var tc = new TemporalConsistency<double>();

        var ramp = new Tensor<double>(new[] { 3, 1, 1, 1 }, new Vector<double>(new[] { 0.0, 0.1, 0.3 }));
        // Pair MSEs 0.01 and 0.04 -> normalised 0.1 and 0.4 -> consistency 1 - 0.25.
        Assert.Equal(0.75, tc.ComputeSimple(ramp), 1e-12);

        var blink = new Tensor<double>(new[] { 3, 1, 1, 1 }, new Vector<double>(new[] { 0.0, 1.0, 0.0 }));
        Assert.Equal(1.0 - Math.Exp(-10.0), tc.ComputeFlicker(blink), 1e-12);
    }
}
