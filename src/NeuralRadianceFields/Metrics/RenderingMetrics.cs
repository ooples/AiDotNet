using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.NeuralRadianceFields.Metrics;

/// <summary>
/// Provides image quality metrics for evaluating neural rendering methods like NeRF.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// These metrics are essential for evaluating the quality of rendered images from
/// neural radiance fields, 3D Gaussian splatting, and other view synthesis methods.
/// </para>
/// <para><b>For Beginners:</b> When a neural network generates an image (like a new view
/// of a 3D scene), we need ways to measure how good that image is compared to a real
/// photograph. These metrics give us numbers that tell us how similar two images are.
/// </para>
/// </remarks>
public static class RenderingMetrics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes Peak Signal-to-Noise Ratio (PSNR) between two images.
    /// </summary>
    /// <param name="predicted">The predicted/rendered image tensor of shape [H, W, C] or [H, W].</param>
    /// <param name="groundTruth">The ground truth image tensor with the same shape.</param>
    /// <param name="maxValue">The maximum possible pixel value. Default is 1.0 for normalized images.</param>
    /// <returns>The PSNR value in decibels (dB). Higher is better.</returns>
    /// <remarks>
    /// <para>
    /// PSNR measures the ratio between the maximum possible signal and the noise (error).
    /// It's calculated as: PSNR = 10 * log10(MAX^2 / MSE)
    /// </para>
    /// <para><b>For Beginners:</b> PSNR tells us how much the generated image differs from
    /// the original. Higher PSNR means less noise/error:
    /// - PSNR &gt; 40 dB: Excellent quality, nearly identical
    /// - PSNR 30-40 dB: Good quality, small differences
    /// - PSNR 20-30 dB: Acceptable quality, noticeable differences
    /// - PSNR &lt; 20 dB: Poor quality, significant differences
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    public static double PSNR(Tensor<T> predicted, Tensor<T> groundTruth, double maxValue = 1.0)
    {
        ValidateShapes(predicted, groundTruth);

        double mse = MSE(predicted, groundTruth);

        if (mse < 1e-10)
        {
            return double.PositiveInfinity;
        }

        return 10.0 * Math.Log10((maxValue * maxValue) / mse);
    }

    /// <summary>
    /// Computes Structural Similarity Index Measure (SSIM) between two images.
    /// </summary>
    /// <param name="predicted">The predicted/rendered image tensor of shape [H, W, C] or [H, W].</param>
    /// <param name="groundTruth">The ground truth image tensor with the same shape.</param>
    /// <param name="windowSize">Size of the sliding window for local statistics. Default is 11.</param>
    /// <param name="k1">Constant to stabilize division. Default is 0.01.</param>
    /// <param name="k2">Constant to stabilize division. Default is 0.03.</param>
    /// <param name="maxValue">The maximum possible pixel value. Default is 1.0.</param>
    /// <returns>The SSIM value in range [0, 1]. Higher is better (1 = identical).</returns>
    /// <remarks>
    /// <para>
    /// SSIM measures the structural similarity between images, considering luminance,
    /// contrast, and structure. It's designed to match human perception better than MSE.
    /// </para>
    /// <para><b>For Beginners:</b> SSIM looks at images the way humans do - it cares about
    /// patterns and structures, not just individual pixel differences:
    /// - SSIM = 1.0: Images are identical
    /// - SSIM &gt; 0.9: Very similar, hard to tell apart
    /// - SSIM 0.7-0.9: Noticeable differences but overall structure preserved
    /// - SSIM &lt; 0.7: Significant structural differences
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    public static double SSIM(
        Tensor<T> predicted,
        Tensor<T> groundTruth,
        int windowSize = 11,
        double k1 = 0.01,
        double k2 = 0.03,
        double maxValue = 1.0)
    {
        ValidateShapes(predicted, groundTruth);
        if (typeof(T) == typeof(Half))
        {
            // Half window sums overflow above 65504, and so does the total over many windows: measure in float.
            return RenderingMetrics<float>.SSIM(
                predicted.Cast<float>(), groundTruth.Cast<float>(), windowSize, k1, k2, maxValue);
        }

        double c1 = (k1 * maxValue) * (k1 * maxValue);
        double c2 = (k2 * maxValue) * (k2 * maxValue);

        int height = predicted.Shape[0];
        int width = predicted.Shape[1];

        int halfWindow = windowSize / 2;
        int numWindows = Math.Max(0, height - 2 * halfWindow) * Math.Max(0, width - 2 * halfWindow);
        T c1T = NumOps.FromDouble(c1);
        T c2T = NumOps.FromDouble(c2);

        if (numWindows == 0)
        {
            return NumOps.ToDouble(ComputeGlobalSSIM(predicted, groundTruth, c1T, c2T));
        }

        T totalSsim = SumWindowSSIM(predicted, groundTruth, halfWindow, c1T, c2T);
        return NumOps.ToDouble(NumOps.Divide(totalSsim, NumOps.FromDouble(numWindows)));
    }

    /// <summary>
    /// Computes a simplified perceptual loss (L1 in feature space) as a proxy for LPIPS.
    /// </summary>
    /// <param name="predicted">The predicted/rendered image tensor of shape [H, W, C].</param>
    /// <param name="groundTruth">The ground truth image tensor with the same shape.</param>
    /// <returns>The perceptual distance. Lower is better (0 = identical).</returns>
    /// <remarks>
    /// <para>
    /// LPIPS (Learned Perceptual Image Patch Similarity) uses deep network features
    /// to measure perceptual similarity. This simplified version uses edge detection
    /// and local statistics as a proxy since we don't have a pre-trained VGG network.
    /// </para>
    /// <para><b>For Beginners:</b> This metric tries to measure whether images "look"
    /// similar to a human, even if the exact pixel values differ. Two images with
    /// the same content but slightly different colors should have low LPIPS.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    public static double SimplifiedLPIPS(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        ValidateShapes(predicted, groundTruth);
        if (typeof(T) == typeof(Half))
        {
            // Half sums and element counts overflow at 65504: measure in float.
            return RenderingMetrics<float>.SimplifiedLPIPS(predicted.Cast<float>(), groundTruth.Cast<float>());
        }

        var predEdges = ComputeEdgeMagnitude(predicted);
        var gtEdges = ComputeEdgeMagnitude(groundTruth);

        T edgeDifference = ComputeMSE(predEdges, gtEdges);

        var predStats = ComputeLocalStatistics(predicted);
        var gtStats = ComputeLocalStatistics(groundTruth);

        T statsDifference = ComputeMSE(predStats, gtStats);

        return NumOps.ToDouble(NumOps.Add(
            NumOps.Sqrt(edgeDifference),
            NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Sqrt(statsDifference))));
    }

    /// <summary>
    /// Computes Mean Squared Error between two tensors.
    /// </summary>
    /// <param name="a">First tensor.</param>
    /// <param name="b">Second tensor.</param>
    /// <returns>The MSE value.</returns>
    public static double MSE(Tensor<T> a, Tensor<T> b)
    {
        ValidateShapes(a, b);
        if (typeof(T) == typeof(Half))
        {
            // TensorMean converts the element count to Half (65,536 becomes infinity): measure in float.
            return RenderingMetrics<float>.MSE(a.Cast<float>(), b.Cast<float>());
        }

        return NumOps.ToDouble(ComputeMSE(a, b));
    }

    /// <summary>
    /// Computes Mean Absolute Error between two tensors.
    /// </summary>
    /// <param name="a">First tensor.</param>
    /// <param name="b">Second tensor.</param>
    /// <returns>The MAE value.</returns>
    public static double MAE(Tensor<T> a, Tensor<T> b)
    {
        ValidateShapes(a, b);
        if (typeof(T) == typeof(Half))
        {
            return RenderingMetrics<float>.MAE(a.Cast<float>(), b.Cast<float>());
        }

        return NumOps.ToDouble(Engine.TensorMean(Engine.TensorAbs(Engine.TensorSubtract(a, b))));
    }

    #region Private Methods

    /// <summary>
    /// Validates that two tensors have the same shape.
    /// </summary>
    private static void ValidateShapes(Tensor<T> a, Tensor<T> b)
    {
        if (a.Rank != b.Rank)
        {
            throw new ArgumentException(
                $"Tensors must have the same rank. Got {a.Rank} and {b.Rank}.");
        }

        for (int i = 0; i < a.Rank; i++)
        {
            if (a.Shape[i] != b.Shape[i])
            {
                throw new ArgumentException(
                    $"Tensors must have the same shape. Dimension {i}: {a.Shape[i]} vs {b.Shape[i]}.");
            }
        }
    }

    /// <summary>
    /// Computes Mean Squared Error.
    /// </summary>
    private static IEngine Engine => AiDotNetEngine.Current;

    private static T ComputeMSE(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorMean(Engine.TensorSquare(Engine.TensorSubtract(a, b)));
    }

    /// <summary>Views a [H, W] image as [H, W, 1]; [H, W, C] images are returned unchanged.</summary>
    private static Tensor<T> AsHeightWidthChannels(Tensor<T> image)
    {
        return image.Rank > 2 ? image : image.Reshape(image.Shape[0], image.Shape[1], 1);
    }

    /// <summary>
    /// Every (2h + 1) x (2h + 1) window of rows <paramref name="firstRow"/>.. of an [H, W, C] image whose centre
    /// row lies in the block, as [rows, W - 2h, C, k, k].
    /// </summary>
    private static Tensor<T> Windows(Tensor<T> image, int halfWindow, int firstRow, int rows)
    {
        int size = 2 * halfWindow + 1;
        int width = image.Shape[1];
        int channels = image.Shape[2];
        var band = Engine.TensorSlice(image, new[] { firstRow, 0, 0 }, new[] { rows + size - 1, width, channels });
        var rowWindows = Engine.TensorUnfold(band, 0, size, 1);
        return Engine.TensorUnfold(rowWindows, 1, size, 1);
    }

    /// <summary>
    /// Sum of the per-window SSIM values over every window whose centre is at least <paramref name="halfWindow"/>
    /// pixels from the border. Each window's mean, variances and covariance use the two-pass (centred) sums.
    /// </summary>
    private static T SumWindowSSIM(Tensor<T> predicted, Tensor<T> groundTruth, int halfWindow, T c1, T c2)
    {
        var x = AsHeightWidthChannels(predicted);
        var y = AsHeightWidthChannels(groundTruth);
        int size = 2 * halfWindow + 1;
        int windowRows = x.Shape[0] - 2 * halfWindow;
        int windowColumns = x.Shape[1] - 2 * halfWindow;
        int perWindow = x.Shape[2] * size * size;
        int blockRows = Math.Max(1, (1 << 22) / Math.Max(1, windowColumns * perWindow));
        var axes = new[] { 2, 3, 4 };
        // A single-sample window (1x1, one channel) has no unbiased divisor; its sums are used as they are, as
        // the global path does for one element.
        T unbiased = NumOps.FromDouble(Math.Max(1, perWindow - 1));
        T two = NumOps.FromDouble(2);

        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < windowRows; start += blockRows)
        {
            int rows = Math.Min(blockRows, windowRows - start);
            var px = Windows(x, halfWindow, start, rows);
            var py = Windows(y, halfWindow, start, rows);
            var meanX = Engine.ReduceMean(px, axes, true);
            var meanY = Engine.ReduceMean(py, axes, true);
            var dx = Engine.TensorSubtract(px, Engine.TensorBroadcastTo(meanX, px.Shape.ToArray()));
            var dy = Engine.TensorSubtract(py, Engine.TensorBroadcastTo(meanY, py.Shape.ToArray()));
            var varX = Engine.TensorDivideScalar(Engine.ReduceSum(Engine.TensorSquare(dx), axes, false), unbiased);
            var varY = Engine.TensorDivideScalar(Engine.ReduceSum(Engine.TensorSquare(dy), axes, false), unbiased);
            var covXY = Engine.TensorDivideScalar(Engine.ReduceSum(Engine.TensorMultiply(dx, dy), axes, false), unbiased);
            var mx = meanX.Reshape(rows, windowColumns);
            var my = meanY.Reshape(rows, windowColumns);
            blocks.Add(Ssim(mx, my, varX, varY, covXY, c1, c2, two));
        }

        return blocks.Count == 0 ? NumOps.Zero : Engine.TensorSum(Engine.TensorConcatenate(blocks.ToArray(), 0));
    }

    /// <summary>((2 mx my + c1)(2 cov + c2)) / ((mx^2 + my^2 + c1)(vx + vy + c2)), element-wise.</summary>
    private static Tensor<T> Ssim(
        Tensor<T> meanX, Tensor<T> meanY, Tensor<T> varX, Tensor<T> varY, Tensor<T> covXY, T c1, T c2, T two)
    {
        var luminance = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(Engine.TensorMultiply(meanX, meanY), two), c1);
        var structure = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(covXY, two), c2);
        var meanEnergy = Engine.TensorAddScalar(
            Engine.TensorAdd(Engine.TensorSquare(meanX), Engine.TensorSquare(meanY)), c1);
        var varianceEnergy = Engine.TensorAddScalar(Engine.TensorAdd(varX, varY), c2);
        return Engine.TensorDivide(
            Engine.TensorMultiply(luminance, structure), Engine.TensorMultiply(meanEnergy, varianceEnergy));
    }

    private static T ComputeGlobalSSIM(
        Tensor<T> predicted,
        Tensor<T> groundTruth,
        T c1,
        T c2)
    {
        int count = predicted.Length;
        T meanX = Engine.TensorMean(predicted);
        T meanY = Engine.TensorMean(groundTruth);
        var dx = Engine.TensorSubtractScalar(predicted, meanX);
        var dy = Engine.TensorSubtractScalar(groundTruth, meanY);
        T varX = Engine.TensorSum(Engine.TensorSquare(dx));
        T varY = Engine.TensorSum(Engine.TensorSquare(dy));
        T covXY = Engine.TensorSum(Engine.TensorMultiply(dx, dy));

        if (count > 1)
        {
            T unbiased = NumOps.FromDouble(count - 1);
            varX = NumOps.Divide(varX, unbiased);
            varY = NumOps.Divide(varY, unbiased);
            covXY = NumOps.Divide(covXY, unbiased);
        }

        T two = NumOps.FromDouble(2);
        T numerator = NumOps.Multiply(
            NumOps.Add(NumOps.Multiply(two, NumOps.Multiply(meanX, meanY)), c1),
            NumOps.Add(NumOps.Multiply(two, covXY), c2));
        T denominator = NumOps.Multiply(
            NumOps.Add(NumOps.Add(NumOps.Multiply(meanX, meanX), NumOps.Multiply(meanY, meanY)), c1),
            NumOps.Add(NumOps.Add(varX, varY), c2));

        return NumOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Channel-averaged Sobel gradient magnitude, [H, W], zero on the one-pixel border. Sobel is linear, so the
    /// per-channel responses summed over channels equal the response of the channel sum.
    /// </summary>
    private static Tensor<T> ComputeEdgeMagnitude(Tensor<T> image)
    {
        var hwc = AsHeightWidthChannels(image);
        int height = hwc.Shape[0];
        int width = hwc.Shape[1];
        int channels = hwc.Shape[2];
        if (height < 3 || width < 3)
        {
            return new Tensor<T>(new[] { height, width });
        }

        var sum = Engine.ReduceSum(hwc, new[] { 2 }, false);
        int innerH = height - 2;
        int innerW = width - 2;
        Tensor<T> P(int dy, int dx) => Engine.TensorSlice(sum, new[] { dy, dx }, new[] { innerH, innerW });
        Tensor<T> Twice(Tensor<T> t) => Engine.TensorAdd(t, t);

        var gx = Engine.TensorSubtract(
            Engine.TensorAdd(Engine.TensorAdd(P(0, 2), Twice(P(1, 2))), P(2, 2)),
            Engine.TensorAdd(Engine.TensorAdd(P(0, 0), Twice(P(1, 0))), P(2, 0)));
        var gy = Engine.TensorSubtract(
            Engine.TensorAdd(Engine.TensorAdd(P(2, 0), Twice(P(2, 1))), P(2, 2)),
            Engine.TensorAdd(Engine.TensorAdd(P(0, 0), Twice(P(0, 1))), P(0, 2)));

        T channelCount = NumOps.FromDouble(channels);
        gx = Engine.TensorDivideScalar(gx, channelCount);
        gy = Engine.TensorDivideScalar(gy, channelCount);
        var magnitude = Engine.TensorSqrt(Engine.TensorAdd(Engine.TensorSquare(gx), Engine.TensorSquare(gy)));
        return Engine.TensorConstantPad(magnitude, new[] { 1, 1, 1, 1 }, NumOps.Zero);
    }

    /// <summary>
    /// Local 3 x 3 (all channels) mean and population standard deviation, [H, W, 2], zero on the border. The
    /// deviation uses the two-pass (centred) sum.
    /// </summary>
    private static Tensor<T> ComputeLocalStatistics(Tensor<T> image)
    {
        var hwc = AsHeightWidthChannels(image);
        int height = hwc.Shape[0];
        int width = hwc.Shape[1];
        const int halfWindow = 1;
        if (height < 3 || width < 3)
        {
            return new Tensor<T>(new[] { height, width, 2 });
        }

        // Windows are expanded in row blocks of at most 1 << 22 elements (as SumWindowSSIM does): the full window
        // tensor is 9 * channels times the image, about 1.8 GB for a 4K RGB double image.
        var axes = new[] { 2, 3, 4 };
        int windowRows = height - 2 * halfWindow;
        int windowColumns = width - 2 * halfWindow;
        int perWindow = hwc.Shape[2] * 9;
        int blockRows = Math.Max(1, (1 << 22) / Math.Max(1, windowColumns * perWindow));
        var meanBlocks = new List<Tensor<T>>();
        var deviationBlocks = new List<Tensor<T>>();
        for (int start = 0; start < windowRows; start += blockRows)
        {
            int rows = Math.Min(blockRows, windowRows - start);
            var windows = Windows(hwc, halfWindow, start, rows);
            var blockMean = Engine.ReduceMean(windows, axes, true);
            var centred = Engine.TensorSubtract(windows, Engine.TensorBroadcastTo(blockMean, windows.Shape.ToArray()));
            meanBlocks.Add(blockMean.Reshape(rows, windowColumns));
            deviationBlocks.Add(Engine.TensorSqrt(Engine.ReduceMean(Engine.TensorSquare(centred), axes, false)).Reshape(rows, windowColumns));
        }

        var mean = meanBlocks.Count == 1 ? meanBlocks[0] : Engine.TensorConcatenate(meanBlocks.ToArray(), 0);
        var deviation = deviationBlocks.Count == 1 ? deviationBlocks[0] : Engine.TensorConcatenate(deviationBlocks.ToArray(), 0);

        var padding = new[] { halfWindow, halfWindow, halfWindow, halfWindow };
        var meanPlane = Engine.TensorConstantPad(mean, padding, NumOps.Zero);
        var deviationPlane = Engine.TensorConstantPad(deviation, padding, NumOps.Zero);
        // Planar layout (all means, then all deviations) under the [H, W, 2] shape, as before.
        var planes = Engine.TensorConcatenate(
            new[] { meanPlane.Reshape(height * width), deviationPlane.Reshape(height * width) }, 0);
        return planes.Reshape(height, width, 2);
    }

    #endregion
}
