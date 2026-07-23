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

        double mse = ComputeMSE(predicted, groundTruth);

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

        double c1 = (k1 * maxValue) * (k1 * maxValue);
        double c2 = (k2 * maxValue) * (k2 * maxValue);

        int height = predicted.Shape[0];
        int width = predicted.Shape[1];
        int channels = predicted.Rank > 2 ? predicted.Shape[2] : 1;

        double totalSsim = 0.0;
        int numWindows = 0;

        int halfWindow = windowSize / 2;

        for (int y = halfWindow; y < height - halfWindow; y++)
        {
            for (int x = halfWindow; x < width - halfWindow; x++)
            {
                double windowSsim = ComputeWindowSSIM(
                    predicted, groundTruth, x, y, halfWindow, channels, c1, c2);
                totalSsim += windowSsim;
                numWindows++;
            }
        }

        if (numWindows == 0)
        {
            return ComputeGlobalSSIM(predicted, groundTruth, c1, c2);
        }

        return totalSsim / numWindows;
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

        var predEdges = ComputeEdgeMagnitude(predicted);
        var gtEdges = ComputeEdgeMagnitude(groundTruth);

        double edgeDifference = ComputeMSE(predEdges, gtEdges);

        var predStats = ComputeLocalStatistics(predicted);
        var gtStats = ComputeLocalStatistics(groundTruth);

        double statsDifference = ComputeMSE(predStats, gtStats);

        return Math.Sqrt(edgeDifference) + 0.5 * Math.Sqrt(statsDifference);
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
        return ComputeMSE(a, b);
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

        double sum = 0.0;
        int total = a.Length;

        var aData = a.ToArray();
        var bData = b.ToArray();

        for (int i = 0; i < total; i++)
        {
            double diff = Math.Abs(NumOps.ToDouble(aData[i]) - NumOps.ToDouble(bData[i]));
            sum += diff;
        }

        return sum / total;
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
    private static double ComputeMSE(Tensor<T> a, Tensor<T> b)
    {
        double sum = 0.0;
        int total = a.Length;

        var aData = a.ToArray();
        var bData = b.ToArray();

        for (int i = 0; i < total; i++)
        {
            double diff = NumOps.ToDouble(aData[i]) - NumOps.ToDouble(bData[i]);
            sum += diff * diff;
        }

        return sum / total;
    }

    /// <summary>
    /// Computes SSIM for a single window.
    /// </summary>
    private static double ComputeWindowSSIM(
        Tensor<T> predicted,
        Tensor<T> groundTruth,
        int centerX,
        int centerY,
        int halfWindow,
        int channels,
        double c1,
        double c2)
    {
        double meanX = 0, meanY = 0;
        double varX = 0, varY = 0, covXY = 0;
        int count = 0;

        for (int dy = -halfWindow; dy <= halfWindow; dy++)
        {
            for (int dx = -halfWindow; dx <= halfWindow; dx++)
            {
                int y = centerY + dy;
                int x = centerX + dx;

                for (int c = 0; c < channels; c++)
                {
                    double px = channels > 1
                        ? NumOps.ToDouble(predicted[y, x, c])
                        : NumOps.ToDouble(predicted[y, x]);
                    double py = channels > 1
                        ? NumOps.ToDouble(groundTruth[y, x, c])
                        : NumOps.ToDouble(groundTruth[y, x]);

                    meanX += px;
                    meanY += py;
                    count++;
                }
            }
        }

        meanX /= count;
        meanY /= count;

        for (int dy = -halfWindow; dy <= halfWindow; dy++)
        {
            for (int dx = -halfWindow; dx <= halfWindow; dx++)
            {
                int y = centerY + dy;
                int x = centerX + dx;

                for (int c = 0; c < channels; c++)
                {
                    double px = channels > 1
                        ? NumOps.ToDouble(predicted[y, x, c])
                        : NumOps.ToDouble(predicted[y, x]);
                    double py = channels > 1
                        ? NumOps.ToDouble(groundTruth[y, x, c])
                        : NumOps.ToDouble(groundTruth[y, x]);

                    varX += (px - meanX) * (px - meanX);
                    varY += (py - meanY) * (py - meanY);
                    covXY += (px - meanX) * (py - meanY);
                }
            }
        }

        varX /= (count - 1);
        varY /= (count - 1);
        covXY /= (count - 1);

        double numerator = (2 * meanX * meanY + c1) * (2 * covXY + c2);
        double denominator = (meanX * meanX + meanY * meanY + c1) * (varX + varY + c2);

        return numerator / denominator;
    }

    /// <summary>
    /// Computes global SSIM when image is too small for windowed approach.
    /// </summary>
    private static double ComputeGlobalSSIM(
        Tensor<T> predicted,
        Tensor<T> groundTruth,
        double c1,
        double c2)
    {
        var pData = predicted.ToArray();
        var gData = groundTruth.ToArray();
        int count = pData.Length;

        double meanX = 0, meanY = 0;
        for (int i = 0; i < count; i++)
        {
            meanX += NumOps.ToDouble(pData[i]);
            meanY += NumOps.ToDouble(gData[i]);
        }
        meanX /= count;
        meanY /= count;

        double varX = 0, varY = 0, covXY = 0;
        for (int i = 0; i < count; i++)
        {
            double px = NumOps.ToDouble(pData[i]);
            double py = NumOps.ToDouble(gData[i]);
            varX += (px - meanX) * (px - meanX);
            varY += (py - meanY) * (py - meanY);
            covXY += (px - meanX) * (py - meanY);
        }

        if (count > 1)
        {
            varX /= (count - 1);
            varY /= (count - 1);
            covXY /= (count - 1);
        }

        double numerator = (2 * meanX * meanY + c1) * (2 * covXY + c2);
        double denominator = (meanX * meanX + meanY * meanY + c1) * (varX + varY + c2);

        return numerator / denominator;
    }

    /// <summary>
    /// Computes edge magnitude using Sobel-like operators.
    /// </summary>
    private static Tensor<T> ComputeEdgeMagnitude(Tensor<T> image)
    {
        int height = image.Shape[0];
        int width = image.Shape[1];
        int channels = image.Rank > 2 ? image.Shape[2] : 1;

        var result = new T[height * width];

        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                double gx = 0, gy = 0;

                for (int c = 0; c < channels; c++)
                {
                    double p00 = GetPixel(image, y - 1, x - 1, c, channels);
                    double p01 = GetPixel(image, y - 1, x, c, channels);
                    double p02 = GetPixel(image, y - 1, x + 1, c, channels);
                    double p10 = GetPixel(image, y, x - 1, c, channels);
                    double p12 = GetPixel(image, y, x + 1, c, channels);
                    double p20 = GetPixel(image, y + 1, x - 1, c, channels);
                    double p21 = GetPixel(image, y + 1, x, c, channels);
                    double p22 = GetPixel(image, y + 1, x + 1, c, channels);

                    gx += (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
                    gy += (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
                }

                gx /= channels;
                gy /= channels;

                double magnitude = Math.Sqrt(gx * gx + gy * gy);
                result[y * width + x] = NumOps.FromDouble(magnitude);
            }
        }

        return new Tensor<T>(result, [height, width]);
    }

    /// <summary>
    /// Gets a pixel value from an image tensor.
    /// </summary>
    private static double GetPixel(Tensor<T> image, int y, int x, int c, int channels)
    {
        if (channels > 1)
        {
            return NumOps.ToDouble(image[y, x, c]);
        }
        return NumOps.ToDouble(image[y, x]);
    }

    /// <summary>
    /// Computes local statistics (mean, variance) as feature maps.
    /// </summary>
    private static Tensor<T> ComputeLocalStatistics(Tensor<T> image)
    {
        int height = image.Shape[0];
        int width = image.Shape[1];
        int channels = image.Rank > 2 ? image.Shape[2] : 1;

        var result = new T[height * width * 2];

        int windowSize = 3;
        int halfWindow = windowSize / 2;

        for (int y = halfWindow; y < height - halfWindow; y++)
        {
            for (int x = halfWindow; x < width - halfWindow; x++)
            {
                double mean = 0;
                double variance = 0;
                int count = 0;

                for (int dy = -halfWindow; dy <= halfWindow; dy++)
                {
                    for (int dx = -halfWindow; dx <= halfWindow; dx++)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            double val = GetPixel(image, y + dy, x + dx, c, channels);
                            mean += val;
                            count++;
                        }
                    }
                }
                mean /= count;

                for (int dy = -halfWindow; dy <= halfWindow; dy++)
                {
                    for (int dx = -halfWindow; dx <= halfWindow; dx++)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            double val = GetPixel(image, y + dy, x + dx, c, channels);
                            variance += (val - mean) * (val - mean);
                        }
                    }
                }
                variance /= count;

                int idx = y * width + x;
                result[idx] = NumOps.FromDouble(mean);
                result[height * width + idx] = NumOps.FromDouble(Math.Sqrt(variance));
            }
        }

        return new Tensor<T>(result, [height, width, 2]);
    }

    #endregion
}
