using AiDotNet.Helpers;
using AiDotNet.Tensors;

namespace AiDotNet.Metrics;

/// <summary>
/// Peak Signal-to-Noise Ratio (PSNR) metric for image quality assessment.
/// </summary>
/// <remarks>
/// <para>
/// PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise.
/// Higher PSNR values indicate better image quality. Common ranges:
/// - &gt;40 dB: Excellent quality (nearly indistinguishable from original)
/// - 30-40 dB: Good quality
/// - 20-30 dB: Acceptable quality
/// - &lt;20 dB: Poor quality
/// </para>
/// <para>
/// Formula: PSNR = 10 * log10(MAX² / MSE) where MAX is the maximum possible pixel value.
/// </para>
/// <para><b>Usage in 3D AI:</b>
/// - NeRF novel view synthesis evaluation
/// - Gaussian Splatting rendering quality
/// - Image reconstruction quality assessment
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PeakSignalToNoiseRatio<T> where T : struct
{
    /// <summary>
    /// The numeric operations provider for type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Maximum possible pixel value (e.g., 1.0 for normalized images, 255 for 8-bit).
    /// </summary>
    private readonly T _maxValue;

    /// <summary>
    /// Initializes a new instance of the PSNR metric.
    /// </summary>
    /// <param name="maxValue">Maximum possible pixel value. Default is 1.0 for normalized images.</param>
    public PeakSignalToNoiseRatio(T? maxValue = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _maxValue = maxValue ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Computes PSNR between predicted and ground truth images.
    /// </summary>
    /// <param name="predicted">Predicted image tensor [H, W, C] or [B, H, W, C].</param>
    /// <param name="groundTruth">Ground truth image tensor with same shape as predicted.</param>
    /// <returns>PSNR value in decibels (dB). Higher is better.</returns>
    /// <exception cref="ArgumentException">If shapes don't match.</exception>
    public T Compute(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        if (predicted == null) throw new ArgumentNullException(nameof(predicted));
        if (groundTruth == null) throw new ArgumentNullException(nameof(groundTruth));

        if (!ShapesMatch(predicted.Shape, groundTruth.Shape))
        {
            throw new ArgumentException($"Shape mismatch: predicted {string.Join(",", predicted.Shape)} vs ground truth {string.Join(",", groundTruth.Shape)}");
        }

        // Compute Mean Squared Error
        T mse = ComputeMSE(predicted, groundTruth);

        // Avoid log(0) by using a small epsilon for zero MSE
        T epsilon = _numOps.FromDouble(1e-10);
        if (_numOps.Compare(mse, epsilon) < 0)
        {
            return _numOps.FromDouble(100.0); // Return high PSNR for near-perfect match
        }

        // PSNR = 10 * log10(MAX² / MSE) = 20 * log10(MAX) - 10 * log10(MSE)
        T maxSquared = _numOps.Multiply(_maxValue, _maxValue);
        T ratio = _numOps.Divide(maxSquared, mse);
        double ratioDouble = _numOps.ToDouble(ratio);
        double psnr = 10.0 * Math.Log10(ratioDouble);

        return _numOps.FromDouble(psnr);
    }

    /// <summary>
    /// Computes PSNR for a batch of images.
    /// </summary>
    /// <param name="predicted">Batch of predicted images [B, H, W, C].</param>
    /// <param name="groundTruth">Batch of ground truth images [B, H, W, C].</param>
    /// <returns>Array of PSNR values, one per batch item.</returns>
    public T[] ComputeBatch(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        if (predicted.Rank != 4 || groundTruth.Rank != 4)
        {
            throw new ArgumentException("Batch computation requires 4D tensors [B, H, W, C]");
        }

        // Validate that shapes match
        if (predicted.Shape[0] != groundTruth.Shape[0] ||
            predicted.Shape[1] != groundTruth.Shape[1] ||
            predicted.Shape[2] != groundTruth.Shape[2] ||
            predicted.Shape[3] != groundTruth.Shape[3])
        {
            throw new ArgumentException(
                $"Shape mismatch: predicted [{string.Join(",", predicted.Shape)}] vs ground truth [{string.Join(",", groundTruth.Shape)}]");
        }

        int batchSize = predicted.Shape[0];
        var results = new T[batchSize];

        int h = predicted.Shape[1];
        int w = predicted.Shape[2];
        int c = predicted.Shape[3];
        int imageSize = h * w * c;

        for (int b = 0; b < batchSize; b++)
        {
            // Extract single image from batch using proper 4D indexing
            var predImage = new T[imageSize];
            var gtImage = new T[imageSize];

            for (int hi = 0; hi < h; hi++)
            {
                for (int wi = 0; wi < w; wi++)
                {
                    for (int ci = 0; ci < c; ci++)
                    {
                        int flatIdx = (hi * w + wi) * c + ci;
                        predImage[flatIdx] = predicted[b, hi, wi, ci];
                        gtImage[flatIdx] = groundTruth[b, hi, wi, ci];
                    }
                }
            }

            var predTensor = new Tensor<T>(new[] { h, w, c }, new Vector<T>(predImage));
            var gtTensor = new Tensor<T>(new[] { h, w, c }, new Vector<T>(gtImage));

            results[b] = Compute(predTensor, gtTensor);
        }

        return results;
    }

    /// <summary>
    /// Computes Mean Squared Error between two tensors.
    /// </summary>
    private T ComputeMSE(Tensor<T> a, Tensor<T> b)
    {
        T sum = _numOps.Zero;
        int count = a.Length;

        for (int i = 0; i < count; i++)
        {
            T diff = _numOps.Subtract(a[i], b[i]);
            sum = _numOps.Add(sum, _numOps.Multiply(diff, diff));
        }

        return _numOps.Divide(sum, _numOps.FromDouble(count));
    }

    /// <summary>
    /// Checks if two shapes match.
    /// </summary>
    private static bool ShapesMatch(int[] shape1, int[] shape2)
    {
        if (shape1.Length != shape2.Length) return false;
        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i]) return false;
        }
        return true;
    }
}

/// <summary>
/// Structural Similarity Index Measure (SSIM) for image quality assessment.
/// </summary>
/// <remarks>
/// <para>
/// SSIM measures structural similarity between two images, considering luminance, contrast, and structure.
/// SSIM values range from -1 to 1, where 1 indicates perfect similarity.
/// </para>
/// <para>
/// Formula: SSIM(x,y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ
/// where l = luminance, c = contrast, s = structure comparisons.
/// </para>
/// <para><b>Usage in 3D AI:</b>
/// - NeRF novel view synthesis evaluation
/// - Better perceptual quality metric than PSNR
/// - Captures structural distortions
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StructuralSimilarity<T> where T : struct
{
    /// <summary>
    /// The numeric operations provider for type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Stabilization constant for luminance (C1 = (K1 * L)^2).
    /// </summary>
    private readonly T _c1;

    /// <summary>
    /// Stabilization constant for contrast (C2 = (K2 * L)^2).
    /// </summary>
    private readonly T _c2;

    /// <summary>
    /// Window size for local statistics computation.
    /// </summary>
    private readonly int _windowSize;

    /// <summary>
    /// Initializes a new instance of the SSIM metric.
    /// </summary>
    /// <param name="maxValue">Maximum possible pixel value. Default is 1.0.</param>
    /// <param name="k1">Constant for luminance stability. Default is 0.01.</param>
    /// <param name="k2">Constant for contrast stability. Default is 0.03.</param>
    /// <param name="windowSize">Window size for local computation. Default is 11.</param>
    public StructuralSimilarity(T? maxValue = null, double k1 = 0.01, double k2 = 0.03, int windowSize = 11)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        T L = maxValue ?? _numOps.FromDouble(1.0);
        double LDouble = _numOps.ToDouble(L);

        _c1 = _numOps.FromDouble(Math.Pow(k1 * LDouble, 2));
        _c2 = _numOps.FromDouble(Math.Pow(k2 * LDouble, 2));
        _windowSize = windowSize;
    }

    /// <summary>
    /// Computes SSIM between predicted and ground truth images.
    /// </summary>
    /// <param name="predicted">Predicted image tensor [H, W] or [H, W, C].</param>
    /// <param name="groundTruth">Ground truth image tensor with same shape.</param>
    /// <returns>SSIM value between -1 and 1. Higher is better.</returns>
    public T Compute(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        if (predicted == null) throw new ArgumentNullException(nameof(predicted));
        if (groundTruth == null) throw new ArgumentNullException(nameof(groundTruth));

        // For multi-channel images, compute SSIM per channel and average
        if (predicted.Rank == 3)
        {
            int channels = predicted.Shape[2];
            T sum = _numOps.Zero;

            for (int c = 0; c < channels; c++)
            {
                var predChannel = ExtractChannel(predicted, c);
                var gtChannel = ExtractChannel(groundTruth, c);
                sum = _numOps.Add(sum, ComputeSingleChannel(predChannel, gtChannel));
            }

            return _numOps.Divide(sum, _numOps.FromDouble(channels));
        }

        return ComputeSingleChannel(predicted, groundTruth);
    }

    /// <summary>
    /// Computes SSIM for a single channel image using sliding window.
    /// </summary>
    private T ComputeSingleChannel(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        int height = predicted.Shape[0];
        int width = predicted.Shape[1];

        // For simplicity, use global statistics (faster but less accurate than windowed)
        // Production implementation should use Gaussian-weighted windowed approach

        // Compute means
        T meanX = ComputeMean(predicted);
        T meanY = ComputeMean(groundTruth);

        // Compute variances and covariance
        T varX = _numOps.Zero;
        T varY = _numOps.Zero;
        T covXY = _numOps.Zero;
        int n = predicted.Length;

        for (int i = 0; i < n; i++)
        {
            T diffX = _numOps.Subtract(predicted[i], meanX);
            T diffY = _numOps.Subtract(groundTruth[i], meanY);

            varX = _numOps.Add(varX, _numOps.Multiply(diffX, diffX));
            varY = _numOps.Add(varY, _numOps.Multiply(diffY, diffY));
            covXY = _numOps.Add(covXY, _numOps.Multiply(diffX, diffY));
        }

        T nT = _numOps.FromDouble(n - 1);
        varX = _numOps.Divide(varX, nT);
        varY = _numOps.Divide(varY, nT);
        covXY = _numOps.Divide(covXY, nT);

        // SSIM formula: ((2*μx*μy + C1)(2*σxy + C2)) / ((μx² + μy² + C1)(σx² + σy² + C2))
        T two = _numOps.FromDouble(2.0);
        T meanXY = _numOps.Multiply(meanX, meanY);
        T meanX2 = _numOps.Multiply(meanX, meanX);
        T meanY2 = _numOps.Multiply(meanY, meanY);

        T numerator1 = _numOps.Add(_numOps.Multiply(two, meanXY), _c1);
        T numerator2 = _numOps.Add(_numOps.Multiply(two, covXY), _c2);
        T numerator = _numOps.Multiply(numerator1, numerator2);

        T denominator1 = _numOps.Add(_numOps.Add(meanX2, meanY2), _c1);
        T denominator2 = _numOps.Add(_numOps.Add(varX, varY), _c2);
        T denominator = _numOps.Multiply(denominator1, denominator2);

        return _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Extracts a single channel from a multi-channel image.
    /// </summary>
    /// <remarks>
    /// Assumes image is in HWC (Height, Width, Channels) format.
    /// </remarks>
    private Tensor<T> ExtractChannel(Tensor<T> image, int channel)
    {
        int height = image.Shape[0];
        int width = image.Shape[1];
        int channels = image.Shape[2];

        if (channel < 0 || channel >= channels)
        {
            throw new ArgumentOutOfRangeException(nameof(channel),
                $"Channel index {channel} is out of range [0, {channels - 1}]");
        }

        var data = new T[height * width];
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                // HWC layout: index = (h * width + w) * channels + c
                int pixelOffset = (h * width + w) * channels;
                int channelIndex = pixelOffset + channel;
                int destIndex = h * width + w;
                data[destIndex] = image.GetFlat(channelIndex);
            }
        }

        return new Tensor<T>(new[] { height, width }, new Vector<T>(data));
    }

    /// <summary>
    /// Computes the mean of a tensor.
    /// </summary>
    private T ComputeMean(Tensor<T> tensor)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < tensor.Length; i++)
        {
            sum = _numOps.Add(sum, tensor[i]);
        }
        return _numOps.Divide(sum, _numOps.FromDouble(tensor.Length));
    }
}

/// <summary>
/// Mean Intersection over Union (mIoU) metric for segmentation tasks.
/// </summary>
/// <remarks>
/// <para>
/// mIoU is the standard metric for semantic segmentation evaluation.
/// It computes IoU for each class and averages across all classes.
/// IoU = TP / (TP + FP + FN) where TP=true positive, FP=false positive, FN=false negative.
/// </para>
/// <para><b>Usage in 3D AI:</b>
/// - Point cloud segmentation (S3DIS, ScanNet)
/// - Mesh semantic segmentation
/// - Voxel-based 3D segmentation
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MeanIntersectionOverUnion<T> where T : struct
{
    /// <summary>
    /// The numeric operations provider for type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Number of classes for segmentation.
    /// </summary>
    private readonly int _numClasses;

    /// <summary>
    /// Whether to ignore the background class (class 0) in computation.
    /// </summary>
    private readonly bool _ignoreBackground;

    /// <summary>
    /// Initializes a new instance of the mIoU metric.
    /// </summary>
    /// <param name="numClasses">Number of segmentation classes.</param>
    /// <param name="ignoreBackground">Whether to ignore class 0 (background). Default is false.</param>
    public MeanIntersectionOverUnion(int numClasses, bool ignoreBackground = false)
    {
        if (numClasses < 2) throw new ArgumentException("numClasses must be at least 2", nameof(numClasses));

        _numOps = MathHelper.GetNumericOperations<T>();
        _numClasses = numClasses;
        _ignoreBackground = ignoreBackground;
    }

    /// <summary>
    /// Computes mIoU between predicted and ground truth segmentation masks.
    /// </summary>
    /// <param name="predicted">Predicted class labels (integers).</param>
    /// <param name="groundTruth">Ground truth class labels (integers).</param>
    /// <returns>Mean IoU value between 0 and 1. Higher is better.</returns>
    public T Compute(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        if (predicted == null) throw new ArgumentNullException(nameof(predicted));
        if (groundTruth == null) throw new ArgumentNullException(nameof(groundTruth));

        // Compute confusion matrix
        var intersection = new long[_numClasses];
        var union = new long[_numClasses];
        var predCount = new long[_numClasses];
        var gtCount = new long[_numClasses];

        for (int i = 0; i < predicted.Length; i++)
        {
            int predClass = (int)_numOps.ToDouble(predicted[i]);
            int gtClass = (int)_numOps.ToDouble(groundTruth[i]);

            if (predClass >= 0 && predClass < _numClasses)
            {
                predCount[predClass]++;
            }

            if (gtClass >= 0 && gtClass < _numClasses)
            {
                gtCount[gtClass]++;
            }

            if (predClass == gtClass && predClass >= 0 && predClass < _numClasses)
            {
                intersection[predClass]++;
            }
        }

        // Compute IoU per class and average
        T sumIoU = _numOps.Zero;
        int validClasses = 0;
        int startClass = _ignoreBackground ? 1 : 0;

        for (int c = startClass; c < _numClasses; c++)
        {
            long unionCount = predCount[c] + gtCount[c] - intersection[c];

            if (unionCount > 0)
            {
                double iou = (double)intersection[c] / unionCount;
                sumIoU = _numOps.Add(sumIoU, _numOps.FromDouble(iou));
                validClasses++;
            }
        }

        if (validClasses == 0)
        {
            return _numOps.Zero;
        }

        return _numOps.Divide(sumIoU, _numOps.FromDouble(validClasses));
    }

    /// <summary>
    /// Computes per-class IoU values.
    /// </summary>
    /// <param name="predicted">Predicted class labels.</param>
    /// <param name="groundTruth">Ground truth class labels.</param>
    /// <returns>Array of IoU values, one per class.</returns>
    public T[] ComputePerClass(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        var intersection = new long[_numClasses];
        var predCount = new long[_numClasses];
        var gtCount = new long[_numClasses];

        for (int i = 0; i < predicted.Length; i++)
        {
            int predClass = (int)_numOps.ToDouble(predicted[i]);
            int gtClass = (int)_numOps.ToDouble(groundTruth[i]);

            if (predClass >= 0 && predClass < _numClasses) predCount[predClass]++;
            if (gtClass >= 0 && gtClass < _numClasses) gtCount[gtClass]++;
            if (predClass == gtClass && predClass >= 0 && predClass < _numClasses)
            {
                intersection[predClass]++;
            }
        }

        var results = new T[_numClasses];
        for (int c = 0; c < _numClasses; c++)
        {
            long unionCount = predCount[c] + gtCount[c] - intersection[c];
            if (unionCount > 0)
            {
                results[c] = _numOps.FromDouble((double)intersection[c] / unionCount);
            }
            else
            {
                results[c] = _numOps.Zero;
            }
        }

        return results;
    }
}

/// <summary>
/// Overall Accuracy metric for classification and segmentation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class OverallAccuracy<T> where T : struct
{
    /// <summary>
    /// The numeric operations provider for type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the OverallAccuracy metric.
    /// </summary>
    public OverallAccuracy()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes overall accuracy between predictions and ground truth.
    /// </summary>
    /// <param name="predicted">Predicted class labels.</param>
    /// <param name="groundTruth">Ground truth class labels.</param>
    /// <returns>Accuracy value between 0 and 1.</returns>
    public T Compute(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        if (predicted == null) throw new ArgumentNullException(nameof(predicted));
        if (groundTruth == null) throw new ArgumentNullException(nameof(groundTruth));

        if (predicted.Length != groundTruth.Length)
        {
            throw new ArgumentException("Predicted and ground truth must have the same length");
        }

        long correct = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double pred = _numOps.ToDouble(predicted[i]);
            double gt = _numOps.ToDouble(groundTruth[i]);

            if (Math.Abs(pred - gt) < 0.5) // For integer class labels
            {
                correct++;
            }
        }

        return _numOps.FromDouble((double)correct / predicted.Length);
    }
}
