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

        if (!ShapesMatch(predicted._shape, groundTruth._shape))
        {
            throw new ArgumentException($"Shape mismatch: predicted {string.Join(",", predicted._shape)} vs ground truth {string.Join(",", groundTruth._shape)}");
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
        return _numOps.Multiply(_numOps.FromDouble(10.0),
            _numOps.Divide(_numOps.Log(ratio), _numOps.Log(_numOps.FromDouble(10.0))));
    }

    /// <summary>
    /// Computes PSNR for a batch of images.
    /// </summary>
    /// <param name="predicted">Batch of predicted images [B, H, W, C].</param>
    /// <param name="groundTruth">Batch of ground truth images [B, H, W, C].</param>
    /// <returns>Array of PSNR values, one per batch item.</returns>
    public T[] ComputeBatch(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        if (predicted == null) throw new ArgumentNullException(nameof(predicted));
        if (groundTruth == null) throw new ArgumentNullException(nameof(groundTruth));

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
                $"Shape mismatch: predicted [{string.Join(",", predicted._shape)}] vs ground truth [{string.Join(",", groundTruth._shape)}]");
        }

        var engine = AiDotNetEngine.Current;

        // Per-image MSE over [H, W, C], then PSNR = 10 * log10(MAX² / MSE) per image.
        var diff = engine.TensorSubtract(predicted, groundTruth);
        var mse = engine.ReduceMean(engine.TensorMultiply(diff, diff), new[] { 1, 2, 3 }, false);

        T maxSquared = _numOps.Multiply(_maxValue, _maxValue);
        var ratio = engine.TensorDivide(engine.TensorAddScalar(engine.TensorZerosLike(mse), maxSquared), mse);
        var psnr = engine.TensorMultiplyScalar(engine.TensorLog10(ratio), _numOps.FromDouble(10.0));

        // Near-perfect matches (MSE below 1e-10) report 100 dB, as Compute does.
        var nearPerfect = engine.TensorLessThan(mse, _numOps.FromDouble(1e-10));
        var capped = engine.TensorAddScalar(engine.TensorZerosLike(mse), _numOps.FromDouble(100.0));
        return engine.TensorWhere(nearPerfect, capped, psnr).ToArray();
    }

    /// <summary>
    /// Computes Mean Squared Error between two tensors.
    /// </summary>
    private T ComputeMSE(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        var diff = engine.TensorSubtract(a, b);
        return _numOps.Divide(engine.TensorSumOfSquares(diff), _numOps.FromDouble(a.Length));
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
    /// Standard deviation of the Gaussian weighting window.
    /// </summary>
    private readonly double _gaussianSigma;

    /// <summary>
    /// Initializes a new instance of the SSIM metric.
    /// </summary>
    /// <param name="maxValue">Maximum possible pixel value. Default is 1.0.</param>
    /// <param name="k1">Constant for luminance stability. Default is 0.01.</param>
    /// <param name="k2">Constant for contrast stability. Default is 0.03.</param>
    /// <param name="windowSize">Window size for local computation. Default is 11 (Wang et al. 2004).</param>
    /// <param name="gaussianSigma">Standard deviation of the Gaussian window. Default is 1.5 (Wang et al. 2004).</param>
    public StructuralSimilarity(T? maxValue = null, double k1 = 0.01, double k2 = 0.03, int windowSize = 11,
        double gaussianSigma = 1.5)
    {
        if (windowSize <= 0) throw new ArgumentOutOfRangeException(nameof(windowSize), "Window size must be positive.");
        if (gaussianSigma <= 0) throw new ArgumentOutOfRangeException(nameof(gaussianSigma), "Sigma must be positive.");

        _numOps = MathHelper.GetNumericOperations<T>();
        T L = maxValue ?? _numOps.FromDouble(1.0);
        double LDouble = _numOps.ToDouble(L);

        _c1 = _numOps.FromDouble(Math.Pow(k1 * LDouble, 2));
        _c2 = _numOps.FromDouble(Math.Pow(k2 * LDouble, 2));
        _windowSize = windowSize;
        _gaussianSigma = gaussianSigma;
    }

    /// <summary>
    /// Computes SSIM between predicted and ground truth images.
    /// </summary>
    /// <param name="predicted">Predicted image tensor [H, W] or [H, W, C].</param>
    /// <param name="groundTruth">Ground truth image tensor with same shape.</param>
    /// <returns>SSIM value between -1 and 1. Higher is better.</returns>
    /// <remarks>
    /// Follows Wang et al. (2004): local statistics under a normalised Gaussian window (eq. 14-16) and the mean
    /// SSIM over all window positions (MSSIM, eq. 17), as scikit-image's structural_similarity computes it with
    /// gaussian_weights=True and use_sample_covariance=False. An image smaller than the window in either dimension
    /// is a single window: the SSIM index of eq. 13 with the unweighted statistics of eq. 2-4 (N - 1 divisor).
    /// Multi-channel images average the per-channel values.
    /// </remarks>
    public T Compute(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        if (predicted == null) throw new ArgumentNullException(nameof(predicted));
        if (groundTruth == null) throw new ArgumentNullException(nameof(groundTruth));

        return ComputeChannelBatch(ToChannelBatch(predicted), ToChannelBatch(groundTruth));
    }

    /// <summary>
    /// Mean SSIM over every frame and channel of a video [T, H, W, C] (or [T, C, H, W] when channels-first),
    /// which equals the mean of the per-frame SSIM values because every frame has the same number of windows.
    /// </summary>
    internal T ComputeFrames(Tensor<T> predicted, Tensor<T> groundTruth, bool isChannelsFirst)
    {
        return ComputeChannelBatch(FramesToChannelBatch(predicted, isChannelsFirst),
            FramesToChannelBatch(groundTruth, isChannelsFirst));
    }

    /// <summary>
    /// Mean SSIM of a [B, 1, H, W] batch of single-channel images.
    /// </summary>
    private T ComputeChannelBatch(Tensor<T> x, Tensor<T> y)
    {
        int height = x.Shape[2];
        int width = x.Shape[3];

        bool windowed = height >= _windowSize && width >= _windowSize;
        Tensor<T> window = windowed ? GaussianWindow() : UniformWindow(height, width);
        int n = height * width;
        T covNorm = windowed ? _numOps.One : _numOps.FromDouble((double)n / (n - 1));

        var engine = AiDotNetEngine.Current;
        var muX = engine.Conv2D(x, window, 1, 0, 1);
        var muY = engine.Conv2D(y, window, 1, 0, 1);
        var muXX = engine.TensorMultiply(muX, muX);
        var muYY = engine.TensorMultiply(muY, muY);
        var muXY = engine.TensorMultiply(muX, muY);

        var varX = engine.TensorMultiplyScalar(
            engine.TensorSubtract(engine.Conv2D(engine.TensorMultiply(x, x), window, 1, 0, 1), muXX), covNorm);
        var varY = engine.TensorMultiplyScalar(
            engine.TensorSubtract(engine.Conv2D(engine.TensorMultiply(y, y), window, 1, 0, 1), muYY), covNorm);
        var covXY = engine.TensorMultiplyScalar(
            engine.TensorSubtract(engine.Conv2D(engine.TensorMultiply(x, y), window, 1, 0, 1), muXY), covNorm);

        // SSIM map: ((2*μx*μy + C1)(2*σxy + C2)) / ((μx² + μy² + C1)(σx² + σy² + C2))
        T two = _numOps.FromDouble(2.0);
        var numerator = engine.TensorMultiply(
            engine.TensorAddScalar(engine.TensorMultiplyScalar(muXY, two), _c1),
            engine.TensorAddScalar(engine.TensorMultiplyScalar(covXY, two), _c2));
        var denominator = engine.TensorMultiply(
            engine.TensorAddScalar(engine.TensorAdd(muXX, muYY), _c1),
            engine.TensorAddScalar(engine.TensorAdd(varX, varY), _c2));

        // Every channel has the same number of windows, so the mean of the map is the mean of per-channel MSSIMs.
        return engine.TensorMean(engine.TensorDivide(numerator, denominator));
    }

    /// <summary>
    /// Reshapes an image to a [channels, 1, H, W] batch: [H, W] is one channel, [H, W, C] (HWC) is split per channel,
    /// and any other rank is treated as one flat channel.
    /// </summary>
    private static Tensor<T> ToChannelBatch(Tensor<T> image)
    {
        if (image.Rank == 2)
        {
            return image.Reshape(new[] { 1, 1, image.Shape[0], image.Shape[1] });
        }

        if (image.Rank == 3)
        {
            var channelsFirst = AiDotNetEngine.Current.TensorPermute(image, new[] { 2, 0, 1 });
            return channelsFirst.Reshape(new[] { image.Shape[2], 1, image.Shape[0], image.Shape[1] });
        }

        return image.Reshape(new[] { 1, 1, 1, image.Length });
    }

    /// <summary>
    /// Reshapes frames [T, H, W, C] (or [T, C, H, W]) to a [T * C, 1, H, W] batch.
    /// </summary>
    private static Tensor<T> FramesToChannelBatch(Tensor<T> frames, bool isChannelsFirst)
    {
        var channelsFirst = isChannelsFirst
            ? frames
            : AiDotNetEngine.Current.TensorPermute(frames, new[] { 0, 3, 1, 2 });
        return channelsFirst.Reshape(new[]
        {
            channelsFirst.Shape[0] * channelsFirst.Shape[1], 1, channelsFirst.Shape[2], channelsFirst.Shape[3]
        });
    }

    /// <summary>
    /// Normalised 2D Gaussian window [1, 1, w, w] with the configured size and sigma.
    /// </summary>
    private Tensor<T> GaussianWindow()
    {
        var engine = AiDotNetEngine.Current;
        double center = (_windowSize - 1) / 2.0;
        var offsets = engine.TensorArange<T>(-center, center + 0.5, 1.0);
        var exponent = engine.TensorMultiplyScalar(engine.TensorMultiply(offsets, offsets),
            _numOps.FromDouble(-1.0 / (2.0 * _gaussianSigma * _gaussianSigma)));
        var g = engine.TensorExp(exponent);
        g = engine.TensorDivideScalar(g, engine.TensorSum(g));
        return engine.TensorOuter(g, g).Reshape(new[] { 1, 1, _windowSize, _windowSize });
    }

    /// <summary>
    /// Uniform window [1, 1, H, W] covering the whole image (weights 1 / (H * W)).
    /// </summary>
    private Tensor<T> UniformWindow(int height, int width)
    {
        var engine = AiDotNetEngine.Current;
        var ones = engine.TensorAddScalar(new Tensor<T>(new[] { 1, 1, height, width }), _numOps.One);
        return engine.TensorDivideScalar(ones, _numOps.FromDouble(height * width));
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

        // Accumulate counts in float for low-precision storage; a valid image can exceed Half's finite range.
        if (typeof(T) == typeof(Half) || typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16))
            return _numOps.FromDouble(new MeanIntersectionOverUnion<float>(_numClasses, _ignoreBackground)
                .Compute(predicted.Cast<float>(), groundTruth.Cast<float>()));

        var engine = AiDotNetEngine.Current;
        var (intersection, union) = ComputeIntersectionAndUnion(predicted, groundTruth);

        // Average IoU over the classes that occur (union > 0), skipping class 0 when background is ignored.
        var present = engine.TensorGreaterThan(union, _numOps.Zero);
        if (_ignoreBackground)
        {
            var notBackground = engine.TensorGreaterThan(
                engine.TensorArange<T>(0, _numClasses, 1), _numOps.Zero);
            present = engine.TensorMultiply(present, notBackground);
        }

        T validClasses = engine.TensorSum(present);
        if (!_numOps.GreaterThan(validClasses, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        var iou = PerClassIoU(intersection, union);
        return _numOps.Divide(engine.TensorSum(engine.TensorMultiply(iou, present)), validClasses);
    }

    /// <summary>
    /// Computes per-class IoU values.
    /// </summary>
    /// <param name="predicted">Predicted class labels.</param>
    /// <param name="groundTruth">Ground truth class labels.</param>
    /// <returns>Array of IoU values, one per class.</returns>
    public T[] ComputePerClass(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        if (typeof(T) == typeof(Half) || typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16))
            return Tensor<float>.FromVector(new Vector<float>(new MeanIntersectionOverUnion<float>(_numClasses, _ignoreBackground)
                .ComputePerClass(predicted.Cast<float>(), groundTruth.Cast<float>()))).Cast<T>().ToArray();

        var (intersection, union) = ComputeIntersectionAndUnion(predicted, groundTruth);
        return PerClassIoU(intersection, union).ToArray();
    }

    /// <summary>
    /// Per-class intersection and union counts [numClasses] from the confusion matrix. Labels are truncated toward
    /// zero; labels outside [0, numClasses) count for no class.
    /// </summary>
    /// <remarks>
    /// The (numClasses + 1)^2 confusion matrix (the extra row and column collect out-of-range labels) is one
    /// index-add (bincount) of ones at pred * (numClasses + 1) + target, so memory is O(pixels + classes^2) rather
    /// than the O(pixels * classes) of one-hot masks. The intersection is its diagonal, the predicted and
    /// ground-truth counts its row and column sums.
    /// </remarks>
    private (Tensor<T> Intersection, Tensor<T> Union) ComputeIntersectionAndUnion(
        Tensor<T> predicted, Tensor<T> groundTruth)
    {
        var engine = AiDotNetEngine.Current;
        int n = predicted.Length;
        int bins = _numClasses + 1;
        var predLabels = ClassIndex(predicted.Reshape(new[] { n }));
        var gtLabels = ClassIndex(groundTruth.Reshape(new[] { n }));

        var cell = engine.TensorAdd(engine.TensorMultiplyScalar(predLabels, _numOps.FromDouble(bins)), gtLabels);
        var confusion = engine.TensorIndexAdd(
            new Tensor<T>(new[] { bins * bins }), 0, cell.Cast<int>(),
            Tensor<T>.CreateDefault(new[] { n }, _numOps.One));

        var diagonal = engine.TensorMultiplyScalar(engine.TensorArange<T>(0, _numClasses, 1), _numOps.FromDouble(bins + 1));
        var intersection = engine.TensorIndexSelect(confusion, diagonal.Cast<int>(), 0);
        var matrix = confusion.Reshape(new[] { bins, bins });
        var predCount = engine.ReduceSum(matrix, new[] { 1 }, false).Slice(0, 0, _numClasses).Contiguous();
        var gtCount = engine.ReduceSum(matrix, new[] { 0 }, false).Slice(0, 0, _numClasses).Contiguous();
        var union = engine.TensorSubtract(engine.TensorAdd(predCount, gtCount), intersection);
        return (intersection, union);
    }

    /// <summary>
    /// Class index [N] of each label: the label truncated toward zero when it lies in [0, numClasses), and
    /// numClasses (no class) otherwise, including NaN.
    /// </summary>
    private Tensor<T> ClassIndex(Tensor<T> labels)
    {
        var engine = AiDotNetEngine.Current;
        var truncated = engine.TensorTrunc(labels);
        var valid = engine.TensorMultiply(
            engine.TensorGreaterThan(truncated, _numOps.FromDouble(-1.0)),
            engine.TensorLessThan(truncated, _numOps.FromDouble(_numClasses)));
        var noClass = Tensor<T>.CreateDefault(new[] { labels.Length }, _numOps.FromDouble(_numClasses));
        return engine.TensorWhere(valid, truncated, noClass);
    }

    /// <summary>
    /// IoU = intersection / union per class, 0 for classes with an empty union.
    /// </summary>
    private Tensor<T> PerClassIoU(Tensor<T> intersection, Tensor<T> union)
    {
        var engine = AiDotNetEngine.Current;
        // An empty union has an empty intersection, so dividing by max(union, 1) gives 0 there.
        return engine.TensorDivide(intersection, engine.TensorMax(union, _numOps.One));
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

        // A prediction is correct when it is within 0.5 of the integer class label.
        var engine = AiDotNetEngine.Current;
        int n = predicted.Length;
        var diff = engine.TensorSubtract(predicted.Reshape(new[] { n }), groundTruth.Reshape(new[] { n }));
        var correct = engine.TensorLessThan(engine.TensorAbs(diff), _numOps.FromDouble(0.5));

        return _numOps.Divide(engine.TensorSum(correct), _numOps.FromDouble(n));
    }
}
