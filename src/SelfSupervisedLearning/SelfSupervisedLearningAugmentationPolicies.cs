using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Provides standard augmentation policies for self-supervised learning methods.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Self-supervised learning methods rely heavily on data augmentation
/// to create different "views" of the same image. These views should be different enough to be
/// challenging, but similar enough that they can still be recognized as the same image.</para>
///
/// <para><b>Common augmentations for SSL:</b></para>
/// <list type="bullet">
/// <item><b>Random crop and resize:</b> Most important - crops different regions</item>
/// <item><b>Color jitter:</b> Changes brightness, contrast, saturation, hue</item>
/// <item><b>Grayscale:</b> Removes color information</item>
/// <item><b>Gaussian blur:</b> Adds blur (important for SimCLR, BYOL)</item>
/// <item><b>Horizontal flip:</b> Random left-right flip</item>
/// <item><b>Solarization:</b> Inverts pixels above threshold (BYOL, DINO)</item>
/// </list>
///
/// <para><b>Method-specific policies:</b></para>
/// <list type="bullet">
/// <item><b>SimCLR:</b> Strong augmentation with blur and color jitter</item>
/// <item><b>MoCo v2:</b> Similar to SimCLR with slightly different parameters</item>
/// <item><b>BYOL:</b> Asymmetric augmentation (view 1 and view 2 different)</item>
/// <item><b>DINO:</b> Multi-crop strategy (global + local crops)</item>
/// </list>
/// </remarks>
public class SelfSupervisedLearningAugmentationPolicies<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the SelfSupervisedLearningAugmentationPolicies class.
    /// </summary>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SelfSupervisedLearningAugmentationPolicies(int? seed = null)
    {
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.Shared;
    }

    /// <summary>
    /// Applies SimCLR-style augmentation to create two views.
    /// </summary>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Two augmented views of the image.</returns>
    /// <remarks>
    /// <para><b>SimCLR augmentation pipeline:</b></para>
    /// <code>
    /// RandomResizedCrop(224, scale=(0.08, 1.0))
    /// RandomHorizontalFlip(p=0.5)
    /// ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8)
    /// RandomGrayscale(p=0.2)
    /// GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5)
    /// </code>
    /// </remarks>
    public (Tensor<T> view1, Tensor<T> view2) ApplySimCLR(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));

        var view1 = ApplySimCLRPipeline(image);
        var view2 = ApplySimCLRPipeline(image);

        return (view1, view2);
    }

    /// <summary>
    /// Applies BYOL-style asymmetric augmentation.
    /// </summary>
    /// <param name="image">Input image tensor.</param>
    /// <returns>Two asymmetrically augmented views.</returns>
    /// <remarks>
    /// <para><b>BYOL uses asymmetric augmentation:</b></para>
    /// <para>View 1: Full augmentation pipeline</para>
    /// <para>View 2: Lighter augmentation (may skip some transforms)</para>
    /// </remarks>
    public (Tensor<T> view1, Tensor<T> view2) ApplyBYOL(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));

        // View 1: Full augmentation (stronger)
        var view1 = ApplyBYOLView1Pipeline(image);

        // View 2: Lighter augmentation
        var view2 = ApplyBYOLView2Pipeline(image);

        return (view1, view2);
    }

    /// <summary>
    /// Applies DINO-style multi-crop augmentation.
    /// </summary>
    /// <param name="image">Input image tensor.</param>
    /// <param name="numGlobalCrops">Number of global (large) crops (default: 2).</param>
    /// <param name="numLocalCrops">Number of local (small) crops (default: 8).</param>
    /// <returns>Global and local crop views.</returns>
    /// <remarks>
    /// <para><b>DINO multi-crop strategy:</b></para>
    /// <list type="bullet">
    /// <item>Global crops: 224x224, covering 50-100% of image</item>
    /// <item>Local crops: 96x96, covering 5-50% of image</item>
    /// <item>Teacher sees only global crops, student sees all crops</item>
    /// </list>
    /// </remarks>
    public (Tensor<T>[] globalViews, Tensor<T>[] localViews) ApplyDINO(
        Tensor<T> image,
        int numGlobalCrops = 2,
        int numLocalCrops = 8)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));

        var globalViews = new Tensor<T>[numGlobalCrops];
        var localViews = new Tensor<T>[numLocalCrops];

        // Generate global crops (larger, 224x224)
        for (int i = 0; i < numGlobalCrops; i++)
        {
            globalViews[i] = ApplyGlobalCrop(image, scaleMin: 0.5, scaleMax: 1.0);
        }

        // Generate local crops (smaller, 96x96)
        for (int i = 0; i < numLocalCrops; i++)
        {
            localViews[i] = ApplyLocalCrop(image, scaleMin: 0.05, scaleMax: 0.5);
        }

        return (globalViews, localViews);
    }

    /// <summary>
    /// Applies MoCo v2 style augmentation.
    /// </summary>
    /// <param name="image">Input image tensor.</param>
    /// <returns>Two augmented views.</returns>
    public (Tensor<T> query, Tensor<T> key) ApplyMoCoV2(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));

        // MoCo v2 uses similar augmentation to SimCLR
        var query = ApplySimCLRPipeline(image);
        var key = ApplySimCLRPipeline(image);

        return (query, key);
    }

    #region Pipeline Implementations

    private Tensor<T> ApplySimCLRPipeline(Tensor<T> image)
    {
        var result = image;

        // Random resized crop (most important augmentation)
        result = RandomResizedCrop(result, scaleMin: 0.08, scaleMax: 1.0);

        // Random horizontal flip
        if (_random.NextDouble() < 0.5)
        {
            result = HorizontalFlip(result);
        }

        // Color jitter with probability 0.8
        if (_random.NextDouble() < 0.8)
        {
            result = ColorJitter(result,
                brightnessRange: 0.8,
                contrastRange: 0.8,
                saturationRange: 0.8,
                hueRange: 0.2);
        }

        // Random grayscale with probability 0.2
        if (_random.NextDouble() < 0.2)
        {
            result = ToGrayscale(result);
        }

        // Gaussian blur with probability 0.5
        if (_random.NextDouble() < 0.5)
        {
            result = GaussianBlur(result, sigmaMin: 0.1, sigmaMax: 2.0);
        }

        return result;
    }

    private Tensor<T> ApplyBYOLView1Pipeline(Tensor<T> image)
    {
        var result = image;

        // Random resized crop
        result = RandomResizedCrop(result, scaleMin: 0.08, scaleMax: 1.0);

        // Random horizontal flip
        if (_random.NextDouble() < 0.5)
        {
            result = HorizontalFlip(result);
        }

        // Color jitter
        if (_random.NextDouble() < 0.8)
        {
            result = ColorJitter(result,
                brightnessRange: 0.4,
                contrastRange: 0.4,
                saturationRange: 0.2,
                hueRange: 0.1);
        }

        // Random grayscale
        if (_random.NextDouble() < 0.2)
        {
            result = ToGrayscale(result);
        }

        // Gaussian blur (always for view 1)
        result = GaussianBlur(result, sigmaMin: 0.1, sigmaMax: 2.0);

        // Solarization with probability 0.0 for view 1
        // (BYOL uses solarization asymmetrically)

        return result;
    }

    private Tensor<T> ApplyBYOLView2Pipeline(Tensor<T> image)
    {
        var result = image;

        // Random resized crop
        result = RandomResizedCrop(result, scaleMin: 0.08, scaleMax: 1.0);

        // Random horizontal flip
        if (_random.NextDouble() < 0.5)
        {
            result = HorizontalFlip(result);
        }

        // Color jitter
        if (_random.NextDouble() < 0.8)
        {
            result = ColorJitter(result,
                brightnessRange: 0.4,
                contrastRange: 0.4,
                saturationRange: 0.2,
                hueRange: 0.1);
        }

        // Random grayscale
        if (_random.NextDouble() < 0.2)
        {
            result = ToGrayscale(result);
        }

        // Gaussian blur with probability 0.1 for view 2
        if (_random.NextDouble() < 0.1)
        {
            result = GaussianBlur(result, sigmaMin: 0.1, sigmaMax: 2.0);
        }

        // Solarization with probability 0.2 for view 2
        if (_random.NextDouble() < 0.2)
        {
            result = Solarize(result, threshold: 0.5);
        }

        return result;
    }

    private Tensor<T> ApplyGlobalCrop(Tensor<T> image, double scaleMin, double scaleMax)
    {
        // Global crop: larger scale, typically 224x224 output
        var result = RandomResizedCrop(image, scaleMin, scaleMax);

        // Standard augmentations
        if (_random.NextDouble() < 0.5)
        {
            result = HorizontalFlip(result);
        }

        if (_random.NextDouble() < 0.8)
        {
            result = ColorJitter(result, 0.4, 0.4, 0.2, 0.1);
        }

        if (_random.NextDouble() < 0.2)
        {
            result = ToGrayscale(result);
        }

        return result;
    }

    private Tensor<T> ApplyLocalCrop(Tensor<T> image, double scaleMin, double scaleMax)
    {
        // Local crop: smaller scale, typically 96x96 output
        var result = RandomResizedCrop(image, scaleMin, scaleMax);

        // Same augmentations as global
        if (_random.NextDouble() < 0.5)
        {
            result = HorizontalFlip(result);
        }

        if (_random.NextDouble() < 0.8)
        {
            result = ColorJitter(result, 0.4, 0.4, 0.2, 0.1);
        }

        if (_random.NextDouble() < 0.2)
        {
            result = ToGrayscale(result);
        }

        return result;
    }

    #endregion

    #region Individual Augmentation Operations

    private Tensor<T> RandomResizedCrop(Tensor<T> image, double scaleMin, double scaleMax)
    {
        // Random scale for the crop area
        var scale = scaleMin + _random.NextDouble() * (scaleMax - scaleMin);
        var aspectRatio = 0.75 + _random.NextDouble() * 0.5; // Random aspect ratio 0.75 to 1.25

        // For [batch, channels, height, width] or [batch, height, width, channels] tensors
        if (image.Shape.Length >= 4)
        {
            var height = image.Shape[2];
            var width = image.Shape[3];

            // Calculate crop dimensions
            var cropArea = height * width * scale;
            var cropHeight = (int)Math.Sqrt(cropArea / aspectRatio);
            var cropWidth = (int)(cropHeight * aspectRatio);
            cropHeight = Math.Min(cropHeight, height);
            cropWidth = Math.Min(cropWidth, width);

            // Random crop position
            var top = _random.Next(0, Math.Max(1, height - cropHeight));
            var left = _random.Next(0, Math.Max(1, width - cropWidth));

            // Perform crop and resize via bilinear interpolation, which is separable: rows then columns
            return ApplySeparable(image,
                BilinearCropMatrix(height, cropHeight, top),
                BilinearCropMatrix(width, cropWidth, left));
        }

        // For 2D tensors, apply scaling transform
        return Engine.TensorMultiplyScalar(image, NumOps.FromDouble(scale));
    }

    /// <summary>
    /// [size, size] bilinear resampling matrix: output o reads source position
    /// offset + o / size * cropSize, weighting floor(pos) by 1 - frac and min(floor(pos) + 1, size - 1) by frac.
    /// </summary>
    private static Tensor<T> BilinearCropMatrix(int size, int cropSize, int offset)
    {
        var positions = Arange(size);
        var source = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(
                Engine.TensorDivideScalar(positions, NumOps.FromDouble(size)), NumOps.FromDouble(cropSize)),
            NumOps.FromDouble(offset));
        var lower = Engine.TensorFloor(source);
        var frac = Engine.TensorSubtract(source, lower).Reshape(size, 1);
        var upper = Engine.TensorClampMax(Engine.TensorAddScalar(lower, NumOps.One), NumOps.FromDouble(size - 1));

        var lowerHit = IndexHits(lower.Reshape(size, 1), positions.Reshape(1, size));
        var upperHit = IndexHits(upper.Reshape(size, 1), positions.Reshape(1, size));
        return Engine.TensorAdd(
            Engine.TensorMultiply(lowerHit, Engine.ScalarMinusTensor(NumOps.One, frac)),
            Engine.TensorMultiply(upperHit, frac));
    }

    /// <summary>
    /// 1 where an integer-valued index equals a candidate position (|index - position| &lt; 0.5), else 0; broadcasts.
    /// </summary>
    private static Tensor<T> IndexHits(Tensor<T> indices, Tensor<T> positions)
    {
        return Engine.TensorLessThan(
            Engine.TensorAbs(Engine.TensorSubtract(indices, positions)), NumOps.FromDouble(0.5));
    }

    /// <summary>[0, 1, ..., count - 1] as T, built from a cumulative sum of ones.</summary>
    private static Tensor<T> Arange(int count)
    {
        var ones = Engine.TensorAddScalar(new Tensor<T>(new[] { count }), NumOps.One);
        return Engine.TensorAddScalar(Engine.TensorCumSum(ones, 0), NumOps.Negate(NumOps.One));
    }

    /// <summary>
    /// out[b, c] = rows @ image[b, c] @ columns^T for a [B, C, H, W] image: columns act along the width first,
    /// then rows along the height.
    /// </summary>
    private static Tensor<T> ApplySeparable(Tensor<T> image, Tensor<T> rows, Tensor<T> columns)
    {
        int batch = image.Shape[0], channels = image.Shape[1], height = image.Shape[2], width = image.Shape[3];

        var alongWidth = Engine.TensorMatMul(
                image.Contiguous().Reshape(batch * channels * height, width), Engine.TensorTranspose(columns))
            .Reshape(batch, channels, height, width);

        var heightLast = Engine.TensorPermute(alongWidth, new[] { 0, 1, 3, 2 }).Contiguous()
            .Reshape(batch * channels * width, height);
        var alongHeight = Engine.TensorMatMul(heightLast, Engine.TensorTranspose(rows))
            .Reshape(batch, channels, width, height);

        return Engine.TensorPermute(alongHeight, new[] { 0, 1, 3, 2 }).Contiguous();
    }

    private Tensor<T> HorizontalFlip(Tensor<T> image)
    {
        var result = new T[image.Length];

        // For [batch, channels, height, width] tensors
        if (image.Shape.Length >= 4)
        {
            var batch = image.Shape[0];
            var channels = image.Shape[1];
            var height = image.Shape[2];
            var width = image.Shape[3];

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            // Flip horizontally: map w to (width - 1 - w)
                            var srcIdx = b * channels * height * width + c * height * width + h * width + (width - 1 - w);
                            var dstIdx = b * channels * height * width + c * height * width + h * width + w;
                            result[dstIdx] = image.Data.Span[srcIdx];
                        }
                    }
                }
            }
        }
        else
        {
            // For 2D tensors, just copy (no spatial dimension to flip)
            Array.Copy(image.Data.ToArray(), result, image.Length);
        }

        return new Tensor<T>(result, image._shape);
    }

    private Tensor<T> ColorJitter(Tensor<T> image,
        double brightnessRange, double contrastRange, double saturationRange, double hueRange)
    {
        // Apply random brightness/contrast adjustments
        var brightness = 1.0 + (_random.NextDouble() * 2 - 1) * brightnessRange;
        var contrast = 1.0 + (_random.NextDouble() * 2 - 1) * contrastRange;

        var half = NumOps.FromDouble(0.5);
        var contrasted = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(Engine.TensorAddScalar(image, NumOps.Negate(half)), NumOps.FromDouble(contrast)),
            half);                                                                       // Contrast
        var brightened = Engine.TensorMultiplyScalar(contrasted, NumOps.FromDouble(brightness)); // Brightness
        return Engine.TensorClamp(brightened, NumOps.Zero, NumOps.One);                  // Clamp
    }

    private Tensor<T> ToGrayscale(Tensor<T> image)
    {
        // For [batch, channels, height, width] tensors
        if (image.Shape.Length >= 4)
        {
            var batch = image.Shape[0];
            var channels = image.Shape[1];
            var height = image.Shape[2];
            var width = image.Shape[3];

            // Standard luminosity weights for RGB to grayscale conversion
            // ITU-R BT.601: Y = 0.299*R + 0.587*G + 0.114*B
            const double rWeight = 0.299;
            const double gWeight = 0.587;
            const double bWeight = 0.114;

            Tensor<T> Channel(int c) =>
                Engine.TensorSlice(image, new[] { 0, c, 0, 0 }, new[] { batch, 1, height, width });

            // Compute grayscale value from RGB channels; a single channel is already grayscale
            var gray = channels >= 3
                ? Engine.TensorAdd(
                    Engine.TensorAdd(
                        Engine.TensorMultiplyScalar(Channel(0), NumOps.FromDouble(rWeight)),
                        Engine.TensorMultiplyScalar(Channel(1), NumOps.FromDouble(gWeight))),
                    Engine.TensorMultiplyScalar(Channel(2), NumOps.FromDouble(bWeight)))
                : Channel(0);

            // Set all channels to the grayscale value (maintains tensor shape)
            return Engine.TensorTile(gray, new[] { 1, channels, 1, 1 });
        }

        // For 2D tensors, just copy (no channel dimension)
        var result = new T[image.Length];
        Array.Copy(image.Data.ToArray(), result, image.Length);
        return new Tensor<T>(result, image._shape);
    }

    private Tensor<T> GaussianBlur(Tensor<T> image, double sigmaMin, double sigmaMax)
    {
        var sigma = sigmaMin + _random.NextDouble() * (sigmaMax - sigmaMin);

        // For [batch, channels, height, width] tensors
        if (image.Shape.Length >= 4)
        {
            var height = image.Shape[2];
            var width = image.Shape[3];

            // Compute kernel size (must be odd, at least 3)
            // Standard practice: kernel_size = ceil(6*sigma) | 1 to make it odd
            var kernelSize = Math.Max(3, (int)Math.Ceiling(sigma * 6) | 1);
            var radius = kernelSize / 2;

            // Generate the normalized 1D Gaussian kernel: exp(-0.5 x^2 / sigma^2), x = -radius..radius
            var offsets = Engine.TensorAddScalar(Arange(kernelSize), NumOps.FromDouble(-radius));
            var kernel = Engine.TensorExp(Engine.TensorDivideScalar(
                Engine.TensorMultiplyScalar(Engine.TensorSquare(offsets), NumOps.FromDouble(-0.5)),
                NumOps.FromDouble(sigma * sigma)));
            kernel = Engine.TensorDivideScalar(kernel, Engine.TensorSum(kernel));

            // Apply separable convolution (horizontal then vertical) with edge clamping
            return ApplySeparable(image,
                ClampedBlurMatrix(height, offsets, kernel),
                ClampedBlurMatrix(width, offsets, kernel));
        }
        else
        {
            // For 2D tensors, just copy (no spatial dimension to blur)
            var result = new T[image.Length];
            Array.Copy(image.Data.ToArray(), result, image.Length);
            return new Tensor<T>(result, image._shape);
        }
    }

    /// <summary>
    /// [size, size] blur matrix: M[o, s] = sum_k kernel[k] where clamp(o + offset[k], 0, size - 1) == s.
    /// </summary>
    private static Tensor<T> ClampedBlurMatrix(int size, Tensor<T> offsets, Tensor<T> kernel)
    {
        var kernelSize = offsets.Shape[0];
        var positions = Arange(size);

        // Source index of tap k for output o: [K, size, 1]
        var taps = Engine.TensorClamp(
            Engine.TensorAdd(positions.Reshape(1, size), offsets.Reshape(kernelSize, 1)),
            NumOps.Zero, NumOps.FromDouble(size - 1)).Reshape(kernelSize, size, 1);

        // Weighted one-hots over the source axis, summed over taps
        var hits = IndexHits(taps, positions.Reshape(1, 1, size));
        var weighted = Engine.TensorMultiply(hits, kernel.Reshape(kernelSize, 1, 1));
        return Engine.ReduceSum(weighted, new[] { 0 }, keepDims: false);
    }

    private Tensor<T> Solarize(Tensor<T> image, double threshold)
    {
        // Invert values above threshold
        var above = Engine.TensorGreaterThan(image, NumOps.FromDouble(threshold));
        return Engine.TensorWhere(above, Engine.ScalarMinusTensor(NumOps.One, image), image);
    }

    #endregion

    /// <summary>
    /// Creates a standard SimCLR augmentation policy.
    /// </summary>
    public static SelfSupervisedLearningAugmentationPolicies<T> ForSimCLR(int? seed = null)
    {
        return new SelfSupervisedLearningAugmentationPolicies<T>(seed);
    }

    /// <summary>
    /// Creates a standard BYOL augmentation policy.
    /// </summary>
    public static SelfSupervisedLearningAugmentationPolicies<T> ForBYOL(int? seed = null)
    {
        return new SelfSupervisedLearningAugmentationPolicies<T>(seed);
    }

    /// <summary>
    /// Creates a standard DINO augmentation policy.
    /// </summary>
    public static SelfSupervisedLearningAugmentationPolicies<T> ForDINO(int? seed = null)
    {
        return new SelfSupervisedLearningAugmentationPolicies<T>(seed);
    }

    /// <summary>
    /// Applies minimal augmentation (used by MAE which relies on masking rather than augmentation).
    /// </summary>
    /// <param name="image">Input image tensor.</param>
    /// <returns>Minimally augmented image.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> MAE uses minimal augmentation because the masking itself
    /// provides the learning signal. Only basic normalization and optional horizontal flip.</para>
    /// </remarks>
    public Tensor<T> ApplyMinimal(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));

        var result = image;

        // Random horizontal flip with probability 0.5
        if (_random.NextDouble() < 0.5)
        {
            result = HorizontalFlip(result);
        }

        return result;
    }
}
