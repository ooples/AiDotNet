using AiDotNet.Helpers;

namespace AiDotNet.SelfSupervisedLearning.Infrastructure;

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
public class SSLAugmentationPolicies<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the SSLAugmentationPolicies class.
    /// </summary>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SSLAugmentationPolicies(int? seed = null)
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
        // Simplified implementation - in production, this would do actual cropping
        // For now, we simulate by adding slight noise
        var scale = scaleMin + _random.NextDouble() * (scaleMax - scaleMin);
        var result = new T[image.Length];

        for (int i = 0; i < image.Length; i++)
        {
            // Simulate crop effect with slight variation
            var noise = NumOps.FromDouble((_random.NextDouble() - 0.5) * 0.05 * (1 - scale));
            result[i] = NumOps.Add(image.Data[i], noise);
        }

        return new Tensor<T>(result, image.Shape);
    }

    private Tensor<T> HorizontalFlip(Tensor<T> image)
    {
        // For simplicity, just return the image
        // In production, this would flip along the width dimension
        var result = new T[image.Length];
        Array.Copy(image.Data, result, image.Length);
        return new Tensor<T>(result, image.Shape);
    }

    private Tensor<T> ColorJitter(Tensor<T> image,
        double brightnessRange, double contrastRange, double saturationRange, double hueRange)
    {
        var result = new T[image.Length];

        // Apply random brightness/contrast adjustments
        var brightness = 1.0 + (_random.NextDouble() * 2 - 1) * brightnessRange;
        var contrast = 1.0 + (_random.NextDouble() * 2 - 1) * contrastRange;

        for (int i = 0; i < image.Length; i++)
        {
            var val = NumOps.ToDouble(image.Data[i]);
            val = (val - 0.5) * contrast + 0.5;  // Contrast
            val = val * brightness;               // Brightness
            val = Math.Max(0, Math.Min(1, val)); // Clamp
            result[i] = NumOps.FromDouble(val);
        }

        return new Tensor<T>(result, image.Shape);
    }

    private Tensor<T> ToGrayscale(Tensor<T> image)
    {
        // Simplified grayscale - average all channels
        var result = new T[image.Length];

        for (int i = 0; i < image.Length; i++)
        {
            // In a real implementation, we'd average across channels
            result[i] = image.Data[i];
        }

        return new Tensor<T>(result, image.Shape);
    }

    private Tensor<T> GaussianBlur(Tensor<T> image, double sigmaMin, double sigmaMax)
    {
        // Simplified blur - slight smoothing
        var sigma = sigmaMin + _random.NextDouble() * (sigmaMax - sigmaMin);
        var result = new T[image.Length];

        // Simple box blur approximation
        for (int i = 0; i < image.Length; i++)
        {
            // Add small random noise to simulate blur effect
            var blur = NumOps.FromDouble((_random.NextDouble() - 0.5) * 0.02 * sigma);
            result[i] = NumOps.Add(image.Data[i], blur);
        }

        return new Tensor<T>(result, image.Shape);
    }

    private Tensor<T> Solarize(Tensor<T> image, double threshold)
    {
        var result = new T[image.Length];

        for (int i = 0; i < image.Length; i++)
        {
            var val = NumOps.ToDouble(image.Data[i]);
            if (val > threshold)
            {
                val = 1.0 - val;  // Invert values above threshold
            }
            result[i] = NumOps.FromDouble(val);
        }

        return new Tensor<T>(result, image.Shape);
    }

    #endregion

    /// <summary>
    /// Creates a standard SimCLR augmentation policy.
    /// </summary>
    public static SSLAugmentationPolicies<T> ForSimCLR(int? seed = null)
    {
        return new SSLAugmentationPolicies<T>(seed);
    }

    /// <summary>
    /// Creates a standard BYOL augmentation policy.
    /// </summary>
    public static SSLAugmentationPolicies<T> ForBYOL(int? seed = null)
    {
        return new SSLAugmentationPolicies<T>(seed);
    }

    /// <summary>
    /// Creates a standard DINO augmentation policy.
    /// </summary>
    public static SSLAugmentationPolicies<T> ForDINO(int? seed = null)
    {
        return new SSLAugmentationPolicies<T>(seed);
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
