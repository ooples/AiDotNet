using AiDotNet.Augmentation.Base;
using AiDotNet.Augmentation.Data;
using AiDotNet.Augmentation.Events;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Image.Regularization;

/// <summary>
/// Cuts a rectangular region from one image and pastes it onto another (CutMix augmentation).
/// </summary>
/// <remarks>
/// <para>
/// CutMix is a regularization technique that combines aspects of Cutout and MixUp. It cuts
/// a rectangular region from one training image and pastes it onto another image. The labels
/// are mixed proportionally to the area of the cut region:
/// y' = λy₁ + (1-λ)y₂
/// where λ is the ratio of the original image area that remains.
/// </para>
/// <para><b>For Beginners:</b> Imagine cutting a square from one photo and pasting it onto
/// another, like cutting a face from one picture and pasting it onto a landscape. The label
/// becomes a mix based on how much of each image is visible. Unlike MixUp which creates
/// "ghostly" blends, CutMix keeps sharp boundaries which can be more natural-looking.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Image classification as a stronger regularizer than Cutout</item>
/// <item>When you want benefits of both Cutout (occlusion) and MixUp (label smoothing)</item>
/// <item>Often combined with MixUp in training pipelines</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Object detection (the cut region might remove important objects entirely)</item>
/// <item>Semantic segmentation (hard to handle cut region masks)</item>
/// <item>When precise localization information is important</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CutMix<T> : LabelMixingAugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the minimum ratio of the image area that should be cut.
    /// </summary>
    public double MinCutRatio { get; }

    /// <summary>
    /// Gets the maximum ratio of the image area that should be cut.
    /// </summary>
    public double MaxCutRatio { get; }

    /// <summary>
    /// Creates a new CutMix augmentation.
    /// </summary>
    /// <param name="alpha">
    /// The alpha parameter for the Beta distribution that samples the cut area ratio.
    /// Industry standard default is 1.0 (uniform distribution).
    /// </param>
    /// <param name="minCutRatio">
    /// The minimum ratio of the image area that can be cut.
    /// Industry standard default is 0.0 (no minimum).
    /// </param>
    /// <param name="maxCutRatio">
    /// The maximum ratio of the image area that can be cut.
    /// Industry standard default is 1.0 (up to entire image).
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 1.0 (always applied during training).
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The alpha parameter controls the size distribution of the
    /// cut region. With alpha=1.0, you might cut anything from a tiny corner to almost the
    /// whole image. The minCutRatio and maxCutRatio let you constrain this - for example,
    /// setting minCutRatio=0.1 and maxCutRatio=0.5 ensures you always cut between 10% and 50%
    /// of the image.
    /// </para>
    /// </remarks>
    public CutMix(
        double alpha = 1.0,
        double minCutRatio = 0.0,
        double maxCutRatio = 1.0,
        double probability = 1.0)
        : base(probability, alpha)
    {
        if (minCutRatio < 0 || minCutRatio > 1)
            throw new ArgumentOutOfRangeException(nameof(minCutRatio), "Must be between 0 and 1");
        if (maxCutRatio < 0 || maxCutRatio > 1)
            throw new ArgumentOutOfRangeException(nameof(maxCutRatio), "Must be between 0 and 1");
        if (minCutRatio > maxCutRatio)
            throw new ArgumentException("minCutRatio must be <= maxCutRatio");

        MinCutRatio = minCutRatio;
        MaxCutRatio = maxCutRatio;
    }

    /// <summary>
    /// Applies CutMix by cutting a region from image2 and pasting it onto image1.
    /// </summary>
    /// <param name="image1">The base image that will receive the cut region.</param>
    /// <param name="image2">The source image from which the region is cut.</param>
    /// <param name="labels1">The labels for the first image.</param>
    /// <param name="labels2">The labels for the second image.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The mixed image with the cut region from image2.</returns>
    /// <remarks>
    /// <para>
    /// The lambda value returned represents the proportion of the original image1 that
    /// remains visible (1 - cut_area_ratio). Labels should be mixed as:
    /// mixed_label = λ * labels1 + (1 - λ) * labels2
    /// </para>
    /// </remarks>
    public (ImageTensor<T> result, int x1, int y1, int x2, int y2) ApplyCutMix(
        ImageTensor<T> image1,
        ImageTensor<T> image2,
        Vector<T>? labels1,
        Vector<T>? labels2,
        AugmentationContext<T> context)
    {
        // Ensure images have same dimensions
        if (image1.Height != image2.Height || image1.Width != image2.Width || image1.Channels != image2.Channels)
        {
            throw new ArgumentException("Images must have the same dimensions for CutMix");
        }

        int height = image1.Height;
        int width = image1.Width;
        int channels = image1.Channels;

        // Sample the cut area ratio from Beta distribution, then clamp to min/max
        double cutRatio = SampleLambda(context);
        cutRatio = 1.0 - cutRatio; // Convert to cut ratio (1 - lambda gives area to cut)
        cutRatio = Math.Max(MinCutRatio, Math.Min(MaxCutRatio, cutRatio));

        // Calculate cut region size based on ratio
        // For a ratio r, we want width_cut * height_cut = r * width * height
        // Using square root to get proportional dimensions
        double sqrtRatio = Math.Sqrt(cutRatio);
        int cutWidth = (int)(width * sqrtRatio);
        int cutHeight = (int)(height * sqrtRatio);

        // Ensure minimum size of 1
        cutWidth = Math.Max(1, Math.Min(width, cutWidth));
        cutHeight = Math.Max(1, Math.Min(height, cutHeight));

        // Sample random center position
        int centerX = context.GetRandomInt(0, width);
        int centerY = context.GetRandomInt(0, height);

        // Calculate bounding box
        int x1 = Math.Max(0, centerX - cutWidth / 2);
        int y1 = Math.Max(0, centerY - cutHeight / 2);
        int x2 = Math.Min(width, x1 + cutWidth);
        int y2 = Math.Min(height, y1 + cutHeight);

        // Calculate actual lambda (ratio of image1 that remains)
        int actualCutArea = (x2 - x1) * (y2 - y1);
        int totalArea = width * height;
        double lambda = 1.0 - ((double)actualCutArea / totalArea);
        LastMixingLambda = (T)Convert.ChangeType(lambda, typeof(T));

        // Create result by copying image1 and pasting region from image2
        var result = image1.Clone();

        for (int y = y1; y < y2; y++)
        {
            for (int x = x1; x < x2; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    result.SetPixel(y, x, c, image2.GetPixel(y, x, c));
                }
            }
        }

        // Raise event for label mixing
        if (labels1 is not null && labels2 is not null)
        {
            var args = new LabelMixingEventArgs<T>(
                labels1,
                labels2,
                LastMixingLambda,
                context.SampleIndex,
                -1, // Will be set by caller if needed
                MixingStrategy.CutMix);
            args.Metadata["cut_x1"] = x1;
            args.Metadata["cut_y1"] = y1;
            args.Metadata["cut_x2"] = x2;
            args.Metadata["cut_y2"] = y2;
            RaiseLabelMixing(args);
        }

        return (result, x1, y1, x2, y2);
    }

    /// <summary>
    /// Applies CutMix augmentation (single image version - requires external pairing).
    /// </summary>
    /// <remarks>
    /// <para>
    /// CutMix requires two images. This method returns the image unchanged.
    /// Use ApplyCutMix with two images for the actual augmentation.
    /// </para>
    /// </remarks>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Single-image CutMix doesn't make sense - return unchanged
        // Use ApplyCutMix with two images instead
        return data;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["mixing_strategy"] = "cutmix";
        parameters["min_cut_ratio"] = MinCutRatio;
        parameters["max_cut_ratio"] = MaxCutRatio;
        return parameters;
    }
}
