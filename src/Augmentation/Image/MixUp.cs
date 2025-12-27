using AiDotNet.Augmentation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Blends two images together by weighted averaging (MixUp augmentation).
/// </summary>
/// <remarks>
/// <para>
/// MixUp is a powerful regularization technique that creates virtual training examples by
/// taking linear combinations of two training samples and their labels. Given two images
/// (x₁, y₁) and (x₂, y₂), MixUp creates a new training sample:
/// x' = λx₁ + (1-λ)x₂
/// y' = λy₁ + (1-λ)y₂
/// where λ is sampled from a Beta distribution.
/// </para>
/// <para><b>For Beginners:</b> Imagine blending two photos together like a double exposure.
/// If you blend a photo of a cat with a photo of a dog, you get something that's partly
/// cat and partly dog. The label becomes a mix too: maybe 70% cat and 30% dog.
/// This teaches the model that the world isn't black-and-white, and improves generalization.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Image classification with sufficient training data</item>
/// <item>When you want smoother decision boundaries</item>
/// <item>As a regularization alternative or complement to dropout</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Object detection (bounding boxes can't be meaningfully mixed)</item>
/// <item>Semantic segmentation (mixed masks don't make sense)</item>
/// <item>When you need hard labels for downstream processing</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MixUp<T> : LabelMixingAugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Creates a new MixUp augmentation.
    /// </summary>
    /// <param name="alpha">
    /// The alpha parameter for the Beta distribution that samples the mixing ratio.
    /// Industry standard default is 1.0 (uniform distribution over [0, 1]).
    /// Lower values (like 0.2) concentrate samples near 0 and 1 (less mixing).
    /// Higher values (above 1.0) concentrate samples near 0.5 (stronger mixing).
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 1.0 (always applied during training).
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The alpha parameter controls how much mixing happens.
    /// With alpha=1.0, you might get 50% of one image and 50% of another.
    /// With alpha=0.2, you'll mostly get one image with just a tiny bit of the other mixed in.
    /// Most papers use alpha=1.0 or alpha=0.4 for good results.
    /// </para>
    /// </remarks>
    public MixUp(double alpha = 1.0, double probability = 1.0)
        : base(probability, alpha)
    {
    }

    /// <summary>
    /// Applies MixUp to blend two images together.
    /// </summary>
    /// <param name="image1">The first image.</param>
    /// <param name="image2">The second image to mix with the first.</param>
    /// <param name="labels1">The labels for the first image.</param>
    /// <param name="labels2">The labels for the second image.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The mixed image.</returns>
    /// <remarks>
    /// <para>
    /// This method blends the two images pixel-by-pixel using the formula:
    /// mixed_pixel = λ * pixel1 + (1 - λ) * pixel2
    /// The labels should be mixed the same way in your training loop.
    /// </para>
    /// </remarks>
    public ImageTensor<T> ApplyMixUp(
        ImageTensor<T> image1,
        ImageTensor<T> image2,
        Vector<T>? labels1,
        Vector<T>? labels2,
        AugmentationContext<T> context)
    {
        // Sample mixing coefficient
        double lambda = SampleLambda(context);
        LastMixingLambda = NumOps.FromDouble(lambda);

        // Ensure images have same dimensions
        if (image1.Height != image2.Height || image1.Width != image2.Width || image1.Channels != image2.Channels)
        {
            throw new ArgumentException("Images must have the same dimensions for MixUp");
        }

        var result = image1.Clone();
        int height = result.Height;
        int width = result.Width;
        int channels = result.Channels;

        // Mix pixels: result = λ * image1 + (1 - λ) * image2
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    double v1 = NumOps.ToDouble(image1.GetPixel(y, x, c));
                    double v2 = NumOps.ToDouble(image2.GetPixel(y, x, c));
                    double mixed = lambda * v1 + (1 - lambda) * v2;
                    result.SetPixel(y, x, c, NumOps.FromDouble(mixed));
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
                MixingStrategy.Mixup);
            RaiseLabelMixing(args);
        }

        return result;
    }

    /// <summary>
    /// Applies MixUp augmentation (single image version - requires external pairing).
    /// </summary>
    /// <remarks>
    /// <para>
    /// MixUp requires two images to blend together. This method stores the current image
    /// for potential pairing. In practice, MixUp is typically applied at the batch level
    /// where all pairs can be formed at once using ApplyMixUp.
    /// </para>
    /// </remarks>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Single-image MixUp doesn't make sense - return unchanged
        // Use ApplyMixUp with two images instead
        return data;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["mixing_strategy"] = "mixup";
        return parameters;
    }
}
