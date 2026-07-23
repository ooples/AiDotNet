namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Base class for image data augmentations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Image augmentation transforms images to improve model
/// robustness to variations in viewpoint, lighting, and appearance. Common techniques include:
/// <list type="bullet">
/// <item>Geometric transforms: flips, rotations, scaling, cropping</item>
/// <item>Color transforms: brightness, contrast, saturation, hue</item>
/// <item>Noise and blur: Gaussian noise, blur, sharpening</item>
/// <item>Regularization: cutout, mixup, cutmix</item>
/// </list>
/// </para>
/// <para>Image data is represented as an ImageTensor with dimensions (height, width, channels).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class ImageAugmenterBase<T> : AugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Initializes a new image augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation (0.0 to 1.0).</param>
    protected ImageAugmenterBase(double probability = 1.0) : base(probability)
    {
    }

    /// <summary>
    /// Gets the height of the image.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <returns>The height in pixels.</returns>
    protected int GetHeight(ImageTensor<T> image) => image.Height;

    /// <summary>
    /// Gets the width of the image.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <returns>The width in pixels.</returns>
    protected int GetWidth(ImageTensor<T> image) => image.Width;

    /// <summary>
    /// Gets the number of channels in the image.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <returns>The number of channels (typically 1, 3, or 4).</returns>
    protected int GetChannels(ImageTensor<T> image) => image.Channels;

    /// <summary>
    /// Checks if the image is grayscale (single channel).
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <returns>True if the image is grayscale.</returns>
    protected bool IsGrayscale(ImageTensor<T> image) => image.Channels == 1;

    /// <summary>
    /// Checks if the image has an alpha channel.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <returns>True if the image has 4 channels (RGBA).</returns>
    protected bool HasAlpha(ImageTensor<T> image) => image.Channels == 4;
}

/// <summary>
/// Base class for image augmentations that transform spatial targets.
/// </summary>
/// <remarks>
/// <para>Use this base class for augmentations that need to also transform
/// bounding boxes, keypoints, or segmentation masks along with the image.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class SpatialImageAugmenterBase<T> : SpatialAugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Initializes a new spatial image augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation.</param>
    protected SpatialImageAugmenterBase(double probability = 1.0) : base(probability)
    {
    }
}

/// <summary>
/// Base class for image augmentations that mix multiple images together.
/// </summary>
/// <remarks>
/// <para>Use this base class for augmentations like MixUp and CutMix that
/// combine two or more images and their labels.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class ImageMixingAugmenterBase<T> : LabelMixingAugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Initializes a new image mixing augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation.</param>
    /// <param name="alpha">The alpha parameter for Beta distribution sampling.</param>
    protected ImageMixingAugmenterBase(double probability = 1.0, double alpha = 1.0)
        : base(probability, alpha)
    {
    }
}
