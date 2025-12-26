using AiDotNet.Augmentation.Base;
using AiDotNet.Augmentation.Data;

namespace AiDotNet.Augmentation.Image.Regularization;

/// <summary>
/// Randomly masks out (cuts out) rectangular regions of an image.
/// </summary>
/// <remarks>
/// <para>
/// Cutout is a regularization technique that randomly removes rectangular patches from
/// training images by filling them with a constant value (usually gray or black). This
/// forces the model to focus on multiple parts of the object rather than relying on
/// a single distinctive feature, improving robustness and generalization.
/// </para>
/// <para><b>For Beginners:</b> Imagine covering parts of a photo with sticky notes.
/// If you can still recognize what's in the photo with pieces hidden, you understand
/// the whole object better, not just one specific feature. Cutout does this automatically
/// during training, teaching the model to recognize objects even when parts are obscured.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Image classification where objects might be partially occluded</item>
/// <item>When you want to prevent the model from overfitting to specific features</item>
/// <item>As a regularization technique to improve generalization</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Object detection or segmentation (might remove the entire object)</item>
/// <item>When fine-grained features are crucial for classification</item>
/// <item>Very small images where cutout would remove too much information</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Cutout<T> : AugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the number of rectangular holes to cut out.
    /// </summary>
    public int NumberOfHoles { get; }

    /// <summary>
    /// Gets the minimum height of each hole.
    /// </summary>
    public int MinHoleHeight { get; }

    /// <summary>
    /// Gets the maximum height of each hole.
    /// </summary>
    public int MaxHoleHeight { get; }

    /// <summary>
    /// Gets the minimum width of each hole.
    /// </summary>
    public int MinHoleWidth { get; }

    /// <summary>
    /// Gets the maximum width of each hole.
    /// </summary>
    public int MaxHoleWidth { get; }

    /// <summary>
    /// Gets the fill value for the cutout regions.
    /// </summary>
    public T FillValue { get; }

    /// <summary>
    /// Creates a new cutout augmentation.
    /// </summary>
    /// <param name="numberOfHoles">
    /// The number of rectangular holes to cut out.
    /// Industry standard default is 1 (one hole per image).
    /// </param>
    /// <param name="minHoleHeight">
    /// The minimum height of each hole in pixels.
    /// Industry standard default is 8 pixels.
    /// </param>
    /// <param name="maxHoleHeight">
    /// The maximum height of each hole in pixels.
    /// Industry standard default is 32 pixels.
    /// </param>
    /// <param name="minHoleWidth">
    /// The minimum width of each hole in pixels.
    /// Industry standard default is 8 pixels.
    /// </param>
    /// <param name="maxHoleWidth">
    /// The maximum width of each hole in pixels.
    /// Industry standard default is 32 pixels.
    /// </param>
    /// <param name="fillValue">
    /// The value to fill the cutout regions with.
    /// Use 0 for black, or 0.5 (for normalized images) for gray.
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.5.
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The hole size should be large enough to hide meaningful
    /// features but not so large that it removes most of the image. For 224x224 images,
    /// holes around 16-32 pixels work well. For 32x32 images (like CIFAR-10), use smaller
    /// holes around 8-16 pixels.
    /// </para>
    /// </remarks>
    public Cutout(
        int numberOfHoles = 1,
        int minHoleHeight = 8,
        int maxHoleHeight = 32,
        int minHoleWidth = 8,
        int maxHoleWidth = 32,
        T? fillValue = default,
        double probability = 0.5) : base(probability)
    {
        if (numberOfHoles < 1)
            throw new ArgumentOutOfRangeException(nameof(numberOfHoles), "Must have at least 1 hole");
        if (minHoleHeight < 1)
            throw new ArgumentOutOfRangeException(nameof(minHoleHeight), "Minimum height must be at least 1");
        if (minHoleWidth < 1)
            throw new ArgumentOutOfRangeException(nameof(minHoleWidth), "Minimum width must be at least 1");
        if (minHoleHeight > maxHoleHeight)
            throw new ArgumentException("minHoleHeight must be <= maxHoleHeight");
        if (minHoleWidth > maxHoleWidth)
            throw new ArgumentException("minHoleWidth must be <= maxHoleWidth");

        NumberOfHoles = numberOfHoles;
        MinHoleHeight = minHoleHeight;
        MaxHoleHeight = maxHoleHeight;
        MinHoleWidth = minHoleWidth;
        MaxHoleWidth = maxHoleWidth;
        FillValue = fillValue ?? default!;
    }

    /// <summary>
    /// Applies the cutout to the image.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int height = result.Height;
        int width = result.Width;
        int channels = result.Channels;

        for (int hole = 0; hole < NumberOfHoles; hole++)
        {
            // Sample random hole size
            int holeHeight = context.GetRandomInt(MinHoleHeight, MaxHoleHeight + 1);
            int holeWidth = context.GetRandomInt(MinHoleWidth, MaxHoleWidth + 1);

            // Sample random center position
            int centerY = context.GetRandomInt(0, height);
            int centerX = context.GetRandomInt(0, width);

            // Calculate hole boundaries
            int y1 = Math.Max(0, centerY - holeHeight / 2);
            int y2 = Math.Min(height, centerY + holeHeight / 2);
            int x1 = Math.Max(0, centerX - holeWidth / 2);
            int x2 = Math.Min(width, centerX + holeWidth / 2);

            // Fill the hole
            for (int y = y1; y < y2; y++)
            {
                for (int x = x1; x < x2; x++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        result.SetPixel(y, x, c, FillValue);
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["number_of_holes"] = NumberOfHoles;
        parameters["min_hole_height"] = MinHoleHeight;
        parameters["max_hole_height"] = MaxHoleHeight;
        parameters["min_hole_width"] = MinHoleWidth;
        parameters["max_hole_width"] = MaxHoleWidth;
        return parameters;
    }
}
