using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Generates masks for outpainting by marking regions outside the original image bounds.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Creates a mask where the original image region is unmasked (0) and extended regions
/// are masked (1). Supports extending in any direction with optional feathering at
/// the boundary for smooth blending.
/// </para>
/// <para>
/// <b>For Beginners:</b> Outpainting extends an image beyond its borders. This utility
/// creates the mask that tells the model which parts are the original image (don't change)
/// and which parts need to be generated (extend). The feathering option makes the
/// transition smoother.
/// </para>
/// <para>
/// When used as an <see cref="IDataTransformer{T, TInput, TOutput}"/>, the Transform method
/// takes the original image tensor and generates an outpainting mask based on the padding
/// configured in the constructor.
/// </para>
/// </remarks>
public class OutpaintingMaskGenerator<T> : IDataTransformer<T, Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _featherRadius;
    private readonly int _padTop;
    private readonly int _padRight;
    private readonly int _padBottom;
    private readonly int _padLeft;

    /// <inheritdoc />
    public bool IsFitted => true;
    /// <inheritdoc />
    public int[]? ColumnIndices => null;
    /// <inheritdoc />
    public bool SupportsInverseTransform => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="OutpaintingMaskGenerator{T}"/> class.
    /// </summary>
    /// <param name="featherRadius">Feathering radius at the boundary. 0 for hard edges.</param>
    /// <param name="padTop">Pixels to extend above the image. Default: 0.</param>
    /// <param name="padRight">Pixels to extend to the right. Default: 0.</param>
    /// <param name="padBottom">Pixels to extend below the image. Default: 0.</param>
    /// <param name="padLeft">Pixels to extend to the left. Default: 0.</param>
    public OutpaintingMaskGenerator(int featherRadius = 8,
        int padTop = 0, int padRight = 0, int padBottom = 0, int padLeft = 0)
    {
        Guard.NonNegative(featherRadius);
        Guard.NonNegative(padTop);
        Guard.NonNegative(padRight);
        Guard.NonNegative(padBottom);
        Guard.NonNegative(padLeft);
        _featherRadius = featherRadius;
        _padTop = padTop;
        _padRight = padRight;
        _padBottom = padBottom;
        _padLeft = padLeft;
    }

    /// <inheritdoc />
    public void Fit(Tensor<T> data) { }

    /// <summary>
    /// Generates an outpainting mask from the input image tensor's dimensions.
    /// The mask covers the extended canvas, with 0 for the original image region
    /// and 1 for the padded regions that need generation.
    /// </summary>
    /// <inheritdoc />
    public Tensor<T> Transform(Tensor<T> data)
    {
        var shape = data.Shape;
        int imageHeight = shape[0];
        int imageWidth = shape.Length > 1 ? shape[1] : 1;
        return Generate(
            imageHeight + _padTop + _padBottom,
            imageWidth + _padLeft + _padRight,
            _padTop, _padLeft, imageHeight, imageWidth);
    }

    /// <inheritdoc />
    public Tensor<T> FitTransform(Tensor<T> data) => Transform(data);
    /// <inheritdoc />
    public Tensor<T> InverseTransform(Tensor<T> data) =>
        throw new NotSupportedException("Outpainting mask generation is not reversible.");
    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) =>
        new[] { "outpainting_mask" };

    /// <summary>
    /// Generates an outpainting mask for the given canvas and image bounds.
    /// </summary>
    /// <param name="canvasHeight">Total canvas height.</param>
    /// <param name="canvasWidth">Total canvas width.</param>
    /// <param name="imageTop">Top offset of the original image on the canvas.</param>
    /// <param name="imageLeft">Left offset of the original image on the canvas.</param>
    /// <param name="imageHeight">Height of the original image.</param>
    /// <param name="imageWidth">Width of the original image.</param>
    /// <returns>A mask tensor where 1 = generate, 0 = keep original.</returns>
    public Tensor<T> Generate(int canvasHeight, int canvasWidth,
        int imageTop, int imageLeft, int imageHeight, int imageWidth)
    {
        var mask = new Tensor<T>(new[] { canvasHeight, canvasWidth });
        var one = NumOps.One;
        var zero = NumOps.Zero;

        // Start with all ones (generate everything)
        for (int y = 0; y < canvasHeight; y++)
            for (int x = 0; x < canvasWidth; x++)
                mask[y, x] = one;

        // Clear the original image region
        for (int y = imageTop; y < Math.Min(imageTop + imageHeight, canvasHeight); y++)
            for (int x = imageLeft; x < Math.Min(imageLeft + imageWidth, canvasWidth); x++)
                mask[y, x] = zero;

        // Apply feathering at the boundary
        if (_featherRadius > 0)
        {
            var featherer = new MaskFeatherer<T>(_featherRadius);
            return featherer.Apply(mask);
        }

        return mask;
    }
}
