using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Generates a mask from one or more bounding box regions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Creates a binary mask where pixels inside any of the specified bounding boxes
/// are masked (1) and all other pixels are unmasked (0). Useful for region-based
/// inpainting from object detection results.
/// </para>
/// <para>
/// <b>For Beginners:</b> If you have a bounding box around an object (like from an
/// object detector), this utility converts that box into a mask you can use for
/// inpainting or editing that specific region.
/// </para>
/// <para>
/// When used as an <see cref="IDataTransformer{T, TInput, TOutput}"/>, the Transform method
/// takes an image tensor, uses its dimensions for height/width, and generates a mask from
/// the bounding boxes configured in the constructor.
/// </para>
/// </remarks>
public class MaskFromBoundingBox<T> : IDataTransformer<T, Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly (int top, int left, int bottom, int right)[] _boxes;

    /// <inheritdoc />
    public bool IsFitted => true;
    /// <inheritdoc />
    public int[]? ColumnIndices => null;
    /// <inheritdoc />
    public bool SupportsInverseTransform => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="MaskFromBoundingBox{T}"/> class.
    /// </summary>
    /// <param name="boxes">Bounding boxes as (top, left, bottom, right) tuples to include in the mask.</param>
    public MaskFromBoundingBox(params (int top, int left, int bottom, int right)[] boxes)
    {
        _boxes = boxes;
    }

    /// <inheritdoc />
    public void Fit(Tensor<T> data) { }

    /// <summary>
    /// Generates a bounding box mask using the input tensor's dimensions and the boxes
    /// configured in the constructor.
    /// </summary>
    /// <inheritdoc />
    public Tensor<T> Transform(Tensor<T> data)
    {
        var shape = data.Shape;
        int height = shape[0];
        int width = shape.Length > 1 ? shape[1] : 1;
        return Generate(height, width, _boxes);
    }

    /// <inheritdoc />
    public Tensor<T> FitTransform(Tensor<T> data) => Transform(data);
    /// <inheritdoc />
    public Tensor<T> InverseTransform(Tensor<T> data) =>
        throw new NotSupportedException("Bounding box mask generation is not reversible.");
    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) =>
        new[] { "bounding_box_mask" };

    /// <summary>
    /// Generates a mask covering the specified bounding boxes.
    /// </summary>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    /// <param name="boxes">Array of bounding boxes as (top, left, bottom, right) tuples.</param>
    /// <returns>A binary mask where bounding box regions are 1.</returns>
    public Tensor<T> Generate(int height, int width, (int top, int left, int bottom, int right)[] boxes)
    {
        var mask = new Tensor<T>(new[] { height, width });

        foreach (var (top, left, bottom, right) in boxes)
        {
            int yStart = Math.Max(0, top);
            int yEnd = Math.Min(height, bottom);
            int xStart = Math.Max(0, left);
            int xEnd = Math.Min(width, right);

            for (int y = yStart; y < yEnd; y++)
                for (int x = xStart; x < xEnd; x++)
                    mask[y, x] = NumOps.One;
        }

        return mask;
    }
}
