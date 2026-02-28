using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Generates a mask from point prompts with circular regions, similar to SAM-style point selection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Creates a mask by placing circles of a specified radius at each point location.
/// Points can be positive (add to mask) or negative (remove from mask). This mirrors
/// the Segment Anything Model (SAM) point prompt interface.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of drawing a mask by hand, you can click on points
/// in an image. Each click creates a circular masked area. You can also use negative
/// points to "un-mask" areas within a masked region.
/// </para>
/// <para>
/// When used as an <see cref="IDataTransformer{T, TInput, TOutput}"/>, the Transform method
/// takes an image tensor, uses its dimensions for height/width, and generates a mask from
/// the points configured in the constructor.
/// </para>
/// </remarks>
public class MaskFromPoints<T> : IDataTransformer<T, Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _pointRadius;
    private readonly (int y, int x)[] _positivePoints;
    private readonly (int y, int x)[]? _negativePoints;

    /// <inheritdoc />
    public bool IsFitted => true;
    /// <inheritdoc />
    public int[]? ColumnIndices => null;
    /// <inheritdoc />
    public bool SupportsInverseTransform => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="MaskFromPoints{T}"/> class.
    /// </summary>
    /// <param name="positivePoints">Points to add to the mask (y, x).</param>
    /// <param name="negativePoints">Points to remove from the mask (y, x). Can be null.</param>
    /// <param name="pointRadius">Radius of each point's circular mask region. Default: 10.</param>
    public MaskFromPoints((int y, int x)[] positivePoints,
        (int y, int x)[]? negativePoints = null,
        int pointRadius = 10)
    {
        Guard.Positive(pointRadius);
        _positivePoints = positivePoints;
        _negativePoints = negativePoints;
        _pointRadius = pointRadius;
    }

    /// <inheritdoc />
    public void Fit(Tensor<T> data) { }

    /// <summary>
    /// Generates a point mask using the input tensor's dimensions and the points
    /// configured in the constructor.
    /// </summary>
    /// <inheritdoc />
    public Tensor<T> Transform(Tensor<T> data)
    {
        var shape = data.Shape;
        int height = shape[0];
        int width = shape.Length > 1 ? shape[1] : 1;
        return Generate(height, width, _positivePoints, _negativePoints);
    }

    /// <inheritdoc />
    public Tensor<T> FitTransform(Tensor<T> data) => Transform(data);
    /// <inheritdoc />
    public Tensor<T> InverseTransform(Tensor<T> data) =>
        throw new NotSupportedException("Point mask generation is not reversible.");
    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) =>
        new[] { "point_mask" };

    /// <summary>
    /// Generates a mask from positive and negative point prompts.
    /// </summary>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    /// <param name="positivePoints">Points to add to the mask (y, x).</param>
    /// <param name="negativePoints">Points to remove from the mask (y, x). Can be null.</param>
    /// <returns>A binary mask where point regions are 1.</returns>
    public Tensor<T> Generate(int height, int width,
        (int y, int x)[] positivePoints, (int y, int x)[]? negativePoints = null)
    {
        var mask = new Tensor<T>(new[] { height, width });
        int radiusSq = _pointRadius * _pointRadius;

        // Apply positive points
        foreach (var (py, px) in positivePoints)
        {
            int yStart = Math.Max(0, py - _pointRadius);
            int yEnd = Math.Min(height, py + _pointRadius + 1);
            int xStart = Math.Max(0, px - _pointRadius);
            int xEnd = Math.Min(width, px + _pointRadius + 1);

            for (int y = yStart; y < yEnd; y++)
            {
                for (int x = xStart; x < xEnd; x++)
                {
                    int dy = y - py;
                    int dx = x - px;
                    if (dy * dy + dx * dx <= radiusSq)
                        mask[y, x] = NumOps.One;
                }
            }
        }

        // Apply negative points
        if (negativePoints != null)
        {
            foreach (var (py, px) in negativePoints)
            {
                int yStart = Math.Max(0, py - _pointRadius);
                int yEnd = Math.Min(height, py + _pointRadius + 1);
                int xStart = Math.Max(0, px - _pointRadius);
                int xEnd = Math.Min(width, px + _pointRadius + 1);

                for (int y = yStart; y < yEnd; y++)
                {
                    for (int x = xStart; x < xEnd; x++)
                    {
                        int dy = y - py;
                        int dx = x - px;
                        if (dy * dy + dx * dx <= radiusSq)
                            mask[y, x] = NumOps.Zero;
                    }
                }
            }
        }

        return mask;
    }
}
