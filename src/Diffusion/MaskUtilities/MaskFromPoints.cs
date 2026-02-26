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
/// </remarks>
public class MaskFromPoints<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _pointRadius;

    /// <summary>
    /// Initializes a new instance of the <see cref="MaskFromPoints{T}"/> class.
    /// </summary>
    /// <param name="pointRadius">Radius of each point's circular mask region. Default: 10.</param>
    public MaskFromPoints(int pointRadius = 10)
    {
        Guard.Positive(pointRadius);
        _pointRadius = pointRadius;
    }

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
