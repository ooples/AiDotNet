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
/// </remarks>
public class MaskFromBoundingBox<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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
