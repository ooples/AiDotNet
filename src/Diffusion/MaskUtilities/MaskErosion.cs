using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Erodes a mask by shrinking masked regions, removing thin protrusions and small artifacts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Morphological erosion shrinks white (masked) regions by taking the minimum value
/// in a local neighborhood. This removes small noise, thin connections, and smooths
/// mask boundaries inward.
/// </para>
/// <para>
/// <b>For Beginners:</b> Erosion makes the white (masked) area smaller. If your mask
/// has small white specks or thin white lines, erosion removes them. It's like
/// "peeling" a layer off the edge of the masked region.
/// </para>
/// </remarks>
public class MaskErosion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _kernelSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="MaskErosion{T}"/> class.
    /// </summary>
    /// <param name="kernelSize">The erosion kernel size (must be odd). Default: 3.</param>
    public MaskErosion(int kernelSize = 3)
    {
        Guard.Positive(kernelSize);
        _kernelSize = kernelSize | 1; // Ensure odd
    }

    /// <summary>
    /// Applies morphological erosion to a mask tensor.
    /// </summary>
    /// <param name="mask">A mask tensor.</param>
    /// <returns>The eroded mask.</returns>
    public Tensor<T> Apply(Tensor<T> mask)
    {
        var shape = mask.Shape;
        var result = new Tensor<T>(shape);
        int height = shape[0];
        int width = shape.Length > 1 ? shape[1] : 1;
        int half = _kernelSize / 2;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                T minVal = NumOps.One;

                for (int dy = -half; dy <= half; dy++)
                {
                    for (int dx = -half; dx <= half; dx++)
                    {
                        int ny = Math.Max(0, Math.Min(y + dy, height - 1));
                        int nx = Math.Max(0, Math.Min(x + dx, width - 1));
                        var val = mask[ny, nx];
                        if (NumOps.LessThan(val, minVal))
                            minVal = val;
                    }
                }

                result[y, x] = minVal;
            }
        }

        return result;
    }
}
