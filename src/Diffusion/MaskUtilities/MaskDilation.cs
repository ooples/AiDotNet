using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Dilates a mask by expanding masked regions, filling small holes and connecting nearby regions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Morphological dilation expands white (masked) regions by taking the maximum value
/// in a local neighborhood. This fills small holes, connects nearby components, and
/// grows mask boundaries outward.
/// </para>
/// <para>
/// <b>For Beginners:</b> Dilation makes the white (masked) area larger. If your mask
/// has small gaps or holes, dilation fills them in. It's like "padding" the edge
/// of the masked region outward. Often used with erosion for mask cleanup.
/// </para>
/// </remarks>
public class MaskDilation<T> : IDataTransformer<T, Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _kernelSize;

    /// <inheritdoc />
    public bool IsFitted => true;
    /// <inheritdoc />
    public int[]? ColumnIndices => null;
    /// <inheritdoc />
    public bool SupportsInverseTransform => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="MaskDilation{T}"/> class.
    /// </summary>
    /// <param name="kernelSize">The dilation kernel size (must be odd). Default: 3.</param>
    public MaskDilation(int kernelSize = 3)
    {
        Guard.Positive(kernelSize);
        _kernelSize = kernelSize | 1; // Ensure odd
    }

    /// <inheritdoc />
    public void Fit(Tensor<T> data) { }
    /// <inheritdoc />
    public Tensor<T> Transform(Tensor<T> data) => Apply(data);
    /// <inheritdoc />
    public Tensor<T> FitTransform(Tensor<T> data) => Apply(data);
    /// <inheritdoc />
    public Tensor<T> InverseTransform(Tensor<T> data) =>
        throw new NotSupportedException("Mask dilation is not reversible.");
    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) =>
        new[] { "dilated_mask" };

    /// <summary>
    /// Applies morphological dilation to a mask tensor.
    /// </summary>
    /// <param name="mask">A mask tensor.</param>
    /// <returns>The dilated mask.</returns>
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
                T maxVal = NumOps.Zero;

                for (int dy = -half; dy <= half; dy++)
                {
                    for (int dx = -half; dx <= half; dx++)
                    {
                        int ny = Math.Max(0, Math.Min(y + dy, height - 1));
                        int nx = Math.Max(0, Math.Min(x + dx, width - 1));
                        var val = mask[ny, nx];
                        if (NumOps.GreaterThan(val, maxVal))
                            maxVal = val;
                    }
                }

                result[y, x] = maxVal;
            }
        }

        return result;
    }
}
