using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Converts a soft mask (continuous values) into a binary mask (0 or 1) using a threshold.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Binarization converts masks with gradual transitions into hard masks where each
/// pixel is either fully masked (1) or fully unmasked (0). Useful when downstream
/// operations require strict binary masks.
/// </para>
/// <para>
/// <b>For Beginners:</b> Some masks have "soft" edges with values like 0.3 or 0.7.
/// Binarizing converts everything above a threshold to 1 (masked) and everything
/// below to 0 (unmasked), creating a clean on/off mask.
/// </para>
/// </remarks>
public class MaskBinarizer<T> : IDataTransformer<T, Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _threshold;

    /// <inheritdoc />
    public bool IsFitted => true;
    /// <inheritdoc />
    public int[]? ColumnIndices => null;
    /// <inheritdoc />
    public bool SupportsInverseTransform => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="MaskBinarizer{T}"/> class.
    /// </summary>
    /// <param name="threshold">The threshold value. Pixels above this become 1, below become 0. Default: 0.5.</param>
    public MaskBinarizer(double threshold = 0.5)
    {
        _threshold = NumOps.FromDouble(threshold);
    }

    /// <inheritdoc />
    public void Fit(Tensor<T> data) { }
    /// <inheritdoc />
    public Tensor<T> Transform(Tensor<T> data) => Apply(data);
    /// <inheritdoc />
    public Tensor<T> FitTransform(Tensor<T> data) => Apply(data);
    /// <inheritdoc />
    public Tensor<T> InverseTransform(Tensor<T> data) =>
        throw new NotSupportedException("Mask binarization is not reversible.");
    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) =>
        new[] { "binary_mask" };

    /// <summary>
    /// Binarizes a mask tensor.
    /// </summary>
    /// <param name="mask">A mask tensor with values in [0, 1].</param>
    /// <returns>A binary mask with values of exactly 0 or 1.</returns>
    public Tensor<T> Apply(Tensor<T> mask)
    {
        var shape = mask.Shape;
        var result = new Tensor<T>(shape);
        int height = shape[0];
        int width = shape.Length > 1 ? shape[1] : 1;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                result[y, x] = NumOps.GreaterThan(mask[y, x], _threshold)
                    ? NumOps.One
                    : NumOps.Zero;
            }
        }

        return result;
    }
}
