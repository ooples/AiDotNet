using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Generates a binary mask from a segmentation map by selecting specific class labels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Takes a segmentation tensor where each pixel holds a class label and produces
/// a binary mask where selected classes are 1 and all others are 0. This bridges
/// semantic segmentation output to inpainting mask input.
/// </para>
/// <para>
/// <b>For Beginners:</b> Segmentation models label every pixel (e.g., "sky", "person",
/// "building"). This utility converts those labels into a mask — for example, you
/// could select "sky" to create a mask that covers only the sky region for replacement.
/// </para>
/// </remarks>
public class MaskFromSegmentation<T> : IDataTransformer<T, Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly HashSet<int> _selectedLabels;

    /// <inheritdoc />
    public bool IsFitted => true;
    /// <inheritdoc />
    public int[]? ColumnIndices => null;
    /// <inheritdoc />
    public bool SupportsInverseTransform => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="MaskFromSegmentation{T}"/> class.
    /// </summary>
    /// <param name="selectedLabels">The segmentation class labels to include in the mask.</param>
    public MaskFromSegmentation(IEnumerable<int> selectedLabels)
    {
        _selectedLabels = new HashSet<int>(selectedLabels);
    }

    /// <inheritdoc />
    public void Fit(Tensor<T> data) { }
    /// <inheritdoc />
    public Tensor<T> Transform(Tensor<T> data) => Apply(data);
    /// <inheritdoc />
    public Tensor<T> FitTransform(Tensor<T> data) => Apply(data);
    /// <inheritdoc />
    public Tensor<T> InverseTransform(Tensor<T> data) =>
        throw new NotSupportedException("Segmentation to mask conversion is not reversible.");
    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) =>
        new[] { "segmentation_mask" };

    /// <summary>
    /// Generates a binary mask from a segmentation map.
    /// </summary>
    /// <param name="segmentationMap">A tensor where each value represents a class label.</param>
    /// <returns>A binary mask where selected labels are 1 and others are 0.</returns>
    public Tensor<T> Apply(Tensor<T> segmentationMap)
    {
        var shape = segmentationMap.Shape;
        var result = new Tensor<T>(shape);
        int height = shape[0];
        int width = shape.Length > 1 ? shape[1] : 1;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int label = (int)NumOps.ToDouble(segmentationMap[y, x]);
                result[y, x] = _selectedLabels.Contains(label)
                    ? NumOps.One
                    : NumOps.Zero;
            }
        }

        return result;
    }
}
