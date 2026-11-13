using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MaskErosion<T> : IDataTransformer<T, Tensor<T>, Tensor<T>>
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
    /// Initializes a new instance of the <see cref="MaskErosion{T}"/> class.
    /// </summary>
    /// <param name="kernelSize">The erosion kernel size (must be odd). Default: 3.</param>
    public MaskErosion(int kernelSize = 3)
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
        throw new NotSupportedException("Mask erosion is not reversible.");
    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) =>
        new[] { "eroded_mask" };

    /// <summary>
    /// Applies morphological erosion to a mask tensor.
    /// </summary>
    /// <param name="mask">A mask tensor.</param>
    /// <returns>The eroded mask.</returns>
    public Tensor<T> Apply(Tensor<T> mask)
    {
        // Neighbourhood minimum (border pixels replicated) = -max(-mask), never above one
        var engine = AiDotNetEngine.Current;
        var minima = engine.TensorNegate(MaskVectorOps.ReplicateMaxFilter2D(engine.TensorNegate(mask), _kernelSize));
        return engine.TensorMin(minima, NumOps.One);
    }
}
