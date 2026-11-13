using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Applies feathering (soft blurring) to mask edges for smooth transitions in inpainting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Feathering smooths the boundary between masked and unmasked regions using a
/// Gaussian-like blur, producing gradual transitions that prevent harsh edges
/// in inpainted results.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you paint a mask for inpainting, the edges can be
/// very sharp (pixel is either 100% masked or 0%). Feathering blurs those edges
/// so the transition is gradual, producing more natural-looking inpainting results.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MaskFeatherer<T> : IDataTransformer<T, Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _radius;

    /// <inheritdoc />
    public bool IsFitted => true;
    /// <inheritdoc />
    public int[]? ColumnIndices => null;
    /// <inheritdoc />
    public bool SupportsInverseTransform => false;

    /// <summary>
    /// Gets the feathering radius in pixels.
    /// </summary>
    public int Radius => _radius;

    /// <summary>
    /// Initializes a new instance of the <see cref="MaskFeatherer{T}"/> class.
    /// </summary>
    /// <param name="radius">The feathering radius in pixels. Larger values produce softer edges.</param>
    public MaskFeatherer(int radius = 5)
    {
        Guard.Positive(radius);
        _radius = radius;
    }

    /// <inheritdoc />
    public void Fit(Tensor<T> data) { }
    /// <inheritdoc />
    public Tensor<T> Transform(Tensor<T> data) => Apply(data);
    /// <inheritdoc />
    public Tensor<T> FitTransform(Tensor<T> data) => Apply(data);
    /// <inheritdoc />
    public Tensor<T> InverseTransform(Tensor<T> data) =>
        throw new NotSupportedException("Mask feathering is not reversible.");
    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) =>
        new[] { "feathered_mask" };

    /// <summary>
    /// Applies feathering to a single-channel mask tensor.
    /// </summary>
    /// <param name="mask">A single-channel mask tensor with values in [0, 1].</param>
    /// <returns>A feathered mask with smooth edge transitions.</returns>
    public Tensor<T> Apply(Tensor<T> mask)
    {
        var engine = AiDotNetEngine.Current;
        int size = 2 * _radius + 1;

        // Box blur approximation of Gaussian feathering: a replicate-padded box sum divided by the window size
        var box = Tensor<T>.CreateDefault(new[] { size, size }, NumOps.One);
        var sum = MaskVectorOps.ReplicateFilter2D(mask, box);
        var avg = engine.TensorDivideScalar(sum, NumOps.FromDouble(size * size));
        return engine.TensorClamp(avg, NumOps.Zero, NumOps.One);
    }
}
