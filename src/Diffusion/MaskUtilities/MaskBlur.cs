using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Applies Gaussian blur to a mask for smooth transitions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Blurs the entire mask using a Gaussian kernel, softening all edges and transitions.
/// Unlike feathering which targets edges specifically, blur affects the entire mask
/// uniformly. Useful for creating soft masks from hard binary masks.
/// </para>
/// <para>
/// <b>For Beginners:</b> This blurs a mask like blurring a photo — sharp edges become
/// soft gradients. It's similar to feathering but affects the whole mask, not just edges.
/// Use this when you want an overall softer mask with gradual transitions everywhere.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MaskBlur<T> : IDataTransformer<T, Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _sigma;
    private readonly int _kernelSize;

    /// <inheritdoc />
    public bool IsFitted => true;
    /// <inheritdoc />
    public int[]? ColumnIndices => null;
    /// <inheritdoc />
    public bool SupportsInverseTransform => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="MaskBlur{T}"/> class.
    /// </summary>
    /// <param name="sigma">Standard deviation of the Gaussian kernel. Default: 2.0.</param>
    public MaskBlur(double sigma = 2.0)
    {
        Guard.Positive(sigma);
        _sigma = sigma;
        _kernelSize = (int)Math.Ceiling(sigma * 3) * 2 + 1;
    }

    /// <inheritdoc />
    public void Fit(Tensor<T> data) { }
    /// <inheritdoc />
    public Tensor<T> Transform(Tensor<T> data) => Apply(data);
    /// <inheritdoc />
    public Tensor<T> FitTransform(Tensor<T> data) => Apply(data);
    /// <inheritdoc />
    public Tensor<T> InverseTransform(Tensor<T> data) =>
        throw new NotSupportedException("Mask blur is not reversible.");
    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) =>
        new[] { "blurred_mask" };

    /// <summary>
    /// Applies Gaussian blur to a mask tensor.
    /// </summary>
    /// <param name="mask">A mask tensor.</param>
    /// <returns>The blurred mask.</returns>
    public Tensor<T> Apply(Tensor<T> mask)
    {
        var engine = AiDotNetEngine.Current;
        int half = _kernelSize / 2;

        // Gaussian kernel exp(-(dy^2 + dx^2) / (2 sigma^2)) = g(dy) g(dx), normalized to sum to 1. Built and
        // normalized in double, then narrowed once: -1 / (2 sigma^2) leaves Half's range for a small sigma, and
        // 0 * -infinity at the center tap is NaN.
        var offsets = engine.TensorArange<double>(-half, half + 1, 1);
        var g = engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorSquare(offsets), -1.0 / (2.0 * _sigma * _sigma)));
        var wide = engine.TensorOuter(g, g);
        var kernel = TensorCopyHelper.CastLogical<double, T>(engine.TensorDivideScalar(wide, engine.TensorSum(wide)));

        var blurred = MaskVectorOps.ReplicateFilter2D(mask, kernel);
        return engine.TensorClamp(blurred, NumOps.Zero, NumOps.One);
    }
}
