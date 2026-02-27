using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

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
        var shape = mask.Shape;
        var result = new Tensor<T>(shape);
        int height = shape[0];
        int width = shape.Length > 1 ? shape[1] : 1;
        int half = _kernelSize / 2;

        // Build Gaussian kernel weights
        var kernel = new double[_kernelSize, _kernelSize];
        double kernelSum = 0;
        double twoSigmaSq = 2.0 * _sigma * _sigma;

        for (int dy = -half; dy <= half; dy++)
        {
            for (int dx = -half; dx <= half; dx++)
            {
                double weight = Math.Exp(-(dy * dy + dx * dx) / twoSigmaSq);
                kernel[dy + half, dx + half] = weight;
                kernelSum += weight;
            }
        }

        // Normalize kernel
        for (int ky = 0; ky < _kernelSize; ky++)
            for (int kx = 0; kx < _kernelSize; kx++)
                kernel[ky, kx] /= kernelSum;

        // Apply convolution
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double sum = 0;

                for (int dy = -half; dy <= half; dy++)
                {
                    for (int dx = -half; dx <= half; dx++)
                    {
                        int ny = Math.Max(0, Math.Min(y + dy, height - 1));
                        int nx = Math.Max(0, Math.Min(x + dx, width - 1));
                        sum += NumOps.ToDouble(mask[ny, nx]) * kernel[dy + half, dx + half];
                    }
                }

                result[y, x] = NumOps.FromDouble(Math.Max(0, Math.Min(sum, 1.0)));
            }
        }

        return result;
    }
}
