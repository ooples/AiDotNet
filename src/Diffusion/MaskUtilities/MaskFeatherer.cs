using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

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
public class MaskFeatherer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _radius;

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

    /// <summary>
    /// Applies feathering to a single-channel mask tensor.
    /// </summary>
    /// <param name="mask">A single-channel mask tensor with values in [0, 1].</param>
    /// <returns>A feathered mask with smooth edge transitions.</returns>
    public Tensor<T> Apply(Tensor<T> mask)
    {
        var shape = mask.Shape;
        int height = shape[0];
        int width = shape.Length > 1 ? shape[1] : 1;
        var result = new Tensor<T>(shape);
        var zero = NumOps.Zero;
        var one = NumOps.One;

        // Box blur approximation of Gaussian feathering
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                T sum = zero;
                int count = 0;

                for (int dy = -_radius; dy <= _radius; dy++)
                {
                    for (int dx = -_radius; dx <= _radius; dx++)
                    {
                        int ny = Math.Max(0, Math.Min(y + dy, height - 1));
                        int nx = Math.Max(0, Math.Min(x + dx, width - 1));
                        sum = NumOps.Add(sum, mask[ny, nx]);
                        count++;
                    }
                }

                var avg = NumOps.Divide(sum, NumOps.FromDouble(count));
                if (NumOps.LessThan(avg, zero)) avg = zero;
                if (NumOps.GreaterThan(avg, one)) avg = one;
                result[y, x] = avg;
            }
        }

        return result;
    }
}
