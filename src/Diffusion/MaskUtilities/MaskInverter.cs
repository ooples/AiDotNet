using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Inverts a mask so masked regions become unmasked and vice versa.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Mask inversion computes (1 - mask) for each pixel, flipping the masked and unmasked
/// regions. This is commonly used when you want to inpaint the background instead of
/// the foreground, or to convert between "keep" and "replace" mask conventions.
/// </para>
/// <para>
/// <b>For Beginners:</b> If your mask highlights a person (white = person, black = background),
/// inverting it highlights the background instead. This is useful when different tools
/// use opposite conventions for what "masked" means.
/// </para>
/// </remarks>
public class MaskInverter<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Inverts a mask tensor by computing (1 - value) for each pixel.
    /// </summary>
    /// <param name="mask">A mask tensor with values in [0, 1].</param>
    /// <returns>The inverted mask.</returns>
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
                result[y, x] = NumOps.Subtract(NumOps.One, mask[y, x]);
            }
        }

        return result;
    }
}
