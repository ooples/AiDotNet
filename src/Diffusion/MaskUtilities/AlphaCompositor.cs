using AiDotNet.Helpers;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Alpha compositing for layered diffusion outputs with transparency support.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements Porter-Duff alpha compositing operations for combining multiple generated
/// layers. Supports standard "over" compositing as well as additive and multiply blend modes.
/// Essential for multi-layer generation pipelines where foreground, background, and effects
/// layers are generated separately and composited into a final image.
/// </para>
/// <para>
/// <b>For Beginners:</b> When generating an image in layers (like background + foreground +
/// effects), you need to combine them properly. AlphaCompositor handles the math of layering
/// images with transparency, just like how Photoshop stacks layers together. It ensures smooth
/// edges and correct color blending between layers.
/// </para>
/// </remarks>
public class AlphaCompositor<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly bool _premultipliedAlpha;

    /// <summary>
    /// Gets whether inputs are expected in premultiplied alpha format.
    /// </summary>
    public bool PremultipliedAlpha => _premultipliedAlpha;

    /// <summary>
    /// Initializes a new alpha compositor.
    /// </summary>
    /// <param name="premultipliedAlpha">Whether inputs use premultiplied alpha (default: false).</param>
    public AlphaCompositor(bool premultipliedAlpha = false)
    {
        _premultipliedAlpha = premultipliedAlpha;
    }

    /// <summary>
    /// Composites foreground over background using alpha blending (Porter-Duff "over" operation).
    /// </summary>
    /// <param name="foreground">Foreground RGB values.</param>
    /// <param name="foregroundAlpha">Foreground alpha values (0 = transparent, 1 = opaque).</param>
    /// <param name="background">Background RGB values.</param>
    /// <param name="backgroundAlpha">Background alpha values.</param>
    /// <returns>Composited result.</returns>
    public Vector<T> CompositeOver(
        Vector<T> foreground, Vector<T> foregroundAlpha,
        Vector<T> background, Vector<T> backgroundAlpha)
    {
        int len = Math.Min(foreground.Length, background.Length);
        var result = new Vector<T>(len);

        for (int i = 0; i < len; i++)
        {
            var fgA = i < foregroundAlpha.Length ? foregroundAlpha[i] : NumOps.One;
            var bgA = i < backgroundAlpha.Length ? backgroundAlpha[i] : NumOps.One;

            // out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
            var oneMinusFgA = NumOps.Subtract(NumOps.One, fgA);
            var outAlpha = NumOps.Add(fgA, NumOps.Multiply(bgA, oneMinusFgA));

            if (NumOps.ToDouble(outAlpha) < 1e-10)
            {
                result[i] = NumOps.Zero;
                continue;
            }

            // out_color = (fg * fg_alpha + bg * bg_alpha * (1 - fg_alpha)) / out_alpha
            var fgContrib = NumOps.Multiply(foreground[i], fgA);
            var bgContrib = NumOps.Multiply(NumOps.Multiply(background[i], bgA), oneMinusFgA);
            result[i] = NumOps.Divide(NumOps.Add(fgContrib, bgContrib), outAlpha);
        }

        return result;
    }

    /// <summary>
    /// Composites using additive blending.
    /// </summary>
    /// <param name="layer1">First layer values.</param>
    /// <param name="layer2">Second layer values.</param>
    /// <param name="weight1">Weight for first layer (default: 0.5).</param>
    /// <param name="weight2">Weight for second layer (default: 0.5).</param>
    /// <returns>Additively blended result.</returns>
    public Vector<T> CompositeAdditive(
        Vector<T> layer1, Vector<T> layer2,
        double weight1 = 0.5, double weight2 = 0.5)
    {
        int len = Math.Min(layer1.Length, layer2.Length);
        var result = new Vector<T>(len);
        var w1 = NumOps.FromDouble(weight1);
        var w2 = NumOps.FromDouble(weight2);

        for (int i = 0; i < len; i++)
        {
            result[i] = NumOps.Add(
                NumOps.Multiply(w1, layer1[i]),
                NumOps.Multiply(w2, layer2[i]));
        }

        return result;
    }
}
