using AiDotNet.Helpers;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Blends latent representations using a mask for seamless inpainting and region editing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Performs per-element blending of two latent tensors using a mask that defines the mixing
/// ratio at each spatial location. Operates in latent space (after VAE encoding) for
/// efficient inpainting. Supports both hard (binary) and soft (feathered) masks for
/// smooth transitions between inpainted and original regions.
/// </para>
/// <para>
/// <b>For Beginners:</b> When editing just part of an image (like replacing a face or
/// removing an object), you need to smoothly blend the new content with the original.
/// LatentMaskBlender does this in the model's internal representation, mixing old and new
/// content according to a mask that says "use new content here, keep old content there."
/// </para>
/// </remarks>
public class LatentMaskBlender<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _blendSharpness;

    /// <summary>
    /// Gets the blend sharpness (higher = sharper mask transitions).
    /// </summary>
    public double BlendSharpness => _blendSharpness;

    /// <summary>
    /// Initializes a new latent mask blender.
    /// </summary>
    /// <param name="blendSharpness">Sharpness of mask transitions (default: 1.0; higher = sharper edges).</param>
    public LatentMaskBlender(double blendSharpness = 1.0)
    {
        _blendSharpness = blendSharpness;
    }

    /// <summary>
    /// Blends two latent vectors using a mask.
    /// </summary>
    /// <param name="original">Original latent (kept where mask is 0).</param>
    /// <param name="generated">Generated latent (used where mask is 1).</param>
    /// <param name="mask">Blending mask (0 = original, 1 = generated).</param>
    /// <returns>Blended latent result.</returns>
    public Vector<T> Blend(Vector<T> original, Vector<T> generated, Vector<T> mask)
    {
        int len = Math.Min(original.Length, generated.Length);
        var result = new Vector<T>(len);

        for (int i = 0; i < len; i++)
        {
            var m = i < mask.Length ? mask[i] : NumOps.Zero;
            var oneMinusM = NumOps.Subtract(NumOps.One, m);
            result[i] = NumOps.Add(
                NumOps.Multiply(oneMinusM, original[i]),
                NumOps.Multiply(m, generated[i]));
        }

        return result;
    }

    /// <summary>
    /// Blends with noise-aware scheduling for diffusion inpainting.
    /// </summary>
    /// <param name="original">Original noisy latent at current timestep.</param>
    /// <param name="generated">Model's denoised prediction.</param>
    /// <param name="mask">Blending mask.</param>
    /// <param name="timestepRatio">Current timestep ratio (1.0 = full noise, 0.0 = clean).</param>
    /// <returns>Blended latent with timestep-aware mask scaling.</returns>
    public Vector<T> BlendWithSchedule(
        Vector<T> original, Vector<T> generated, Vector<T> mask, double timestepRatio)
    {
        int len = Math.Min(original.Length, generated.Length);
        var result = new Vector<T>(len);
        var scheduleScale = NumOps.FromDouble(Math.Pow(timestepRatio, _blendSharpness));

        for (int i = 0; i < len; i++)
        {
            var baseMask = i < mask.Length ? mask[i] : NumOps.Zero;
            var m = NumOps.Multiply(baseMask, scheduleScale);
            var oneMinusM = NumOps.Subtract(NumOps.One, m);
            result[i] = NumOps.Add(
                NumOps.Multiply(oneMinusM, original[i]),
                NumOps.Multiply(m, generated[i]));
        }

        return result;
    }
}
