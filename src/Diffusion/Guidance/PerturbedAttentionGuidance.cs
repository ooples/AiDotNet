using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Guidance;

/// <summary>
/// Perturbed Attention Guidance (PAG) for diffusion model inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PAG replaces the unconditional prediction with a "perturbed" prediction where
/// self-attention maps are modified (e.g., replaced with identity). This produces
/// better guidance than standard CFG without needing a separate unconditional pass.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of comparing "with prompt" vs "without prompt",
/// PAG compares "normal attention" vs "broken attention." This gives better image
/// quality, especially at high guidance scales where CFG can cause artifacts.
/// </para>
/// <para>
/// Reference: Ahn et al., "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance", ECCV 2024
/// </para>
/// </remarks>
public class PerturbedAttentionGuidance<T> : IGuidanceMethod<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _perturbationScale;

    /// <inheritdoc />
    public GuidanceType GuidanceType => GuidanceType.PerturbedAttention;

    /// <summary>
    /// Initializes a new Perturbed Attention Guidance instance.
    /// </summary>
    /// <param name="perturbationScale">Scale of the attention perturbation. Default: 1.0.</param>
    public PerturbedAttentionGuidance(double perturbationScale = 1.0)
    {
        _perturbationScale = perturbationScale;
    }

    /// <inheritdoc />
    public Tensor<T> Apply(Tensor<T> unconditional, Tensor<T> conditional, double scale, double timestep)
    {
        var result = new Tensor<T>(unconditional.Shape);
        var uncondSpan = unconditional.AsSpan();
        var condSpan = conditional.AsSpan();
        var resultSpan = result.AsWritableSpan();

        // PAG: guided = cond + scale * perturbation_scale * (cond - uncond_perturbed)
        var scaleT = NumOps.FromDouble(scale * _perturbationScale);

        for (int i = 0; i < resultSpan.Length; i++)
        {
            var diff = NumOps.Subtract(condSpan[i], uncondSpan[i]);
            resultSpan[i] = NumOps.Add(condSpan[i], NumOps.Multiply(scaleT, diff));
        }

        return result;
    }
}
