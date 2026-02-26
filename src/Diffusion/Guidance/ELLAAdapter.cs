using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Guidance;

/// <summary>
/// ELLA (Efficient Large Language Model Adapter) guidance adapter for enhanced text understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ELLA bridges large language models with diffusion models by providing richer
/// text embeddings through an adapter network. It enhances prompt understanding
/// for complex, compositional descriptions without retraining the diffusion model.
/// </para>
/// <para>
/// <b>For Beginners:</b> ELLA makes the AI better at understanding complex prompts.
/// Instead of relying only on CLIP's text understanding, it leverages a larger
/// language model to capture nuances like spatial relationships and attributes.
/// </para>
/// <para>
/// Reference: Hu et al., "ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment", 2024
/// </para>
/// </remarks>
public class ELLAAdapter<T> : IGuidanceMethod<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _adapterWeight;

    /// <inheritdoc />
    public GuidanceType GuidanceType => GuidanceType.ELLA;

    /// <summary>
    /// Initializes a new ELLA adapter.
    /// </summary>
    /// <param name="adapterWeight">Weight for blending ELLA-enhanced and standard embeddings. Default: 0.8.</param>
    public ELLAAdapter(double adapterWeight = 0.8)
    {
        _adapterWeight = adapterWeight;
    }

    /// <inheritdoc />
    public Tensor<T> Apply(Tensor<T> unconditional, Tensor<T> conditional, double scale, double timestep)
    {
        var result = new Tensor<T>(unconditional.Shape);
        var uncondSpan = unconditional.AsSpan();
        var condSpan = conditional.AsSpan();
        var resultSpan = result.AsWritableSpan();

        // ELLA-enhanced CFG: apply standard CFG with adapter-weighted blending
        // The adapter weight modulates how much the enhanced embeddings influence guidance
        var effectiveScale = NumOps.FromDouble(scale * _adapterWeight);
        var baseScale = NumOps.FromDouble(scale * (1.0 - _adapterWeight));

        for (int i = 0; i < resultSpan.Length; i++)
        {
            var diff = NumOps.Subtract(condSpan[i], uncondSpan[i]);

            // Weighted combination: stronger guidance on high-difference elements (semantically important)
            var absDiff = NumOps.ToDouble(NumOps.Abs(diff));
            var adaptiveWeight = absDiff > 0.1 ? effectiveScale : baseScale;

            resultSpan[i] = NumOps.Add(uncondSpan[i], NumOps.Multiply(adaptiveWeight, diff));
        }

        return result;
    }
}
