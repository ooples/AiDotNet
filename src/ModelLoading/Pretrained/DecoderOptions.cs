using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// Per-family knobs for <see cref="LlamaModelBuilder{T}"/> that capture the small, faithful differences
/// between LLaMA-family decoders (LLaMA/Mistral/Qwen2 use the defaults; Gemma flips the norm and embedding
/// options; Phi-3 pre-splits its fused projections before build so it also uses the defaults).
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
public sealed class DecoderOptions<T>
{
    /// <summary>The gated-FFN activation applied on the gate path. Default <see cref="SiLUActivation{T}"/>
    /// (LLaMA/Mistral/Qwen2/Phi-3); Gemma uses the tanh-approximation <see cref="GELUActivation{T}"/> (GeGLU).</summary>
    public IActivationFunction<T> FfnActivation { get; init; } = new SiLUActivation<T>();

    /// <summary>
    /// When true, RMSNorm scales by <c>(1 + weight)</c> rather than <c>weight</c> — the Gemma convention.
    /// Implemented by loading <c>weight + 1</c> into the norm's gamma.
    /// </summary>
    public bool RmsNormAddsOne { get; init; }

    /// <summary>
    /// When true, the token embeddings are multiplied by <c>sqrt(hidden_size)</c> — the Gemma embedding
    /// normalizer. Baked into the embedding table on load; a tied LM head still uses the UNSCALED embedding.
    /// </summary>
    public bool ScaleEmbeddingBySqrtHidden { get; init; }

    /// <summary>The LLaMA/Mistral/Qwen2 defaults (SiLU gate, plain RMSNorm, no embedding scaling).</summary>
    public static DecoderOptions<T> Llama { get; } = new();

    /// <summary>The Gemma decoder options: GeGLU (tanh GELU), <c>(1 + weight)</c> RMSNorm, √hidden embedding scale.</summary>
    public static DecoderOptions<T> Gemma { get; } = new()
    {
        FfnActivation = new GELUActivation<T>(),
        RmsNormAddsOne = true,
        ScaleEmbeddingBySqrtHidden = true,
    };
}
