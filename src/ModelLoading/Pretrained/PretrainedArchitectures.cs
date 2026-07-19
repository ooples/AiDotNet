using AiDotNet.Agentic.Models.Local;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// Registry mapping Hugging Face architecture / model-type names to the factory that reconstructs and
/// weight-loads that architecture. Seeded with the LLaMA-family decoders; extend it to teach the
/// pretrained loader new architectures without touching the facade.
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> When a model is loaded, AiDotNet looks up its architecture name (from
/// <c>config.json</c>) here to find the code that knows how to build that particular kind of model.
/// Adding support for a new family is a single <see cref="Register"/> call.
/// </para>
/// </remarks>
public static class PretrainedArchitectures<T>
{
    /// <summary>Builds a weight-loaded network from a parsed config and a tensor source.</summary>
    /// <param name="config">The parsed <c>config.json</c>.</param>
    /// <param name="weights">The checkpoint's tensor source (safetensors/GGUF).</param>
    public delegate NeuralNetworkBase<T> DecoderFactory(HuggingFaceConfig config, INamedTensorSource weights);

    private static readonly object Gate = new();
    private static readonly Dictionary<string, DecoderFactory> Registry =
        new(StringComparer.OrdinalIgnoreCase);

    static PretrainedArchitectures()
    {
        // LLaMA / Mistral: the default bias-free gated-SwiGLU decoder.
        DecoderFactory llama = (config, weights) => LlamaModelBuilder<T>.Build(config, weights);
        foreach (var name in LlamaModelBuilder<T>.SupportedArchitectures)
            Registry[name] = llama;

        // Qwen2: LLaMA-style but with q/k/v attention projection biases.
        DecoderFactory qwen2 = (config, weights) => LlamaModelBuilder<T>.Build(config, weights, DecoderOptions<T>.Qwen2);
        Registry["qwen2"] = qwen2;
        Registry["Qwen2ForCausalLM"] = qwen2;

        // Gemma: GeGLU + (1 + weight) RMSNorm + sqrt(hidden) embedding scale.
        DecoderFactory gemma = (config, weights) => LlamaModelBuilder<T>.Build(config, weights, DecoderOptions<T>.Gemma);
        Registry["gemma"] = gemma;
        Registry["GemmaForCausalLM"] = gemma;

        // Phi-3: LLaMA-style, but its qkv_proj / gate_up_proj projections are fused — split them on read.
        DecoderFactory phi3 = (config, weights) =>
            LlamaModelBuilder<T>.Build(config, new FusedProjectionSource(weights, config));
        Registry["phi3"] = phi3;
        Registry["Phi3ForCausalLM"] = phi3;

        // Mixtral: sparse mixture-of-experts FFN (top-k of E gated experts).
        DecoderFactory mixtral = (config, weights) => MoEModelBuilder<T>.Build(config, weights);
        foreach (var name in MoEModelBuilder<T>.SupportedArchitectures)
            Registry[name] = mixtral;

        // Qwen2-MoE: routed experts + an always-on shared expert, with Qwen2 q/k/v attention biases.
        DecoderFactory qwen2Moe = (config, weights) => Qwen2MoEModelBuilder<T>.Build(config, weights);
        foreach (var name in Qwen2MoEModelBuilder<T>.SupportedArchitectures)
            Registry[name] = qwen2Moe;

        // Gemma-2: sandwiched dual RMSNorms + GeGLU + (1+w) norm + embed scale + final-logit soft-capping.
        DecoderFactory gemma2 = (config, weights) => Gemma2ModelBuilder<T>.Build(config, weights);
        foreach (var name in Gemma2ModelBuilder<T>.SupportedArchitectures)
            Registry[name] = gemma2;

        // Cohere (Command-R): LayerNorm + parallel residual + gated SwiGLU + tied embeddings + logit scale.
        DecoderFactory cohere = (config, weights) => CohereModelBuilder<T>.Build(config, weights);
        foreach (var name in CohereModelBuilder<T>.SupportedArchitectures)
            Registry[name] = cohere;

        // StarCoder2: LayerNorm (with bias) + biased attention projections + non-gated GELU MLP (with bias).
        DecoderFactory starcoder2 = (config, weights) => StarCoder2ModelBuilder<T>.Build(config, weights);
        foreach (var name in StarCoder2ModelBuilder<T>.SupportedArchitectures)
            Registry[name] = starcoder2;
    }

    /// <summary>Registers (or replaces) the factory for an architecture / model-type name.</summary>
    /// <param name="architectureName">The <c>architectures[]</c> class name or <c>model_type</c>
    /// (matched case-insensitively), e.g. <c>"LlamaForCausalLM"</c> or <c>"llama"</c>.</param>
    /// <param name="factory">The factory that builds and weight-loads the network.</param>
    public static void Register(string architectureName, DecoderFactory factory)
    {
        if (string.IsNullOrWhiteSpace(architectureName))
            throw new ArgumentException("Architecture name must be non-empty.", nameof(architectureName));
        Guard.NotNull(factory);
        lock (Gate)
            Registry[architectureName] = factory;
    }

    /// <summary>The registered architecture names, for diagnostics / error messages.</summary>
    public static IReadOnlyCollection<string> RegisteredNames
    {
        get { lock (Gate) return new List<string>(Registry.Keys); }
    }

    /// <summary>
    /// Resolves the factory for a config, trying (in order) an explicit override, each declared
    /// architecture class, then the model type.
    /// </summary>
    /// <param name="config">The parsed config.</param>
    /// <param name="architectureOverride">An optional explicit architecture name.</param>
    /// <param name="factory">The resolved factory, when found.</param>
    /// <returns><c>true</c> when a factory was found.</returns>
    public static bool TryResolve(HuggingFaceConfig config, string? architectureOverride, out DecoderFactory factory)
    {
        Guard.NotNull(config);
        lock (Gate)
        {
            if (architectureOverride is not null && architectureOverride.Trim().Length > 0 &&
                Registry.TryGetValue(architectureOverride, out var byOverride))
            {
                factory = byOverride;
                return true;
            }

            foreach (var arch in config.Architectures)
            {
                if (Registry.TryGetValue(arch, out var byArch))
                {
                    factory = byArch;
                    return true;
                }
            }

            if (!string.IsNullOrWhiteSpace(config.ModelType) &&
                Registry.TryGetValue(config.ModelType, out var byType))
            {
                factory = byType;
                return true;
            }
        }

        factory = (_, _) => throw new InvalidOperationException("unreachable");
        return false;
    }
}
