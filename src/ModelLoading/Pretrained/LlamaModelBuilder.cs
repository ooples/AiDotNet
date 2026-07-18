using AiDotNet.ActivationFunctions;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// Reconstructs a LLaMA-family decoder (LLaMA / Mistral / Qwen2) from a Hugging Face
/// <see cref="HuggingFaceConfig"/> and loads its pretrained weights from a safetensors/GGUF
/// tensor source. The result is a plain <see cref="NeuralNetwork{T}"/> whose layer stack is the
/// canonical servable decoder — <c>[EmbeddingLayer, PreLNTransformerBlock×N, RMSNorm, DenseLayer]</c> —
/// so it runs through the existing paged / continuous-batching / tensor-parallel serving path unchanged.
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
/// <remarks>
/// <para>
/// Faithfulness: RMSNorm + RoPE grouped-query attention + gated SwiGLU FFN, all bias-free, matching
/// the reference architecture (Touvron 2023; Shazeer 2020). Weight layouts follow the AiDotNet layer
/// conventions: attention and dense projections store <c>[in, out]</c>, so Hugging Face's <c>[out, in]</c>
/// matrices are transposed on load; the embedding table is <c>[vocab, hidden]</c> in both.
/// </para>
/// </remarks>
public static class LlamaModelBuilder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The architecture family names this builder recognizes (matched against config.json's
    /// <c>architectures[0]</c> / <c>model_type</c>, case-insensitively).</summary>
    public static IReadOnlyList<string> SupportedArchitectures { get; } = new[]
    {
        "LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM",
        "llama", "mistral", "qwen2",
    };

    /// <summary>
    /// Builds and weight-loads a decoder for <paramref name="config"/> from <paramref name="weights"/>.
    /// </summary>
    /// <param name="config">The parsed <c>config.json</c>.</param>
    /// <param name="weights">The tensor source (a loaded safetensors file or GGUF), keyed by Hugging Face
    /// parameter names (e.g. <c>model.layers.0.self_attn.q_proj.weight</c>).</param>
    /// <param name="warmupSequenceLength">Sequence length of the one-shot warmup forward used to
    /// materialize lazily-shaped weights before loading. Must be ≥ 1.</param>
    /// <returns>A ready-to-serve <see cref="NeuralNetwork{T}"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="config"/> or
    /// <paramref name="weights"/> is null.</exception>
    /// <exception cref="InvalidDataException">Thrown when a required tensor is absent, has an
    /// unsupported shape, or the config's head geometry cannot be represented.</exception>
    public static NeuralNetwork<T> Build(HuggingFaceConfig config, INamedTensorSource weights, int warmupSequenceLength = 8)
    {
        Guard.NotNull(config);
        Guard.NotNull(weights);
        if (warmupSequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(warmupSequenceLength));

        int hidden = config.HiddenSize;
        int numHeads = config.NumAttentionHeads;
        int numKVHeads = config.NumKeyValueHeads;
        // Honor an explicit head_dim (e.g. Gemma-style, where numHeads*headDim != hidden); the attention
        // layer is built with this head dimension and the projection widths follow from it.
        int headDim = config.HeadDim;
        bool explicitHeadDim = headDim != hidden / numHeads;
        int kvDim = numKVHeads * headDim;
        int intermediate = config.IntermediateSize;
        int vocab = config.VocabSize;
        int maxPos = config.MaxPositionEmbeddings;

        // ---- assemble the canonical servable decoder stack ----
        var embedding = new EmbeddingLayer<T>(vocab, hidden);
        var layers = new List<ILayer<T>> { embedding };

        var blocks = new PreLNTransformerBlock<T>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var attention = new GroupedQueryAttentionLayer<T>(
                sequenceLength: maxPos, embeddingDimension: hidden, numHeads: numHeads, numKVHeads: numKVHeads,
                headDimension: explicitHeadDim ? headDim : null);
            attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, config.RopeTheta, maxPos);

            // Gated SwiGLU FFN with SiLU on the gate path (bias-free), matching LLaMA/Mistral/Qwen2.
            var block = new PreLNTransformerBlock<T>(
                hiddenSize: hidden, ffnDim: intermediate, attention: attention,
                ffnActivation: new SiLUActivation<T>(), gated: true);
            blocks[i] = block;
            layers.Add(block);
        }

        var finalNorm = new RMSNormalizationLayer<T>(hidden, config.RmsNormEps);
        layers.Add(finalNorm);

        var lmHead = new DenseLayer<T>(vocab, activationFunction: new IdentityActivation<T>());
        layers.Add(lmHead);

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple,
            inputSize: 1, outputSize: vocab, layers: layers);
        var network = new NeuralNetwork<T>(architecture);

        // Materialize every lazily-shaped weight (GQA projections, FFN dense matrices, LM head) with a
        // single warmup forward, so the per-layer SetParameters/gamma writes below have real tensors to fill.
        var warmup = new Tensor<T>(new[] { 1, warmupSequenceLength });
        for (int p = 0; p < warmupSequenceLength; p++)
            warmup[0, p] = NumOps.Zero;
        network.Predict(warmup);

        // ---- load pretrained weights ----
        // Embedding: HF model.embed_tokens.weight is [vocab, hidden] row-major — same layout AiDotNet uses.
        var embedData = ReadTensor(weights, EmbedName, vocab * hidden);
        embedding.SetParameters(new Vector<T>(embedData));

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var block = blocks[i];
            string p = $"model.layers.{i}.";

            // RMSNorm gammas ([hidden]) load directly into the live gamma tensors.
            WriteGamma(block.Norm1, ReadTensor(weights, p + "input_layernorm.weight", hidden));
            WriteGamma(block.Norm2, ReadTensor(weights, p + "post_attention_layernorm.weight", hidden));

            // Attention: HF q/k/v/o_proj are [out, in]; the GQA layer stores [in, out]; load order is
            // Q, K, V, O, then the (zero) output bias.
            var qT = TransposeOutInToInOut(weights, p + "self_attn.q_proj.weight", outDim: numHeads * headDim, inDim: hidden);
            var kT = TransposeOutInToInOut(weights, p + "self_attn.k_proj.weight", outDim: kvDim, inDim: hidden);
            var vT = TransposeOutInToInOut(weights, p + "self_attn.v_proj.weight", outDim: kvDim, inDim: hidden);
            var oT = TransposeOutInToInOut(weights, p + "self_attn.o_proj.weight", outDim: hidden, inDim: numHeads * headDim);
            var attnParams = Concat(qT, kT, vT, oT, new T[hidden]); // trailing zeros = output bias
            ((GroupedQueryAttentionLayer<T>)block.AttentionLayer).SetParameters(new Vector<T>(attnParams));

            // Gated SwiGLU FFN: gate/up are [intermediate, hidden] → [hidden, intermediate]; down is
            // [hidden, intermediate] → [intermediate, hidden]. Each DenseLayer stores [in, out] + zero bias.
            LoadDense(block.FfnGate ?? throw new InvalidOperationException("gated block missing FfnGate"),
                weights, p + "mlp.gate_proj.weight", outDim: intermediate, inDim: hidden);
            LoadDense(block.FfnUp, weights, p + "mlp.up_proj.weight", outDim: intermediate, inDim: hidden);
            LoadDense(block.FfnDown, weights, p + "mlp.down_proj.weight", outDim: hidden, inDim: intermediate);
        }

        WriteGamma(finalNorm, ReadTensor(weights, "model.norm.weight", hidden));

        // LM head: HF lm_head.weight is [vocab, hidden]; when tie_word_embeddings, reuse the embedding.
        string headName = HasTensor(weights, LmHeadName) ? LmHeadName : EmbedName;
        if (!config.TieWordEmbeddings && !HasTensor(weights, LmHeadName))
            throw new InvalidDataException(
                "config does not tie word embeddings but lm_head.weight is absent from the checkpoint.");
        LoadDense(lmHead, weights, headName, outDim: vocab, inDim: hidden);

        return network;
    }

    private const string EmbedName = "model.embed_tokens.weight";
    private const string LmHeadName = "lm_head.weight";

    // Loads a DenseLayer's weights ([in, out] after transposing HF's [out, in]) and a zero bias.
    private static void LoadDense(DenseLayer<T> dense, INamedTensorSource weights, string name, int outDim, int inDim)
    {
        var wInOut = TransposeOutInToInOut(weights, name, outDim, inDim);
        var full = Concat(wInOut, new T[outDim]); // trailing zeros = bias
        dense.SetParameters(new Vector<T>(full));
    }

    // Writes a [featureSize] gamma vector into an RMSNorm layer's live gamma tensor.
    private static void WriteGamma(RMSNormalizationLayer<T> norm, T[] gamma)
    {
        var live = norm.GetGammaTensor();
        if (live.Length != gamma.Length)
            throw new InvalidDataException(
                $"RMSNorm gamma length {live.Length} does not match loaded weight length {gamma.Length}.");
        var span = live.AsWritableSpan();
        for (int i = 0; i < gamma.Length; i++)
            span[i] = gamma[i];
    }

    // Reads a named tensor's values (row-major) as T, verifying the element count.
    private static T[] ReadTensor(INamedTensorSource weights, string name, int expectedCount)
    {
        if (!HasTensor(weights, name))
            throw new InvalidDataException($"checkpoint is missing required tensor '{name}'.");
        double[] raw = weights.ReadAsDouble(name);
        if (raw.Length != expectedCount)
            throw new InvalidDataException(
                $"tensor '{name}' has {raw.Length} elements, expected {expectedCount}.");
        var result = new T[raw.Length];
        for (int i = 0; i < raw.Length; i++)
            result[i] = NumOps.FromDouble(raw[i]);
        return result;
    }

    // Reads an HF [outDim, inDim] row-major matrix and returns it transposed to [inDim, outDim] row-major.
    private static T[] TransposeOutInToInOut(INamedTensorSource weights, string name, int outDim, int inDim)
    {
        var src = ReadTensor(weights, name, outDim * inDim);
        var dst = new T[inDim * outDim];
        for (int o = 0; o < outDim; o++)
            for (int i = 0; i < inDim; i++)
                dst[i * outDim + o] = src[o * inDim + i];
        return dst;
    }

    private static bool HasTensor(INamedTensorSource weights, string name)
    {
        foreach (var n in weights.TensorNames)
            if (string.Equals(n, name, StringComparison.Ordinal))
                return true;
        return false;
    }

    private static T[] Concat(params T[][] parts)
    {
        int total = 0;
        foreach (var part in parts) total += part.Length;
        var result = new T[total];
        int offset = 0;
        foreach (var part in parts)
        {
            Array.Copy(part, 0, result, offset, part.Length);
            offset += part.Length;
        }
        return result;
    }
}
