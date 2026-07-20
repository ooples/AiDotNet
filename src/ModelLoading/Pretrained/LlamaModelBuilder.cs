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
        "LlamaForCausalLM", "MistralForCausalLM",
        "llama", "mistral",
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
    public static NeuralNetwork<T> Build(HuggingFaceConfig config, INamedTensorSource weights,
        DecoderOptions<T>? options = null, int warmupSequenceLength = 8)
    {
        Guard.NotNull(config);
        Guard.NotNull(weights);
        if (warmupSequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(warmupSequenceLength));
        var opt = options ?? DecoderOptions<T>.Llama;

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
                headDimension: explicitHeadDim ? headDim : null, useProjectionBias: opt.UseAttentionQkvBias,
                useCausalMask: true);
            attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, config.RopeTheta, maxPos);

            // Gated FFN (bias-free): SiLU gate for LLaMA/Mistral/Qwen2/Phi-3, GELU (GeGLU) for Gemma.
            var block = new PreLNTransformerBlock<T>(
                hiddenSize: hidden, ffnDim: intermediate, attention: attention,
                ffnActivation: opt.FfnActivation, gated: true);
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
        // Gemma multiplies the embeddings by sqrt(hidden); bake that into the embedding table (a tied LM head
        // below still reads the UNSCALED embedding from the source).
        var embedData = ReadTensor(weights, EmbedName, vocab * hidden);
        if (opt.ScaleEmbeddingBySqrtHidden)
        {
            var scale = NumOps.FromDouble(Math.Sqrt(hidden));
            var scaled = new T[embedData.Length];
            for (int k = 0; k < embedData.Length; k++)
                scaled[k] = NumOps.Multiply(embedData[k], scale);
            embedding.SetParameters(new Vector<T>(scaled));
        }
        else
        {
            embedding.SetParameters(new Vector<T>(embedData));
        }

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var block = blocks[i];
            string p = $"model.layers.{i}.";

            // RMSNorm gammas ([hidden]) load directly into the live gamma tensors (Gemma: gamma = weight + 1).
            LoadGamma(block.Norm1, weights, p + "input_layernorm.weight", hidden, opt.RmsNormAddsOne);
            LoadGamma(block.Norm2, weights, p + "post_attention_layernorm.weight", hidden, opt.RmsNormAddsOne);

            if (opt.UseAttentionQkvBias)
                LoadAttentionBiased((GroupedQueryAttentionLayer<T>)block.AttentionLayer, weights, p, numHeads, numKVHeads, headDim, hidden);
            else
                LoadAttention((GroupedQueryAttentionLayer<T>)block.AttentionLayer, weights, p, numHeads, numKVHeads, headDim, hidden);

            // Gated SwiGLU FFN: gate/up are [intermediate, hidden] → [hidden, intermediate]; down is
            // [hidden, intermediate] → [intermediate, hidden]. Each DenseLayer stores [in, out] + zero bias.
            LoadDense(block.FfnGate ?? throw new InvalidOperationException("gated block missing FfnGate"),
                weights, p + "mlp.gate_proj.weight", outDim: intermediate, inDim: hidden);
            LoadDense(block.FfnUp, weights, p + "mlp.up_proj.weight", outDim: intermediate, inDim: hidden);
            LoadDense(block.FfnDown, weights, p + "mlp.down_proj.weight", outDim: hidden, inDim: intermediate);
        }

        LoadGamma(finalNorm, weights, "model.norm.weight", hidden, opt.RmsNormAddsOne);

        // LM head: HF lm_head.weight is [vocab, hidden]; when tie_word_embeddings, reuse the embedding.
        string headName = HasTensor(weights, LmHeadName) ? LmHeadName : EmbedName;
        if (!config.TieWordEmbeddings && !HasTensor(weights, LmHeadName))
            throw new InvalidDataException(
                "config does not tie word embeddings but lm_head.weight is absent from the checkpoint.");
        LoadDense(lmHead, weights, headName, outDim: vocab, inDim: hidden);

        return network;
    }

    internal const string EmbedName = "model.embed_tokens.weight";
    internal const string LmHeadName = "lm_head.weight";

    // Loads a GQA layer's Q/K/V/O projections (HF [out,in] -> layer [in,out]) + zero output bias, in the
    // layer's SetParameters order (Q, K, V, O, bias). Shared by the dense and MoE decoder builders.
    internal static void LoadAttention(GroupedQueryAttentionLayer<T> attn, INamedTensorSource weights,
        string layerPrefix, int numHeads, int numKVHeads, int headDim, int hidden)
    {
        int kvDim = numKVHeads * headDim;
        var qT = TransposeOutInToInOut(weights, layerPrefix + "self_attn.q_proj.weight", outDim: numHeads * headDim, inDim: hidden);
        var kT = TransposeOutInToInOut(weights, layerPrefix + "self_attn.k_proj.weight", outDim: kvDim, inDim: hidden);
        var vT = TransposeOutInToInOut(weights, layerPrefix + "self_attn.v_proj.weight", outDim: kvDim, inDim: hidden);
        var oT = TransposeOutInToInOut(weights, layerPrefix + "self_attn.o_proj.weight", outDim: hidden, inDim: numHeads * headDim);
        var attnParams = Concat(qT, kT, vT, oT, new T[hidden]); // trailing zeros = output bias
        attn.SetParameters(new Vector<T>(attnParams));
    }

    // Loads a DenseLayer's weights ([in, out] after transposing HF's [out, in]) and a zero bias.
    internal static void LoadDense(DenseLayer<T> dense, INamedTensorSource weights, string name, int outDim, int inDim)
    {
        var wInOut = TransposeOutInToInOut(weights, name, outDim, inDim);
        var full = Concat(wInOut, new T[outDim]); // trailing zeros = bias
        dense.SetParameters(new Vector<T>(full));
    }

    // Reads a bias tensor if present, otherwise returns a zero vector of the given length.
    internal static T[] OptionalBias(INamedTensorSource weights, string name, int length)
        => HasTensor(weights, name) ? ReadTensor(weights, name, length) : new T[length];

    // Loads a DenseLayer's weights ([in, out]) and its real bias ([out]) — for biased models (StarCoder2).
    internal static void LoadDenseBiased(DenseLayer<T> dense, INamedTensorSource weights, string weightName, string biasName, int outDim, int inDim)
    {
        var wInOut = TransposeOutInToInOut(weights, weightName, outDim, inDim);
        var bias = ReadTensor(weights, biasName, outDim);
        dense.SetParameters(new Vector<T>(Concat(wInOut, bias)));
    }

    // Loads a GQA layer's Q/K/V/O weights AND their real q/k/v/o biases (StarCoder2), in the layer's
    // biased SetParameters order (Q, K, V, O weights, then q, k, v biases, then the output bias).
    internal static void LoadAttentionBiased(GroupedQueryAttentionLayer<T> attn, INamedTensorSource weights,
        string layerPrefix, int numHeads, int numKVHeads, int headDim, int hidden)
    {
        int kvDim = numKVHeads * headDim, qDim = numHeads * headDim;
        var qT = TransposeOutInToInOut(weights, layerPrefix + "self_attn.q_proj.weight", outDim: qDim, inDim: hidden);
        var kT = TransposeOutInToInOut(weights, layerPrefix + "self_attn.k_proj.weight", outDim: kvDim, inDim: hidden);
        var vT = TransposeOutInToInOut(weights, layerPrefix + "self_attn.v_proj.weight", outDim: kvDim, inDim: hidden);
        var oT = TransposeOutInToInOut(weights, layerPrefix + "self_attn.o_proj.weight", outDim: hidden, inDim: qDim);
        // All projection biases are loaded when present and left zero otherwise (Qwen2 biases q/k/v but not o;
        // StarCoder2 biases all four).
        var qB = OptionalBias(weights, layerPrefix + "self_attn.q_proj.bias", qDim);
        var kB = OptionalBias(weights, layerPrefix + "self_attn.k_proj.bias", kvDim);
        var vB = OptionalBias(weights, layerPrefix + "self_attn.v_proj.bias", kvDim);
        var oB = OptionalBias(weights, layerPrefix + "self_attn.o_proj.bias", hidden);
        attn.SetParameters(new Vector<T>(Concat(qT, kT, vT, oT, qB, kB, vB, oB)));
    }

    // Reads an RMSNorm weight and writes it into the layer's gamma, optionally as (weight + 1) for the
    // Gemma convention where the norm scales by (1 + weight).
    internal static void LoadGamma(RMSNormalizationLayer<T> norm, INamedTensorSource weights, string name, int hidden, bool addOne)
    {
        var gamma = ReadTensor(weights, name, hidden);
        if (addOne)
        {
            for (int i = 0; i < gamma.Length; i++)
                gamma[i] = NumOps.Add(gamma[i], NumOps.One);
        }
        WriteGamma(norm, gamma);
    }

    // Writes a [featureSize] gamma vector into an RMSNorm layer's gamma parameter. Routes through the
    // layer's SetParameters (an in-place copy that ALSO calls Engine.InvalidatePersistentTensor) instead of
    // writing the raw AsWritableSpan directly: the raw span is a documented escape hatch that does NOT bump
    // the tensor Version, so a gamma loaded under a live GPU engine would leave the persistent GPU buffer at
    // its construction-time value (gamma = 1) and every RMSNorm forward would run against the wrong scale.
    private static void WriteGamma(RMSNormalizationLayer<T> norm, T[] gamma)
    {
        long expected = norm.ParameterCount;
        if (expected != gamma.Length)
            throw new InvalidDataException(
                $"RMSNorm gamma length {expected} does not match loaded weight length {gamma.Length}.");
        norm.SetParameters(new Vector<T>(gamma));
    }

    // Reads a named tensor's values (row-major) as T, verifying the element count.
    internal static T[] ReadTensor(INamedTensorSource weights, string name, int expectedCount)
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
    internal static T[] TransposeOutInToInOut(INamedTensorSource weights, string name, int outDim, int inDim)
    {
        var src = ReadTensor(weights, name, outDim * inDim);
        var dst = new T[inDim * outDim];
        for (int o = 0; o < outDim; o++)
            for (int i = 0; i < inDim; i++)
                dst[i * outDim + o] = src[o * inDim + i];
        return dst;
    }

    internal static bool HasTensor(INamedTensorSource weights, string name)
    {
        foreach (var n in weights.TensorNames)
            if (string.Equals(n, name, StringComparison.Ordinal))
                return true;
        return false;
    }

    internal static T[] Concat(params T[][] parts)
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
