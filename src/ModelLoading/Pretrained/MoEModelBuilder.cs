using AiDotNet.ActivationFunctions;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// Reconstructs a sparse mixture-of-experts decoder (Mixtral) from a Hugging Face
/// <see cref="HuggingFaceConfig"/> and loads its pretrained weights. Same pre-LN RMSNorm + RoPE-GQA
/// attention as the dense LLaMA builder, but the FFN is a <see cref="MoEFeedForwardLayer{T}"/>
/// (top-<c>k</c>-of-<c>E</c> gated experts). Produces a servable <see cref="NeuralNetwork{T}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
/// <remarks>
/// Weight names follow Mixtral: <c>block_sparse_moe.gate.weight</c> is the router, and each expert has
/// <c>experts.{e}.w1</c> (gate), <c>w3</c> (up), <c>w2</c> (down). All projections are bias-free and stored
/// <c>[out, in]</c>, so they are transposed to the layer's <c>[in, out]</c> on load. The MoE blocks are not
/// recognized by the tensor-parallel partitioner, so imported MoE models serve through the base decode path.
/// <para>OLMoE (<c>OlmoeForCausalLM</c>) is the same decoder with three differences: an RMSNorm over the whole
/// query and key projections (<c>self_attn.q_norm</c>, <c>k_norm</c>) before the rotary embedding; the router keeps
/// the softmax-over-all-experts probabilities of its top-k unless <c>norm_topk_prob</c> is true; and the weights
/// are named <c>mlp.gate</c> and <c>mlp.experts.{e}.gate_proj</c> / <c>up_proj</c> / <c>down_proj</c>.</para>
/// </remarks>
public static class MoEModelBuilder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The MoE architecture family names this builder recognizes.</summary>
    public static IReadOnlyList<string> SupportedArchitectures { get; } = new[]
    {
        "MixtralForCausalLM", "mixtral", "OlmoeForCausalLM", "olmoe",
    };

    private static bool IsOlmoe(HuggingFaceConfig config)
        => string.Equals(config.ModelType, "olmoe", StringComparison.OrdinalIgnoreCase)
            || config.Architectures.Any(a => string.Equals(a, "OlmoeForCausalLM", StringComparison.Ordinal));

    /// <summary>Builds and weight-loads a Mixtral-style MoE decoder.</summary>
    /// <param name="config">The parsed <c>config.json</c> (must declare <c>num_local_experts</c> and
    /// <c>num_experts_per_tok</c>).</param>
    /// <param name="weights">The tensor source keyed by Hugging Face parameter names.</param>
    /// <param name="warmupSequenceLength">Warmup forward length for lazy-weight materialization (≥ 1).</param>
    public static NeuralNetwork<T> Build(HuggingFaceConfig config, INamedTensorSource weights, int warmupSequenceLength = 8)
        => BuildCore(config, weights, weightStore: null, warmupSequenceLength);

    /// <summary>
    /// Builds and weight-loads a Mixtral-style MoE decoder whose weights go straight into a streaming store
    /// (inference; process-global, like every weight-streaming configuration).
    /// </summary>
    /// <param name="config">The parsed <c>config.json</c>.</param>
    /// <param name="weights">The tensor source keyed by Hugging Face parameter names.</param>
    /// <param name="weightStore">
    /// The store encoding. Each decoder
    /// block's experts are allocated, loaded and encoded in turn, so the load never holds every weight at full
    /// precision: a bf16 checkpoint peaks near its bf16 size plus one block, where loading at full precision and
    /// streaming afterwards held the whole fp32 model first (a 6.9B-parameter MoE: 27.6 GB of weights, 48.1 GB peak
    /// private bytes).
    /// </param>
    /// <param name="warmupSequenceLength">Warmup forward length for lazy-weight materialization (≥ 1).</param>
    public static NeuralNetwork<T> BuildStreamed(HuggingFaceConfig config, INamedTensorSource weights,
        AiDotNet.Tensors.LinearAlgebra.StreamingStoreDtype weightStore, int warmupSequenceLength = 8)
        => BuildCore(config, weights, weightStore, warmupSequenceLength);

    private static NeuralNetwork<T> BuildCore(HuggingFaceConfig config, INamedTensorSource weights,
        AiDotNet.Tensors.LinearAlgebra.StreamingStoreDtype? weightStore, int warmupSequenceLength)
    {
        Guard.NotNull(config);
        Guard.NotNull(weights);
        // The store as the caller had it, before building the network (whose warmup forward already sets the store's
        // training/inference hint): a streamed load that fails puts this back.
        var previousStore = WeightRegistry.CurrentOptions;
        bool? previousExecutionTraining = WeightRegistry.StreamingExecutionTraining;
        if (warmupSequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(warmupSequenceLength));
        if (config.NumLocalExperts <= 0 || config.NumExpertsPerTok <= 0)
            throw new InvalidDataException(
                "MoE decoder requires num_local_experts and num_experts_per_tok in config.json.");

        int hidden = config.HiddenSize;
        int numHeads = config.NumAttentionHeads;
        int numKVHeads = config.NumKeyValueHeads;
        int headDim = config.HeadDim;
        bool explicitHeadDim = headDim != hidden / numHeads;
        int intermediate = config.IntermediateSize;
        int vocab = config.VocabSize;
        int maxPos = config.MaxPositionEmbeddings;
        int numExperts = config.NumLocalExperts;
        int topK = config.NumExpertsPerTok;
        bool olmoe = IsOlmoe(config);
        // Mixtral always renormalizes the kept probabilities; OlmoeConfig's norm_topk_prob defaults to false.
        bool normalizeTopK = !olmoe || (config.NormTopKProb ?? false);

        var embedding = new EmbeddingLayer<T>(vocab, hidden);
        var layers = new List<ILayer<T>> { embedding };

        var blocks = new MoEDecoderBlock<T>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var attention = new GroupedQueryAttentionLayer<T>(
                sequenceLength: maxPos, embeddingDimension: hidden, numHeads: numHeads, numKVHeads: numKVHeads,
                headDimension: explicitHeadDim ? headDim : null, useCausalMask: true);
            attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, config.RopeTheta, maxPos);

            var moe = new MoEFeedForwardLayer<T>(hidden, intermediate, numExperts, topK, new SiLUActivation<T>(),
                sharedFfnDim: 0, normalizeTopK: normalizeTopK);
            var block = new MoEDecoderBlock<T>(hidden, attention, moe, config.RmsNormEps);
            blocks[i] = block;
            layers.Add(block);
        }

        var finalNorm = new RMSNormalizationLayer<T>(hidden, config.RmsNormEps);
        layers.Add(finalNorm);
        var lmHead = LlamaModelBuilder<T>.CreateLmHead(embedding, config.TieWordEmbeddings, vocab);
        layers.Add(lmHead);

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple, inputSize: 1, outputSize: vocab, layers: layers);
        var network = LlamaModelBuilder<T>.CreateImportNetwork(architecture);

        // Warmup materializes embed/attention/final-norm/lm-head + the router; a warmup forward only routes
        // to the top-k experts, so materialize every expert explicitly before loading their weights.
        var warmup = new Tensor<T>(new[] { 1, warmupSequenceLength });
        warmup.Fill(NumOps.Zero);
        network.Predict(warmup);
        bool incremental = weightStore is not null;
        // Resident: materialize every expert now. Streamed: one block at a time, inside the load loop.
        if (!incremental)
        {
            foreach (var b in blocks) b.Moe.Materialize();
            LoadWeights(config, weights, network, embedding, blocks, finalNorm, lmHead, incremental: false);
            return network;
        }

        // The store is process-global: a load that throws partway puts it back as it was, without the blocks
        // already loaded, rather than leaving it configured (options and training/inference hint) for a network the caller never received.
        network.BeginIncrementalWeightStreaming(weightStore!.Value);
        try
        {
            LoadWeights(config, weights, network, embedding, blocks, finalNorm, lmHead, incremental: true);
        }
        catch (Exception loadFailure)
        {
            try { network.AbortIncrementalWeightStreaming(previousStore, previousExecutionTraining); }
            catch (Exception restoreFailure)
            {
                throw new AggregateException(
                    "The streamed load failed, and the streaming store could not be restored afterwards.",
                    loadFailure, restoreFailure);
            }
            throw;
        }
        return network;
    }

    private static void LoadWeights(HuggingFaceConfig config, INamedTensorSource weights, NeuralNetwork<T> network,
        EmbeddingLayer<T> embedding, MoEDecoderBlock<T>[] blocks, RMSNormalizationLayer<T> finalNorm, ILayer<T> lmHead,
        bool incremental)
    {
        int hidden = config.HiddenSize;
        int numHeads = config.NumAttentionHeads;
        int numKVHeads = config.NumKeyValueHeads;
        int headDim = config.HeadDim;
        int intermediate = config.IntermediateSize;
        int vocab = config.VocabSize;
        int numExperts = config.NumLocalExperts;
        bool olmoe = IsOlmoe(config);
        var scratch = new WeightLoadScratch<T>();

        // ---- load pretrained weights ----
        embedding.SetParameters(new Vector<T>(
            LlamaModelBuilder<T>.ReadTensor(weights, LlamaModelBuilder<T>.EmbedName, vocab * hidden)));
        if (incremental) network.RegisterLoadedLayerWeights(embedding);

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var block = blocks[i];
            string p = $"model.layers.{i}.";
            if (incremental) block.Moe.Materialize();

            LlamaModelBuilder<T>.LoadGamma(block.Norm1, weights, p + "input_layernorm.weight", hidden, addOne: false);
            LlamaModelBuilder<T>.LoadGamma(block.Norm2, weights, p + "post_attention_layernorm.weight", hidden, addOne: false);
            LlamaModelBuilder<T>.LoadAttention((GroupedQueryAttentionLayer<T>)block.AttentionLayer, weights, p, numHeads, numKVHeads, headDim, hidden, LlamaModelBuilder<T>.ShouldPermuteRope(weights));

            if (olmoe)
            {
                // q_norm / k_norm: RMSNorm over the whole projection ([heads * headDim] and [kvHeads * headDim]).
                // Their weights index the projection's output rows, so they take the q/k rows' rotary permutation.
                int qWidth = numHeads * headDim, kWidth = numKVHeads * headDim;
                bool permute = LlamaModelBuilder<T>.ShouldPermuteRope(weights);
                var qNormValues = LlamaModelBuilder<T>.ReadTensor(weights, p + "self_attn.q_norm.weight", qWidth);
                var kNormValues = LlamaModelBuilder<T>.ReadTensor(weights, p + "self_attn.k_norm.weight", kWidth);
                if (permute)
                {
                    qNormValues = LlamaModelBuilder<T>.PermuteRopeRowsHalfToInterleaved(qNormValues, numHeads, headDim, inDim: 1);
                    kNormValues = LlamaModelBuilder<T>.PermuteRopeRowsHalfToInterleaved(kNormValues, numKVHeads, headDim, inDim: 1);
                }
                var qNorm = new Tensor<T>(new[] { qWidth }, new Vector<T>(qNormValues));
                var kNorm = new Tensor<T>(new[] { kWidth }, new Vector<T>(kNormValues));
                ((GroupedQueryAttentionLayer<T>)block.AttentionLayer).ConfigureProjectionNorm(qNorm, kNorm, config.RmsNormEps);
            }

            // Router: [numExperts, hidden]. Experts: gate and up [ffn, hidden]; down [hidden, ffn].
            string moePrefix = p + (olmoe ? "mlp." : "block_sparse_moe.");
            string gateName = olmoe ? "gate_proj" : "w1", upName = olmoe ? "up_proj" : "w3", downName = olmoe ? "down_proj" : "w2";
            LlamaModelBuilder<T>.LoadDense(block.Moe.Router, weights, moePrefix + "gate.weight", outDim: numExperts, inDim: hidden, scratch);
            for (int e = 0; e < numExperts; e++)
            {
                string ep = moePrefix + $"experts.{e}.";
                LlamaModelBuilder<T>.LoadDense(block.Moe.ExpertGate(e), weights, ep + gateName + ".weight", outDim: intermediate, inDim: hidden, scratch);
                LlamaModelBuilder<T>.LoadDense(block.Moe.ExpertUp(e), weights, ep + upName + ".weight", outDim: intermediate, inDim: hidden, scratch);
                LlamaModelBuilder<T>.LoadDense(block.Moe.ExpertDown(e), weights, ep + downName + ".weight", outDim: hidden, inDim: intermediate, scratch);
            }
            if (incremental) network.RegisterLoadedLayerWeights(block);
        }

        LlamaModelBuilder<T>.LoadGamma(finalNorm, weights, "model.norm.weight", hidden, addOne: false);

        LlamaModelBuilder<T>.LoadLmHead(lmHead, weights, vocab, hidden);
        if (incremental) network.FinishIncrementalWeightStreaming();
    }
}
