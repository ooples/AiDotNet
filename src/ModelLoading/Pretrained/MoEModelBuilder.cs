using AiDotNet.ActivationFunctions;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;

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
/// </remarks>
public static class MoEModelBuilder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The MoE architecture family names this builder recognizes.</summary>
    public static IReadOnlyList<string> SupportedArchitectures { get; } = new[]
    {
        "MixtralForCausalLM", "mixtral",
    };

    /// <summary>Builds and weight-loads a Mixtral-style MoE decoder.</summary>
    /// <param name="config">The parsed <c>config.json</c> (must declare <c>num_local_experts</c> and
    /// <c>num_experts_per_tok</c>).</param>
    /// <param name="weights">The tensor source keyed by Hugging Face parameter names.</param>
    /// <param name="warmupSequenceLength">Warmup forward length for lazy-weight materialization (≥ 1).</param>
    public static NeuralNetwork<T> Build(HuggingFaceConfig config, INamedTensorSource weights, int warmupSequenceLength = 8)
    {
        Guard.NotNull(config);
        Guard.NotNull(weights);
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

        var embedding = new EmbeddingLayer<T>(vocab, hidden);
        var layers = new List<ILayer<T>> { embedding };

        var blocks = new MoEDecoderBlock<T>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var attention = new GroupedQueryAttentionLayer<T>(
                sequenceLength: maxPos, embeddingDimension: hidden, numHeads: numHeads, numKVHeads: numKVHeads,
                headDimension: explicitHeadDim ? headDim : null);
            attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, config.RopeTheta, maxPos);

            var moe = new MoEFeedForwardLayer<T>(hidden, intermediate, numExperts, topK, new SiLUActivation<T>());
            var block = new MoEDecoderBlock<T>(hidden, attention, moe, config.RmsNormEps);
            blocks[i] = block;
            layers.Add(block);
        }

        var finalNorm = new RMSNormalizationLayer<T>(hidden, config.RmsNormEps);
        layers.Add(finalNorm);
        var lmHead = new DenseLayer<T>(vocab, activationFunction: new IdentityActivation<T>());
        layers.Add(lmHead);

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple, inputSize: 1, outputSize: vocab, layers: layers);
        var network = new NeuralNetwork<T>(architecture);

        // Warmup materializes embed/attention/final-norm/lm-head + the router; a warmup forward only routes
        // to the top-k experts, so materialize every expert explicitly before loading their weights.
        var warmup = new Tensor<T>(new[] { 1, warmupSequenceLength });
        for (int p = 0; p < warmupSequenceLength; p++) warmup[0, p] = NumOps.Zero;
        network.Predict(warmup);
        foreach (var b in blocks) b.Moe.Materialize();

        // ---- load pretrained weights ----
        embedding.SetParameters(new Vector<T>(
            LlamaModelBuilder<T>.ReadTensor(weights, LlamaModelBuilder<T>.EmbedName, vocab * hidden)));

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var block = blocks[i];
            string p = $"model.layers.{i}.";

            LlamaModelBuilder<T>.LoadGamma(block.Norm1, weights, p + "input_layernorm.weight", hidden, addOne: false);
            LlamaModelBuilder<T>.LoadGamma(block.Norm2, weights, p + "post_attention_layernorm.weight", hidden, addOne: false);
            LlamaModelBuilder<T>.LoadAttention((GroupedQueryAttentionLayer<T>)block.AttentionLayer, weights, p, numHeads, numKVHeads, headDim, hidden);

            // Router: block_sparse_moe.gate.weight is [numExperts, hidden].
            LlamaModelBuilder<T>.LoadDense(block.Moe.Router, weights, p + "block_sparse_moe.gate.weight", outDim: numExperts, inDim: hidden);

            // Experts: w1 = gate, w3 = up (both [ffn, hidden]); w2 = down ([hidden, ffn]).
            for (int e = 0; e < numExperts; e++)
            {
                string ep = p + $"block_sparse_moe.experts.{e}.";
                LlamaModelBuilder<T>.LoadDense(block.Moe.ExpertGate(e), weights, ep + "w1.weight", outDim: intermediate, inDim: hidden);
                LlamaModelBuilder<T>.LoadDense(block.Moe.ExpertUp(e), weights, ep + "w3.weight", outDim: intermediate, inDim: hidden);
                LlamaModelBuilder<T>.LoadDense(block.Moe.ExpertDown(e), weights, ep + "w2.weight", outDim: hidden, inDim: intermediate);
            }
        }

        LlamaModelBuilder<T>.LoadGamma(finalNorm, weights, "model.norm.weight", hidden, addOne: false);

        string headName = LlamaModelBuilder<T>.HasTensor(weights, LlamaModelBuilder<T>.LmHeadName)
            ? LlamaModelBuilder<T>.LmHeadName : LlamaModelBuilder<T>.EmbedName;
        if (!config.TieWordEmbeddings && !LlamaModelBuilder<T>.HasTensor(weights, LlamaModelBuilder<T>.LmHeadName))
            throw new InvalidDataException(
                "config does not tie word embeddings but lm_head.weight is absent from the checkpoint.");
        LlamaModelBuilder<T>.LoadDense(lmHead, weights, headName, outDim: vocab, inDim: hidden);

        return network;
    }
}
