using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// Reconstructs a DBRX decoder from a Hugging Face <see cref="HuggingFaceConfig"/> and loads its weights.
/// DBRX is a LayerNorm sparse-MoE decoder that fuses q/k/v (Wqkv) and stacks all experts into single tensors;
/// a <see cref="DbrxTensorSource"/> (wrapped in a <see cref="FusedProjectionSource"/>) exposes the weights
/// under standard Hugging Face names so the usual loaders apply.
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
/// <remarks>
/// Bias-free attention + LayerNorm norms. The <c>clip_qkv</c> post-projection clamp is not applied (it rarely
/// binds at DBRX's default clip and would require attention-layer support).
/// </remarks>
public static class DbrxModelBuilder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The architecture family names this builder recognizes.</summary>
    public static IReadOnlyList<string> SupportedArchitectures { get; } = new[]
    {
        "DbrxForCausalLM", "dbrx",
    };

    /// <summary>Builds and weight-loads a DBRX decoder.</summary>
    public static NeuralNetwork<T> Build(HuggingFaceConfig config, INamedTensorSource rawWeights, int warmupSequenceLength = 8)
    {
        Guard.NotNull(config);
        Guard.NotNull(rawWeights);
        if (warmupSequenceLength < 1) throw new ArgumentOutOfRangeException(nameof(warmupSequenceLength));
        if (config.NumLocalExperts <= 0 || config.NumExpertsPerTok <= 0)
            throw new InvalidDataException("DBRX requires moe_num_experts and moe_top_k.");

        int hidden = config.HiddenSize;
        int numHeads = config.NumAttentionHeads;
        int numKVHeads = config.NumKeyValueHeads;
        int headDim = config.HeadDim;
        bool explicitHeadDim = headDim != hidden / numHeads;
        int vocab = config.VocabSize;
        int maxPos = config.MaxPositionEmbeddings;
        int numExperts = config.NumLocalExperts;
        int topK = config.NumExpertsPerTok;
        int ffn = config.MoeIntermediateSize > 0 ? config.MoeIntermediateSize : config.IntermediateSize;
        double normEps = config.RmsNormEps; // DBRX norm epsilon lands in this slot

        // Rename/unstack DBRX tensors to HF names, then split the fused Wqkv. A raw DBRX safetensors source
        // names blocks transformer.blocks.* and needs DbrxTensorSource; a GGUF source already presents the
        // Hugging Face names (and slices its own stacked experts), so only the fused-qkv split is applied there.
        bool rawDbrx = LlamaModelBuilder<T>.HasTensor(rawWeights, "transformer.wte.weight");
        INamedTensorSource translated = rawDbrx ? new DbrxTensorSource(rawWeights, config) : rawWeights;
        var weights = new FusedProjectionSource(translated, config);

        var embedding = new EmbeddingLayer<T>(vocab, hidden);
        var layers = new List<ILayer<T>> { embedding };

        var blocks = new DbrxDecoderBlock<T>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var attention = new GroupedQueryAttentionLayer<T>(
                sequenceLength: maxPos, embeddingDimension: hidden, numHeads: numHeads, numKVHeads: numKVHeads,
                headDimension: explicitHeadDim ? headDim : null, useCausalMask: true);
            attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, config.RopeTheta, maxPos);

            var moe = new MoEFeedForwardLayer<T>(hidden, ffn, numExperts, topK, new SiLUActivation<T>());
            var block = new DbrxDecoderBlock<T>(hidden, attention, moe, normEps);
            blocks[i] = block;
            layers.Add(block);
        }

        var finalNorm = new LayerNormalizationLayer<T>(hidden, normEps);
        layers.Add(finalNorm);
        var lmHead = new DenseLayer<T>(vocab, activationFunction: new IdentityActivation<T>());
        layers.Add(lmHead);

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple, inputSize: 1, outputSize: vocab, layers: layers);
        var network = new NeuralNetwork<T>(architecture);

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

            WriteGamma(block.Norm1, LlamaModelBuilder<T>.ReadTensor(weights, p + "input_layernorm.weight", hidden));
            LlamaModelBuilder<T>.LoadAttention((GroupedQueryAttentionLayer<T>)block.AttentionLayer, weights, p, numHeads, numKVHeads, headDim, hidden);
            WriteGamma(block.Norm2, LlamaModelBuilder<T>.ReadTensor(weights, p + "post_attention_layernorm.weight", hidden));

            var moe = block.Moe;
            LlamaModelBuilder<T>.LoadDense(moe.Router, weights, p + "mlp.gate.weight", outDim: numExperts, inDim: hidden);
            for (int e = 0; e < numExperts; e++)
            {
                string ep = p + $"mlp.experts.{e}.";
                LlamaModelBuilder<T>.LoadDense(moe.ExpertGate(e), weights, ep + "gate_proj.weight", outDim: ffn, inDim: hidden);
                LlamaModelBuilder<T>.LoadDense(moe.ExpertUp(e), weights, ep + "up_proj.weight", outDim: ffn, inDim: hidden);
                LlamaModelBuilder<T>.LoadDense(moe.ExpertDown(e), weights, ep + "down_proj.weight", outDim: hidden, inDim: ffn);
            }
        }

        WriteGamma(finalNorm, LlamaModelBuilder<T>.ReadTensor(weights, "model.norm.weight", hidden));

        string headName = LlamaModelBuilder<T>.HasTensor(weights, LlamaModelBuilder<T>.LmHeadName)
            ? LlamaModelBuilder<T>.LmHeadName : LlamaModelBuilder<T>.EmbedName;
        if (!config.TieWordEmbeddings && !LlamaModelBuilder<T>.HasTensor(weights, LlamaModelBuilder<T>.LmHeadName))
            throw new InvalidDataException(
                "config does not tie word embeddings but lm_head.weight is absent from the checkpoint.");
        LlamaModelBuilder<T>.LoadDense(lmHead, weights, headName, outDim: vocab, inDim: hidden);

        return network;
    }

    // Writes a LayerNorm's weight (gamma) into its live tensor; DBRX LayerNorms are bias-free (beta stays zero).
    private static void WriteGamma(LayerNormalizationLayer<T> norm, T[] gamma)
    {
        var live = norm.GetGammaTensor();
        if (live.Length != gamma.Length)
            throw new InvalidDataException(
                $"LayerNorm gamma length {live.Length} does not match loaded weight length {gamma.Length}.");
        var span = live.AsWritableSpan();
        for (int i = 0; i < gamma.Length; i++) span[i] = gamma[i];
    }
}
