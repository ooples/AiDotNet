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
/// Reconstructs a Qwen2-MoE decoder from a Hugging Face <see cref="HuggingFaceConfig"/> and loads its
/// weights. Like Mixtral it routes each token through the top-k of many experts, but adds an <em>always-on
/// shared expert</em> gated by <c>sigmoid(shared_expert_gate(x))</c>, uses <c>moe_intermediate_size</c> for
/// the routed experts and <c>shared_expert_intermediate_size</c> for the shared one, and (Qwen2) biases the
/// q/k/v attention projections.
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
public static class Qwen2MoEModelBuilder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The architecture family names this builder recognizes.</summary>
    public static IReadOnlyList<string> SupportedArchitectures { get; } = new[]
    {
        "Qwen2MoeForCausalLM", "qwen2_moe",
    };

    /// <summary>Builds and weight-loads a Qwen2-MoE decoder.</summary>
    public static NeuralNetwork<T> Build(HuggingFaceConfig config, INamedTensorSource weights, int warmupSequenceLength = 8)
    {
        Guard.NotNull(config);
        Guard.NotNull(weights);
        if (warmupSequenceLength < 1) throw new ArgumentOutOfRangeException(nameof(warmupSequenceLength));
        if (config.NumLocalExperts <= 0 || config.NumExpertsPerTok <= 0)
            throw new InvalidDataException("Qwen2-MoE requires num_experts and num_experts_per_tok.");

        int hidden = config.HiddenSize;
        int numHeads = config.NumAttentionHeads;
        int numKVHeads = config.NumKeyValueHeads;
        int headDim = config.HeadDim;
        bool explicitHeadDim = headDim != hidden / numHeads;
        int vocab = config.VocabSize;
        int maxPos = config.MaxPositionEmbeddings;
        int numExperts = config.NumLocalExperts;
        int topK = config.NumExpertsPerTok;
        int routedFfn = config.MoeIntermediateSize > 0 ? config.MoeIntermediateSize : config.IntermediateSize;
        int sharedFfn = config.SharedExpertIntermediateSize; // 0 => no shared expert

        var embedding = new EmbeddingLayer<T>(vocab, hidden);
        var layers = new List<ILayer<T>> { embedding };

        var blocks = new MoEDecoderBlock<T>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var attention = new GroupedQueryAttentionLayer<T>(
                sequenceLength: maxPos, embeddingDimension: hidden, numHeads: numHeads, numKVHeads: numKVHeads,
                headDimension: explicitHeadDim ? headDim : null, useProjectionBias: true, useCausalMask: true); // Qwen2 q/k/v biases
            attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, config.RopeTheta, maxPos);

            var moe = new MoEFeedForwardLayer<T>(hidden, routedFfn, numExperts, topK, new SiLUActivation<T>(), sharedFfnDim: sharedFfn);
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
            LlamaModelBuilder<T>.LoadAttentionBiased((GroupedQueryAttentionLayer<T>)block.AttentionLayer, weights, p, numHeads, numKVHeads, headDim, hidden);

            var moe = block.Moe;
            LlamaModelBuilder<T>.LoadDense(moe.Router, weights, p + "mlp.gate.weight", outDim: numExperts, inDim: hidden);
            for (int e = 0; e < numExperts; e++)
            {
                string ep = p + $"mlp.experts.{e}.";
                LlamaModelBuilder<T>.LoadDense(moe.ExpertGate(e), weights, ep + "gate_proj.weight", outDim: routedFfn, inDim: hidden);
                LlamaModelBuilder<T>.LoadDense(moe.ExpertUp(e), weights, ep + "up_proj.weight", outDim: routedFfn, inDim: hidden);
                LlamaModelBuilder<T>.LoadDense(moe.ExpertDown(e), weights, ep + "down_proj.weight", outDim: hidden, inDim: routedFfn);
            }

            if (moe.SharedGate is { } sGate && moe.SharedUp is { } sUp && moe.SharedDown is { } sDown && moe.SharedGateLogit is { } sGateLogit)
            {
                LlamaModelBuilder<T>.LoadDense(sGate, weights, p + "mlp.shared_expert.gate_proj.weight", outDim: sharedFfn, inDim: hidden);
                LlamaModelBuilder<T>.LoadDense(sUp, weights, p + "mlp.shared_expert.up_proj.weight", outDim: sharedFfn, inDim: hidden);
                LlamaModelBuilder<T>.LoadDense(sDown, weights, p + "mlp.shared_expert.down_proj.weight", outDim: hidden, inDim: sharedFfn);
                LlamaModelBuilder<T>.LoadDense(sGateLogit, weights, p + "mlp.shared_expert_gate.weight", outDim: 1, inDim: hidden);
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
