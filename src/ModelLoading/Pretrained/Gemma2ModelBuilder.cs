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
/// Reconstructs a Gemma-2 decoder from a Hugging Face <see cref="HuggingFaceConfig"/> and loads its weights.
/// Gemma-2 differs from the dense LLaMA stack by <em>sandwiched</em> RMSNorms (norm before AND after each
/// sublayer), GeGLU FFN, the Gemma <c>(1 + weight)</c> RMSNorm and √hidden embedding scale, and final-logit
/// soft-capping.
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
/// <remarks>
/// Applied faithfully: 4-norm sandwich blocks, (1+weight) norms, √hidden embedding scale, GeGLU, explicit
/// head_dim, and final-logit soft-capping (<c>cap·tanh(logits/cap)</c>). Not yet applied (numerical
/// refinements): attention-logit soft-capping and alternating sliding-window attention — within the trained
/// window the sliding mask equals the causal mask, so short-context generation is unaffected.
/// </remarks>
public static class Gemma2ModelBuilder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The architecture family names this builder recognizes.</summary>
    public static IReadOnlyList<string> SupportedArchitectures { get; } = new[]
    {
        "Gemma2ForCausalLM", "gemma2",
    };

    /// <summary>Builds and weight-loads a Gemma-2 decoder.</summary>
    public static NeuralNetwork<T> Build(HuggingFaceConfig config, INamedTensorSource weights, int warmupSequenceLength = 8)
    {
        Guard.NotNull(config);
        Guard.NotNull(weights);
        if (warmupSequenceLength < 1) throw new ArgumentOutOfRangeException(nameof(warmupSequenceLength));

        int hidden = config.HiddenSize;
        int numHeads = config.NumAttentionHeads;
        int numKVHeads = config.NumKeyValueHeads;
        int headDim = config.HeadDim;
        bool explicitHeadDim = headDim != hidden / numHeads;
        int intermediate = config.IntermediateSize;
        int vocab = config.VocabSize;
        int maxPos = config.MaxPositionEmbeddings;

        var embedding = new EmbeddingLayer<T>(vocab, hidden);
        var layers = new List<ILayer<T>> { embedding };

        var blocks = new Gemma2DecoderBlock<T>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var attention = new GroupedQueryAttentionLayer<T>(
                sequenceLength: maxPos, embeddingDimension: hidden, numHeads: numHeads, numKVHeads: numKVHeads,
                headDimension: explicitHeadDim ? headDim : null,
                attnLogitSoftcap: config.AttnLogitSoftcapping ?? 0.0);
            attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, config.RopeTheta, maxPos);

            var block = new Gemma2DecoderBlock<T>(hidden, intermediate, attention, config.RmsNormEps);
            blocks[i] = block;
            layers.Add(block);
        }

        var finalNorm = new RMSNormalizationLayer<T>(hidden, config.RmsNormEps);
        layers.Add(finalNorm);
        var lmHead = new DenseLayer<T>(vocab, activationFunction: new IdentityActivation<T>());
        layers.Add(lmHead);

        LogitSoftcapLayer<T>? softcap = null;
        if (config.FinalLogitSoftcapping is { } cap && cap > 0.0)
        {
            softcap = new LogitSoftcapLayer<T>(cap);
            layers.Add(softcap);
        }

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple, inputSize: 1, outputSize: vocab, layers: layers);
        var network = new NeuralNetwork<T>(architecture);

        var warmup = new Tensor<T>(new[] { 1, warmupSequenceLength });
        for (int p = 0; p < warmupSequenceLength; p++) warmup[0, p] = NumOps.Zero;
        network.Predict(warmup);

        // ---- load pretrained weights ----
        // Embedding x sqrt(hidden); a tied LM head keeps the UNSCALED embedding (read from source below).
        var embedData = LlamaModelBuilder<T>.ReadTensor(weights, LlamaModelBuilder<T>.EmbedName, vocab * hidden);
        var scale = NumOps.FromDouble(Math.Sqrt(hidden));
        var scaled = new T[embedData.Length];
        for (int k = 0; k < embedData.Length; k++) scaled[k] = NumOps.Multiply(embedData[k], scale);
        embedding.SetParameters(new Vector<T>(scaled));

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var block = blocks[i];
            string p = $"model.layers.{i}.";

            // Gemma (1 + weight) RMSNorms — the four sandwich norms.
            LlamaModelBuilder<T>.LoadGamma(block.NormInput, weights, p + "input_layernorm.weight", hidden, addOne: true);
            LlamaModelBuilder<T>.LoadGamma(block.NormPostAttn, weights, p + "post_attention_layernorm.weight", hidden, addOne: true);
            LlamaModelBuilder<T>.LoadGamma(block.NormPreFfn, weights, p + "pre_feedforward_layernorm.weight", hidden, addOne: true);
            LlamaModelBuilder<T>.LoadGamma(block.NormPostFfn, weights, p + "post_feedforward_layernorm.weight", hidden, addOne: true);

            LlamaModelBuilder<T>.LoadAttention((GroupedQueryAttentionLayer<T>)block.AttentionLayer, weights, p, numHeads, numKVHeads, headDim, hidden);

            LlamaModelBuilder<T>.LoadDense(block.FfnGate, weights, p + "mlp.gate_proj.weight", outDim: intermediate, inDim: hidden);
            LlamaModelBuilder<T>.LoadDense(block.FfnUp, weights, p + "mlp.up_proj.weight", outDim: intermediate, inDim: hidden);
            LlamaModelBuilder<T>.LoadDense(block.FfnDown, weights, p + "mlp.down_proj.weight", outDim: hidden, inDim: intermediate);
        }

        LlamaModelBuilder<T>.LoadGamma(finalNorm, weights, "model.norm.weight", hidden, addOne: true);

        string headName = LlamaModelBuilder<T>.HasTensor(weights, LlamaModelBuilder<T>.LmHeadName)
            ? LlamaModelBuilder<T>.LmHeadName : LlamaModelBuilder<T>.EmbedName;
        if (!config.TieWordEmbeddings && !LlamaModelBuilder<T>.HasTensor(weights, LlamaModelBuilder<T>.LmHeadName))
            throw new InvalidDataException(
                "config does not tie word embeddings but lm_head.weight is absent from the checkpoint.");
        LlamaModelBuilder<T>.LoadDense(lmHead, weights, headName, outDim: vocab, inDim: hidden);

        return network;
    }
}
