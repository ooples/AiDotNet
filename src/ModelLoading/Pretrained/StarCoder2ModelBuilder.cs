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
/// Reconstructs a StarCoder2 decoder from a Hugging Face <see cref="HuggingFaceConfig"/> and loads its
/// weights. StarCoder2 differs from the dense LLaMA stack by using true LayerNorm <em>with bias</em>,
/// biased attention projections, and a non-gated two-matrix GELU MLP (<c>c_fc</c> → GELU → <c>c_proj</c>)
/// with biases.
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
public static class StarCoder2ModelBuilder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The architecture family names this builder recognizes.</summary>
    public static IReadOnlyList<string> SupportedArchitectures { get; } = new[]
    {
        "Starcoder2ForCausalLM", "starcoder2",
    };

    /// <summary>Builds and weight-loads a StarCoder2 decoder.</summary>
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
        double normEps = config.RmsNormEps; // config's norm_epsilon lands in this field slot

        var embedding = new EmbeddingLayer<T>(vocab, hidden);
        var layers = new List<ILayer<T>> { embedding };

        var blocks = new StarCoder2DecoderBlock<T>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var attention = new GroupedQueryAttentionLayer<T>(
                sequenceLength: maxPos, embeddingDimension: hidden, numHeads: numHeads, numKVHeads: numKVHeads,
                headDimension: explicitHeadDim ? headDim : null, useProjectionBias: true, useCausalMask: true);
            attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, config.RopeTheta, maxPos);

            var block = new StarCoder2DecoderBlock<T>(hidden, intermediate, attention, normEps);
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

        // ---- load pretrained weights (biased) ----
        embedding.SetParameters(new Vector<T>(
            LlamaModelBuilder<T>.ReadTensor(weights, LlamaModelBuilder<T>.EmbedName, vocab * hidden)));

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var block = blocks[i];
            string p = $"model.layers.{i}.";

            LoadLayerNorm(block.Norm1, weights, p + "input_layernorm", hidden);
            LlamaModelBuilder<T>.LoadAttentionBiased((GroupedQueryAttentionLayer<T>)block.AttentionLayer, weights, p, numHeads, numKVHeads, headDim, hidden);
            LoadLayerNorm(block.Norm2, weights, p + "post_attention_layernorm", hidden);
            LlamaModelBuilder<T>.LoadDenseBiased(block.CFc, weights, p + "mlp.c_fc.weight", p + "mlp.c_fc.bias", outDim: intermediate, inDim: hidden);
            LlamaModelBuilder<T>.LoadDenseBiased(block.CProj, weights, p + "mlp.c_proj.weight", p + "mlp.c_proj.bias", outDim: hidden, inDim: intermediate);
        }

        LoadLayerNorm(finalNorm, weights, "model.norm", hidden);

        string headName = LlamaModelBuilder<T>.HasTensor(weights, LlamaModelBuilder<T>.LmHeadName)
            ? LlamaModelBuilder<T>.LmHeadName : LlamaModelBuilder<T>.EmbedName;
        if (!config.TieWordEmbeddings && !LlamaModelBuilder<T>.HasTensor(weights, LlamaModelBuilder<T>.LmHeadName))
            throw new InvalidDataException(
                "config does not tie word embeddings but lm_head.weight is absent from the checkpoint.");
        LlamaModelBuilder<T>.LoadDense(lmHead, weights, headName, outDim: vocab, inDim: hidden);

        return network;
    }

    // Writes a LayerNorm's weight (gamma) and bias (beta) into its live tensors.
    private static void LoadLayerNorm(LayerNormalizationLayer<T> norm, INamedTensorSource weights, string prefix, int hidden)
    {
        var gamma = LlamaModelBuilder<T>.ReadTensor(weights, prefix + ".weight", hidden);
        var beta = LlamaModelBuilder<T>.ReadTensor(weights, prefix + ".bias", hidden);
        var gSpan = norm.GetGammaTensor().AsWritableSpan();
        var bSpan = norm.GetBetaTensor().AsWritableSpan();
        for (int i = 0; i < hidden; i++) { gSpan[i] = gamma[i]; bSpan[i] = beta[i]; }
    }
}
