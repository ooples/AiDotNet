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
/// Reconstructs a Cohere (Command-R) decoder from a Hugging Face <see cref="HuggingFaceConfig"/> and loads
/// its weights. Cohere differs from the dense LLaMA stack by using true (mean-centered) LayerNorm instead of
/// RMSNorm, a <em>parallel</em> residual (one norm feeds both attention and FFN), tied embeddings, and a
/// <c>logit_scale</c> multiplier on the output.
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
/// <remarks>
/// Bias-free (matching Command-R). The <c>logit_scale</c> is baked into the (tied) LM-head weights on load.
/// QK-normalization (some Command-R+ variants) is not applied.
/// </remarks>
public static class CohereModelBuilder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The architecture family names this builder recognizes.</summary>
    public static IReadOnlyList<string> SupportedArchitectures { get; } = new[]
    {
        "CohereForCausalLM", "cohere", "command-r",
    };

    /// <summary>Builds and weight-loads a Cohere decoder.</summary>
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
        double normEps = config.RmsNormEps; // Cohere config reuses layer_norm_eps in the rms field slot

        var embedding = new EmbeddingLayer<T>(vocab, hidden);
        var layers = new List<ILayer<T>> { embedding };

        var blocks = new CohereDecoderBlock<T>[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var attention = new GroupedQueryAttentionLayer<T>(
                sequenceLength: maxPos, embeddingDimension: hidden, numHeads: numHeads, numKVHeads: numKVHeads,
                headDimension: explicitHeadDim ? headDim : null);
            attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, config.RopeTheta, maxPos);

            var block = new CohereDecoderBlock<T>(hidden, intermediate, attention, normEps);
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

        // ---- load pretrained weights (bias-free) ----
        embedding.SetParameters(new Vector<T>(
            LlamaModelBuilder<T>.ReadTensor(weights, LlamaModelBuilder<T>.EmbedName, vocab * hidden)));

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            var block = blocks[i];
            string p = $"model.layers.{i}.";

            WriteGamma(block.Norm, LlamaModelBuilder<T>.ReadTensor(weights, p + "input_layernorm.weight", hidden));
            LlamaModelBuilder<T>.LoadAttention((GroupedQueryAttentionLayer<T>)block.AttentionLayer, weights, p, numHeads, numKVHeads, headDim, hidden);
            LlamaModelBuilder<T>.LoadDense(block.FfnGate, weights, p + "mlp.gate_proj.weight", outDim: intermediate, inDim: hidden);
            LlamaModelBuilder<T>.LoadDense(block.FfnUp, weights, p + "mlp.up_proj.weight", outDim: intermediate, inDim: hidden);
            LlamaModelBuilder<T>.LoadDense(block.FfnDown, weights, p + "mlp.down_proj.weight", outDim: hidden, inDim: intermediate);
        }

        WriteGamma(finalNorm, LlamaModelBuilder<T>.ReadTensor(weights, "model.norm.weight", hidden));

        // Tied, logit_scale-multiplied LM head: load embed transposed to [hidden, vocab] and scale.
        string headName = LlamaModelBuilder<T>.HasTensor(weights, LlamaModelBuilder<T>.LmHeadName)
            ? LlamaModelBuilder<T>.LmHeadName : LlamaModelBuilder<T>.EmbedName;
        double logitScale = config.LogitScale ?? 1.0;
        var headInOut = LlamaModelBuilder<T>.TransposeOutInToInOut(weights, headName, outDim: vocab, inDim: hidden);
        if (logitScale != 1.0)
        {
            var s = NumOps.FromDouble(logitScale);
            for (int k = 0; k < headInOut.Length; k++) headInOut[k] = NumOps.Multiply(headInOut[k], s);
        }
        var full = new T[headInOut.Length + vocab]; // + zero bias
        Array.Copy(headInOut, full, headInOut.Length);
        lmHead.SetParameters(new Vector<T>(full));

        return network;
    }

    // Writes a [featureSize] gamma vector into a LayerNorm layer's live gamma tensor (beta stays zero,
    // matching Cohere's bias-free norm).
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
