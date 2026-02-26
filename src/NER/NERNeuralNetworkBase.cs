using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;

namespace AiDotNet.NER;

/// <summary>
/// Base class for NER-focused neural networks that can operate in both ONNX inference and native training modes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class extends <see cref="NeuralNetworkBase{T}"/> to provide NER-specific functionality
/// while maintaining full integration with the AiDotNet neural network infrastructure.
/// </para>
/// <para>
/// <b>For Beginners:</b> NER neural networks process sequences of tokens (words or subwords) to
/// identify and classify named entities. This base class provides:
///
/// - Support for pre-trained ONNX models (fast inference with existing models)
/// - Full training capability from scratch (like other neural networks)
/// - Token embedding preprocessing utilities
/// - BIO/BIOES label scheme handling
///
/// You can use derived classes in two ways:
/// 1. Load a pre-trained ONNX model for quick inference
/// 2. Build and train a new model from scratch
/// </para>
/// </remarks>
public abstract class NERNeuralNetworkBase<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the number of entity label classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default is 9 for CoNLL-2003 BIO scheme:
    /// O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
    /// </para>
    /// </remarks>
    public int NumLabels { get; protected set; } = 9;

    /// <summary>
    /// Gets or sets the embedding dimension for input token representations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values: 100 (GloVe), 300 (Word2Vec/GloVe), 768 (BERT base), 1024 (BERT large).
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; protected set; } = 100;

    /// <summary>
    /// Gets or sets the maximum sequence length this model supports.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Sequences longer than this will be truncated. Common values: 128, 256, 512.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength { get; protected set; } = 256;

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    public bool IsOnnxMode => OnnxEncoder is not null || OnnxDecoder is not null || OnnxModel is not null;

    /// <summary>
    /// Gets or sets the ONNX encoder model (for encoder-decoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxEncoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX decoder model (for encoder-decoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxDecoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX model (for single-model architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxModel { get; set; }

    /// <summary>
    /// Initializes a new instance of the NERNeuralNetworkBase class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, a default cross-entropy loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected NERNeuralNetworkBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => !IsOnnxMode;

    /// <summary>
    /// Preprocesses raw token embeddings for model input.
    /// </summary>
    /// <param name="rawEmbeddings">Raw token embeddings tensor [sequenceLength, embeddingDim] or [batch, sequenceLength, embeddingDim].</param>
    /// <returns>Preprocessed embeddings suitable for model input.</returns>
    protected abstract Tensor<T> PreprocessTokens(Tensor<T> rawEmbeddings);

    /// <summary>
    /// Postprocesses model output into label predictions.
    /// </summary>
    /// <param name="modelOutput">Raw output from the model.</param>
    /// <returns>Postprocessed output with label scores or predictions.</returns>
    protected abstract Tensor<T> PostprocessOutput(Tensor<T> modelOutput);

    /// <summary>
    /// Runs inference using ONNX model(s).
    /// </summary>
    /// <param name="input">Preprocessed input tensor.</param>
    /// <returns>Model output tensor.</returns>
    protected virtual Tensor<T> RunOnnxInference(Tensor<T> input)
    {
        if (OnnxModel is not null)
        {
            return OnnxModel.Run(input);
        }

        if (OnnxEncoder is not null)
        {
            var encoded = OnnxEncoder.Run(input);
            if (OnnxDecoder is not null)
            {
                return OnnxDecoder.Run(encoded);
            }
            return encoded;
        }

        throw new InvalidOperationException("No ONNX model is loaded.");
    }

    /// <summary>
    /// Performs a forward pass through the native neural network layers.
    /// </summary>
    /// <param name="input">Preprocessed input tensor.</param>
    /// <returns>Model output tensor.</returns>
    protected virtual Tensor<T> Forward(Tensor<T> input)
    {
        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    /// <summary>
    /// Gets the default loss function for this model.
    /// </summary>
    public override ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Normalizes token embeddings to unit length (L2 normalization).
    /// </summary>
    /// <param name="embeddings">Token embeddings tensor.</param>
    /// <returns>Normalized embeddings tensor.</returns>
    protected Tensor<T> NormalizeEmbeddings(Tensor<T> embeddings)
    {
        var normalized = new Tensor<T>(embeddings.Shape);

        int seqLen, embDim;
        int batchSize;
        if (embeddings.Rank == 3)
        {
            batchSize = embeddings.Shape[0];
            seqLen = embeddings.Shape[1];
            embDim = embeddings.Shape[2];
        }
        else
        {
            batchSize = 1;
            seqLen = embeddings.Shape[0];
            embDim = embeddings.Shape[1];
        }

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                double norm = 0;
                int baseIdx = embeddings.Rank == 3
                    ? b * seqLen * embDim + s * embDim
                    : s * embDim;

                for (int e = 0; e < embDim; e++)
                {
                    double val = NumOps.ToDouble(embeddings.Data.Span[baseIdx + e]);
                    norm += val * val;
                }

                norm = Math.Sqrt(norm);
                if (norm < 1e-12) norm = 1e-12;

                for (int e = 0; e < embDim; e++)
                {
                    double val = NumOps.ToDouble(embeddings.Data.Span[baseIdx + e]);
                    normalized.Data.Span[baseIdx + e] = NumOps.FromDouble(val / norm);
                }
            }
        }

        return normalized;
    }

    /// <summary>
    /// Extracts a single sequence from a batched tensor.
    /// </summary>
    /// <param name="batch">Batched tensor [batch, sequenceLength, embeddingDim].</param>
    /// <param name="index">Index of the sequence to extract.</param>
    /// <returns>Single sequence tensor [sequenceLength, embeddingDim].</returns>
    protected Tensor<T> ExtractSequence(Tensor<T> batch, int index)
    {
        if (index < 0 || index >= batch.Shape[0])
            throw new ArgumentOutOfRangeException(nameof(index), $"Sequence index {index} is out of range [0, {batch.Shape[0]}).");

        int seqLen = batch.Shape[1];
        int embDim = batch.Shape[2];
        var sequence = new Tensor<T>([seqLen, embDim]);
        int srcOffset = index * seqLen * embDim;

        for (int i = 0; i < seqLen * embDim; i++)
        {
            sequence.Data.Span[i] = batch.Data.Span[srcOffset + i];
        }

        return sequence;
    }

    /// <summary>
    /// Disposes of resources used by this model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxEncoder?.Dispose();
            OnnxDecoder?.Dispose();
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
