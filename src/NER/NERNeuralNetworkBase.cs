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
/// while maintaining full integration with the AiDotNet neural network infrastructure. It serves
/// as the domain-level base class for all NER models, analogous to how
/// <c>VideoNeuralNetworkBase&lt;T&gt;</c> serves video models and
/// <c>AudioNeuralNetworkBase&lt;T&gt;</c> serves audio models.
///
/// NER (Named Entity Recognition) is a sequence labeling task where the model assigns a label
/// to each token in a text sequence. The labels identify whether each token is part of a named
/// entity (person, organization, location, etc.) or not. This is distinct from text classification
/// (which assigns a single label to an entire document) and from relation extraction (which
/// identifies relationships between entities).
///
/// This base class provides:
/// - Dual-mode support: ONNX inference for deployment and native training for model development
/// - Token embedding preprocessing utilities (normalization, batch handling)
/// - Sequence extraction helpers for batched input
/// - Common properties for embedding dimensions, label counts, and sequence lengths
/// </para>
/// <para>
/// <b>For Beginners:</b> NER neural networks read text and identify important entities like
/// people's names, company names, and places. This base class provides the shared foundation
/// that all NER models build upon.
///
/// You can use derived NER models in two ways:
/// 1. <b>ONNX mode:</b> Load a pre-trained model for fast inference (identifying entities in text)
/// 2. <b>Native mode:</b> Build and train a new model from scratch on your own labeled data
///
/// The model processes text as numerical vectors called "embeddings" - each word is represented
/// by a list of numbers that capture its meaning. Common embedding sources include GloVe
/// (100-300 dimensions), Word2Vec (300 dimensions), and BERT (768 dimensions).
/// </para>
/// </remarks>
public abstract class NERNeuralNetworkBase<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the number of entity label classes this model predicts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This represents the total number of labels in the BIO (or BIOES) tagging scheme.
    /// For the standard CoNLL-2003 dataset with 4 entity types (PER, ORG, LOC, MISC),
    /// there are 9 labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC.
    ///
    /// The formula for BIO scheme is: numLabels = 2 * numEntityTypes + 1 (the +1 is for the O label).
    /// For BIOES scheme: numLabels = 4 * numEntityTypes + 1 (B, I, O, E, S prefixes).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how many different labels the model can assign to each word.
    /// With 9 labels, the model can identify 4 types of entities (person, organization, location,
    /// miscellaneous) and also mark words that aren't part of any entity.
    /// </para>
    /// </remarks>
    public int NumLabels { get; protected set; } = 9;

    /// <summary>
    /// Gets or sets the embedding dimension for input token representations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Token embeddings are dense vector representations of words. Each word is mapped to a
    /// fixed-size vector that captures semantic and syntactic information. The embedding dimension
    /// must match the pre-trained word vectors being used as input.
    ///
    /// Common values:
    /// - <b>100:</b> GloVe-100d - compact, good for BiLSTM-CRF baseline (used in Lample et al., 2016)
    /// - <b>300:</b> GloVe-300d or Word2Vec - richer representations, standard for NER research
    /// - <b>768:</b> BERT-base hidden states - contextual embeddings from a pre-trained transformer
    /// - <b>1024:</b> BERT-large or RoBERTa-large hidden states
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Words need to be converted to numbers before a neural network can
    /// process them. An "embedding" is a list of numbers that represents a word's meaning.
    /// Words with similar meanings have similar embeddings. The embedding dimension is how many
    /// numbers are in each word's vector. A dimension of 100 means each word is represented by
    /// 100 numbers. Larger dimensions capture more information but require more computation.
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; protected set; } = 100;

    /// <summary>
    /// Gets or sets the maximum sequence length this model supports.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Sentences longer than this will be truncated to fit. Shorter sentences are typically
    /// padded with zero vectors. The CRF layer's sequence dimension is fixed to this value.
    ///
    /// Common values: 128 (for short texts), 256 (standard), 512 (for long documents).
    /// Most NER datasets have sentences under 50 tokens, so 256 provides ample headroom.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the maximum number of words the model can process in
    /// a single sentence. If you have a sentence with 300 words but the max is 256, the
    /// last 44 words will be cut off. For most use cases, 256 is more than enough since
    /// typical sentences are 10-50 words long.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength { get; protected set; } = 256;

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the model uses a pre-trained ONNX model for fast inference. ONNX (Open Neural
    /// Network Exchange) is an industry-standard format for neural network models that enables
    /// hardware-accelerated inference. In ONNX mode, training is not supported.
    ///
    /// When false, the model uses native C# layers and supports both training and inference.
    /// Native mode is slower for inference but allows you to train on your own data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> ONNX mode is like using a finished, pre-built model - it's fast
    /// but you can't modify it. Native mode is like having the blueprints - it's slower but
    /// you can train and customize the model to your needs.
    /// </para>
    /// </remarks>
    public bool IsOnnxMode => OnnxEncoder is not null || OnnxDecoder is not null || OnnxModel is not null;

    /// <summary>
    /// Gets or sets the ONNX encoder model for encoder-decoder NER architectures.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Some NER models (like BERT-NER) use an encoder-decoder architecture where the encoder
    /// produces contextualized token representations and the decoder produces label predictions.
    /// This property holds the encoder portion when using ONNX inference.
    /// </para>
    /// </remarks>
    protected OnnxModel<T>? OnnxEncoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX decoder model for encoder-decoder NER architectures.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Holds the decoder portion (typically the CRF or classification head) when using ONNX
    /// inference with encoder-decoder architectures.
    /// </para>
    /// </remarks>
    protected OnnxModel<T>? OnnxDecoder { get; set; }

    /// <summary>
    /// Gets or sets the single ONNX model for end-to-end NER architectures.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For models that are exported as a single ONNX graph (e.g., a complete BiLSTM-CRF model),
    /// this holds the entire model. Input is token embeddings, output is label predictions.
    /// </para>
    /// </remarks>
    protected OnnxModel<T>? OnnxModel { get; set; }

    /// <summary>
    /// Initializes a new instance of the NERNeuralNetworkBase class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="lossFunction">The loss function for training. If null, cross-entropy loss is used,
    /// which is the standard loss for classification tasks including NER.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping, which prevents
    /// exploding gradients during training. Default is 1.0, following standard NER practice.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The architecture defines the model's structure (how many inputs
    /// and outputs). The loss function measures how wrong the model's predictions are during
    /// training. Gradient clipping prevents the model from making wild, unstable updates
    /// during training.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// In ONNX mode, training is not supported because the model weights are frozen in the
    /// exported format. In native mode, all layers have trainable parameters that are updated
    /// during backpropagation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> You can only train the model when it's in native mode (created
    /// without an ONNX model path). If you loaded a pre-trained ONNX model, you can only
    /// use it for predictions, not for further training.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => !IsOnnxMode;

    /// <summary>
    /// Preprocesses raw token embeddings for model input.
    /// </summary>
    /// <param name="rawEmbeddings">Raw token embeddings tensor with shape
    /// [sequenceLength, embeddingDim] or [batch, sequenceLength, embeddingDim].</param>
    /// <returns>Preprocessed embeddings suitable for model input. The preprocessing may include
    /// normalization, padding, or format conversion depending on the model requirements.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different NER models may expect their input in slightly different
    /// formats. This method converts your raw word embeddings into whatever format the specific
    /// model needs. For most models, it simply passes the input through unchanged.
    /// </para>
    /// </remarks>
    protected abstract Tensor<T> PreprocessTokens(Tensor<T> rawEmbeddings);

    /// <summary>
    /// Postprocesses model output into label predictions.
    /// </summary>
    /// <param name="modelOutput">Raw output from the model's forward pass.</param>
    /// <returns>Postprocessed output containing label predictions or scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model's raw output may be in a format that needs conversion
    /// to get actual label predictions. For example, the model might output probability scores
    /// for each label, and this method picks the most likely label for each word.
    /// </para>
    /// </remarks>
    protected abstract Tensor<T> PostprocessOutput(Tensor<T> modelOutput);

    /// <summary>
    /// Runs inference using the loaded ONNX model(s).
    /// </summary>
    /// <param name="input">Preprocessed input tensor containing token embeddings.</param>
    /// <returns>Model output tensor containing label scores or predictions.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no ONNX model is loaded.</exception>
    /// <remarks>
    /// <para>
    /// This method supports three ONNX configurations:
    /// 1. Single model (OnnxModel): The entire NER pipeline in one ONNX graph
    /// 2. Encoder only (OnnxEncoder): Just the feature extraction (e.g., BERT encoder)
    /// 3. Encoder + Decoder: Separate ONNX models for feature extraction and classification
    ///
    /// Override this method in derived classes to implement model-specific ONNX inference logic.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When using a pre-trained ONNX model, this method feeds your input
    /// through the model and returns the predictions. The ONNX runtime handles all the
    /// computation efficiently, potentially using GPU acceleration.
    /// </para>
    /// </remarks>
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
    /// <param name="input">Preprocessed input tensor containing token embeddings.</param>
    /// <returns>Model output tensor after processing through all layers.</returns>
    /// <remarks>
    /// <para>
    /// In native mode, this passes the input sequentially through each layer in the network:
    /// typically LSTM layers, dropout, a linear projection, and optionally a CRF layer.
    /// Each layer transforms the tensor, with the final output being either emission scores
    /// (if no CRF) or decoded label indices (if CRF is present).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This sends your input through each layer of the neural network
    /// in order, like an assembly line. Each layer processes the data and passes it to the
    /// next layer. The final output contains the model's predictions.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// For NER models, the default loss function is cross-entropy loss, which measures the
    /// difference between predicted label probabilities and the true labels. When a CRF layer
    /// is used, the CRF's own negative log-likelihood loss takes precedence during training.
    /// </para>
    /// </remarks>
    public override ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Normalizes token embeddings to unit length using L2 normalization.
    /// </summary>
    /// <param name="embeddings">Token embeddings tensor with shape [sequenceLength, embeddingDim]
    /// or [batch, sequenceLength, embeddingDim].</param>
    /// <returns>Normalized embeddings where each token's embedding vector has unit L2 norm.</returns>
    /// <remarks>
    /// <para>
    /// L2 normalization scales each embedding vector so that its Euclidean length equals 1.
    /// This can improve training stability by ensuring all embeddings have the same magnitude,
    /// preventing tokens with larger embedding norms from dominating the model's attention.
    ///
    /// Note: This is optional and not used by default in BiLSTM-CRF. It's provided as a
    /// utility for models that benefit from normalized inputs (e.g., some transformer-based models).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Some word embeddings have different "sizes" (magnitudes), which
    /// can confuse the model. Normalization makes all embeddings the same size while preserving
    /// their direction (which captures the word's meaning). Think of it like adjusting the volume
    /// so all words are equally loud, but still sound different from each other.
    /// </para>
    /// </remarks>
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
    /// Extracts a single sentence's embeddings from a batched tensor.
    /// </summary>
    /// <param name="batch">Batched tensor with shape [batch, sequenceLength, embeddingDim].</param>
    /// <param name="index">Zero-based index of the sentence to extract.</param>
    /// <returns>Single sentence tensor with shape [sequenceLength, embeddingDim].</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is outside the valid range.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When processing multiple sentences at once (a "batch"), this method
    /// lets you pull out one specific sentence by its position in the batch. This is useful
    /// when you need to examine or process individual sentences after batched inference.
    /// </para>
    /// </remarks>
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
    /// Disposes of resources used by this model, including any loaded ONNX models.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources; false if called from finalizer.</param>
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
