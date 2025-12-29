using AiDotNet.Audio.Features;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Base class for speaker recognition models (embedding extraction, verification, diarization).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Speaker recognition encompasses tasks that identify or verify speakers based on their voice.
/// This base class provides common functionality for:
/// - Speaker embedding extraction (d-vectors, x-vectors)
/// - Speaker verification (is this the claimed speaker?)
/// - Speaker diarization (who spoke when?)
/// </para>
/// <para>
/// <b>For Beginners:</b> Speaker recognition is like voice fingerprinting.
/// Just as fingerprints are unique to each person, voice characteristics (pitch,
/// speaking style, accent) can identify individuals.
///
/// This base class provides:
/// - Feature extraction utilities (MFCCs, spectral features)
/// - Embedding dimension management
/// - Similarity computation methods
/// </para>
/// </remarks>
public abstract class SpeakerRecognitionBase<T> : AudioNeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the dimension of output speaker embeddings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values: 192, 256, or 512. Higher dimensions may capture more nuance
    /// but require more storage and computation.
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; protected set; } = 256;

    /// <summary>
    /// Gets the MFCC extractor for preprocessing.
    /// </summary>
    protected MfccExtractor<T>? MfccExtractor { get; set; }

    /// <summary>
    /// Initializes a new instance of the SpeakerRecognitionBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, a default MSE loss is used.</param>
    protected SpeakerRecognitionBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction)
    {
        // Default sample rate for speaker recognition
        SampleRate = 16000;
    }

    /// <summary>
    /// Computes cosine similarity between two speaker embeddings.
    /// </summary>
    /// <param name="embedding1">First speaker embedding vector.</param>
    /// <param name="embedding2">Second speaker embedding vector.</param>
    /// <returns>Cosine similarity score between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cosine similarity measures how similar two embeddings are.
    /// - Score close to 1.0: Very similar (likely same speaker)
    /// - Score close to 0.0: Not similar
    /// - Score close to -1.0: Opposite (very different)
    /// </para>
    /// </remarks>
    protected T ComputeCosineSimilarity(Vector<T> embedding1, Vector<T> embedding2)
    {
        if (embedding1.Length != embedding2.Length)
        {
            throw new ArgumentException("Embeddings must have the same dimension.");
        }

        T dotProduct = NumOps.Zero;
        T norm1 = NumOps.Zero;
        T norm2 = NumOps.Zero;

        for (int i = 0; i < embedding1.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(embedding1[i], embedding2[i]));
            norm1 = NumOps.Add(norm1, NumOps.Multiply(embedding1[i], embedding1[i]));
            norm2 = NumOps.Add(norm2, NumOps.Multiply(embedding2[i], embedding2[i]));
        }

        T normProduct = NumOps.Multiply(NumOps.Sqrt(norm1), NumOps.Sqrt(norm2));

        if (NumOps.Equals(normProduct, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        return NumOps.Divide(dotProduct, normProduct);
    }

    /// <summary>
    /// Computes cosine similarity between two speaker embedding tensors.
    /// </summary>
    /// <param name="embedding1">First speaker embedding tensor.</param>
    /// <param name="embedding2">Second speaker embedding tensor.</param>
    /// <returns>Cosine similarity score.</returns>
    protected T ComputeCosineSimilarity(Tensor<T> embedding1, Tensor<T> embedding2)
    {
        var vec1 = embedding1.ToVector();
        var vec2 = embedding2.ToVector();
        return ComputeCosineSimilarity(vec1, vec2);
    }

    /// <summary>
    /// Normalizes an embedding to unit length (L2 normalization).
    /// </summary>
    /// <param name="embedding">The embedding to normalize.</param>
    /// <returns>Normalized embedding with unit length.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Normalizing embeddings makes them easier to compare.
    /// After normalization, all embeddings have length 1, so cosine similarity
    /// becomes equivalent to a simple dot product.
    /// </para>
    /// </remarks>
    protected Tensor<T> NormalizeEmbedding(Tensor<T> embedding)
    {
        T sumSquares = NumOps.Zero;
        var data = embedding.ToArray();

        for (int i = 0; i < data.Length; i++)
        {
            sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(data[i], data[i]));
        }

        T norm = NumOps.Sqrt(sumSquares);

        if (NumOps.Equals(norm, NumOps.Zero))
        {
            return embedding;
        }

        // Create new tensor with same shape and copy normalized values
        var result = new Tensor<T>(embedding.Shape);
        for (int i = 0; i < data.Length; i++)
        {
            result[i] = NumOps.Divide(data[i], norm);
        }

        return result;
    }

    /// <summary>
    /// Aggregates multiple embeddings into a single representative embedding.
    /// </summary>
    /// <param name="embeddings">Collection of embeddings to aggregate.</param>
    /// <returns>Aggregated embedding (normalized mean).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you have multiple recordings of the same person,
    /// this combines them into one stronger voiceprint by averaging and normalizing.
    /// </para>
    /// </remarks>
    protected Tensor<T> AggregateEmbeddings(IReadOnlyList<Tensor<T>> embeddings)
    {
        if (embeddings.Count == 0)
        {
            throw new ArgumentException("At least one embedding is required.");
        }

        if (embeddings.Count == 1)
        {
            return NormalizeEmbedding(embeddings[0]);
        }

        var firstShape = embeddings[0].Shape;
        int totalSize = embeddings[0].Length;
        T[] sum = new T[totalSize];

        // Initialize with zeros
        for (int i = 0; i < totalSize; i++)
        {
            sum[i] = NumOps.Zero;
        }

        // Sum all embeddings
        foreach (var embedding in embeddings)
        {
            var data = embedding.ToArray();
            for (int i = 0; i < totalSize; i++)
            {
                sum[i] = NumOps.Add(sum[i], data[i]);
            }
        }

        // Compute mean
        T count = NumOps.FromDouble(embeddings.Count);
        for (int i = 0; i < totalSize; i++)
        {
            sum[i] = NumOps.Divide(sum[i], count);
        }

        // Create tensor and copy mean values
        var meanTensor = new Tensor<T>(firstShape);
        for (int i = 0; i < totalSize; i++)
        {
            meanTensor[i] = sum[i];
        }

        return NormalizeEmbedding(meanTensor);
    }

    /// <summary>
    /// Creates an MFCC extractor for preprocessing speaker audio.
    /// </summary>
    /// <param name="sampleRate">Sample rate of input audio.</param>
    /// <param name="numCoeffs">Number of MFCC coefficients.</param>
    /// <returns>A configured MFCC extractor.</returns>
    protected MfccExtractor<T> CreateMfccExtractor(int sampleRate = 16000, int numCoeffs = 40)
    {
        var options = new MfccOptions
        {
            SampleRate = sampleRate,
            NumCoefficients = numCoeffs
        };
        return new MfccExtractor<T>(options);
    }
}
