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
public abstract partial class SpeakerRecognitionBase<T> : AudioNeuralNetworkBase<T>
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

        // The norms are reduced in double (TensorNorm), so squares above T's range (65504 in Half) do not
        // overflow, and the dot product is taken between the unit vectors.
        T norm1 = Engine.TensorNorm(embedding1.AsTensor()).GetFlat(0);
        T norm2 = Engine.TensorNorm(embedding2.AsTensor()).GetFlat(0);

        if (NumOps.Equals(norm1, NumOps.Zero) || NumOps.Equals(norm2, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        return Engine.DotProduct(Engine.Divide(embedding1, norm1), Engine.Divide(embedding2, norm2));
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
        var data = embedding.ToArray();

        // L2 norm reduced in double, so the sum of squares cannot overflow T.
        T norm = Engine.TensorNorm(new Tensor<T>(data, [data.Length])).GetFlat(0);

        if (NumOps.Equals(norm, NumOps.Zero))
        {
            // Return a defensive copy to avoid aliasing issues
            return new Tensor<T>(data, embedding._shape);
        }

        // New tensor with the same shape holding the normalized values
        return Engine.TensorDivideScalar(new Tensor<T>(data, embedding._shape), norm);
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

        var firstShape = embeddings[0]._shape;
        int totalSize = embeddings[0].Length;

        // Validate all embeddings have the same shape
        for (int e = 1; e < embeddings.Count; e++)
        {
            if (embeddings[e].Length != totalSize)
            {
                throw new ArgumentException(
                    $"All embeddings must have the same shape. Expected length {totalSize}, " +
                    $"got {embeddings[e].Length} at index {e}.");
            }
        }

        // Sum all embeddings (flattened, since only the lengths are required to match)
        var sum = new Tensor<T>(embeddings[0].ToArray(), [totalSize]);
        for (int e = 1; e < embeddings.Count; e++)
        {
            sum = Engine.TensorAdd(sum, new Tensor<T>(embeddings[e].ToArray(), [totalSize]));
        }

        // Compute mean and restore the first embedding's shape
        var mean = Engine.TensorDivideScalar(sum, NumOps.FromDouble(embeddings.Count));
        var meanTensor = new Tensor<T>(mean.ToArray(), firstShape);

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

    /// <summary>
    /// Creates the MFCC front end described by a speaker model's own options.
    /// </summary>
    /// <param name="options">The model's options.</param>
    /// <returns>An extractor configured to match the model.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is null.</exception>
    /// <remarks>
    /// <para>
    /// The overload above defaults the frame geometry independently of the model, so a model that
    /// changed its coefficient count got layers of one width fed by features of another. Taking
    /// the options directly keeps the front end and the network configured from one place.
    /// </para>
    /// </remarks>
    protected MfccExtractor<T> CreateMfccExtractor(SpeakerEmbeddingOptions options)
    {
        if (options is null)
            throw new ArgumentNullException(nameof(options));

        return new MfccExtractor<T>(new MfccOptions
        {
            SampleRate = options.SampleRate,
            NumCoefficients = options.NumMfcc,
            FftSize = options.FftSize,
            HopLength = options.HopLength
        });
    }
}
