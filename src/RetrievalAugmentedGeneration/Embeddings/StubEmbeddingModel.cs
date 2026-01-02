using System.Security.Cryptography;
using System.Text;
using AiDotNet.Extensions;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.Embeddings;

/// <summary>
/// A deterministic stub embedding model for testing and development that uses hash-based vector generation.
/// </summary>
/// <remarks>
/// <para>
/// This implementation creates deterministic vector embeddings based on text hashing.
/// It's designed for testing, development, and prototyping RAG systems before real embedding
/// models are available. The same input text always produces the same embedding vector,
/// making it suitable for unit tests and demonstrations.
/// </para>
/// <para><b>For Beginners:</b> This is a simple placeholder until real embedding models are ready.
/// 
/// Think of it like using a stick figure placeholder in a drawing:
/// - It's not a real embedding model (like BERT or GPT)
/// - It converts text to numbers in a simple, predictable way (using hashing)
/// - Same text always gets same numbers (deterministic)
/// - Good enough for testing your RAG pipeline structure
/// - Replace it with a real embedding model for production
/// 
/// For example:
/// - "hello" always becomes the same vector
/// - "hello" and "world" get different vectors
/// - But similarity isn't semantically meaningful (unlike real embeddings)
/// 
/// This enables development work on Issue #284 without waiting for Issue #12.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector calculations (typically float or double).</typeparam>
public class StubEmbeddingModel<T> : EmbeddingModelBase<T>
{
    private readonly int _embeddingDimension;
    private readonly int _maxTokens;

    /// <summary>
    /// Gets the dimensionality of the embedding vectors produced by this model.
    /// </summary>
    public override int EmbeddingDimension => _embeddingDimension;

    /// <summary>
    /// Gets the maximum length of text (in tokens) that this model can process.
    /// </summary>
    public override int MaxTokens => _maxTokens;

    /// <summary>
    /// Initializes a new instance of the StubEmbeddingModel class.
    /// </summary>
    /// <param name="embeddingDimension">The dimensionality of the embeddings (default: 768, matching common models like BERT).</param>
    /// <param name="maxTokens">The maximum token length (default: 512).</param>
    public StubEmbeddingModel(int embeddingDimension = 768, int maxTokens = 512)
    {
        if (embeddingDimension <= 0)
            throw new ArgumentException("Embedding dimension must be greater than zero", nameof(embeddingDimension));

        if (maxTokens <= 0)
            throw new ArgumentException("MaxTokens must be greater than zero", nameof(maxTokens));

        _embeddingDimension = embeddingDimension;
        _maxTokens = maxTokens;
    }

    /// <summary>
    /// Core embedding logic that generates a deterministic vector from text using hashing.
    /// </summary>
    /// <param name="text">The validated text to embed.</param>
    /// <returns>A normalized vector representing the text.</returns>
    protected override Vector<T> EmbedCore(string text)
    {
        // Use SHA256 to generate deterministic hash from text
        using var sha256 = SHA256.Create();
        var textBytes = Encoding.UTF8.GetBytes(text);
        var hashBytes = sha256.ComputeHash(textBytes);

        // Generate vector values from hash
        var values = new T[_embeddingDimension];
        var random = RandomHelper.CreateSeededRandom(BitConverter.ToInt32(hashBytes, 0));

        // Generate values with normal distribution (mean=0, stddev=1)
        for (int i = 0; i < _embeddingDimension; i++)
        {
            values[i] = (T)Convert.ChangeType(random.NextGaussian(), typeof(T));
        }

        var vector = new Vector<T>(values);

        // Normalize the vector to unit length for cosine similarity
        return NormalizeVector(vector);
    }

    /// <summary>
    /// Normalizes a vector to unit length.
    /// </summary>
    /// <param name="vector">The vector to normalize.</param>
    /// <returns>The normalized vector.</returns>
    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        double magnitude = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            var value = Convert.ToDouble(vector[i]);
            magnitude += value * value;
        }
        magnitude = Math.Sqrt(magnitude);

        const double epsilon = 1e-8;
        if (Math.Abs(magnitude) < epsilon)
            return vector;

        var normalized = new T[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            var value = Convert.ToDouble(vector[i]);
            normalized[i] = (T)Convert.ChangeType(value / magnitude, typeof(T));
        }

        return new Vector<T>(normalized);
    }
}
