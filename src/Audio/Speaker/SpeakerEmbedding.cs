using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Represents a speaker embedding vector.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class SpeakerEmbedding<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the embedding vector.
    /// </summary>
    public required T[] Vector { get; set; }

    /// <summary>
    /// Gets or sets the duration of the source audio in seconds.
    /// </summary>
    public double Duration { get; set; }

    /// <summary>
    /// Gets or sets the number of frames used.
    /// </summary>
    public int NumFrames { get; set; }

    /// <summary>
    /// Computes cosine similarity with another embedding.
    /// </summary>
    public double CosineSimilarity(SpeakerEmbedding<T> other)
    {
        double dot = 0;
        double norm1 = 0;
        double norm2 = 0;

        int len = Math.Min(Vector.Length, other.Vector.Length);
        for (int i = 0; i < len; i++)
        {
            double v1 = NumOps.ToDouble(Vector[i]);
            double v2 = NumOps.ToDouble(other.Vector[i]);
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        double denominator = Math.Sqrt(norm1 * norm2);
        if (denominator < 1e-10) return 0;

        return dot / denominator;
    }

    /// <summary>
    /// Computes Euclidean distance from another embedding.
    /// </summary>
    public double EuclideanDistance(SpeakerEmbedding<T> other)
    {
        double sumSquared = 0;
        int len = Math.Min(Vector.Length, other.Vector.Length);

        for (int i = 0; i < len; i++)
        {
            double diff = NumOps.ToDouble(Vector[i]) - NumOps.ToDouble(other.Vector[i]);
            sumSquared += diff * diff;
        }

        return Math.Sqrt(sumSquared);
    }
}
