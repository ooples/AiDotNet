using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Represents a speaker embedding vector.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SpeakerEmbedding provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
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
    /// <remarks>
    /// Only the common prefix of the two vectors is compared (the shorter length). Each vector is divided by its
    /// largest magnitude before it is squared, so the norms stay inside <typeparamref name="T"/>'s range (a norm
    /// above 65504 is infinite in Half, and dividing by it zeroes the score); a norm product below 1e-10 returns
    /// zero. The dot product is taken between the unit vectors, which keeps it inside [-1, 1].
    /// </remarks>
    public double CosineSimilarity(SpeakerEmbedding<T> other)
    {
        int len = Math.Min(Vector.Length, other.Vector.Length);
        var a = AiDotNet.Tensors.LinearAlgebra.Vector<T>.Wrap(Vector, 0, len).AsTensor();
        var b = AiDotNet.Tensors.LinearAlgebra.Vector<T>.Wrap(other.Vector, 0, len).AsTensor();

        T cosine = SpeakerScoring.Cosine(AiDotNetEngine.Current, a, b, out double denominator);
        if (!(denominator >= 1e-10))
        {
            return 0.0;
        }

        return NumOps.ToDouble(cosine);
    }

    /// <summary>
    /// Computes Euclidean distance from another embedding.
    /// </summary>
    /// <remarks>Only the common prefix of the two vectors is compared (the shorter length). The differences, their
    /// squares and the root are all taken in double, the type this method returns: in <typeparamref name="T"/> the
    /// difference itself can overflow (Half: 4e4 - (-4e4)), and so can a norm above 65504.</remarks>
    public double EuclideanDistance(SpeakerEmbedding<T> other)
    {
        int len = Math.Min(Vector.Length, other.Vector.Length);
        var engine = AiDotNetEngine.Current;
        var a = AiDotNet.Tensors.LinearAlgebra.Vector<T>.Wrap(Vector, 0, len).AsTensor().Cast<double>();
        var b = AiDotNet.Tensors.LinearAlgebra.Vector<T>.Wrap(other.Vector, 0, len).AsTensor().Cast<double>();
        return Math.Sqrt(engine.TensorSum(engine.TensorSquare(engine.TensorSubtract(a, b))));
    }
}
