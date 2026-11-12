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
    /// Only the common prefix of the two vectors is compared (the shorter length). The result stays in
    /// <typeparamref name="T"/> until the final scalar; a zero (or below 1e-10) norm product returns zero.
    /// </remarks>
    public double CosineSimilarity(SpeakerEmbedding<T> other)
    {
        int len = Math.Min(Vector.Length, other.Vector.Length);
        var a = AiDotNet.Tensors.LinearAlgebra.Vector<T>.Wrap(Vector, 0, len);
        var b = AiDotNet.Tensors.LinearAlgebra.Vector<T>.Wrap(other.Vector, 0, len);
        var engine = AiDotNetEngine.Current;

        T dot = engine.DotProduct(a, b);
        T denominator = NumOps.Sqrt(NumOps.Multiply(engine.DotProduct(a, a), engine.DotProduct(b, b)));
        if (NumOps.Equals(denominator, NumOps.Zero) || NumOps.LessThan(denominator, NumOps.FromDouble(1e-10)))
        {
            return 0.0;
        }

        return NumOps.ToDouble(NumOps.Divide(dot, denominator));
    }

    /// <summary>
    /// Computes Euclidean distance from another embedding.
    /// </summary>
    /// <remarks>Only the common prefix of the two vectors is compared (the shorter length).</remarks>
    public double EuclideanDistance(SpeakerEmbedding<T> other)
    {
        int len = Math.Min(Vector.Length, other.Vector.Length);
        var engine = AiDotNetEngine.Current;
        var diff = engine.Subtract(AiDotNet.Tensors.LinearAlgebra.Vector<T>.Wrap(Vector, 0, len), AiDotNet.Tensors.LinearAlgebra.Vector<T>.Wrap(other.Vector, 0, len));
        return NumOps.ToDouble(NumOps.Sqrt(engine.DotProduct(diff, diff)));
    }
}
