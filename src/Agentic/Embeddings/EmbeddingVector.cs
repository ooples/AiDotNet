using AiDotNet.Validation;

namespace AiDotNet.Agentic.Embeddings;

/// <summary>An immutable, finite-valued embedding vector with a cached magnitude.</summary>
/// <remarks>
/// <para>
/// The vector is copied on construction and never handed out as a mutable array, so a vector held in a cache cannot
/// be altered by whoever received it. Every component is validated as finite: a provider that returns a NaN or an
/// infinity is a broken response, and admitting one would silently poison every later cosine comparison with a NaN
/// that compares false against every threshold. <see cref="Magnitude"/> is computed once, which is what makes a
/// repeated similarity scan against a fixed set of neighbours cheap.
/// </para>
/// <para>
/// <see cref="CosineSimilarity"/> returns zero for a mismatched length or a zero-magnitude operand rather than
/// throwing, matching the reference implementation's guard so that a degenerate vector reads as "not similar to
/// anything" instead of aborting a search. It never returns NaN.
/// </para>
/// <para><b>For Beginners:</b> An embedding is a list of numbers that stands in for a piece of text, arranged so
/// that texts meaning similar things get similar lists. Cosine similarity measures the angle between two such
/// lists: 1.0 means they point the same way (very similar text), 0.0 means unrelated. This class holds one such
/// list, checks the numbers are sane, and does the similarity arithmetic for you.</para>
/// </remarks>
public sealed class EmbeddingVector
{
    private readonly double[] _components;

    /// <summary>Initializes an embedding vector from a sequence of finite components.</summary>
    /// <param name="components">The vector components; at least one is required.</param>
    /// <exception cref="ArgumentNullException"><paramref name="components"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="components"/> is empty or holds a non-finite value.</exception>
    public EmbeddingVector(IEnumerable<double> components)
    {
        Guard.NotNull(components);
        _components = components.ToArray();
        if (_components.Length == 0)
        {
            throw new ArgumentException("An embedding vector needs at least one component.", nameof(components));
        }

        double sumOfSquares = 0;
        foreach (double component in _components)
        {
            if (double.IsNaN(component) || double.IsInfinity(component))
            {
                throw new ArgumentException(
                    "An embedding vector cannot contain a NaN or an infinite component.", nameof(components));
            }

            sumOfSquares += component * component;
        }

        Magnitude = Math.Sqrt(sumOfSquares);
    }

    /// <summary>Gets the vector components in provider order.</summary>
    public IReadOnlyList<double> Components => _components;

    /// <summary>Gets the number of components.</summary>
    public int Dimensions => _components.Length;

    /// <summary>Gets the Euclidean length of the vector, computed once at construction.</summary>
    public double Magnitude { get; }

    /// <summary>Computes the cosine similarity of two vectors.</summary>
    /// <param name="first">The first vector.</param>
    /// <param name="second">The second vector.</param>
    /// <returns>
    /// A value in the range -1 to 1, or zero when the lengths differ or either vector has zero magnitude. Never NaN.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="first"/> or <paramref name="second"/> is <c>null</c>.</exception>
    public static double CosineSimilarity(EmbeddingVector first, EmbeddingVector second)
    {
        Guard.NotNull(first);
        Guard.NotNull(second);
        if (first._components.Length != second._components.Length) return 0.0;
        if (first.Magnitude == 0.0 || second.Magnitude == 0.0) return 0.0;

        double dot = 0;
        for (int index = 0; index < first._components.Length; index++)
        {
            dot += first._components[index] * second._components[index];
        }

        double similarity = dot / (first.Magnitude * second.Magnitude);
        return similarity < -1.0 ? -1.0 : similarity > 1.0 ? 1.0 : similarity;
    }

    /// <summary>Returns the dimension count only, never the components, so a log stays bounded.</summary>
    /// <returns>A short diagnostic label for this vector.</returns>
    public override string ToString() =>
        "embedding(" + Dimensions.ToString(System.Globalization.CultureInfo.InvariantCulture) + ")";
}
