using System;

using AiDotNet.Enums;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Maps an AiDotNet <see cref="DistanceMetricType"/> onto the corresponding pgvector distance
/// operator and converts the raw operator distance into a higher-is-better similarity score.
/// Extracted from <see cref="PostgresVectorDocumentStore{T}"/> for unit testing.
/// </summary>
/// <remarks>
/// pgvector distance operators:
/// <list type="bullet">
///   <item><description><see cref="DistanceMetricType.Cosine"/> → <c>&lt;=&gt;</c> (cosine distance = 1 − cosine similarity).</description></item>
///   <item><description><see cref="DistanceMetricType.Euclidean"/> → <c>&lt;-&gt;</c> (L2 distance).</description></item>
///   <item><description><see cref="DistanceMetricType.Manhattan"/> → <c>&lt;+&gt;</c> (L1 / taxicab distance).</description></item>
/// </list>
/// pgvector also exposes <c>&lt;#&gt;</c> (negative inner product); there is no matching
/// <see cref="DistanceMetricType"/> member, so it is not surfaced here.
/// </remarks>
public static class PgVectorMetric
{
    /// <summary>Returns the pgvector distance operator for <paramref name="metric"/>.</summary>
    /// <exception cref="NotSupportedException">Thrown for metrics pgvector cannot express.</exception>
    public static string Operator(DistanceMetricType metric)
    {
        switch (metric)
        {
            case DistanceMetricType.Cosine:
                return "<=>";
            case DistanceMetricType.Euclidean:
                return "<->";
            case DistanceMetricType.Manhattan:
                return "<+>";
            default:
                throw new NotSupportedException(
                    $"Distance metric '{metric}' is not supported by pgvector. Use Cosine, Euclidean or Manhattan.");
        }
    }

    /// <summary>
    /// Converts a raw pgvector operator distance into a similarity score where higher means more similar.
    /// Cosine → <c>1 − distance</c>; Euclidean/Manhattan → <c>1 / (1 + distance)</c>.
    /// </summary>
    public static double ToSimilarity(DistanceMetricType metric, double distance)
    {
        switch (metric)
        {
            case DistanceMetricType.Cosine:
                return 1.0 - distance;
            case DistanceMetricType.Euclidean:
            case DistanceMetricType.Manhattan:
                return 1.0 / (1.0 + distance);
            default:
                throw new NotSupportedException(
                    $"Distance metric '{metric}' is not supported by pgvector.");
        }
    }
}
