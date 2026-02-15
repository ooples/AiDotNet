namespace AiDotNet.Clustering.Interfaces;

/// <summary>
/// Defines an interface for computing distance or similarity between vectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Distance metrics are fundamental to many clustering algorithms as they define
/// what it means for two points to be "similar" or "close together". Different
/// metrics are suitable for different types of data and applications.
/// </para>
/// <para><b>For Beginners:</b> A distance metric tells us how to measure
/// "how far apart" two data points are.
///
/// Common examples:
/// - Euclidean: Straight-line distance (what you'd measure with a ruler)
/// - Manhattan: City-block distance (sum of differences along each axis)
/// - Cosine: Angle between vectors (useful for text/documents)
///
/// The choice of distance metric can significantly affect clustering results.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("DistanceMetric")]
public interface IDistanceMetric<T>
{
    /// <summary>
    /// Gets the name of this distance metric.
    /// </summary>
    /// <value>A human-readable name like "Euclidean" or "Manhattan".</value>
    string Name { get; }

    /// <summary>
    /// Computes the distance between two vectors.
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The computed distance. Always non-negative for valid metrics.</returns>
    /// <exception cref="ArgumentException">Thrown if vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates how far apart two points are.
    ///
    /// For example, with Euclidean distance:
    /// - Points [0, 0] and [3, 4] have distance 5 (Pythagorean theorem)
    /// - Points [1, 1] and [1, 1] have distance 0 (same point)
    ///
    /// Different metrics will give different values for the same pair of points.
    /// </para>
    /// </remarks>
    T Compute(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes distances from a single point to all rows in a matrix.
    /// </summary>
    /// <param name="point">The reference point.</param>
    /// <param name="data">A matrix where each row is a data point.</param>
    /// <returns>A vector of distances from the point to each row in data.</returns>
    /// <remarks>
    /// <para>
    /// This batch operation can be optimized by implementations for better performance
    /// compared to calling Compute repeatedly.
    /// </para>
    /// </remarks>
    Vector<T> ComputeToAll(Vector<T> point, Matrix<T> data);

    /// <summary>
    /// Computes the full pairwise distance matrix between all rows.
    /// </summary>
    /// <param name="data">A matrix where each row is a data point.</param>
    /// <returns>
    /// A symmetric matrix where element [i, j] is the distance between row i and row j.
    /// The diagonal is always zero.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This operation has O(n^2) complexity where n is the number of rows.
    /// For large datasets, consider using spatial indexing structures like KD-Trees.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a table showing the distance between
    /// every pair of points. It's like a mileage chart showing distances between cities.
    ///
    /// The result is symmetric (distance A to B equals distance B to A) and the
    /// diagonal is all zeros (distance from a point to itself is zero).
    /// </para>
    /// </remarks>
    Matrix<T> ComputePairwise(Matrix<T> data);

    /// <summary>
    /// Computes pairwise distances between rows of two different matrices.
    /// </summary>
    /// <param name="x">First matrix where each row is a data point.</param>
    /// <param name="y">Second matrix where each row is a data point.</param>
    /// <returns>
    /// A matrix where element [i, j] is the distance between row i of x and row j of y.
    /// Shape is [x.Rows, y.Rows].
    /// </returns>
    Matrix<T> ComputePairwise(Matrix<T> x, Matrix<T> y);
}
