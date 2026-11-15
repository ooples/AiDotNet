using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Base;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Clustering.DistanceMetrics;

/// <summary>
/// Abstract base class for distance metrics providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This base class provides default implementations for batch distance computations
/// that can be overridden by derived classes for optimized implementations.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for all distance metrics.
/// It provides common code so each specific distance metric only needs to
/// define how to compute the distance between two points.
/// </para>
/// </remarks>
public abstract class DistanceMetricBase<T> : IDistanceMetric<T>
{
    /// <summary>
    /// The numeric operations provider for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Provides hardware-accelerated tensor/vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Initializes a new instance of the distance metric base class.
    /// </summary>
    protected DistanceMetricBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public abstract T Compute(Vector<T> a, Vector<T> b);

    /// <inheritdoc />
    /// <remarks>
    /// Wraps the arrays without copying and evaluates <see cref="Compute"/>, so every metric (including custom
    /// ones that only override <see cref="Compute"/>) measures its own distance here.
    /// </remarks>
    public virtual T ComputeInline(T[] a, T[] b, int length)
    {
        if (length < 0 || length > a.Length || length > b.Length)
            throw new ArgumentOutOfRangeException(nameof(length),
                $"Length ({length}) must be non-negative and within bounds of both arrays (a={a.Length}, b={b.Length}).");

        return Compute(Vector<T>.Wrap(a, 0, length), Vector<T>.Wrap(b, 0, length));
    }

    /// <inheritdoc />
    public virtual Vector<T> ComputeToAll(Vector<T> point, Matrix<T> data)
    {
        if (point.Length != data.Columns)
        {
            throw new ArgumentException(
                $"Point dimension ({point.Length}) must match data columns ({data.Columns}).");
        }

        return ClusteringTensorOps.PointRowReduce(point, data, ComputeRowPairs);
    }

    /// <inheritdoc />
    public virtual Matrix<T> ComputePairwise(Matrix<T> data)
    {
        // Distances above the diagonal mirrored below it, with a zero diagonal.
        var full = Tensor<T>.FromMatrix(ComputePairwise(data, data));
        var upper = Engine.TensorTriu(full, 1);
        return Engine.TensorAdd(upper, Engine.TensorTranspose(upper)).ToMatrix();
    }

    /// <inheritdoc />
    public virtual Matrix<T> ComputePairwise(Matrix<T> x, Matrix<T> y)
    {
        if (x.Columns != y.Columns)
        {
            throw new ArgumentException(
                $"Matrix columns must match: x has {x.Columns}, y has {y.Columns}.");
        }

        return ClusteringTensorOps.PairwiseRowReduce(x, y, ComputeRowPairs);
    }

    /// <summary>
    /// Computes the distance between every pair of corresponding rows of two [p, d] tensors.
    /// </summary>
    /// <param name="a">The first set of points, one per row.</param>
    /// <param name="b">The second set of points, one per row, paired with <paramref name="a"/> by row index.</param>
    /// <returns>A [p] tensor whose entry i is the distance between row i of a and row i of b.</returns>
    /// <remarks>
    /// The built-in metrics override this with whole-tensor engine operations. The default evaluates the
    /// metric's own <see cref="Compute"/> once per row pair, which keeps custom metrics that only implement
    /// <see cref="Compute"/> working with every batched API.
    /// </remarks>
    protected virtual Tensor<T> ComputeRowPairs(Tensor<T> a, Tensor<T> b)
    {
        var left = a.ToMatrix();
        var right = b.ToMatrix();
        return Vector<T>.FromEnumerable(left.GetRows().Zip(right.GetRows(), Compute)).AsTensor();
    }

    /// <summary>
    /// Gets a row from a matrix as a vector.
    /// </summary>
    /// <param name="matrix">The source matrix.</param>
    /// <param name="rowIndex">The row index to extract.</param>
    /// <returns>A vector containing the row data.</returns>
    protected Vector<T> GetRow(Matrix<T> matrix, int rowIndex) => matrix.GetRow(rowIndex);

    /// <summary>
    /// Computes the sum of elements in a vector.
    /// </summary>
    /// <param name="vector">The vector to sum.</param>
    /// <returns>The sum of all elements.</returns>
    protected T Sum(Vector<T> vector) => Engine.Sum(vector);

    /// <summary>
    /// Computes the square root of a value.
    /// </summary>
    /// <param name="value">The value to compute square root of.</param>
    /// <returns>The square root.</returns>
    protected T Sqrt(T value)
    {
        double doubleValue = NumOps.ToDouble(value);
        return NumOps.FromDouble(Math.Sqrt(doubleValue));
    }

    /// <summary>
    /// Computes the absolute value.
    /// </summary>
    /// <param name="value">The value to compute absolute value of.</param>
    /// <returns>The absolute value.</returns>
    protected T Abs(T value)
    {
        double doubleValue = NumOps.ToDouble(value);
        return NumOps.FromDouble(Math.Abs(doubleValue));
    }

    /// <summary>
    /// Computes the power of a value.
    /// </summary>
    /// <param name="baseValue">The base value.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The result of base raised to exponent.</returns>
    protected T Pow(T baseValue, double exponent)
    {
        double doubleBase = NumOps.ToDouble(baseValue);
        return NumOps.FromDouble(Math.Pow(doubleBase, exponent));
    }

    /// <summary>
    /// Returns the maximum of two values.
    /// </summary>
    /// <param name="a">First value.</param>
    /// <param name="b">Second value.</param>
    /// <returns>The larger of the two values.</returns>
    protected T Max(T a, T b)
    {
        double aDouble = NumOps.ToDouble(a);
        double bDouble = NumOps.ToDouble(b);
        return NumOps.FromDouble(Math.Max(aDouble, bDouble));
    }
}
