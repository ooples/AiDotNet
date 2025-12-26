using AiDotNet.Clustering.Interfaces;

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
    public virtual Vector<T> ComputeToAll(Vector<T> point, Matrix<T> data)
    {
        if (point.Length != data.Columns)
        {
            throw new ArgumentException(
                $"Point dimension ({point.Length}) must match data columns ({data.Columns}).");
        }

        var distances = new Vector<T>(data.Rows);
        for (int i = 0; i < data.Rows; i++)
        {
            var row = GetRow(data, i);
            distances[i] = Compute(point, row);
        }

        return distances;
    }

    /// <inheritdoc />
    public virtual Matrix<T> ComputePairwise(Matrix<T> data)
    {
        int n = data.Rows;
        var distances = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            var rowI = GetRow(data, i);
            distances[i, i] = NumOps.Zero;

            for (int j = i + 1; j < n; j++)
            {
                var rowJ = GetRow(data, j);
                var dist = Compute(rowI, rowJ);
                distances[i, j] = dist;
                distances[j, i] = dist; // Symmetric
            }
        }

        return distances;
    }

    /// <inheritdoc />
    public virtual Matrix<T> ComputePairwise(Matrix<T> x, Matrix<T> y)
    {
        if (x.Columns != y.Columns)
        {
            throw new ArgumentException(
                $"Matrix columns must match: x has {x.Columns}, y has {y.Columns}.");
        }

        var distances = new Matrix<T>(x.Rows, y.Rows);

        for (int i = 0; i < x.Rows; i++)
        {
            var rowX = GetRow(x, i);
            for (int j = 0; j < y.Rows; j++)
            {
                var rowY = GetRow(y, j);
                distances[i, j] = Compute(rowX, rowY);
            }
        }

        return distances;
    }

    /// <summary>
    /// Gets a row from a matrix as a vector.
    /// </summary>
    /// <param name="matrix">The source matrix.</param>
    /// <param name="rowIndex">The row index to extract.</param>
    /// <returns>A vector containing the row data.</returns>
    protected Vector<T> GetRow(Matrix<T> matrix, int rowIndex)
    {
        var row = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            row[j] = matrix[rowIndex, j];
        }
        return row;
    }

    /// <summary>
    /// Computes the sum of elements in a vector.
    /// </summary>
    /// <param name="vector">The vector to sum.</param>
    /// <returns>The sum of all elements.</returns>
    protected T Sum(Vector<T> vector)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sum = NumOps.Add(sum, vector[i]);
        }
        return sum;
    }

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
