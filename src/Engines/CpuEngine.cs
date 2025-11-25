using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;

namespace AiDotNet.Engines;

/// <summary>
/// CPU-based execution engine using INumericOperations for type-generic operations.
/// </summary>
/// <remarks>
/// <para>
/// CpuEngine provides the default execution backend for AiDotNet. It works with
/// any numeric type that implements INumericOperations{T}, including decimal,
/// BigInteger, and custom numeric types.
/// </para>
/// <para><b>For Beginners:</b> This is the standard, "always works" mode.
///
/// CpuEngine characteristics:
/// - Works with ANY numeric type (float, double, decimal, BigInteger, custom types)
/// - No special hardware required
/// - Good performance for small-to-medium datasets
/// - Single-threaded by default (can be parallelized in future versions)
///
/// When to use:
/// - You need decimal or high-precision arithmetic
/// - You don't have a GPU
/// - Your datasets are small (< 100K parameters)
/// - You're using custom numeric types
/// </para>
/// </remarks>
public class CpuEngine : IEngine
{
    /// <inheritdoc/>
    public string Name => "CPU Engine";

    /// <inheritdoc/>
    public bool SupportsGpu => false;

    /// <inheritdoc/>
    public Vector<T> Add<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10× speedup for float)
        return TensorPrimitivesHelper<T>.Add(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Subtract<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10× speedup for float)
        return TensorPrimitivesHelper<T>.Subtract(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10× speedup for float)
        return TensorPrimitivesHelper<T>.Multiply(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> vector, T scalar)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Create scalar vector and use SIMD-optimized multiplication
        var scalarVector = Vector<T>.CreateDefault(vector.Length, scalar);
        return TensorPrimitivesHelper<T>.Multiply(vector, scalarVector);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Check for division by zero before calling TensorPrimitivesHelper
        var numOps = MathHelper.GetNumericOperations<T>();
        var bArray = b.ToArray();
        for (int i = 0; i < bArray.Length; i++)
        {
            if (numOps.Equals(bArray[i], numOps.Zero))
            {
                throw new DivideByZeroException($"Division by zero at index {i}");
            }
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10× speedup for float)
        return TensorPrimitivesHelper<T>.Divide(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> vector, T scalar)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Check for division by zero
        if (numOps.Equals(scalar, numOps.Zero))
        {
            throw new DivideByZeroException("Cannot divide by zero");
        }

        // Create scalar vector and use SIMD-optimized division
        var scalarVector = Vector<T>.CreateDefault(vector.Length, scalar);
        return TensorPrimitivesHelper<T>.Divide(vector, scalarVector);
    }

    /// <inheritdoc/>
    public Vector<T> Sqrt<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (5-10× speedup for float)
        return TensorPrimitivesHelper<T>.Sqrt(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Power<T>(Vector<T> vector, T exponent)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Power(vector[i], exponent);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Max<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = numOps.GreaterThan(a[i], b[i]) ? a[i] : b[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Min<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = numOps.LessThan(a[i], b[i]) ? a[i] : b[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Abs<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Abs(vector[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Exp<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (3-6× speedup for float)
        return TensorPrimitivesHelper<T>.Exp(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Log<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (3-6× speedup for float)
        return TensorPrimitivesHelper<T>.Log(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Sign<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            // Sign returns -1, 0, or +1
            if (numOps.GreaterThan(vector[i], numOps.Zero))
            {
                result[i] = numOps.One;
            }
            else if (numOps.LessThan(vector[i], numOps.Zero))
            {
                result[i] = numOps.Negate(numOps.One);
            }
            else
            {
                result[i] = numOps.Zero;
            }
        }

        return result;
    }

    #region Reduction Operations

    /// <inheritdoc/>
    public T Sum<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = numOps.Zero;

        for (int i = 0; i < vector.Length; i++)
        {
            sum = numOps.Add(sum, vector[i]);
        }

        return sum;
    }

    /// <inheritdoc/>
    public T DotProduct<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vectors must have the same length for dot product. Got lengths {a.Length} and {b.Length}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        T result = numOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            result = numOps.Add(result, numOps.Multiply(a[i], b[i]));
        }

        return result;
    }

    /// <inheritdoc/>
    public T Mean<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (vector.Length == 0) throw new ArgumentException("Cannot compute mean of empty vector.");

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = Sum(vector);
        T length = numOps.FromDouble(vector.Length);
        return numOps.Divide(sum, length);
    }
/// <inheritdoc/>
    public Vector<T> Fill<T>(int length, T value)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = value;
        }
        return result;
    }

    /// <inheritdoc/>
    public Vector<T> FillZero<T>(int length)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        return new Vector<T>(length); // Vector constructor already initializes to zero
    }

    /// <inheritdoc/>
    public Vector<T> GenerateDropoutMask<T>(int length, T dropoutRate, T scale, int? seed = null)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        var random = seed.HasValue ? new Random(seed.Value) : new Random();
        var numOps = MathHelper.GetNumericOperations<T>();
        double dropoutRateDouble = Convert.ToDouble(dropoutRate);
        var mask = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            mask[i] = random.NextDouble() > dropoutRateDouble ? scale : numOps.Zero;
        }
        return mask;
    }

    /// <inheritdoc/>
    public void CopyVectorToTensor<T>(Vector<T> source, Tensor<T> destination)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (source.Length != destination.Length)
        {
            throw new ArgumentException(
                $"Vector length ({source.Length}) must equal tensor total elements ({destination.Length}).");
        }
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = source[i];
        }
    }
    /// <inheritdoc/>
    public Vector<T> GenerateGaussianNoise<T>(int length, T mean, T standardDeviation, int? seed = null)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        var random = seed.HasValue ? new Random(seed.Value) : new Random();
        var numOps = MathHelper.GetNumericOperations<T>();
        var noise = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            // Box-Muller transform to generate Gaussian random numbers
            T u1 = numOps.FromDouble(random.NextDouble());
            T u2 = numOps.FromDouble(random.NextDouble());
            T z = numOps.Multiply(
                numOps.Sqrt(numOps.Multiply(numOps.FromDouble(-2.0), numOps.Log(u1))),
                numOps.FromDouble(Math.Cos(2.0 * Math.PI * Convert.ToDouble(u2))));
            noise[i] = numOps.Add(mean, numOps.Multiply(standardDeviation, z));
        }
        return noise;
    }

    #endregion

    #region Matrix Operations (Phase B: Epic 2)

    /// <inheritdoc/>
    public Matrix<T> MatrixMultiply<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible for multiplication. " +
                $"First matrix is {a.Rows}x{a.Columns}, second is {b.Rows}x{b.Columns}. " +
                $"First matrix columns ({a.Columns}) must equal second matrix rows ({b.Rows}).");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(a.Rows, b.Columns);

        // Standard O(n³) matrix multiplication
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < b.Columns; j++)
            {
                T sum = numOps.Zero;
                for (int k = 0; k < a.Columns; k++)
                {
                    sum = numOps.Add(sum, numOps.Multiply(a[i, k], b[k, j]));
                }
                result[i, j] = sum;
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> MatrixVectorMultiply<T>(Matrix<T> matrix, Vector<T> vector)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (matrix.Columns != vector.Length)
        {
            throw new ArgumentException(
                $"Matrix-vector dimensions incompatible. " +
                $"Matrix is {matrix.Rows}x{matrix.Columns}, vector has {vector.Length} elements. " +
                $"Matrix columns ({matrix.Columns}) must equal vector length ({vector.Length}).");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(matrix.Rows);

        for (int i = 0; i < matrix.Rows; i++)
        {
            T sum = numOps.Zero;
            for (int j = 0; j < matrix.Columns; j++)
            {
                sum = numOps.Add(sum, numOps.Multiply(matrix[i, j], vector[j]));
            }
            result[i] = sum;
        }

        return result;
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixTranspose<T>(Matrix<T> matrix)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        var result = new Matrix<T>(matrix.Columns, matrix.Rows);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[j, i] = matrix[i, j];
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixAdd<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rows != b.Rows || a.Columns != b.Columns)
        {
            throw new ArgumentException(
                $"Matrix dimensions must match for addition. " +
                $"First matrix is {a.Rows}x{a.Columns}, second is {b.Rows}x{b.Columns}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(a.Rows, a.Columns);

        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                result[i, j] = numOps.Add(a[i, j], b[i, j]);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixMultiplyScalar<T>(Matrix<T> matrix, T scalar)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = numOps.Multiply(matrix[i, j], scalar);
            }
        }

        return result;
    }

    public Matrix<T> MatrixSubtract<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArgumentException("Matrix dimensions must match for subtraction");

        var result = new Matrix<T>(a.Rows, a.Columns);

        // VECTORIZED: Use existing Vector Subtract operation on each row
        for (int i = 0; i < a.Rows; i++)
        {
            var rowA = a.GetRow(i);
            var rowB = b.GetRow(i);
            var diffRow = Subtract(rowA, rowB); // Reuse vectorized Vector Subtract
            result.SetRow(i, diffRow);
        }

        return result;
    }

    public T MatrixSumOfSquares<T>(Matrix<T> matrix)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = numOps.Zero;

        // VECTORIZED: Use existing DotProduct operation on each row
        for (int i = 0; i < matrix.Rows; i++)
        {
            var row = matrix.GetRow(i);
            T rowSumSquares = DotProduct(row, row); // row · row = sum of squares for row
            sum = numOps.Add(sum, rowSumSquares);
        }

        return sum;
    }

    public void SwapColumns<T>(Matrix<T> matrix, int col1, int col2)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        // Direct element swap - no vectorization benefit for column swaps due to strided access
        for (int i = 0; i < matrix.Rows; i++)
        {
            T temp = matrix[i, col1];
            matrix[i, col1] = matrix[i, col2];
            matrix[i, col2] = temp;
        }
    }

    public void SwapRows<T>(Matrix<T> matrix, int row1, int row2)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        // Use vectorized operations for row swapping
        var tempRow1 = GetRow(matrix, row1);
        var tempRow2 = GetRow(matrix, row2);

        SetRow(matrix, row1, tempRow2);
        SetRow(matrix, row2, tempRow1);
    }

    public Matrix<T> OuterProduct<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        var result = new Matrix<T>(a.Length, b.Length);
        var aArray = a.ToArray();
        var bArray = b.ToArray();

        // Use SIMD-optimized TensorPrimitives for float type
        if (typeof(T) == typeof(float) && bArray.Length >= 16)
        {
            var bFloat = (float[])(object)bArray;
            var aFloat = (float[])(object)aArray;

            for (int i = 0; i < aFloat.Length; i++)
            {
                var rowData = new float[bFloat.Length];
                // SIMD vectorized: multiply vector b by scalar a[i]
                TensorPrimitives.Multiply(bFloat, aFloat[i], rowData);

                // Copy result to matrix
                for (int j = 0; j < bFloat.Length; j++)
                {
                    result[i, j] = (T)(object)rowData[j];
                }
            }
        }
        else
        {
            // Fallback using NumOps
            var numOps = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < aArray.Length; i++)
            {
                for (int j = 0; j < bArray.Length; j++)
                {
                    result[i, j] = numOps.Multiply(aArray[i], bArray[j]);
                }
            }
        }

        return result;
    }

    public Vector<T> GetColumn<T>(Matrix<T> matrix, int columnIndex)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        // No vectorization benefit - column access is strided
        var result = new T[matrix.Rows];
        for (int i = 0; i < matrix.Rows; i++)
        {
            result[i] = matrix[i, columnIndex];
        }
        return new Vector<T>(result);
    }

    public Vector<T> GetRow<T>(Matrix<T> matrix, int rowIndex)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        // Row access is contiguous - can use direct array copy
        var result = new T[matrix.Columns];
        for (int j = 0; j < matrix.Columns; j++)
        {
            result[j] = matrix[rowIndex, j];
        }
        return new Vector<T>(result);
    }

    public void SetColumn<T>(Matrix<T> matrix, int columnIndex, Vector<T> values)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (values == null) throw new ArgumentNullException(nameof(values));

        // No vectorization benefit - column access is strided
        var valuesArray = values.ToArray();
        for (int i = 0; i < Math.Min(matrix.Rows, valuesArray.Length); i++)
        {
            matrix[i, columnIndex] = valuesArray[i];
        }
    }

    public void SetRow<T>(Matrix<T> matrix, int rowIndex, Vector<T> values)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (values == null) throw new ArgumentNullException(nameof(values));

        // Row access is contiguous - direct assignment
        var valuesArray = values.ToArray();
        for (int j = 0; j < Math.Min(matrix.Columns, valuesArray.Length); j++)
        {
            matrix[rowIndex, j] = valuesArray[j];
        }
    }

    #endregion

    #region Tensor Operations (Phase B: Epic 3)

    /// <inheritdoc/>
    public Tensor<T> BatchMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rank != 3 || b.Rank != 3)
        {
            throw new ArgumentException(
                $"BatchMatMul requires 3D tensors. Got ranks {a.Rank} and {b.Rank}.");
        }

        int batchSize = a.Shape[0];
        int m = a.Shape[1];
        int k = a.Shape[2];
        int k2 = b.Shape[1];
        int n = b.Shape[2];

        if (b.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Batch sizes must match. Got {batchSize} and {b.Shape[0]}.");
        }
        if (k != k2)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible for multiplication. " +
                $"First tensor has shape [{batchSize}, {m}, {k}], " +
                $"second has shape [{b.Shape[0]}, {k2}, {n}]. " +
                $"Inner dimensions must match ({k} != {k2}).");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(new[] { batchSize, m, n });

        // Process each batch
        for (int batch = 0; batch < batchSize; batch++)
        {
            // Standard matrix multiplication for this batch: C[batch] = A[batch] @ B[batch]
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    T sum = numOps.Zero;
                    for (int p = 0; p < k; p++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(
                            a[batch, i, p],
                            b[batch, p, j]));
                    }
                    result[batch, i, j] = sum;
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorAdd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = numOps.Add(a[i], b[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSubtract<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = numOps.Subtract(a[i], b[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = numOps.Multiply(a[i], b[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMultiplyScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        for (int i = 0; i < tensor.Length; i++)
        {
            result[i] = numOps.Multiply(tensor[i], scalar);
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        for (int i = 0; i < a.Length; i++)
        {
            // Check for division by zero
            if (numOps.Equals(b[i], numOps.Zero))
            {
                throw new DivideByZeroException($"Division by zero at index {i}");
            }

            result[i] = numOps.Divide(a[i], b[i]);
        }

        return result;
    }

    public Tensor<T> TensorTranspose<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        
        // Verify tensor is 2D
        if (tensor.Shape.Length != 2)
        {
            throw new ArgumentException(
                $"TensorTranspose requires a 2D tensor, but got {tensor.Shape.Length}D tensor with shape {FormatShape(tensor.Shape)}.",
                nameof(tensor));
        }

        int rows = tensor.Shape[0];
        int cols = tensor.Shape[1];
        
        // Create result tensor with transposed dimensions
        var result = new Tensor<T>(new int[] { cols, rows });
        
        // Perform transpose: result[j, i] = tensor[i, j]
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int sourceIdx = i * cols + j;
                int destIdx = j * rows + i;
                result[destIdx] = tensor[sourceIdx];
            }
        }
        
        return result;
    }

    public Tensor<T> TensorMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        
        // Verify both tensors are 2D
        if (a.Shape.Length != 2)
        {
            throw new ArgumentException(
                $"TensorMatMul requires 2D tensors, but first tensor is {a.Shape.Length}D with shape {FormatShape(a.Shape)}.",
                nameof(a));
        }
        if (b.Shape.Length != 2)
        {
            throw new ArgumentException(
                $"TensorMatMul requires 2D tensors, but second tensor is {b.Shape.Length}D with shape {FormatShape(b.Shape)}.",
                nameof(b));
        }
        
        int M = a.Shape[0];  // Rows in A
        int N = a.Shape[1];  // Cols in A (must equal rows in B)
        int P = b.Shape[1];  // Cols in B
        
        // Verify inner dimensions match
        if (b.Shape[0] != N)
        {
            throw new ArgumentException(
                $"Matrix multiplication requires inner dimensions to match. " +
                $"Got A: {FormatShape(a.Shape)} and B: {FormatShape(b.Shape)}. " +
                $"A has {N} columns but B has {b.Shape[0]} rows.");
        }
        
        var numOps = MathHelper.GetNumericOperations<T>();
        
        // Create result tensor with shape [M, P]
        var result = new Tensor<T>(new int[] { M, P });
        
        // Perform matrix multiplication: C[i,k] = sum(A[i,j] * B[j,k])
        for (int i = 0; i < M; i++)
        {
            for (int k = 0; k < P; k++)
            {
                T sum = numOps.Zero;
                
                for (int j = 0; j < N; j++)
                {
                    int aIdx = i * N + j;      // A[i,j]
                    int bIdx = j * P + k;      // B[j,k]
                    
                    T product = numOps.Multiply(a[aIdx], b[bIdx]);
                    sum = numOps.Add(sum, product);
                }
                
                int resultIdx = i * P + k;
                result[resultIdx] = sum;
            }
        }
        
        return result;
    }

    /// <summary>
    /// Helper method to check if two shapes match.
    /// </summary>
    private bool ShapesMatch(int[] shape1, int[] shape2)
    {
        if (shape1.Length != shape2.Length)
            return false;

        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Helper method to format a shape for error messages.
    /// </summary>
    private string FormatShape(int[] shape)
    {
        return "[" + string.Join(", ", shape) + "]";
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
        {
            throw new ArgumentException($"MaxPool2D requires a 4D tensor [batch, channels, height, width]. Got rank {input.Rank}.");
        }
        if (poolSize <= 0) throw new ArgumentException("Pool size must be positive.");

        if (stride == 0) stride = poolSize; // Default stride equals pool size

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outputHeight = (height + 2 * padding - poolSize) / stride + 1;
        int outputWidth = (width + 2 * padding - poolSize) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid pooling parameters. Output dimensions would be {outputHeight}x{outputWidth}. " +
                $"Ensure poolSize={poolSize}, stride={stride}, padding={padding} are compatible with input size {height}x{width}.");
        }

        var result = new Tensor<T>(new[] { batch, channels, outputHeight, outputWidth });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        // Use MinValue for type-safe initialization (works for all numeric types)
                        T maxValue = numOps.MinValue;

                        for (int kh = 0; kh < poolSize; kh++)
                        {
                            for (int kw = 0; kw < poolSize; kw++)
                            {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                // Check bounds (handle padding)
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    T value = input[b, c, ih, iw];
                                    if (numOps.GreaterThan(value, maxValue))
                                    {
                                        maxValue = value;
                                    }
                                }
                            }
                        }

                        result[b, c, oh, ow] = maxValue;
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
        {
            throw new ArgumentException($"AvgPool2D requires a 4D tensor [batch, channels, height, width]. Got rank {input.Rank}.");
        }
        if (poolSize <= 0) throw new ArgumentException("Pool size must be positive.");

        if (stride == 0) stride = poolSize; // Default stride equals pool size

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outputHeight = (height + 2 * padding - poolSize) / stride + 1;
        int outputWidth = (width + 2 * padding - poolSize) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid pooling parameters. Output dimensions would be {outputHeight}x{outputWidth}. " +
                $"Ensure poolSize={poolSize}, stride={stride}, padding={padding} are compatible with input size {height}x{width}.");
        }

        var result = new Tensor<T>(new[] { batch, channels, outputHeight, outputWidth });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;
                        int count = 0;

                        for (int kh = 0; kh < poolSize; kh++)
                        {
                            for (int kw = 0; kw < poolSize; kw++)
                            {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                // Check bounds (handle padding)
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    sum = numOps.Add(sum, input[b, c, ih, iw]);
                                    count++;
                                }
                            }
                        }

                        // Calculate average
                        if (count > 0)
                        {
                            var countValue = numOps.FromDouble(count);
                            result[b, c, oh, ow] = numOps.Divide(sum, countValue);
                        }
                        else
                        {
                            result[b, c, oh, ow] = numOps.Zero;
                        }
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 4)
        {
            throw new ArgumentException($"Conv2D input requires a 4D tensor [batch, in_channels, height, width]. Got rank {input.Rank}.");
        }
        if (kernel.Rank != 4)
        {
            throw new ArgumentException($"Conv2D kernel requires a 4D tensor [out_channels, in_channels, kernel_height, kernel_width]. Got rank {kernel.Rank}.");
        }
        if (stride <= 0) throw new ArgumentException("Stride must be positive.");
        if (dilation <= 0) throw new ArgumentException("Dilation must be positive.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelInChannels = kernel.Shape[1];
        int kernelHeight = kernel.Shape[2];
        int kernelWidth = kernel.Shape[3];

        if (inChannels != kernelInChannels)
        {
            throw new ArgumentException(
                $"Input channels ({inChannels}) must match kernel input channels ({kernelInChannels}).");
        }

        int effectiveKernelHeight = dilation * (kernelHeight - 1) + 1;
        int effectiveKernelWidth = dilation * (kernelWidth - 1) + 1;

        int outputHeight = (height + 2 * padding - effectiveKernelHeight) / stride + 1;
        int outputWidth = (width + 2 * padding - effectiveKernelWidth) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid convolution parameters. Output dimensions would be {outputHeight}x{outputWidth}. " +
                $"Ensure stride={stride}, padding={padding}, dilation={dilation} are compatible with input size {height}x{width} and kernel size {kernelHeight}x{kernelWidth}.");
        }

        var result = new Tensor<T>(new[] { batch, outChannels, outputHeight, outputWidth });

        // Perform convolution
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;

                        // Sum over all input channels
                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            // Sum over kernel window
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int ih = oh * stride + kh * dilation - padding;
                                    int iw = ow * stride + kw * dilation - padding;

                                    // Check bounds (handle padding)
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    {
                                        T inputVal = input[b, ic, ih, iw];
                                        T kernelVal = kernel[oc, ic, kh, kw];
                                        sum = numOps.Add(sum, numOps.Multiply(inputVal, kernelVal));
                                    }
                                }
                            }
                        }

                        result[b, oc, oh, ow] = sum;
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be a 2-element array [strideH, strideW].");
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be a 2-element array [padH, padW].");
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be a 2-element array [dilationH, dilationW].");
        if (input.Rank != 4)
        {
            throw new ArgumentException($"Conv2D input requires a 4D tensor [batch, in_channels, height, width]. Got rank {input.Rank}.");
        }
        if (kernel.Rank != 4)
        {
            throw new ArgumentException($"Conv2D kernel requires a 4D tensor [out_channels, in_channels, kernel_height, kernel_width]. Got rank {kernel.Rank}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelInChannels = kernel.Shape[1];
        int kernelHeight = kernel.Shape[2];
        int kernelWidth = kernel.Shape[3];

        if (inChannels != kernelInChannels)
        {
            throw new ArgumentException(
                $"Input channels ({inChannels}) must match kernel input channels ({kernelInChannels}).");
        }

        int strideH = stride[0];
        int strideW = stride[1];
        int padH = padding[0];
        int padW = padding[1];
        int dilationH = dilation[0];
        int dilationW = dilation[1];

        int effectiveKernelHeight = dilationH * (kernelHeight - 1) + 1;
        int effectiveKernelWidth = dilationW * (kernelWidth - 1) + 1;

        int outputHeight = (height + 2 * padH - effectiveKernelHeight) / strideH + 1;
        int outputWidth = (width + 2 * padW - effectiveKernelWidth) / strideW + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid convolution parameters. Output dimensions would be {outputHeight}x{outputWidth}.");
        }

        var result = new Tensor<T>(new[] { batch, outChannels, outputHeight, outputWidth });

        // Perform convolution with asymmetric parameters
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;

                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int ih = oh * strideH + kh * dilationH - padH;
                                    int iw = ow * strideW + kw * dilationW - padW;

                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    {
                                        T inputVal = input[b, ic, ih, iw];
                                        T kernelVal = kernel[oc, ic, kh, kw];
                                        sum = numOps.Add(sum, numOps.Multiply(inputVal, kernelVal));
                                    }
                                }
                            }
                        }

                        result[b, oc, oh, ow] = sum;
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (inputShape == null || inputShape.Length != 4) throw new ArgumentException("InputShape must be a 4-element array.");
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be a 2-element array.");
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be a 2-element array.");
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be a 2-element array.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];

        int outChannels = kernel.Shape[0];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        int outH = gradOutput.Shape[2];
        int outW = gradOutput.Shape[3];

        int strideH = stride[0];
        int strideW = stride[1];
        int padH = padding[0];
        int padW = padding[1];
        int dilationH = dilation[0];
        int dilationW = dilation[1];

        var gradInput = new Tensor<T>(inputShape);

        // Compute gradient w.r.t. input using transposed convolution logic
        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int ih = 0; ih < inH; ih++)
                {
                    for (int iw = 0; iw < inW; iw++)
                    {
                        T sum = numOps.Zero;

                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    // Compute output position that used this input position
                                    int ohShifted = ih + padH - kh * dilationH;
                                    int owShifted = iw + padW - kw * dilationW;

                                    if (ohShifted % strideH == 0 && owShifted % strideW == 0)
                                    {
                                        int oh = ohShifted / strideH;
                                        int ow = owShifted / strideW;

                                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW)
                                        {
                                            T gradVal = gradOutput[b, oc, oh, ow];
                                            T kernelVal = kernel[oc, ic, kh, kw];
                                            sum = numOps.Add(sum, numOps.Multiply(gradVal, kernelVal));
                                        }
                                    }
                                }
                            }
                        }

                        gradInput[b, ic, ih, iw] = sum;
                    }
                }
            }
        }

        return gradInput;
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelShape == null || kernelShape.Length != 4) throw new ArgumentException("KernelShape must be a 4-element array.");
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be a 2-element array.");
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be a 2-element array.");
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be a 2-element array.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inH = input.Shape[2];
        int inW = input.Shape[3];

        int outChannels = kernelShape[0];
        int kernelH = kernelShape[2];
        int kernelW = kernelShape[3];

        int outH = gradOutput.Shape[2];
        int outW = gradOutput.Shape[3];

        int strideH = stride[0];
        int strideW = stride[1];
        int padH = padding[0];
        int padW = padding[1];
        int dilationH = dilation[0];
        int dilationW = dilation[1];

        var gradKernel = new Tensor<T>(kernelShape);

        // Compute gradient w.r.t. kernel using cross-correlation
        for (int oc = 0; oc < outChannels; oc++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int kh = 0; kh < kernelH; kh++)
                {
                    for (int kw = 0; kw < kernelW; kw++)
                    {
                        T sum = numOps.Zero;

                        for (int b = 0; b < batch; b++)
                        {
                            for (int oh = 0; oh < outH; oh++)
                            {
                                for (int ow = 0; ow < outW; ow++)
                                {
                                    int ih = oh * strideH + kh * dilationH - padH;
                                    int iw = ow * strideW + kw * dilationW - padW;

                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                    {
                                        T gradVal = gradOutput[b, oc, oh, ow];
                                        T inputVal = input[b, ic, ih, iw];
                                        sum = numOps.Add(sum, numOps.Multiply(gradVal, inputVal));
                                    }
                                }
                            }
                        }

                        gradKernel[oc, ic, kh, kw] = sum;
                    }
                }
            }
        }

        return gradKernel;
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool2DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,] maxIndices)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
            throw new ArgumentException($"MaxPool2D requires a 4D tensor [batch, channels, height, width]. Got rank {input.Rank}.");
        if (poolSize == null || poolSize.Length != 2)
            throw new ArgumentException("PoolSize must be a 2-element array [poolH, poolW].");
        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be a 2-element array [strideH, strideW].");

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int poolH = poolSize[0];
        int poolW = poolSize[1];
        int strideH = stride[0];
        int strideW = stride[1];

        int outputHeight = (height - poolH) / strideH + 1;
        int outputWidth = (width - poolW) / strideW + 1;

        var result = new Tensor<T>(new[] { batch, channels, outputHeight, outputWidth });
        maxIndices = new int[batch, channels, outputHeight, outputWidth, 2];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T maxValue = numOps.MinValue;
                        int maxIh = 0, maxIw = 0;

                        for (int kh = 0; kh < poolH; kh++)
                        {
                            for (int kw = 0; kw < poolW; kw++)
                            {
                                int ih = oh * strideH + kh;
                                int iw = ow * strideW + kw;

                                if (ih < height && iw < width)
                                {
                                    T value = input[b, c, ih, iw];
                                    if (numOps.GreaterThan(value, maxValue))
                                    {
                                        maxValue = value;
                                        maxIh = ih;
                                        maxIw = iw;
                                    }
                                }
                            }
                        }

                        result[b, c, oh, ow] = maxValue;
                        maxIndices[b, c, oh, ow, 0] = maxIh;
                        maxIndices[b, c, oh, ow, 1] = maxIw;
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool2DBackward<T>(Tensor<T> gradOutput, int[,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (maxIndices == null) throw new ArgumentNullException(nameof(maxIndices));
        if (inputShape == null || inputShape.Length != 4)
            throw new ArgumentException("InputShape must be a 4-element array [batch, channels, height, width].");

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = inputShape[0];
        int channels = inputShape[1];
        int outH = gradOutput.Shape[2];
        int outW = gradOutput.Shape[3];

        var gradInput = new Tensor<T>(inputShape);

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int ih = maxIndices[b, c, oh, ow, 0];
                        int iw = maxIndices[b, c, oh, ow, 1];
                        T gradVal = gradOutput[b, c, oh, ow];

                        // Accumulate gradient at max position
                        gradInput[b, c, ih, iw] = numOps.Add(gradInput[b, c, ih, iw], gradVal);
                    }
                }
            }
        }

        return gradInput;
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool2D<T>(Tensor<T> input, int[] poolSize, int[] stride)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
            throw new ArgumentException($"AvgPool2D requires a 4D tensor [batch, channels, height, width]. Got rank {input.Rank}.");
        if (poolSize == null || poolSize.Length != 2)
            throw new ArgumentException("PoolSize must be a 2-element array [poolH, poolW].");
        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be a 2-element array [strideH, strideW].");

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int poolH = poolSize[0];
        int poolW = poolSize[1];
        int strideH = stride[0];
        int strideW = stride[1];

        int outputHeight = (height - poolH) / strideH + 1;
        int outputWidth = (width - poolW) / strideW + 1;

        var result = new Tensor<T>(new[] { batch, channels, outputHeight, outputWidth });
        T poolArea = numOps.FromDouble(poolH * poolW);

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;

                        for (int kh = 0; kh < poolH; kh++)
                        {
                            for (int kw = 0; kw < poolW; kw++)
                            {
                                int ih = oh * strideH + kh;
                                int iw = ow * strideW + kw;

                                if (ih < height && iw < width)
                                {
                                    sum = numOps.Add(sum, input[b, c, ih, iw]);
                                }
                            }
                        }

                        result[b, c, oh, ow] = numOps.Divide(sum, poolArea);
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool2DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (inputShape == null || inputShape.Length != 4)
            throw new ArgumentException("InputShape must be a 4-element array [batch, channels, height, width].");
        if (poolSize == null || poolSize.Length != 2)
            throw new ArgumentException("PoolSize must be a 2-element array [poolH, poolW].");
        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be a 2-element array [strideH, strideW].");

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = inputShape[0];
        int channels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];

        int poolH = poolSize[0];
        int poolW = poolSize[1];
        int strideH = stride[0];
        int strideW = stride[1];

        int outH = gradOutput.Shape[2];
        int outW = gradOutput.Shape[3];

        var gradInput = new Tensor<T>(inputShape);
        T poolArea = numOps.FromDouble(poolH * poolW);

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        T gradVal = numOps.Divide(gradOutput[b, c, oh, ow], poolArea);

                        for (int kh = 0; kh < poolH; kh++)
                        {
                            for (int kw = 0; kw < poolW; kw++)
                            {
                                int ih = oh * strideH + kh;
                                int iw = ow * strideW + kw;

                                if (ih < inH && iw < inW)
                                {
                                    gradInput[b, c, ih, iw] = numOps.Add(gradInput[b, c, ih, iw], gradVal);
                                }
                            }
                        }
                    }
                }
            }
        }

        return gradInput;
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 4)
            throw new ArgumentException($"DepthwiseConv2D input requires a 4D tensor [batch, in_channels, height, width]. Got rank {input.Rank}.");
        if (kernel.Rank != 4)
            throw new ArgumentException($"DepthwiseConv2D kernel requires a 4D tensor [in_channels, multiplier, kernel_height, kernel_width]. Got rank {kernel.Rank}.");
        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be a 2-element array [strideH, strideW].");
        if (padding == null || padding.Length != 2)
            throw new ArgumentException("Padding must be a 2-element array [padH, padW].");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int kernelChannels = kernel.Shape[0];
        int multiplier = kernel.Shape[1];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        if (inChannels != kernelChannels)
            throw new ArgumentException($"Input channels ({inChannels}) must match kernel channels ({kernelChannels}).");

        int strideH = stride[0];
        int strideW = stride[1];
        int padH = padding[0];
        int padW = padding[1];

        int outputHeight = (height + 2 * padH - kernelH) / strideH + 1;
        int outputWidth = (width + 2 * padW - kernelW) / strideW + 1;
        int outChannels = inChannels * multiplier;

        var result = new Tensor<T>(new[] { batch, outChannels, outputHeight, outputWidth });

        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int m = 0; m < multiplier; m++)
                {
                    int oc = ic * multiplier + m;

                    for (int oh = 0; oh < outputHeight; oh++)
                    {
                        for (int ow = 0; ow < outputWidth; ow++)
                        {
                            T sum = numOps.Zero;

                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int ih = oh * strideH + kh - padH;
                                    int iw = ow * strideW + kw - padW;

                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    {
                                        T inputVal = input[b, ic, ih, iw];
                                        T kernelVal = kernel[ic, m, kh, kw];
                                        sum = numOps.Add(sum, numOps.Multiply(inputVal, kernelVal));
                                    }
                                }
                            }

                            result[b, oc, oh, ow] = sum;
                        }
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (inputShape == null || inputShape.Length != 4)
            throw new ArgumentException("InputShape must be a 4-element array [batch, in_channels, height, width].");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];

        int multiplier = kernel.Shape[1];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        int strideH = stride[0];
        int strideW = stride[1];
        int padH = padding[0];
        int padW = padding[1];

        int outH = gradOutput.Shape[2];
        int outW = gradOutput.Shape[3];

        var gradInput = new Tensor<T>(inputShape);

        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int ih = 0; ih < inH; ih++)
                {
                    for (int iw = 0; iw < inW; iw++)
                    {
                        T sum = numOps.Zero;

                        for (int m = 0; m < multiplier; m++)
                        {
                            int oc = ic * multiplier + m;

                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int ohShifted = ih + padH - kh;
                                    int owShifted = iw + padW - kw;

                                    if (ohShifted % strideH == 0 && owShifted % strideW == 0)
                                    {
                                        int oh = ohShifted / strideH;
                                        int ow = owShifted / strideW;

                                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW)
                                        {
                                            T gradVal = gradOutput[b, oc, oh, ow];
                                            T kernelVal = kernel[ic, m, kh, kw];
                                            sum = numOps.Add(sum, numOps.Multiply(gradVal, kernelVal));
                                        }
                                    }
                                }
                            }
                        }

                        gradInput[b, ic, ih, iw] = sum;
                    }
                }
            }
        }

        return gradInput;
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelShape == null || kernelShape.Length != 4)
            throw new ArgumentException("KernelShape must be a 4-element array [in_channels, multiplier, kernelH, kernelW].");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = kernelShape[0];
        int multiplier = kernelShape[1];
        int kernelH = kernelShape[2];
        int kernelW = kernelShape[3];

        int inH = input.Shape[2];
        int inW = input.Shape[3];

        int strideH = stride[0];
        int strideW = stride[1];
        int padH = padding[0];
        int padW = padding[1];

        int outH = gradOutput.Shape[2];
        int outW = gradOutput.Shape[3];

        var gradKernel = new Tensor<T>(kernelShape);

        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int m = 0; m < multiplier; m++)
            {
                int oc = ic * multiplier + m;

                for (int kh = 0; kh < kernelH; kh++)
                {
                    for (int kw = 0; kw < kernelW; kw++)
                    {
                        T sum = numOps.Zero;

                        for (int b = 0; b < batch; b++)
                        {
                            for (int oh = 0; oh < outH; oh++)
                            {
                                for (int ow = 0; ow < outW; ow++)
                                {
                                    int ih = oh * strideH + kh - padH;
                                    int iw = ow * strideW + kw - padW;

                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                    {
                                        T gradVal = gradOutput[b, oc, oh, ow];
                                        T inputVal = input[b, ic, ih, iw];
                                        sum = numOps.Add(sum, numOps.Multiply(gradVal, inputVal));
                                    }
                                }
                            }
                        }

                        gradKernel[ic, m, kh, kw] = sum;
                    }
                }
            }
        }

        return gradKernel;
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] outputPadding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 4)
            throw new ArgumentException($"ConvTranspose2D input requires a 4D tensor [batch, in_channels, height, width]. Got rank {input.Rank}.");
        if (kernel.Rank != 4)
            throw new ArgumentException($"ConvTranspose2D kernel requires a 4D tensor [in_channels, out_channels, kernel_height, kernel_width]. Got rank {kernel.Rank}.");
        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be a 2-element array [strideH, strideW].");
        if (padding == null || padding.Length != 2)
            throw new ArgumentException("Padding must be a 2-element array [padH, padW].");
        if (outputPadding == null || outputPadding.Length != 2)
            throw new ArgumentException("OutputPadding must be a 2-element array [outPadH, outPadW].");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inH = input.Shape[2];
        int inW = input.Shape[3];

        int kernelInChannels = kernel.Shape[0];
        int outChannels = kernel.Shape[1];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        if (inChannels != kernelInChannels)
            throw new ArgumentException($"Input channels ({inChannels}) must match kernel input channels ({kernelInChannels}).");

        int strideH = stride[0];
        int strideW = stride[1];
        int padH = padding[0];
        int padW = padding[1];
        int outPadH = outputPadding[0];
        int outPadW = outputPadding[1];

        // Output size formula for transposed convolution
        int outputHeight = (inH - 1) * strideH - 2 * padH + kernelH + outPadH;
        int outputWidth = (inW - 1) * strideW - 2 * padW + kernelW + outPadW;

        var result = new Tensor<T>(new[] { batch, outChannels, outputHeight, outputWidth });

        // Transposed convolution: scatter input values through kernel
        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int ih = 0; ih < inH; ih++)
                {
                    for (int iw = 0; iw < inW; iw++)
                    {
                        T inputVal = input[b, ic, ih, iw];

                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int oh = ih * strideH + kh - padH;
                                    int ow = iw * strideW + kw - padW;

                                    if (oh >= 0 && oh < outputHeight && ow >= 0 && ow < outputWidth)
                                    {
                                        T kernelVal = kernel[ic, oc, kh, kw];
                                        result[b, oc, oh, ow] = numOps.Add(
                                            result[b, oc, oh, ow],
                                            numOps.Multiply(inputVal, kernelVal));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (inputShape == null || inputShape.Length != 4)
            throw new ArgumentException("InputShape must be a 4-element array [batch, in_channels, height, width].");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];

        int outChannels = kernel.Shape[1];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        int strideH = stride[0];
        int strideW = stride[1];
        int padH = padding[0];
        int padW = padding[1];

        int outH = gradOutput.Shape[2];
        int outW = gradOutput.Shape[3];

        var gradInput = new Tensor<T>(inputShape);

        // Backward of ConvTranspose2D w.r.t. input is a regular Conv2D
        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int ih = 0; ih < inH; ih++)
                {
                    for (int iw = 0; iw < inW; iw++)
                    {
                        T sum = numOps.Zero;

                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int oh = ih * strideH + kh - padH;
                                    int ow = iw * strideW + kw - padW;

                                    if (oh >= 0 && oh < outH && ow >= 0 && ow < outW)
                                    {
                                        T gradVal = gradOutput[b, oc, oh, ow];
                                        T kernelVal = kernel[ic, oc, kh, kw];
                                        sum = numOps.Add(sum, numOps.Multiply(gradVal, kernelVal));
                                    }
                                }
                            }
                        }

                        gradInput[b, ic, ih, iw] = sum;
                    }
                }
            }
        }

        return gradInput;
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelShape == null || kernelShape.Length != 4)
            throw new ArgumentException("KernelShape must be a 4-element array [in_channels, out_channels, kernelH, kernelW].");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = kernelShape[0];
        int outChannels = kernelShape[1];
        int kernelH = kernelShape[2];
        int kernelW = kernelShape[3];

        int inH = input.Shape[2];
        int inW = input.Shape[3];

        int strideH = stride[0];
        int strideW = stride[1];
        int padH = padding[0];
        int padW = padding[1];

        int outH = gradOutput.Shape[2];
        int outW = gradOutput.Shape[3];

        var gradKernel = new Tensor<T>(kernelShape);

        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int kh = 0; kh < kernelH; kh++)
                {
                    for (int kw = 0; kw < kernelW; kw++)
                    {
                        T sum = numOps.Zero;

                        for (int b = 0; b < batch; b++)
                        {
                            for (int ih = 0; ih < inH; ih++)
                            {
                                for (int iw = 0; iw < inW; iw++)
                                {
                                    int oh = ih * strideH + kh - padH;
                                    int ow = iw * strideW + kw - padW;

                                    if (oh >= 0 && oh < outH && ow >= 0 && ow < outW)
                                    {
                                        T gradVal = gradOutput[b, oc, oh, ow];
                                        T inputVal = input[b, ic, ih, iw];
                                        sum = numOps.Add(sum, numOps.Multiply(gradVal, inputVal));
                                    }
                                }
                            }
                        }

                        gradKernel[ic, oc, kh, kw] = sum;
                    }
                }
            }
        }

        return gradKernel;
    }

    #endregion

    #region Normalization and Activation Operations

    public Tensor<T> Softmax<T>(Tensor<T> input, int axis = -1)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = input.Shape;
        int rank = shape.Length;

        // Handle negative axis
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {rank} dimensions");

        var outputData = new T[input.Length];
        var inputData = input.ToArray();

        // Calculate the size of each dimension
        int outerSize = 1;
        for (int i = 0; i < axis; i++)
            outerSize *= shape[i];

        int axisSize = shape[axis];

        int innerSize = 1;
        for (int i = axis + 1; i < rank; i++)
            innerSize *= shape[i];

        // Apply softmax along the specified axis
        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Find max for numerical stability
            T max = inputData[outer * axisSize * innerSize + inner];
            for (int i = 1; i < axisSize; i++)
            {
                int inputIdx = outer * axisSize * innerSize + i * innerSize + inner;
                if (numOps.GreaterThan(inputData[inputIdx], max))
                    max = inputData[inputIdx];
            }

            // Compute exp(x - max) and sum
            T sum = numOps.Zero;
            var expValues = new T[axisSize];
            for (int i = 0; i < axisSize; i++)
            {
                int inputIdx = outer * axisSize * innerSize + i * innerSize + inner;
                expValues[i] = numOps.Exp(numOps.Subtract(inputData[inputIdx], max));
                sum = numOps.Add(sum, expValues[i]);
            }

            // Normalize
            for (int i = 0; i < axisSize; i++)
            {
                int outputIdx = outer * axisSize * innerSize + i * innerSize + inner;
                outputData[outputIdx] = numOps.Divide(expValues[i], sum);
            }
        });

        return new Tensor<T>(shape, new Vector<T>(outputData));
    }

    public Tensor<T> SoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis = -1)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = output.Shape;
        int rank = shape.Length;

        // Handle negative axis
        if (axis < 0) axis = rank + axis;

        var gradInputData = new T[output.Length];
        var gradOutputData = gradOutput.ToArray();
        var outputData = output.ToArray();

        // Calculate the size of each dimension
        int outerSize = 1;
        for (int i = 0; i < axis; i++)
            outerSize *= shape[i];

        int axisSize = shape[axis];

        int innerSize = 1;
        for (int i = axis + 1; i < rank; i++)
            innerSize *= shape[i];

        // Compute gradient: dL/dx_i = sum_j(dL/dy_j * dy_j/dx_i)
        // For softmax: dy_j/dx_i = y_j * (delta_ij - y_i)
        // So: dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Compute dot product: sum_j(dL/dy_j * y_j)
            T dotProduct = numOps.Zero;
            for (int j = 0; j < axisSize; j++)
            {
                int index = outer * axisSize * innerSize + j * innerSize + inner;
                dotProduct = numOps.Add(dotProduct, numOps.Multiply(gradOutputData[index], outputData[index]));
            }

            // Compute gradient for each element
            for (int i = 0; i < axisSize; i++)
            {
                int index = outer * axisSize * innerSize + i * innerSize + inner;
                // dL/dx_i = y_i * (dL/dy_i - dot_product)
                gradInputData[index] = numOps.Multiply(outputData[index],
                    numOps.Subtract(gradOutputData[index], dotProduct));
            }
        });

        return new Tensor<T>(shape, new Vector<T>(gradInputData));
    }

    public Tensor<T> BatchNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon,
        out Tensor<T> mean, out Tensor<T> variance)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = input.Shape;

        if (shape.Length != 2)
            throw new ArgumentException("BatchNorm expects 2D tensor [batch, features]");

        int batchSize = shape[0];
        int features = shape[1];

        var inputData = input.ToArray();
        var gammaData = gamma.ToArray();
        var betaData = beta.ToArray();

        var meanData = new T[features];
        var varianceData = new T[features];
        var outputData = new T[batchSize * features];

        T eps = numOps.FromDouble(epsilon);

        // Compute mean and variance for each feature
        for (int f = 0; f < features; f++)
        {
            // Mean
            T sum = numOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                sum = numOps.Add(sum, inputData[b * features + f]);
            }
            meanData[f] = numOps.Divide(sum, numOps.FromDouble(batchSize));

            // Variance
            T varSum = numOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                T diff = numOps.Subtract(inputData[b * features + f], meanData[f]);
                varSum = numOps.Add(varSum, numOps.Multiply(diff, diff));
            }
            varianceData[f] = numOps.Divide(varSum, numOps.FromDouble(batchSize));
        }

        // Normalize and apply scale/shift
        Parallel.For(0, batchSize, b =>
        {
            for (int f = 0; f < features; f++)
            {
                int idx = b * features + f;
                // x_norm = (x - mean) / sqrt(variance + eps)
                T xNorm = numOps.Divide(
                    numOps.Subtract(inputData[idx], meanData[f]),
                    numOps.Sqrt(numOps.Add(varianceData[f], eps)));
                // y = gamma * x_norm + beta
                outputData[idx] = numOps.Add(numOps.Multiply(gammaData[f], xNorm), betaData[f]);
            }
        });

        mean = new Tensor<T>([features], new Vector<T>(meanData));
        variance = new Tensor<T>([features], new Vector<T>(varianceData));
        return new Tensor<T>(shape, new Vector<T>(outputData));
    }

    public Tensor<T> BatchNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma,
        Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = input.Shape;

        int batchSize = shape[0];
        int features = shape[1];

        var gradOutputData = gradOutput.ToArray();
        var inputData = input.ToArray();
        var gammaData = gamma.ToArray();
        var meanData = mean.ToArray();
        var varianceData = variance.ToArray();

        var gradInputData = new T[batchSize * features];
        var gradGammaData = new T[features];
        var gradBetaData = new T[features];

        T eps = numOps.FromDouble(epsilon);
        T batchSizeT = numOps.FromDouble(batchSize);

        // Compute gradients for gamma and beta, and intermediate values
        for (int f = 0; f < features; f++)
        {
            T std = numOps.Sqrt(numOps.Add(varianceData[f], eps));
            T invStd = numOps.Divide(numOps.One, std);

            T sumGradBeta = numOps.Zero;
            T sumGradGamma = numOps.Zero;
            T sumGradXNorm = numOps.Zero;
            T sumGradXNormTimesXNorm = numOps.Zero;

            for (int b = 0; b < batchSize; b++)
            {
                int idx = b * features + f;
                T xNorm = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[f]), invStd);

                sumGradBeta = numOps.Add(sumGradBeta, gradOutputData[idx]);
                sumGradGamma = numOps.Add(sumGradGamma, numOps.Multiply(gradOutputData[idx], xNorm));
                T gradXNorm = numOps.Multiply(gradOutputData[idx], gammaData[f]);
                sumGradXNorm = numOps.Add(sumGradXNorm, gradXNorm);
                sumGradXNormTimesXNorm = numOps.Add(sumGradXNormTimesXNorm, numOps.Multiply(gradXNorm, xNorm));
            }

            gradBetaData[f] = sumGradBeta;
            gradGammaData[f] = sumGradGamma;

            // Compute gradient with respect to input
            for (int b = 0; b < batchSize; b++)
            {
                int idx = b * features + f;
                T xNorm = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[f]), invStd);
                T gradXNorm = numOps.Multiply(gradOutputData[idx], gammaData[f]);

                // dL/dx = (1/N) * gamma * invStd * (N * dL/dxNorm - sum(dL/dxNorm) - xNorm * sum(dL/dxNorm * xNorm))
                T term1 = numOps.Multiply(batchSizeT, gradXNorm);
                T term2 = sumGradXNorm;
                T term3 = numOps.Multiply(xNorm, sumGradXNormTimesXNorm);

                gradInputData[idx] = numOps.Multiply(
                    numOps.Divide(numOps.Multiply(gammaData[f], invStd), batchSizeT),
                    numOps.Subtract(numOps.Subtract(term1, term2), term3));
            }
        }

        gradGamma = new Tensor<T>([features], new Vector<T>(gradGammaData));
        gradBeta = new Tensor<T>([features], new Vector<T>(gradBetaData));
        return new Tensor<T>(shape, new Vector<T>(gradInputData));
    }

    public Tensor<T> LayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon,
        out Tensor<T> mean, out Tensor<T> variance)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = input.Shape;

        if (shape.Length != 2)
            throw new ArgumentException("LayerNorm expects 2D tensor [batch, features]");

        int batchSize = shape[0];
        int features = shape[1];

        var inputData = input.ToArray();
        var gammaData = gamma.ToArray();
        var betaData = beta.ToArray();

        var meanData = new T[batchSize];
        var varianceData = new T[batchSize];
        var outputData = new T[batchSize * features];

        T eps = numOps.FromDouble(epsilon);

        // Compute mean and variance for each sample (along features dimension)
        Parallel.For(0, batchSize, b =>
        {
            // Mean
            T sum = numOps.Zero;
            for (int f = 0; f < features; f++)
            {
                sum = numOps.Add(sum, inputData[b * features + f]);
            }
            meanData[b] = numOps.Divide(sum, numOps.FromDouble(features));

            // Variance
            T varSum = numOps.Zero;
            for (int f = 0; f < features; f++)
            {
                T diff = numOps.Subtract(inputData[b * features + f], meanData[b]);
                varSum = numOps.Add(varSum, numOps.Multiply(diff, diff));
            }
            varianceData[b] = numOps.Divide(varSum, numOps.FromDouble(features));

            // Normalize and apply scale/shift
            T std = numOps.Sqrt(numOps.Add(varianceData[b], eps));
            for (int f = 0; f < features; f++)
            {
                int idx = b * features + f;
                T xNorm = numOps.Divide(numOps.Subtract(inputData[idx], meanData[b]), std);
                outputData[idx] = numOps.Add(numOps.Multiply(gammaData[f], xNorm), betaData[f]);
            }
        });

        mean = new Tensor<T>([batchSize], new Vector<T>(meanData));
        variance = new Tensor<T>([batchSize], new Vector<T>(varianceData));
        return new Tensor<T>(shape, new Vector<T>(outputData));
    }

    public Tensor<T> LayerNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma,
        Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = input.Shape;

        int batchSize = shape[0];
        int features = shape[1];

        var gradOutputData = gradOutput.ToArray();
        var inputData = input.ToArray();
        var gammaData = gamma.ToArray();
        var meanData = mean.ToArray();
        var varianceData = variance.ToArray();

        var gradInputData = new T[batchSize * features];
        var gradGammaData = new T[features];
        var gradBetaData = new T[features];

        T eps = numOps.FromDouble(epsilon);
        T featuresT = numOps.FromDouble(features);

        // First compute gradGamma and gradBeta
        for (int f = 0; f < features; f++)
        {
            T sumGradBeta = numOps.Zero;
            T sumGradGamma = numOps.Zero;

            for (int b = 0; b < batchSize; b++)
            {
                int idx = b * features + f;
                T std = numOps.Sqrt(numOps.Add(varianceData[b], eps));
                T xNorm = numOps.Divide(numOps.Subtract(inputData[idx], meanData[b]), std);

                sumGradBeta = numOps.Add(sumGradBeta, gradOutputData[idx]);
                sumGradGamma = numOps.Add(sumGradGamma, numOps.Multiply(gradOutputData[idx], xNorm));
            }

            gradBetaData[f] = sumGradBeta;
            gradGammaData[f] = sumGradGamma;
        }

        // Compute gradient with respect to input
        Parallel.For(0, batchSize, b =>
        {
            T std = numOps.Sqrt(numOps.Add(varianceData[b], eps));
            T invStd = numOps.Divide(numOps.One, std);

            T sumGradXNorm = numOps.Zero;
            T sumGradXNormTimesXNorm = numOps.Zero;

            for (int f = 0; f < features; f++)
            {
                int idx = b * features + f;
                T xNorm = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[b]), invStd);
                T gradXNorm = numOps.Multiply(gradOutputData[idx], gammaData[f]);
                sumGradXNorm = numOps.Add(sumGradXNorm, gradXNorm);
                sumGradXNormTimesXNorm = numOps.Add(sumGradXNormTimesXNorm, numOps.Multiply(gradXNorm, xNorm));
            }

            for (int f = 0; f < features; f++)
            {
                int idx = b * features + f;
                T xNorm = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[b]), invStd);
                T gradXNorm = numOps.Multiply(gradOutputData[idx], gammaData[f]);

                T term1 = numOps.Multiply(featuresT, gradXNorm);
                T term2 = sumGradXNorm;
                T term3 = numOps.Multiply(xNorm, sumGradXNormTimesXNorm);

                gradInputData[idx] = numOps.Multiply(
                    numOps.Divide(invStd, featuresT),
                    numOps.Subtract(numOps.Subtract(term1, term2), term3));
            }
        });

        gradGamma = new Tensor<T>([features], new Vector<T>(gradGammaData));
        gradBeta = new Tensor<T>([features], new Vector<T>(gradBetaData));
        return new Tensor<T>(shape, new Vector<T>(gradInputData));
    }

    #endregion

    #region Reduction Operations

    public Tensor<T> ReduceMax<T>(Tensor<T> input, int[] axes, bool keepDims, out int[] maxIndices)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Shape;
        var inputData = input.ToArray();

        // Normalize axes
        var normalizedAxes = axes.Select(a => a < 0 ? inputShape.Length + a : a).OrderBy(a => a).ToArray();

        // Compute output shape
        var outputShapeList = new List<int>();
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (normalizedAxes.Contains(i))
            {
                if (keepDims) outputShapeList.Add(1);
            }
            else
            {
                outputShapeList.Add(inputShape[i]);
            }
        }
        var outputShape = outputShapeList.Count > 0 ? outputShapeList.ToArray() : [1];

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];
        maxIndices = new int[outputSize];

        // Initialize with minimum values
        T minVal = numOps.MinValue;
        for (int i = 0; i < outputSize; i++)
        {
            outputData[i] = minVal;
            maxIndices[i] = -1;
        }

        // Compute strides for input and output
        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(outputShape);

        // Iterate through all input elements
        for (int i = 0; i < input.Length; i++)
        {
            // Compute multi-dimensional index from flat index
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            // Compute output index by removing reduced dimensions
            var outputMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (keepDims) outputMultiIndex.Add(0);
                }
                else
                {
                    outputMultiIndex.Add(multiIndex[d]);
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], outputShape, outputStrides);

            if (numOps.GreaterThan(inputData[i], outputData[outputIdx]))
            {
                outputData[outputIdx] = inputData[i];
                maxIndices[outputIdx] = i;
            }
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    public Tensor<T> ReduceMaxBackward<T>(Tensor<T> gradOutput, int[] maxIndices, int[] inputShape)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int inputSize = inputShape.Aggregate(1, (a, b) => a * b);
        var gradInputData = new T[inputSize];

        // Initialize with zeros
        for (int i = 0; i < inputSize; i++)
            gradInputData[i] = numOps.Zero;

        var gradOutputData = gradOutput.ToArray();

        // Route gradients to max positions
        for (int i = 0; i < maxIndices.Length; i++)
        {
            if (maxIndices[i] >= 0 && maxIndices[i] < inputSize)
            {
                gradInputData[maxIndices[i]] = numOps.Add(gradInputData[maxIndices[i]], gradOutputData[i]);
            }
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    public Tensor<T> ReduceMean<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Shape;
        var inputData = input.ToArray();

        // Normalize axes
        var normalizedAxes = axes.Select(a => a < 0 ? inputShape.Length + a : a).OrderBy(a => a).ToArray();

        // Compute output shape
        var outputShapeList = new List<int>();
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (normalizedAxes.Contains(i))
            {
                if (keepDims) outputShapeList.Add(1);
            }
            else
            {
                outputShapeList.Add(inputShape[i]);
            }
        }
        var outputShape = outputShapeList.Count > 0 ? outputShapeList.ToArray() : [1];

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];
        var counts = new int[outputSize];

        // Initialize
        for (int i = 0; i < outputSize; i++)
        {
            outputData[i] = numOps.Zero;
            counts[i] = 0;
        }

        // Compute strides
        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(outputShape);

        // Sum values
        for (int i = 0; i < input.Length; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            var outputMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (keepDims) outputMultiIndex.Add(0);
                }
                else
                {
                    outputMultiIndex.Add(multiIndex[d]);
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], outputShape, outputStrides);
            outputData[outputIdx] = numOps.Add(outputData[outputIdx], inputData[i]);
            counts[outputIdx]++;
        }

        // Divide by count to get mean
        for (int i = 0; i < outputSize; i++)
        {
            if (counts[i] > 0)
            {
                outputData[i] = numOps.Divide(outputData[i], numOps.FromDouble(counts[i]));
            }
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    public Tensor<T> ReduceMeanBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] axes)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int inputSize = inputShape.Aggregate(1, (a, b) => a * b);
        var gradInputData = new T[inputSize];

        // Normalize axes
        var normalizedAxes = axes.Select(a => a < 0 ? inputShape.Length + a : a).ToArray();

        // Count elements in reduced dimensions
        int reduceCount = 1;
        foreach (var ax in normalizedAxes)
        {
            reduceCount *= inputShape[ax];
        }
        T scale = numOps.Divide(numOps.One, numOps.FromDouble(reduceCount));

        var gradOutputData = gradOutput.ToArray();
        var gradOutputShape = gradOutput.Shape;
        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(gradOutputShape);

        // Broadcast gradient to input shape
        for (int i = 0; i < inputSize; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            var outputMultiIndex = new List<int>();
            int d2 = 0;
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (d2 < gradOutputShape.Length && gradOutputShape[d2] == 1)
                    {
                        outputMultiIndex.Add(0);
                        d2++;
                    }
                }
                else
                {
                    if (d2 < gradOutputShape.Length)
                    {
                        outputMultiIndex.Add(multiIndex[d]);
                        d2++;
                    }
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);

            // Clamp to valid range
            while (outputMultiIndex.Count < gradOutputShape.Length)
                outputMultiIndex.Add(0);
            while (outputMultiIndex.Count > gradOutputShape.Length)
                outputMultiIndex.RemoveAt(outputMultiIndex.Count - 1);

            int outputIdx = Math.Min(MultiToFlatIndex([.. outputMultiIndex], gradOutputShape, outputStrides), gradOutputData.Length - 1);
            gradInputData[i] = numOps.Multiply(gradOutputData[outputIdx], scale);
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    // Helper methods for reduction operations
    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int[] FlatToMultiIndex(int flatIndex, int[] shape, int[] strides)
    {
        var multiIndex = new int[shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            multiIndex[i] = flatIndex / strides[i];
            flatIndex %= strides[i];
        }
        return multiIndex;
    }

    private static int MultiToFlatIndex(int[] multiIndex, int[] shape, int[] strides)
    {
        int flatIndex = 0;
        for (int i = 0; i < multiIndex.Length; i++)
        {
            flatIndex += multiIndex[i] * strides[i];
        }
        return flatIndex;
    }

    #endregion

    #region Spatial Operations

    public Tensor<T> Upsample<T>(Tensor<T> input, int scaleH, int scaleW)
    {
        var shape = input.Shape;
        if (shape.Length != 4)
            throw new ArgumentException("Upsample expects 4D tensor [batch, channels, height, width]");

        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];

        int newHeight = height * scaleH;
        int newWidth = width * scaleW;

        var inputData = input.ToArray();
        var outputData = new T[batch * channels * newHeight * newWidth];

        // Nearest neighbor upsampling
        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int oh = 0; oh < newHeight; oh++)
            {
                int ih = oh / scaleH;
                for (int ow = 0; ow < newWidth; ow++)
                {
                    int iw = ow / scaleW;
                    int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                    int outputIdx = ((b * channels + c) * newHeight + oh) * newWidth + ow;
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        });

        return new Tensor<T>([batch, channels, newHeight, newWidth], new Vector<T>(outputData));
    }

    public Tensor<T> UpsampleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int scaleH, int scaleW)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int newHeight = height * scaleH;
        int newWidth = width * scaleW;

        var gradOutputData = gradOutput.ToArray();
        var gradInputData = new T[batch * channels * height * width];

        // Initialize to zero
        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        // Sum gradients from all positions that map to each input position
        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int oh = 0; oh < newHeight; oh++)
            {
                int ih = oh / scaleH;
                for (int ow = 0; ow < newWidth; ow++)
                {
                    int iw = ow / scaleW;
                    int gradOutputIdx = ((b * channels + c) * newHeight + oh) * newWidth + ow;
                    int gradInputIdx = ((b * channels + c) * height + ih) * width + iw;

                    lock (gradInputData)
                    {
                        gradInputData[gradInputIdx] = numOps.Add(gradInputData[gradInputIdx], gradOutputData[gradOutputIdx]);
                    }
                }
            }
        });

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    public Tensor<T> PixelShuffle<T>(Tensor<T> input, int upscaleFactor)
    {
        var shape = input.Shape;
        if (shape.Length != 4)
            throw new ArgumentException("PixelShuffle expects 4D tensor [batch, channels, height, width]");

        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];

        int r = upscaleFactor;
        if (channels % (r * r) != 0)
            throw new ArgumentException($"Number of channels ({channels}) must be divisible by r^2 ({r * r})");

        int newChannels = channels / (r * r);
        int newHeight = height * r;
        int newWidth = width * r;

        var inputData = input.ToArray();
        var outputData = new T[batch * newChannels * newHeight * newWidth];

        // Rearrange channels to spatial dimensions
        Parallel.For(0, batch, b =>
        {
            for (int oc = 0; oc < newChannels; oc++)
            {
                for (int oh = 0; oh < newHeight; oh++)
                {
                    for (int ow = 0; ow < newWidth; ow++)
                    {
                        int ih = oh / r;
                        int iw = ow / r;
                        int subH = oh % r;
                        int subW = ow % r;
                        int ic = oc * r * r + subH * r + subW;

                        int inputIdx = ((b * channels + ic) * height + ih) * width + iw;
                        int outputIdx = ((b * newChannels + oc) * newHeight + oh) * newWidth + ow;
                        outputData[outputIdx] = inputData[inputIdx];
                    }
                }
            }
        });

        return new Tensor<T>([batch, newChannels, newHeight, newWidth], new Vector<T>(outputData));
    }

    public Tensor<T> PixelShuffleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int upscaleFactor)
    {
        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int r = upscaleFactor;
        int newChannels = channels / (r * r);
        int newHeight = height * r;
        int newWidth = width * r;

        var gradOutputData = gradOutput.ToArray();
        var gradInputData = new T[batch * channels * height * width];

        // Reverse the rearrangement
        Parallel.For(0, batch, b =>
        {
            for (int oc = 0; oc < newChannels; oc++)
            {
                for (int oh = 0; oh < newHeight; oh++)
                {
                    for (int ow = 0; ow < newWidth; ow++)
                    {
                        int ih = oh / r;
                        int iw = ow / r;
                        int subH = oh % r;
                        int subW = ow % r;
                        int ic = oc * r * r + subH * r + subW;

                        int gradInputIdx = ((b * channels + ic) * height + ih) * width + iw;
                        int gradOutputIdx = ((b * newChannels + oc) * newHeight + oh) * newWidth + ow;
                        gradInputData[gradInputIdx] = gradOutputData[gradOutputIdx];
                    }
                }
            }
        });

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    public Tensor<T> Crop<T>(Tensor<T> input, int top, int left, int height, int width)
    {
        var shape = input.Shape;
        if (shape.Length != 4)
            throw new ArgumentException("Crop expects 4D tensor [batch, channels, height, width]");

        int batch = shape[0];
        int channels = shape[1];
        int inputHeight = shape[2];
        int inputWidth = shape[3];

        if (top < 0 || left < 0 || top + height > inputHeight || left + width > inputWidth)
            throw new ArgumentException("Crop region is out of bounds");

        var inputData = input.ToArray();
        var outputData = new T[batch * channels * height * width];

        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int oh = 0; oh < height; oh++)
            {
                int ih = top + oh;
                for (int ow = 0; ow < width; ow++)
                {
                    int iw = left + ow;
                    int inputIdx = ((b * channels + c) * inputHeight + ih) * inputWidth + iw;
                    int outputIdx = ((b * channels + c) * height + oh) * width + ow;
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        });

        return new Tensor<T>([batch, channels, height, width], new Vector<T>(outputData));
    }

    public Tensor<T> CropBackward<T>(Tensor<T> gradOutput, int[] inputShape, int top, int left)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];

        var gradOutputShape = gradOutput.Shape;
        int cropHeight = gradOutputShape[2];
        int cropWidth = gradOutputShape[3];

        var gradOutputData = gradOutput.ToArray();
        var gradInputData = new T[batch * channels * inputHeight * inputWidth];

        // Initialize to zero
        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        // Copy gradients to the cropped region
        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int oh = 0; oh < cropHeight; oh++)
            {
                int ih = top + oh;
                for (int ow = 0; ow < cropWidth; ow++)
                {
                    int iw = left + ow;
                    int gradOutputIdx = ((b * channels + c) * cropHeight + oh) * cropWidth + ow;
                    int gradInputIdx = ((b * channels + c) * inputHeight + ih) * inputWidth + iw;
                    gradInputData[gradInputIdx] = gradOutputData[gradOutputIdx];
                }
            }
        });

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    public Tensor<T> Pad<T>(Tensor<T> input, int padTop, int padBottom, int padLeft, int padRight, T padValue)
    {
        var shape = input.Shape;
        if (shape.Length < 2)
            throw new ArgumentException("Pad expects at least 2D tensor");

        // Assume last two dimensions are height and width
        int rank = shape.Length;
        int height = shape[rank - 2];
        int width = shape[rank - 1];

        int newHeight = height + padTop + padBottom;
        int newWidth = width + padLeft + padRight;

        // Calculate batch dimensions
        int batchSize = 1;
        for (int i = 0; i < rank - 2; i++)
            batchSize *= shape[i];

        var inputData = input.ToArray();
        var outputData = new T[batchSize * newHeight * newWidth];

        // Initialize with pad value
        for (int i = 0; i < outputData.Length; i++)
            outputData[i] = padValue;

        // Copy input data
        Parallel.For(0, batchSize, b =>
        {
            for (int ih = 0; ih < height; ih++)
            {
                int oh = ih + padTop;
                for (int iw = 0; iw < width; iw++)
                {
                    int ow = iw + padLeft;
                    int inputIdx = b * height * width + ih * width + iw;
                    int outputIdx = b * newHeight * newWidth + oh * newWidth + ow;
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        });

        var newShape = (int[])shape.Clone();
        newShape[rank - 2] = newHeight;
        newShape[rank - 1] = newWidth;

        return new Tensor<T>(newShape, new Vector<T>(outputData));
    }

    public Tensor<T> PadBackward<T>(Tensor<T> gradOutput, int padTop, int padLeft, int[] inputShape)
    {
        int rank = inputShape.Length;
        int height = inputShape[rank - 2];
        int width = inputShape[rank - 1];

        int batchSize = 1;
        for (int i = 0; i < rank - 2; i++)
            batchSize *= inputShape[i];

        var gradOutputShape = gradOutput.Shape;
        int paddedHeight = gradOutputShape[rank - 2];
        int paddedWidth = gradOutputShape[rank - 1];

        var gradOutputData = gradOutput.ToArray();
        var gradInputData = new T[batchSize * height * width];

        // Extract gradient from padded region
        Parallel.For(0, batchSize, b =>
        {
            for (int ih = 0; ih < height; ih++)
            {
                int oh = ih + padTop;
                for (int iw = 0; iw < width; iw++)
                {
                    int ow = iw + padLeft;
                    int gradOutputIdx = b * paddedHeight * paddedWidth + oh * paddedWidth + ow;
                    int gradInputIdx = b * height * width + ih * width + iw;
                    gradInputData[gradInputIdx] = gradOutputData[gradOutputIdx];
                }
            }
        });

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    public Tensor<T> Concat<T>(IReadOnlyList<Tensor<T>> tensors, int axis)
    {
        if (tensors == null || tensors.Count == 0)
            throw new ArgumentException("At least one tensor required for concatenation");

        var firstShape = tensors[0].Shape;
        int rank = firstShape.Length;

        // Normalize axis
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {rank} dimensions");

        // Validate shapes and compute total size along concatenation axis
        int totalAxisSize = 0;
        foreach (var tensor in tensors)
        {
            if (tensor.Shape.Length != rank)
                throw new ArgumentException("All tensors must have the same number of dimensions");

            for (int i = 0; i < rank; i++)
            {
                if (i != axis && tensor.Shape[i] != firstShape[i])
                    throw new ArgumentException($"All tensors must have the same shape except along axis {axis}");
            }

            totalAxisSize += tensor.Shape[axis];
        }

        // Build output shape
        var outputShape = (int[])firstShape.Clone();
        outputShape[axis] = totalAxisSize;

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];

        // Compute strides
        var outputStrides = ComputeStrides(outputShape);

        // Copy data from each tensor
        int axisOffset = 0;
        foreach (var tensor in tensors)
        {
            var tensorData = tensor.ToArray();
            var tensorShape = tensor.Shape;
            var tensorStrides = ComputeStrides(tensorShape);

            for (int i = 0; i < tensor.Length; i++)
            {
                var multiIndex = FlatToMultiIndex(i, tensorShape, tensorStrides);
                multiIndex[axis] += axisOffset;
                int outputIdx = MultiToFlatIndex(multiIndex, outputShape, outputStrides);
                outputData[outputIdx] = tensorData[i];
            }

            axisOffset += tensor.Shape[axis];
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    #endregion

    #region Activation Functions

    public Vector<T> Tanh<T>(Vector<T> vector)
    {
        // Use SIMD-optimized Tanh (3-6× speedup for float)
        return TensorPrimitivesHelper<T>.Tanh(vector);
    }

    public Vector<T> Sigmoid<T>(Vector<T> vector)
    {
        // Use SIMD-optimized Sigmoid (3-6× speedup for float)
        return TensorPrimitivesHelper<T>.Sigmoid(vector);
    }

    public Vector<T> ReLU<T>(Vector<T> vector)
    {
        // ReLU(x) = max(0, x)
        // TensorPrimitives doesn't have ReLU directly, but has Max
        // For now, use element-wise max with zero
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputArray = vector.ToArray();
        var outputArray = new T[inputArray.Length];

        // For float, we could use TensorPrimitives.Max with scalar zero
        // For now, manual implementation that works for all types
        for (int i = 0; i < inputArray.Length; i++)
        {
            outputArray[i] = numOps.GreaterThan(inputArray[i], numOps.Zero)
                ? inputArray[i]
                : numOps.Zero;
        }

        return new Vector<T>(outputArray);
    }

    public Tensor<T> Tanh<T>(Tensor<T> tensor)
    {
        // Convert tensor to vector, apply SIMD-optimized Tanh, convert back
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Tanh(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> Sigmoid<T>(Tensor<T> tensor)
    {
        // Convert tensor to vector, apply SIMD-optimized Sigmoid, convert back
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Sigmoid(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> ReLU<T>(Tensor<T> tensor)
    {
        // ReLU(x) = max(0, x)
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputArray = tensor.ToArray();
        var outputArray = new T[inputArray.Length];

        // Manual implementation that works for all types
        for (int i = 0; i < inputArray.Length; i++)
        {
            outputArray[i] = numOps.GreaterThan(inputArray[i], numOps.Zero)
                ? inputArray[i]
                : numOps.Zero;
        }

        return new Tensor<T>(tensor.Shape, new Vector<T>(outputArray));
    }

    public Vector<T> GELU<T>(Vector<T> vector)
    {
        return TensorPrimitivesHelper<T>.GELU(vector);
    }

    public Vector<T> Mish<T>(Vector<T> vector)
    {
        return TensorPrimitivesHelper<T>.Mish(vector);
    }

    public Vector<T> Swish<T>(Vector<T> vector)
    {
        return TensorPrimitivesHelper<T>.Swish(vector);
    }

    public Vector<T> ELU<T>(Vector<T> vector, double alpha = 1.0)
    {
        return TensorPrimitivesHelper<T>.ELU(vector, alpha);
    }

    public Tensor<T> GELU<T>(Tensor<T> tensor)
    {
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.GELU(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> Mish<T>(Tensor<T> tensor)
    {
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Mish(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> Swish<T>(Tensor<T> tensor)
    {
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Swish(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> ELU<T>(Tensor<T> tensor, double alpha = 1.0)
    {
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.ELU(flatVector, alpha);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    #endregion
}
