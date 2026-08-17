using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// Small matrix utilities the control algorithms share.
/// </summary>
/// <remarks>
/// <para>
/// These sit here rather than on <see cref="Matrix{T}"/> because they encode conventions specific to
/// control — symmetrizing a matrix that theory guarantees is symmetric, measuring a residual the way
/// a convergence test needs it — rather than general linear algebra.
/// </para>
/// </remarks>
internal static class ControlMath<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Inverts a square matrix, or returns <c>null</c> when it is singular.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The result is checked for finiteness rather than the factorization merely being allowed to
    /// throw. An LU factorization of a singular matrix divides by a zero pivot, and in floating
    /// point that produces infinities and NaNs instead of an exception — which would then propagate
    /// silently through every subsequent matrix product and turn a detectable "this problem has no
    /// solution" into a returned answer made entirely of NaN.
    /// </para>
    /// </remarks>
    public static Matrix<T>? TryInvert(Matrix<T> matrix)
    {
        Matrix<T> inverse;
        try
        {
            inverse = new LuDecomposition<T>(matrix).Invert();
        }
        catch (Exception)
        {
            return null;
        }

        for (int r = 0; r < inverse.Rows; r++)
        {
            for (int c = 0; c < inverse.Columns; c++)
            {
                double value = NumOps.ToDouble(inverse[r, c]);
                if (double.IsNaN(value) || double.IsInfinity(value)) return null;
            }
        }

        return inverse;
    }

    /// <summary>
    /// Averages a matrix with its own transpose.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The solution of a Riccati equation is symmetric as a matter of theory, but rounding makes the
    /// computed one very slightly asymmetric. Left alone that asymmetry compounds across iterations;
    /// projecting it back onto the symmetric matrices each step costs nothing and keeps the iteration
    /// on the surface it is supposed to live on.
    /// </para>
    /// </remarks>
    public static Matrix<T> Symmetrize(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var result = new Matrix<T>(n, n);
        T half = NumOps.FromDouble(0.5);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = NumOps.Multiply(half, NumOps.Add(matrix[i, j], matrix[j, i]));
            }
        }

        return result;
    }

    /// <summary>
    /// Returns the Frobenius norm of a matrix.
    /// </summary>
    public static double FrobeniusNorm(Matrix<T> matrix)
    {
        double total = 0.0;
        for (int r = 0; r < matrix.Rows; r++)
        {
            for (int c = 0; c < matrix.Columns; c++)
            {
                double value = NumOps.ToDouble(matrix[r, c]);
                total += value * value;
            }
        }

        return Math.Sqrt(total);
    }

    /// <summary>
    /// Returns the Frobenius norm of the difference between two matrices.
    /// </summary>
    public static double Distance(Matrix<T> left, Matrix<T> right)
    {
        double total = 0.0;
        for (int r = 0; r < left.Rows; r++)
        {
            for (int c = 0; c < left.Columns; c++)
            {
                double difference =
                    NumOps.ToDouble(left[r, c]) - NumOps.ToDouble(right[r, c]);
                total += difference * difference;
            }
        }

        return Math.Sqrt(total);
    }

    /// <summary>
    /// Subtracts one matrix from another.
    /// </summary>
    public static Matrix<T> Subtract(Matrix<T> left, Matrix<T> right)
    {
        var result = new Matrix<T>(left.Rows, left.Columns);
        for (int r = 0; r < left.Rows; r++)
        {
            for (int c = 0; c < left.Columns; c++)
            {
                result[r, c] = NumOps.Subtract(left[r, c], right[r, c]);
            }
        }

        return result;
    }

    /// <summary>
    /// Adds two matrices.
    /// </summary>
    public static Matrix<T> Add(Matrix<T> left, Matrix<T> right)
    {
        var result = new Matrix<T>(left.Rows, left.Columns);
        for (int r = 0; r < left.Rows; r++)
        {
            for (int c = 0; c < left.Columns; c++)
            {
                result[r, c] = NumOps.Add(left[r, c], right[r, c]);
            }
        }

        return result;
    }

    /// <summary>
    /// Adds two vectors.
    /// </summary>
    public static Vector<T> Add(Vector<T> left, Vector<T> right)
    {
        var result = new Vector<T>(left.Length);
        for (int i = 0; i < left.Length; i++) result[i] = NumOps.Add(left[i], right[i]);
        return result;
    }

    /// <summary>
    /// Subtracts one vector from another.
    /// </summary>
    public static Vector<T> Subtract(Vector<T> left, Vector<T> right)
    {
        var result = new Vector<T>(left.Length);
        for (int i = 0; i < left.Length; i++) result[i] = NumOps.Subtract(left[i], right[i]);
        return result;
    }

    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    public static Matrix<T> Multiply(Matrix<T> left, Matrix<T> right)
    {
        var result = new Matrix<T>(left.Rows, right.Columns);
        for (int r = 0; r < left.Rows; r++)
        {
            for (int c = 0; c < right.Columns; c++)
            {
                T accumulator = NumOps.Zero;
                for (int k = 0; k < left.Columns; k++)
                {
                    accumulator = NumOps.Add(
                        accumulator, NumOps.Multiply(left[r, k], right[k, c]));
                }

                result[r, c] = accumulator;
            }
        }

        return result;
    }

    /// <summary>
    /// Scales every entry of a matrix.
    /// </summary>
    public static Matrix<T> Scale(Matrix<T> matrix, double factor)
    {
        T scale = NumOps.FromDouble(factor);
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);
        for (int r = 0; r < matrix.Rows; r++)
        {
            for (int c = 0; c < matrix.Columns; c++)
            {
                result[r, c] = NumOps.Multiply(scale, matrix[r, c]);
            }
        }

        return result;
    }

    /// <summary>
    /// Transposes a matrix.
    /// </summary>
    public static Matrix<T> Transpose(Matrix<T> matrix)
    {
        var result = new Matrix<T>(matrix.Columns, matrix.Rows);
        for (int r = 0; r < matrix.Rows; r++)
        {
            for (int c = 0; c < matrix.Columns; c++) result[c, r] = matrix[r, c];
        }

        return result;
    }

    /// <summary>
    /// Multiplies a matrix by a vector.
    /// </summary>
    public static Vector<T> Multiply(Matrix<T> matrix, Vector<T> vector)
    {
        var result = new Vector<T>(matrix.Rows);
        for (int r = 0; r < matrix.Rows; r++)
        {
            T accumulator = NumOps.Zero;
            for (int c = 0; c < matrix.Columns; c++)
            {
                accumulator = NumOps.Add(accumulator, NumOps.Multiply(matrix[r, c], vector[c]));
            }

            result[r] = accumulator;
        }

        return result;
    }

    /// <summary>
    /// Copies an <c>n</c>-by-<c>n</c> block out of a larger matrix.
    /// </summary>
    public static Matrix<T> Block(Matrix<T> matrix, int rowOffset, int columnOffset, int size)
    {
        var result = new Matrix<T>(size, size);
        for (int r = 0; r < size; r++)
        {
            for (int c = 0; c < size; c++)
            {
                result[r, c] = matrix[rowOffset + r, columnOffset + c];
            }
        }

        return result;
    }
}
