using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Exceptions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
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
    private static IEngine Engine => AiDotNetEngine.Current;

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
        catch (MatrixFactorizationException)
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
        => Engine.MatrixSubtract(left, right);

    /// <summary>
    /// Adds two matrices.
    /// </summary>
    public static Matrix<T> Add(Matrix<T> left, Matrix<T> right)
        => Engine.MatrixAdd(left, right);

    /// <summary>
    /// Adds two vectors.
    /// </summary>
    public static Vector<T> Add(Vector<T> left, Vector<T> right)
        => (Vector<T>)Engine.Add(left, right);

    /// <summary>
    /// Subtracts one vector from another.
    /// </summary>
    public static Vector<T> Subtract(Vector<T> left, Vector<T> right)
        => (Vector<T>)Engine.Subtract(left, right);

    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    public static Matrix<T> Multiply(Matrix<T> left, Matrix<T> right)
        => (Matrix<T>)Engine.MatrixMultiply(left, right);

    /// <summary>
    /// Scales every entry of a matrix.
    /// </summary>
    public static Matrix<T> Scale(Matrix<T> matrix, double factor)
        => Engine.MatrixMultiplyScalar(matrix, NumOps.FromDouble(factor));

    /// <summary>
    /// Scales every entry of a vector.
    /// </summary>
    public static Vector<T> Scale(Vector<T> vector, double factor)
        => vector.Multiply(NumOps.FromDouble(factor));

    /// <summary>
    /// Transposes a matrix.
    /// </summary>
    public static Matrix<T> Transpose(Matrix<T> matrix)
        => Engine.MatrixTranspose(matrix);

    /// <summary>
    /// Multiplies a matrix by a vector.
    /// </summary>
    public static Vector<T> Multiply(Matrix<T> matrix, Vector<T> vector)
        => Engine.MatrixVectorMultiply(matrix, vector);

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
