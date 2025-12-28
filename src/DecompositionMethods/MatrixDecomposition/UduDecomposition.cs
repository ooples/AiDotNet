namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Represents a UDU' decomposition of a matrix, which factorizes a symmetric matrix A into U*D*U',
/// where U is an upper triangular matrix with ones on the diagonal, D is a diagonal matrix,
/// and U' is the transpose of U.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float)</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> UDU' decomposition is a way to break down a complex matrix into simpler parts.
/// Think of it like factoring a number (e.g., 12 = 3 * 4). This decomposition is particularly
/// useful for solving systems of linear equations and for numerical stability in calculations.
/// </para>
/// </remarks>
public class UduDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the upper triangular matrix U from the decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An upper triangular matrix has non-zero values only on and above the main diagonal
    /// (the diagonal from top-left to bottom-right). All values below this diagonal are zero.
    /// </para>
    /// </remarks>
    public Matrix<T> U { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the diagonal matrix D represented as a vector of its diagonal elements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A diagonal matrix has non-zero values only along its main diagonal.
    /// We store it as a vector to save memory since all other values are zero.
    /// </para>
    /// </remarks>
    public Vector<T> D { get; private set; } = new Vector<T>(0);

    private readonly UduAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the UduDecomposition class and performs the decomposition.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <param name="algorithm">The algorithm to use for decomposition (default is Crout)</param>
    /// <exception cref="ArgumentException">Thrown when the matrix is not square</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your original matrix and immediately breaks it down
    /// into the U and D components. You can choose between two different methods (algorithms)
    /// for doing this decomposition.
    /// </para>
    /// </remarks>
    public UduDecomposition(Matrix<T> matrix, UduAlgorithmType algorithm = UduAlgorithmType.Crout)
        : base(matrix)
    {
        if (!matrix.IsSquareMatrix())
            throw new ArgumentException("Matrix must be square for UDU decomposition.");

        _algorithm = algorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the UDU' decomposition.
    /// </summary>
    protected override void Decompose()
    {
        int n = A.Rows;
        U = new Matrix<T>(n, n);
        D = new Vector<T>(n);

        ComputeDecomposition(_algorithm);
    }

    /// <summary>
    /// Computes the UDU' decomposition using the specified algorithm.
    /// </summary>
    /// <param name="algorithm">The algorithm to use for decomposition</param>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified</exception>
    private void ComputeDecomposition(UduAlgorithmType algorithm)
    {
        switch (algorithm)
        {
            case UduAlgorithmType.Crout:
                DecomposeCrout();
                break;
            case UduAlgorithmType.Doolittle:
                DecomposeDoolittle();
                break;
            default:
                throw new ArgumentException("Unsupported UDU decomposition algorithm.");
        }
    }

    /// <summary>
    /// Performs the UDU' decomposition using the Crout algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Crout algorithm is one method for breaking down a matrix into simpler parts.
    /// It works by calculating the elements of U and D in a specific order, starting from the top-left
    /// and working across rows and down columns.
    /// </para>
    /// </remarks>
    private void DecomposeCrout()
    {
        int n = A.Rows;

        // Work backwards from the last row for UDU^T decomposition
        for (int j = n - 1; j >= 0; j--)
        {
            // Set diagonal of U to 1
            U[j, j] = NumOps.One;

            // Calculate D[j] = A[j,j] - sum_{k=j+1}^{n-1} U[j,k]^2 * D[k]
            T sum = NumOps.Zero;
            for (int k = j + 1; k < n; k++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(U[j, k], U[j, k]), D[k]));
            }
            D[j] = NumOps.Subtract(A[j, j], sum);

            // Calculate U[i,j] for i < j
            for (int i = 0; i < j; i++)
            {
                // U[i,j] = (A[i,j] - sum_{k=j+1}^{n-1} U[i,k]*U[j,k]*D[k]) / D[j]
                sum = NumOps.Zero;
                for (int k = j + 1; k < n; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(U[i, k], U[j, k]), D[k]));
                }
                U[i, j] = NumOps.Divide(NumOps.Subtract(A[i, j], sum), D[j]);
            }
        }
    }

    /// <summary>
    /// Performs the UDU' decomposition using the Doolittle algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Doolittle algorithm is another method for matrix decomposition.
    /// It's similar to Crout but processes the matrix in a slightly different order.
    /// Different algorithms may be more efficient or stable depending on the specific matrix.
    /// </para>
    /// </remarks>
    private void DecomposeDoolittle()
    {
        int n = A.Rows;

        // Doolittle variant - also works backwards for UDU^T
        // Uses the same recurrence as Crout but with a different initialization strategy
        for (int j = n - 1; j >= 0; j--)
        {
            // Set diagonal of U to 1
            U[j, j] = NumOps.One;

            // Calculate D[j] = A[j,j] - sum_{k=j+1}^{n-1} U[j,k]^2 * D[k]
            T sum = NumOps.Zero;
            for (int k = j + 1; k < n; k++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(U[j, k], U[j, k]), D[k]));
            }
            D[j] = NumOps.Subtract(A[j, j], sum);

            // For each row i < j, calculate U[i,j]
            for (int i = j - 1; i >= 0; i--)
            {
                // U[i,j] = (A[i,j] - sum_{k=j+1}^{n-1} U[i,k]*U[j,k]*D[k]) / D[j]
                sum = NumOps.Zero;
                for (int k = j + 1; k < n; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(U[i, k], U[j, k]), D[k]));
                }
                U[i, j] = NumOps.Divide(NumOps.Subtract(A[i, j], sum), D[j]);
            }
        }
    }

    /// <summary>
    /// Solves the linear system Ax = b using the UDU' decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector</param>
    /// <returns>The solution vector x</returns>
    /// <exception cref="ArgumentException">Thrown when the length of vector b doesn't match the matrix dimensions</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the values of x in the equation Ax = b.
    /// Think of it like solving for x in the equation 3x = 6 (where x = 2),
    /// but with matrices instead of simple numbers. The decomposition makes
    /// this process more efficient by breaking it into three simpler steps:
    /// forward substitution, diagonal scaling, and backward substitution.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        if (b.Length != A.Rows)
            throw new ArgumentException("Vector b must have the same length as the number of rows in matrix A.");

        // Forward substitution
        Vector<T> y = new Vector<T>(b.Length);
        for (int i = 0; i < b.Length; i++)
        {
            // VECTORIZED: Use dot product for sum computation
            T sum = NumOps.Zero;
            if (i > 0)
            {
                var colSlice = new T[i];
                var ySlice = new T[i];
                for (int k = 0; k < i; k++)
                {
                    colSlice[k] = U[k, i];
                    ySlice[k] = y[k];
                }
                var colVec = new Vector<T>(colSlice);
                var yVec = new Vector<T>(ySlice);
                sum = colVec.DotProduct(yVec);
            }
            y[i] = NumOps.Subtract(b[i], sum);
        }

        // VECTORIZED: Diagonal scaling using vector division
        y = y.ElementwiseDivide(D);

        // Backward substitution
        Vector<T> x = new Vector<T>(b.Length);
        for (int i = b.Length - 1; i >= 0; i--)
        {
            // VECTORIZED: Use dot product for sum computation
            T sum = NumOps.Zero;
            if (i < b.Length - 1)
            {
                int remaining = b.Length - i - 1;
                var rowSlice = new T[remaining];
                var xSlice = new T[remaining];
                for (int k = 0; k < remaining; k++)
                {
                    rowSlice[k] = U[i, i + 1 + k];
                    xSlice[k] = x[i + 1 + k];
                }
                var rowVec = new Vector<T>(rowSlice);
                var xVec = new Vector<T>(xSlice);
                sum = rowVec.DotProduct(xVec);
            }
            x[i] = NumOps.Subtract(y[i], sum);
        }

        return x;
    }

    /// <summary>
    /// Calculates the inverse of the original matrix A using the UDU' decomposition.
    /// </summary>
    /// <returns>The inverse of matrix A</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number. Just as 1/3 is the
    /// reciprocal of 3 (because 3 * 1/3 = 1), the inverse of a matrix A is another matrix that,
    /// when multiplied by A, gives the identity matrix (the matrix equivalent of the number 1).
    /// This method finds the inverse by solving multiple equation systems, one for each column.
    /// </para>
    /// </remarks>
    public override Matrix<T> Invert()
    {
        int n = A.Rows;
        Matrix<T> inverse = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            Vector<T> ei = new Vector<T>(n);
            ei[i] = NumOps.One;
            Vector<T> column = Solve(ei);
            for (int j = 0; j < n; j++)
            {
                inverse[j, i] = column[j];
            }
        }

        return inverse;
    }

    /// <summary>
    /// Returns the U and D factors from the UDU' decomposition.
    /// </summary>
    /// <returns>A tuple containing the U matrix and D vector that represent the decomposition factors</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method gives you direct access to the two main components of the decomposition:
    /// the U matrix and the D vector. These are the "building blocks" that, when combined properly,
    /// can reconstruct your original matrix. Think of it like getting the individual ingredients
    /// that went into making a cake. You might need these separate components for further
    /// mathematical operations or to better understand the structure of your original matrix.
    /// </para>
    /// </remarks>
    public (Matrix<T> U, Vector<T> D) GetFactors()
    {
        return (U, D);
    }
}
