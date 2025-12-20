global using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements LU decomposition for matrices, which factorizes a matrix into a product of lower and upper triangular matrices.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// LU decomposition factors a matrix A into the product of a lower triangular matrix L and an upper
/// triangular matrix U, often with row permutations represented by a permutation matrix P (P*A = L*U).
/// This decomposition is fundamental in numerical linear algebra and is used for solving linear systems,
/// computing determinants, and matrix inversion.
/// </para>
/// <para>
/// <b>For Beginners:</b> LU decomposition breaks a matrix into two simpler triangular matrices:
/// L (lower triangular, with values only on and below the diagonal) and U (upper triangular, with
/// values only on and above the diagonal). Think of it like factoring a number into simpler parts,
/// but for matrices. This makes solving equations much faster.
/// </para>
/// <para>
/// Real-world applications:
/// - Solving systems of linear equations efficiently
/// - Computing matrix determinants
/// - Finding matrix inverses in numerical algorithms
/// </para>
/// </remarks>
public class LuDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the lower triangular matrix from the decomposition.
    /// </summary>
    public Matrix<T> L { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the upper triangular matrix from the decomposition.
    /// </summary>
    public Matrix<T> U { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the permutation vector that tracks row exchanges during pivoting.
    /// Each value represents the original row index for the current row position.
    /// </summary>
    public Vector<int> P { get; private set; } = new Vector<int>(0);

    /// <summary>
    /// The algorithm to use for LU decomposition.
    /// </summary>
    private readonly LuAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the LuDecomposition class and performs the decomposition.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="luAlgorithm">The algorithm to use for LU decomposition. Defaults to partial pivoting.</param>
    /// <remarks>
    /// <para>
    /// LU decomposition factorizes a matrix A into the product of a lower triangular matrix L and an upper triangular matrix U,
    /// possibly with row permutations (P*A = L*U). This is useful for solving linear systems and calculating determinants.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your input matrix and immediately breaks it down into
    /// the L and U components using the specified algorithm. Different algorithms have different trade-offs
    /// in terms of speed and numerical stability.
    /// </para>
    /// </remarks>
    public LuDecomposition(Matrix<T> matrix, LuAlgorithmType luAlgorithm = LuAlgorithmType.PartialPivoting)
        : base(matrix)
    {
        _algorithm = luAlgorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the LU decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (L, U, P) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Solves the linear system Ax = b using the LU decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// This method solves the equation in two steps:
    /// 1. Forward substitution to solve Ly = Pb
    /// 2. Back substitution to solve Ux = y
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        var pb = PermutateVector(b, P);
        var y = ForwardSubstitution(L, pb);

        return BackSubstitution(U, y);
    }

    /// <summary>
    /// Performs the matrix decomposition using the specified algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <returns>A tuple containing the L matrix, U matrix, and permutation vector P.</returns>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeDecomposition(Matrix<T> matrix, LuAlgorithmType algorithm)
    {
        return algorithm switch
        {
            LuAlgorithmType.Doolittle => ComputeLuDoolittle(matrix),
            LuAlgorithmType.Crout => ComputeLuCrout(matrix),
            LuAlgorithmType.PartialPivoting => ComputeLuPartialPivoting(matrix),
            LuAlgorithmType.CompletePivoting => ComputeLuCompletePivoting(matrix),
            LuAlgorithmType.Cholesky => ComputeCholesky(matrix),
            _ => throw new ArgumentException("Unsupported LU decomposition algorithm."),
        };
    }

    /// <summary>
    /// Computes LU decomposition with partial pivoting (Gaussian elimination with row pivoting).
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the L matrix, U matrix, and permutation vector P.</returns>
    /// <remarks>
    /// Partial pivoting selects the largest element in the current column as the pivot,
    /// which improves numerical stability by reducing round-off errors.
    /// </remarks>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuPartialPivoting(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        Matrix<T> A = matrix.Clone();
        Matrix<T> L = new(n, n);
        Vector<int> P = new(n);

        // Initialize permutation vector to identity
        for (int i = 0; i < n; i++)
            P[i] = i;

        for (int k = 0; k < n - 1; k++)
        {
            // Find pivot (largest absolute value in current column)
            int pivotRow = k;
            T pivotValue = NumOps.Abs(A[k, k]);
            for (int i = k + 1; i < n; i++)
            {
                T absValue = NumOps.Abs(A[i, k]);
                if (NumOps.GreaterThan(absValue, pivotValue))
                {
                    pivotRow = i;
                    pivotValue = absValue;
                }
            }

            // VECTORIZED: Use Engine.SwapRows for efficient row swapping
            if (pivotRow != k)
            {
                Engine.SwapRows(A, k, pivotRow);
                (P[pivotRow], P[k]) = (P[k], P[pivotRow]);
            }

            // VECTORIZED: Perform elimination using Engine operations
            for (int i = k + 1; i < n; i++)
            {
                T factor = NumOps.Divide(A[i, k], A[k, k]);
                L[i, k] = factor;
                // VECTORIZED: A[i,:] = A[i,:] - factor * A[k,:]
                var rowK = Engine.GetRow(A, k);
                var rowI = Engine.GetRow(A, i);
                var scaled = (Vector<T>)Engine.Multiply(rowK, factor);
                var updated = (Vector<T>)Engine.Subtract(rowI, scaled);
                Engine.SetRow(A, i, updated);
            }
        }

        // Extract L and U matrices from the modified A
        Matrix<T> U = new(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i > j)
                    L[i, j] = A[i, j];
                else if (i == j)
                {
                    L[i, j] = NumOps.One;
                    U[i, j] = A[i, j];  // Also copy diagonal to U
                }
                else
                    U[i, j] = A[i, j];
            }
        }

        return (L, U, P);
    }

    /// - U: Upper triangular matrix
    /// - P: Permutation vector for rows
    /// </returns>
    /// <remarks>
    /// Complete pivoting searches for the largest element in the entire remaining submatrix
    /// to use as a pivot. This improves numerical stability compared to partial pivoting.
    /// The method rearranges both rows and columns during the decomposition process.
    /// </remarks>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuCompletePivoting(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        Matrix<T> A = matrix.Clone();
        Matrix<T> L = new(n, n);
        Vector<int> P = new(n);
        Vector<int> Q = new(n);

        for (int i = 0; i < n; i++)
        {
            P[i] = i;
            Q[i] = i;
        }

        for (int k = 0; k < n - 1; k++)
        {
            int pivotRow = k, pivotCol = k;
            T pivotValue = NumOps.Abs(A[k, k]);

            for (int i = k; i < n; i++)
            {
                for (int j = k; j < n; j++)
                {
                    T absValue = NumOps.Abs(A[i, j]);
                    if (NumOps.GreaterThan(absValue, pivotValue))
                    {
                        pivotRow = i;
                        pivotCol = j;
                        pivotValue = absValue;
                    }
                }
            }

            // VECTORIZED: Use Engine.SwapRows for efficient row swapping
            if (pivotRow != k)
            {
                Engine.SwapRows(A, k, pivotRow);
                (P[pivotRow], P[k]) = (P[k], P[pivotRow]);
            }

            // VECTORIZED: Use Engine.SwapColumns for efficient column swapping
            if (pivotCol != k)
            {
                Engine.SwapColumns(A, k, pivotCol);
                (Q[pivotCol], Q[k]) = (Q[k], Q[pivotCol]);
            }

            // VECTORIZED: Use Engine operations for row updates
            for (int i = k + 1; i < n; i++)
            {
                T factor = NumOps.Divide(A[i, k], A[k, k]);
                L[i, k] = factor;
                // VECTORIZED: A[i,:] = A[i,:] - factor * A[k,:]
                var rowK = Engine.GetRow(A, k);
                var rowI = Engine.GetRow(A, i);
                var scaled = (Vector<T>)Engine.Multiply(rowK, factor);
                var updated = (Vector<T>)Engine.Subtract(rowI, scaled);
                Engine.SetRow(A, i, updated);
            }
        }

        Matrix<T> U = new(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i > j)
                    L[i, j] = A[i, j];
                else if (i == j)
                {
                    L[i, j] = NumOps.One;
                    U[i, j] = A[i, j];  // Also copy diagonal to U
                }
                else
                    U[i, j] = A[i, j];
            }
        }

        // Adjust U and P for column permutations
        Matrix<T> adjustedU = new(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                adjustedU[i, Q[j]] = U[i, j];
            }
        }

        return (L, adjustedU, P);
    }

    /// <summary>
    /// Computes the Cholesky decomposition of a symmetric positive-definite matrix.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose (must be symmetric positive-definite).</param>
    /// <returns>
    /// A tuple containing:
    /// - L: Lower triangular matrix where L * L^T = matrix
    /// - U: Upper triangular matrix (transpose of L)
    /// - P: Identity permutation vector
    /// </returns>
    /// <remarks>
    /// Cholesky decomposition is a special form of LU decomposition for symmetric positive-definite matrices.
    /// It is more efficient than standard LU decomposition for these types of matrices.
    /// A symmetric positive-definite matrix is a matrix that is symmetric (equal to its transpose)
    /// and has all positive eigenvalues (a mathematical property that ensures stability).
    /// </remarks>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeCholesky(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");

        Matrix<T> L = new(n, n);
        Vector<int> P = new(n);

        for (int i = 0; i < n; i++)
            P[i] = i;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = NumOps.Zero;

                if (j == i)
                {
                    for (int k = 0; k < j; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(L[j, k], L[j, k]));
                    }
                    L[j, j] = NumOps.Sqrt(NumOps.Subtract(matrix[j, j], sum));
                }
                else
                {
                    for (int k = 0; k < j; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(L[i, k], L[j, k]));
                    }

                    L[i, j] = NumOps.Divide(NumOps.Subtract(matrix[i, j], sum), L[j, j]);
                }
            }
        }

        Matrix<T> U = L.Transpose();

        return (L, U, P);
    }

    /// <summary>
    /// Computes the LU decomposition using Doolittle's method.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose.</param>
    /// <returns>
    /// A tuple containing:
    /// - L: Lower triangular matrix with 1's on the diagonal
    /// - U: Upper triangular matrix
    /// - P: Identity permutation vector
    /// </returns>
    /// <remarks>
    /// Doolittle's method is a form of LU decomposition where the diagonal elements of L are all 1's.
    /// This method does not use pivoting, so it may be less numerically stable for some matrices.
    /// It's best used when the matrix is known to be well-conditioned (not close to singular).
    /// </remarks>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuDoolittle(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        Matrix<T> L = new(n, n);
        Matrix<T> U = new(n, n);
        Vector<int> P = new(n);

        // Initialize P as [0, 1, 2, ..., n-1]
        for (int i = 0; i < n; i++)
            P[i] = i;

        for (int i = 0; i < n; i++)
        {
            // Upper Triangular
            for (int k = i; k < n; k++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < i; j++)
                    sum = NumOps.Add(sum, NumOps.Multiply(L[i, j], U[j, k]));
                U[i, k] = NumOps.Subtract(matrix[i, k], sum);
            }

            // Lower Triangular
            for (int k = i; k < n; k++)
            {
                if (i == k)
                    L[i, i] = NumOps.One;
                else
                {
                    T sum = NumOps.Zero;
                    for (int j = 0; j < i; j++)
                        sum = NumOps.Add(sum, NumOps.Multiply(L[k, j], U[j, i]));
                    L[k, i] = NumOps.Divide(NumOps.Subtract(matrix[k, i], sum), U[i, i]);
                }
            }
        }

        return (L, U, P);
    }

    /// <summary>
    /// Computes the LU decomposition using Crout's method.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose.</param>
    /// <returns>
    /// A tuple containing:
    /// - L: Lower triangular matrix
    /// - U: Upper triangular matrix with 1's on the diagonal
    /// - P: Identity permutation vector
    /// </returns>
    /// <remarks>
    /// <para>
    /// Crout's method is a form of LU decomposition where the diagonal elements of U are all 1's.
    /// This method does not use pivoting, so it may be less numerically stable for some matrices.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Crout's method is a way to break down a complex matrix into simpler parts.
    /// Think of it like factoring a number (e.g., 12 = 3 * 4). Here, we're factoring a matrix into 
    /// two triangular matrices - one with values only below the diagonal (L) and one with values 
    /// only above the diagonal and 1's on the diagonal itself (U). This makes solving equations much easier.
    /// </para>
    /// </remarks>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuCrout(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        Matrix<T> L = new(n, n);
        Matrix<T> U = new(n, n);
        Vector<int> P = new(n);

        // Initialize P as [0, 1, 2, ..., n-1]
        for (int i = 0; i < n; i++)
            P[i] = i;

        for (int j = 0; j < n; j++)
        {
            U[j, j] = NumOps.One;
        }

        for (int j = 0; j < n; j++)
        {
            for (int i = j; i < n; i++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(L[i, k], U[k, j]));
                }
                L[i, j] = NumOps.Subtract(matrix[i, j], sum);
            }

            for (int i = j; i < n; i++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(L[j, k], U[k, i]));
                }
                if (!NumOps.Equals(L[j, j], NumOps.Zero))
                {
                    U[j, i] = NumOps.Divide(NumOps.Subtract(matrix[j, i], sum), L[j, j]);
                }
                else
                {
                    U[j, i] = NumOps.Zero;
                }
            }
        }

        return (L, U, P);
    }

    /// <summary>
    /// Rearranges a vector according to the permutation vector P.
    /// </summary>
    /// <param name="b">The original vector to permutate.</param>
    /// <param name="P">The permutation vector that defines the new order.</param>
    /// <returns>The permutated vector.</returns>
    /// <remarks>
    /// <para>
    /// This method is used internally to apply row permutations to the right-hand side vector
    /// when solving linear systems.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When we rearrange rows in our matrix during decomposition, we need to 
    /// make the same changes to our right-hand side vector. This method does that rearrangement.
    /// It's like if you reordered the questions in a test, you'd need to reorder the answers too!
    /// </para>
    /// </remarks>
    private Vector<T> PermutateVector(Vector<T> b, Vector<int> P)
    {
        var pb = new Vector<T>(b.Length);
        for (int i = 0; i < b.Length; i++)
        {
            pb[i] = b[P[i]];
        }

        return pb;
    }

    /// <summary>
    /// Performs forward substitution to solve the system Ly = b where L is lower triangular.
    /// </summary>
    /// <param name="L">The lower triangular matrix.</param>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector y.</returns>
    /// <remarks>
    /// <para>
    /// Forward substitution is used to solve a system of equations where the coefficient matrix
    /// is lower triangular. It works by solving for one variable at a time, starting from the first.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Forward substitution is like solving a cascade of simple equations.
    /// Since L is lower triangular (has zeros above the diagonal), we can solve for the first 
    /// variable directly, then use that to solve for the second, and so on. It's like a domino 
    /// effect where each solution helps us find the next one.
    /// </para>
    /// </remarks>
    private Vector<T> ForwardSubstitution(Matrix<T> L, Vector<T> b)
    {
        var y = new Vector<T>(L.Rows);
        for (int i = 0; i < L.Rows; i++)
        {
            // VECTORIZED: Use dot product for sum computation
            T sum = NumOps.Zero;
            if (i > 0)
            {
                var rowSlice = new T[i];
                var ySlice = new T[i];
                for (int k = 0; k < i; k++)
                {
                    rowSlice[k] = L[i, k];
                    ySlice[k] = y[k];
                }
                var rowVec = new Vector<T>(rowSlice);
                var yVec = new Vector<T>(ySlice);
                sum = rowVec.DotProduct(yVec);
            }

            T rhs = NumOps.Subtract(b[i], sum);
            T diag = L[i, i];
            if (NumOps.Equals(diag, NumOps.Zero))
            {
                throw new InvalidOperationException("Lower triangular factor is singular.");
            }
            y[i] = NumOps.Divide(rhs, diag);
        }

        return y;
    }

    /// <summary>
    /// Performs back substitution to solve the system Ux = y where U is upper triangular.
    /// </summary>
    /// <param name="U">The upper triangular matrix.</param>
    /// <param name="y">The right-hand side vector.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <para>
    /// Back substitution is used to solve a system of equations where the coefficient matrix
    /// is upper triangular. It works by solving for one variable at a time, starting from the last.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Back substitution is like solving a waterfall of equations from bottom to top.
    /// Since U is upper triangular (has zeros below the diagonal), we can solve for the last 
    /// variable directly, then use that to solve for the second-to-last, and so on. It's the 
    /// reverse process of forward substitution, working backwards through our variables.
    /// </para>
    /// </remarks>
    private Vector<T> BackSubstitution(Matrix<T> U, Vector<T> y)
    {
        var x = new Vector<T>(U.Columns);
        for (int i = U.Columns - 1; i >= 0; i--)
        {
            // VECTORIZED: Use dot product for sum computation
            T sum = NumOps.Zero;
            if (i < U.Columns - 1)
            {
                int remaining = U.Columns - i - 1;
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

            x[i] = NumOps.Divide(NumOps.Subtract(y[i], sum), U[i, i]);
        }

        return x;
    }

    /// <summary>
    /// Calculates the inverse of the original matrix using the LU decomposition.
    /// </summary>
    /// <returns>The inverse of the original matrix.</returns>
    /// <remarks>
    /// Matrix inversion is computationally expensive and numerically less stable than directly
    /// solving a system. When possible, use the Solve method instead of calculating the inverse.
    /// </remarks>
    // Uses default implementation from base class (MatrixDecompositionBase.Invert)
}
