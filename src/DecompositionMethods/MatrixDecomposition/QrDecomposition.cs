namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs QR decomposition on a matrix, factoring it into an orthogonal matrix Q and an upper triangular matrix R.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// QR decomposition factors a matrix A into the product of an orthogonal matrix Q and an upper triangular
/// matrix R. This decomposition is widely used in solving linear systems, computing eigenvalues, and least
/// squares problems. Multiple algorithms are available, each with different performance characteristics.
/// </para>
/// <para>
/// <b>For Beginners:</b> QR decomposition breaks down a matrix into two parts - Q (which has perpendicular columns with length 1)
/// and R (which is triangular with zeros below the diagonal). This is useful for solving equations and other matrix operations.
/// Think of it like factoring a number into its prime components, but for matrices.
/// </para>
/// <para>
/// Real-world applications:
/// - Solving systems of linear equations
/// - Computing eigenvalues and eigenvectors
/// - Least squares regression in statistics
/// </para>
/// </remarks>
public class QrDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the orthogonal matrix Q from the decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Q matrix has special properties - its columns are perpendicular to each other
    /// (orthogonal) and each column has a length of 1. This makes it useful for many calculations.
    /// </para>
    /// </remarks>
    public Matrix<T> Q { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the upper triangular matrix R from the decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The R matrix is upper triangular, which means it has numbers on and above the diagonal,
    /// and zeros below the diagonal. This structure makes solving equations much easier.
    /// </para>
    /// </remarks>
    public Matrix<T> R { get; private set; } = new Matrix<T>(0, 0);

    private readonly QrAlgorithmType _algorithm;

    /// <summary>
    /// Creates a new QR decomposition of the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="qrAlgorithm">The algorithm to use for QR decomposition (default is Householder).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different algorithms can be used to perform QR decomposition. Each has advantages in terms of
    /// speed, accuracy, or memory usage. Householder is generally a good default choice for most applications.
    /// </para>
    /// </remarks>
    public QrDecomposition(Matrix<T> matrix, QrAlgorithmType qrAlgorithm = QrAlgorithmType.Householder)
        : base(matrix)
    {
        _algorithm = qrAlgorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the QR decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (Q, R) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Solves the linear system Ax = b using the QR decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector x such that Ax * b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method solves equations of the form Ax = b, where A is a matrix, and x and b are vectors.
    /// It finds the values of x that make the equation true. Using QR decomposition often gives more accurate results
    /// than other methods, especially for complex matrices.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        var y = Q.Transpose().Multiply(b);
        return BackSubstitution(R, y);
    }

    /// <summary>
    /// Selects and applies the appropriate QR decomposition algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    private (Matrix<T> Q, Matrix<T> R) ComputeDecomposition(Matrix<T> matrix, QrAlgorithmType algorithm)
    {
        return algorithm switch
        {
            QrAlgorithmType.GramSchmidt => ComputeQrGramSchmidt(matrix),
            QrAlgorithmType.Householder => ComputeQrHouseholder(matrix),
            QrAlgorithmType.Givens => ComputeQrGivens(matrix),
            QrAlgorithmType.ModifiedGramSchmidt => ComputeQrModifiedGramSchmidt(matrix),
            QrAlgorithmType.IterativeGramSchmidt => ComputeQrIterativeGramSchmidt(matrix),
            _ => throw new ArgumentException("Unsupported QR decomposition algorithm.")
        };
    }

    /// <summary>
    /// Computes QR decomposition using the classical Gram-Schmidt process.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gram-Schmidt process transforms a set of vectors into a set of perpendicular vectors.
    /// It works by taking each column of the matrix and removing any components that point in the same direction
    /// as previous columns. This is like ensuring that each new direction is completely different from all previous ones.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n);
        Matrix<T> R = new(n, n);

        for (int j = 0; j < n; j++)
        {
            Vector<T> v = Engine.GetColumn(matrix, j);
            for (int i = 0; i < j; i++)
            {
                var qCol = Engine.GetColumn(Q, i);
                R[i, j] = qCol.DotProduct(v);
                // VECTORIZED: Subtract the projection using Engine operations
                var projection = (Vector<T>)Engine.Multiply(qCol, R[i, j]);
                v = (Vector<T>)Engine.Subtract(v, projection);
            }
            R[j, j] = v.Norm();
            if (!NumOps.Equals(R[j, j], NumOps.Zero))
            {
                for (int i = 0; i < m; i++)
                {
                    Q[i, j] = NumOps.Divide(v[i], R[j, j]);
                }
            }
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes QR decomposition using Householder reflections.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Householder reflections use special matrices that act like mirrors, reflecting vectors
    /// across a plane. This method is more numerically stable than Gram-Schmidt, meaning it gives more accurate
    /// results, especially for large or complex matrices. It works by systematically creating zeros below the diagonal
    /// of the matrix.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrHouseholder(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = Matrix<T>.CreateIdentityMatrix(m);
        Matrix<T> R = matrix.Clone();

        for (int k = 0; k < n; k++)
        {
            Vector<T> x = R.GetSubColumn(k, k, m - k);
            T normX = x.Norm();
            Vector<T> e = new(m - k)
            {
                [0] = NumOps.One
            };

            Vector<T> u = x.Add(e.Multiply(normX));
            T normU = u.Norm();

            if (!NumOps.Equals(normU, NumOps.Zero))
            {
                u = u.Divide(normU);
                Matrix<T> H = Matrix<T>.CreateIdentityMatrix(m - k)
                    .Subtract(u.OuterProduct(u).Multiply(NumOps.FromDouble(2)));

                Matrix<T> QkTranspose = Matrix<T>.CreateIdentityMatrix(m);
                for (int i = k; i < m; i++)
                {
                    for (int j = k; j < m; j++)
                    {
                        QkTranspose[i, j] = H[i - k, j - k];
                    }
                }

                Q = Q.Multiply(QkTranspose);
                R = QkTranspose.Multiply(R);
            }
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes QR decomposition using Givens rotations.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Givens rotations work by rotating the matrix in a plane to create zeros in specific positions.
    /// Think of it like carefully turning a dial to make certain values become zero. This method is particularly useful
    /// for sparse matrices (matrices with many zeros) because it can target specific elements without changing others.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrGivens(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = Matrix<T>.CreateIdentityMatrix(m);
        Matrix<T> R = matrix.Clone();

        for (int j = 0; j < n; j++)
        {
            for (int i = m - 1; i > j; i--)
            {
                if (!NumOps.Equals(R[i, j], NumOps.Zero))
                {
                    T a = R[i - 1, j];
                    T b = R[i, j];
                    T r = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(a, a), NumOps.Multiply(b, b)));
                    T c = NumOps.Divide(a, r);
                    T s = NumOps.Divide(b, r);

                    Matrix<T> G = Matrix<T>.CreateIdentityMatrix(m);
                    G[i - 1, i - 1] = c;
                    G[i, i] = c;
                    G[i - 1, i] = s;
                    G[i, i - 1] = NumOps.Negate(s);

                    R = G.Multiply(R);
                    Q = Q.Multiply(G.Transpose());
                }
            }
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes QR decomposition using the Modified Gram-Schmidt process, which is more numerically stable.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Modified Gram-Schmidt process is an improved version of the classical Gram-Schmidt method.
    /// It performs the same task (creating perpendicular vectors) but does it in a way that reduces rounding errors
    /// in calculations. This makes it more reliable for complex problems or when high precision is needed.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrModifiedGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n);
        Matrix<T> R = new(n, n);

        // Clone the matrix since Modified Gram-Schmidt modifies columns in place
        Matrix<T> workMatrix = matrix.Clone();

        for (int k = 0; k < n; k++)
        {
            Vector<T> v = Engine.GetColumn(workMatrix, k);
            R[k, k] = v.Norm();
            // VECTORIZED: Normalize using Engine division
            var normalized = (Vector<T>)Engine.Divide(v, R[k, k]);
            Engine.SetColumn(Q, k, normalized);

            for (int j = k + 1; j < n; j++)
            {
                var qCol = Engine.GetColumn(Q, k);
                var matCol = Engine.GetColumn(workMatrix, j);
                R[k, j] = qCol.DotProduct(matCol);
                var subtracted = matCol.Subtract(qCol.Multiply(R[k, j]));
                Engine.SetColumn(workMatrix, j, subtracted);
            }
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes QR decomposition using the Iterative Gram-Schmidt process for enhanced numerical stability.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Iterative Gram-Schmidt method repeats the orthogonalization process multiple times
    /// to achieve better accuracy. Think of it like double-checking your work to make sure the vectors are truly
    /// perpendicular to each other. This method is useful when working with matrices that might cause precision problems.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrIterativeGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n);
        Matrix<T> R = new(n, n);

        for (int k = 0; k < n; k++)
        {
            Vector<T> v = matrix.GetColumn(k);
            for (int i = 0; i < 2; i++) // Perform two iterations
            {
                for (int j = 0; j < k; j++)
                {
                    T r = Q.GetColumn(j).DotProduct(v);
                    R[j, k] = NumOps.Add(R[j, k], r);
                    // VECTORIZED: Subtract the projection using Engine operations
                    var qCol = Q.GetColumn(j);
                    var projection = (Vector<T>)Engine.Multiply(qCol, r);
                    v = (Vector<T>)Engine.Subtract(v, projection);
                }
            }
            R[k, k] = v.Norm();
            // VECTORIZED: Normalize using Engine division
            var normalized = (Vector<T>)Engine.Divide(v, R[k, k]);
            Q.SetColumn(k, normalized);
        }

        return (Q, R);
    }

    /// <summary>
    /// Solves a system of linear equations using back substitution on an upper triangular matrix.
    /// </summary>
    /// <param name="R">The upper triangular matrix.</param>
    /// <param name="y">The right-hand side vector.</param>
    /// <returns>The solution vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Back substitution is a simple way to solve equations when your matrix is triangular.
    /// It works by starting at the bottom row (which has only one unknown) and solving for that value.
    /// Then it moves up row by row, using the values already found to solve for each new unknown.
    /// This is like solving a puzzle where each piece you place helps you figure out where the next one goes.
    /// </para>
    /// </remarks>
    private Vector<T> BackSubstitution(Matrix<T> R, Vector<T> y)
    {
        var x = new Vector<T>(R.Columns);
        for (int i = R.Columns - 1; i >= 0; i--)
        {
            // VECTORIZED: Use dot product for sum computation
            T sum = NumOps.Zero;
            if (i < R.Columns - 1)
            {
                int remaining = R.Columns - i - 1;
                var rowSlice = new T[remaining];
                var xSlice = new T[remaining];
                for (int k = 0; k < remaining; k++)
                {
                    rowSlice[k] = R[i, i + 1 + k];
                    xSlice[k] = x[i + 1 + k];
                }
                var rowVec = new Vector<T>(rowSlice);
                var xVec = new Vector<T>(xSlice);
                sum = rowVec.DotProduct(xVec);
            }

            x[i] = NumOps.Divide(NumOps.Subtract(y[i], sum), R[i, i]);
        }

        return x;
    }
}
