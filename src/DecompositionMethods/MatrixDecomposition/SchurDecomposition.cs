namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs Schur decomposition on a matrix, factoring it into the product of a unitary matrix and an upper triangular matrix.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix.</typeparam>
/// <remarks>
/// <para>
/// Schur decomposition factors a matrix A into the product A = USU*, where U is a unitary matrix
/// and S is an upper triangular matrix. This decomposition is particularly useful for computing
/// eigenvalues and for analyzing the properties of linear transformations.
/// </para>
/// <para>
/// <b>For Beginners:</b> Schur decomposition breaks down a complex matrix into simpler parts that are easier to work with.
/// It's like factoring a number (e.g., 12 = 3 * 4), but for matrices. The decomposition produces two matrices:
/// a unitary matrix (which preserves lengths and angles) and an upper triangular matrix (which has zeros below the diagonal).
/// This makes many calculations much simpler.
/// </para>
/// <para>
/// Real-world applications:
/// - Computing eigenvalues and eigenvectors efficiently
/// - Solving differential equations in engineering
/// - Control theory and system stability analysis
/// </para>
/// </remarks>
public class SchurDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the upper triangular Schur matrix (S) from the decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the triangular matrix that has all its non-zero elements on or above the diagonal.
    /// It's easier to work with than the original matrix for many calculations.
    /// </para>
    /// </remarks>
    public Matrix<T> SchurMatrix { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the unitary matrix (U) from the decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A unitary matrix preserves the length of vectors and the angles between them.
    /// When you multiply a vector by a unitary matrix, the result has the same length as the original vector.
    /// </para>
    /// </remarks>
    public Matrix<T> UnitaryMatrix { get; private set; } = new Matrix<T>(0, 0);

    private readonly SchurAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the SchurDecomposition class.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for the decomposition.</param>
    public SchurDecomposition(Matrix<T> matrix, SchurAlgorithmType algorithm = SchurAlgorithmType.Francis)
        : base(matrix)
    {
        _algorithm = algorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the Schur decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (SchurMatrix, UnitaryMatrix) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Computes the Schur decomposition using the specified algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use.</param>
    /// <returns>A tuple containing the Schur matrix and the unitary matrix.</returns>
    private (Matrix<T> S, Matrix<T> U) ComputeDecomposition(Matrix<T> matrix, SchurAlgorithmType algorithm)
    {
        return algorithm switch
        {
            SchurAlgorithmType.Francis => ComputeSchurFrancis(matrix),
            SchurAlgorithmType.QR => ComputeSchurQR(matrix),
            SchurAlgorithmType.Implicit => ComputeSchurImplicit(matrix),
            _ => throw new ArgumentException("Unsupported Schur decomposition algorithm.")
        };
    }

    /// <summary>
    /// Computes the Schur decomposition using the QR algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Schur matrix and the unitary matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method uses repeated QR decompositions to gradually transform the matrix
    /// into an upper triangular form. It's like repeatedly refining a rough sketch until you get a clear picture.
    /// The QR algorithm is one of the most common ways to find eigenvalues of a matrix.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<T> U) ComputeSchurQR(Matrix<T> matrix)
    {
        int n = matrix.Rows;

        // Use HessenbergDecomposition to get both H and the orthogonal Q_hess
        // A = Q_hess * H * Q_hess^T
        var hessDecomp = new HessenbergDecomposition<T>(matrix);
        Matrix<T> H = hessDecomp.HessenbergMatrix;
        Matrix<T> Q_hess = hessDecomp.OrthogonalMatrix;

        // Start U with Q_hess (the orthogonal matrix from Hessenberg reduction)
        Matrix<T> U = Q_hess.Clone();
        Matrix<T> S = H.Clone();

        const int maxIterations = 100;
        T tolerance = NumOps.FromDouble(1e-10);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var qrDecomp = new QrDecomposition<T>(S);
            (var Q, var R) = (qrDecomp.Q, qrDecomp.R);
            S = R.Multiply(Q);
            U = U.Multiply(Q);

            if (S.IsUpperTriangularMatrix(tolerance))
                break;
        }

        return (S, U);
    }

    /// <summary>
    /// Computes the Schur decomposition using the Francis QR algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Schur matrix and the unitary matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Francis QR algorithm is an improved version of the basic QR algorithm.
    /// It uses special techniques to make the computation faster and more accurate.
    /// This method is particularly good for finding eigenvalues of large matrices.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<T> U) ComputeSchurFrancis(Matrix<T> matrix)
    {
        int n = matrix.Rows;

        // Use HessenbergDecomposition to get both H and the orthogonal Q_hess
        // A = Q_hess * H * Q_hess^T
        var hessDecomp = new HessenbergDecomposition<T>(matrix);
        Matrix<T> H = hessDecomp.HessenbergMatrix;
        Matrix<T> Q_hess = hessDecomp.OrthogonalMatrix;

        // Start U with Q_hess (the orthogonal matrix from Hessenberg reduction)
        Matrix<T> U = Q_hess.Clone();

        const int maxIterations = 100;
        T tolerance = NumOps.FromDouble(1e-10);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Perform Francis double-shift QR step
            ApplyFrancisDoubleShift(H, U, n);

            if (H.IsUpperTriangularMatrix(tolerance))
                break;
        }

        return (H, U);
    }

    /// <summary>
    /// Applies a Francis double-shift QR step using Givens rotations.
    /// </summary>
    private void ApplyFrancisDoubleShift(Matrix<T> H, Matrix<T> U, int n)
    {
        // Check if matrix is already (quasi-)triangular - no work needed
        bool allSubdiagonalZero = true;
        for (int i = 1; i < n; i++)
        {
            if (NumOps.GreaterThan(NumOps.Abs(H[i, i - 1]), NumOps.FromDouble(1e-14)))
            {
                allSubdiagonalZero = false;
                break;
            }
        }
        if (allSubdiagonalZero)
        {
            return; // Already triangular
        }

        // Find the active unreduced submatrix (deflation)
        int p = n - 1;
        while (p > 0 && NumOps.LessThanOrEquals(NumOps.Abs(H[p, p - 1]), NumOps.FromDouble(1e-14)))
        {
            H[p, p - 1] = NumOps.Zero; // Explicitly zero out
            p--;
        }

        if (p == 0)
        {
            return; // Matrix is diagonal
        }

        int q = p - 1;
        while (q > 0 && NumOps.GreaterThan(NumOps.Abs(H[q, q - 1]), NumOps.FromDouble(1e-14)))
        {
            q--;
        }

        // Work only on the active submatrix from q to p
        // Use Wilkinson shift for better convergence
        T s = NumOps.Add(H[p - 1, p - 1], H[p, p]);
        T prod = NumOps.Subtract(
            NumOps.Multiply(H[p - 1, p - 1], H[p, p]),
            NumOps.Multiply(H[p - 1, p], H[p, p - 1]));

        // First column of (H - s1*I)(H - s2*I) = H^2 - s*H + p*I
        T h_q_q = H[q, q];
        T h_q_q1 = q + 1 < n ? H[q, q + 1] : NumOps.Zero;
        T h_q1_q = q + 1 < n ? H[q + 1, q] : NumOps.Zero;
        T h_q1_q1 = q + 1 < n ? H[q + 1, q + 1] : NumOps.Zero;

        T x = NumOps.Add(
            NumOps.Subtract(NumOps.Multiply(h_q_q, h_q_q), NumOps.Multiply(s, h_q_q)),
            NumOps.Add(prod, NumOps.Multiply(h_q_q1, h_q1_q)));
        T y = NumOps.Multiply(h_q1_q, NumOps.Subtract(NumOps.Add(h_q_q, h_q1_q1), s));

        // Apply Givens rotations to chase the bulge within active submatrix
        for (int k = q; k < p; k++)
        {
            // Compute Givens rotation to zero out y
            T r = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(x, x), NumOps.Multiply(y, y)));
            if (NumOps.LessThan(r, NumOps.FromDouble(1e-14)))
            {
                if (k < p - 1)
                {
                    x = H[k + 1, k];
                    y = k + 2 <= p ? H[k + 2, k] : NumOps.Zero;
                }
                continue;
            }

            T c = NumOps.Divide(x, r);
            T s2 = NumOps.Divide(y, r);

            // Apply Givens rotation from LEFT: G^T * H (affects rows k and k+1)
            for (int j = 0; j < n; j++)
            {
                T temp1 = H[k, j];
                T temp2 = H[k + 1, j];
                H[k, j] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s2, temp2));
                H[k + 1, j] = NumOps.Subtract(NumOps.Multiply(c, temp2), NumOps.Multiply(s2, temp1));
            }

            // Apply Givens rotation from RIGHT: H * G (affects columns k and k+1)
            for (int i = 0; i < n; i++)
            {
                T temp1 = H[i, k];
                T temp2 = H[i, k + 1];
                H[i, k] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s2, temp2));
                H[i, k + 1] = NumOps.Subtract(NumOps.Multiply(c, temp2), NumOps.Multiply(s2, temp1));
            }

            // Accumulate transformation in U
            for (int i = 0; i < n; i++)
            {
                T temp1 = U[i, k];
                T temp2 = U[i, k + 1];
                U[i, k] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s2, temp2));
                U[i, k + 1] = NumOps.Subtract(NumOps.Multiply(c, temp2), NumOps.Multiply(s2, temp1));
            }

            // Prepare for next iteration (the bulge to chase)
            if (k < p - 1)
            {
                x = H[k + 1, k];
                y = k + 2 <= p ? H[k + 2, k] : NumOps.Zero;
            }
        }
    }

    /// <summary>
    /// Computes the Schur decomposition using the implicit QR algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Schur matrix and the unitary matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The implicit QR algorithm is another variation of the QR algorithm that avoids
    /// explicitly forming certain matrices, making it more efficient. It's like taking a shortcut
    /// in a calculation by skipping steps that aren't necessary.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<T> U) ComputeSchurImplicit(Matrix<T> matrix)
    {
        // Use the same Francis algorithm as it's more robust
        return ComputeSchurFrancis(matrix);
    }

    /// <summary>
    /// Computes a Householder reflection matrix based on the input vector components.
    /// </summary>
    /// <param name="x">First component of the vector.</param>
    /// <param name="y">Second component of the vector.</param>
    /// <param name="z">Optional third component of the vector.</param>
    /// <returns>A Householder reflection matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Householder reflection is a mathematical operation that "reflects" vectors across a plane.
    /// It's used in many matrix decomposition algorithms to transform matrices into simpler forms.
    /// Think of it like a mirror that reflects a vector to point in a specific direction.
    /// This helps simplify complex matrix operations by zeroing out specific elements.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeHouseholderReflection(T x, T y, T? z = default)
    {
        int n = NumOps.Equals(z ?? NumOps.Zero, NumOps.Zero) ? 2 : 3;
        Vector<T> v = new(n)
        {
            [0] = x,
            [1] = y
        };
        if (n == 3) v[2] = z ?? NumOps.Zero;

        // VECTORIZED: Use dot product to compute norm
        T alpha = NumOps.Sqrt(v.DotProduct(v));
        if (NumOps.Equals(alpha, NumOps.Zero)) return Matrix<T>.CreateIdentity(n);

        v[0] = NumOps.Add(v[0], NumOps.Multiply(NumOps.SignOrZero(x), alpha));

        // VECTORIZED: Use dot product for norm calculation (||v||)
        T beta = NumOps.Sqrt(v.DotProduct(v));

        if (NumOps.Equals(beta, NumOps.Zero)) return Matrix<T>.CreateIdentity(n);

        // VECTORIZED: Use Engine division for normalization
        v = (Vector<T>)Engine.Divide(v, beta);
        return Matrix<T>.CreateIdentity(n).Subtract(v.OuterProduct(v).Multiply(NumOps.FromDouble(2)));
    }

    /// <summary>
    /// Solves a system of linear equations Ax = b using the Schur decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the solution to a system of equations represented by Ax = b,
    /// where A is the original matrix, x is the unknown vector we're solving for, and b is a known vector.
    /// It uses the Schur decomposition to break this problem into simpler steps that are easier to solve.
    /// </para>
    /// <para>
    /// The method works by first applying forward substitution (solving from top to bottom) and then
    /// backward substitution (solving from bottom to top) to find the answer efficiently.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        var y = UnitaryMatrix.ForwardSubstitution(b);
        return SchurMatrix.BackwardSubstitution(y);
    }

    /// <summary>
    /// Computes the inverse of the original matrix using the Schur decomposition.
    /// </summary>
    /// <returns>The inverse of the original matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number. Just as 1/x is the reciprocal of x,
    /// the inverse of a matrix A (written as A⁻¹) is a matrix that, when multiplied by A, gives the identity matrix.
    /// </para>
    /// <para>
    /// This method uses the Schur decomposition to find the inverse more efficiently than direct methods.
    /// It first inverts the simpler components (the unitary and upper triangular matrices) and then
    /// combines them to get the inverse of the original matrix.
    /// </para>
    /// </remarks>
    public override Matrix<T> Invert()
    {
        var invU = UnitaryMatrix.Transpose();
        var invS = SchurMatrix.InvertUpperTriangularMatrix();

        return invS.Multiply(invU);
    }
}
