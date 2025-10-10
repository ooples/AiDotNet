namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs Schur decomposition on a matrix, factoring it into the product of a unitary matrix and an upper triangular matrix.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Schur decomposition breaks down a complex matrix into simpler parts that are easier to work with.
/// It's like factoring a number (e.g., 12 = 3 × 4), but for matrices. The decomposition produces two matrices:
/// a unitary matrix (which preserves lengths and angles) and an upper triangular matrix (which has zeros below the diagonal).
/// This makes many calculations much simpler.
/// </para>
/// </remarks>
public class SchurDecomposition<T> : IMatrixDecomposition<T>
{
    /// <summary>
    /// Gets the upper triangular Schur matrix (S) from the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the triangular matrix that has all its non-zero elements on or above the diagonal.
    /// It's easier to work with than the original matrix for many calculations.
    /// </remarks>
    public Matrix<T> SchurMatrix { get; private set; }
    
    /// <summary>
    /// Gets the unitary matrix (U) from the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A unitary matrix preserves the length of vectors and the angles between them.
    /// When you multiply a vector by a unitary matrix, the result has the same length as the original vector.
    /// </remarks>
    public Matrix<T> UnitaryMatrix { get; private set; }
    
    /// <summary>
    /// Gets the original matrix that was decomposed.
    /// </summary>
    public Matrix<T> A { get; private set; }

    /// <summary>
    /// Provides operations for the numeric type being used.
    /// </summary>
    private readonly INumericOperations<T> _numOps = default!;

    /// <summary>
    /// Initializes a new instance of the SchurDecomposition class.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for the decomposition.</param>
    public SchurDecomposition(Matrix<T> matrix, SchurAlgorithmType algorithm = SchurAlgorithmType.Francis)
    {
        A = matrix;
        _numOps = MathHelper.GetNumericOperations<T>();
        (SchurMatrix, UnitaryMatrix) = Decompose(matrix, algorithm);
    }

    /// <summary>
    /// Selects and applies the appropriate decomposition algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use.</param>
    /// <returns>A tuple containing the Schur matrix and the unitary matrix.</returns>
    private (Matrix<T> S, Matrix<T> U) Decompose(Matrix<T> matrix, SchurAlgorithmType algorithm)
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
        Matrix<T> H = MatrixHelper<T>.ReduceToHessenbergFormat(matrix);
        Matrix<T> U = Matrix<T>.CreateIdentity(n);
        Matrix<T> S = H.Clone();

        const int maxIterations = 100;
        T tolerance = _numOps.FromDouble(1e-10);

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
        Matrix<T> H = MatrixHelper<T>.ReduceToHessenbergFormat(matrix);
        Matrix<T> U = Matrix<T>.CreateIdentity(n);

        const int maxIterations = 100;
        T tolerance = _numOps.FromDouble(1e-10);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            for (int i = 0; i < n - 1; i++)
            {
                T s = _numOps.Add(H[n - 2, n - 2], H[n - 1, n - 1]);
                T t = _numOps.Subtract(_numOps.Multiply(H[n - 2, n - 2], H[n - 1, n - 1]), _numOps.Multiply(H[n - 2, n - 1], H[n - 1, n - 2]));

                Matrix<T> Q = ComputeFrancisQRStep(H, s, t);
                H = Q.Transpose().Multiply(H).Multiply(Q);
                U = U.Multiply(Q);
            }

            if (H.IsUpperTriangularMatrix(tolerance))
                break;
        }

        return (H, U);
    }

    /// <summary>
    /// Computes a single step of the Francis QR algorithm.
    /// </summary>
    /// <param name="H">The Hessenberg matrix.</param>
    /// <param name="s">The sum of the last two diagonal elements.</param>
    /// <param name="t">The determinant of the bottom-right 2x2 submatrix.</param>
    /// <returns>The transformation matrix for this step.</returns>
    private Matrix<T> ComputeFrancisQRStep(Matrix<T> H, T s, T t)
    {
        int n = H.Rows;
        T x = _numOps.Subtract(_numOps.Subtract(H[0, 0], s), _numOps.Divide(_numOps.Multiply(H[0, 1], H[1, 0]), _numOps.Subtract(H[1, 1], s)));
        T y = H[1, 0];

        for (int k = 0; k < n - 1; k++)
        {
            Matrix<T> Q = ComputeHouseholderReflection(x, y);
            H = Q.Transpose().Multiply(H).Multiply(Q);

            if (k < n - 2)
            {
                x = H[k + 1, k];
                y = H[k + 2, k];
            }
        }

        return H;
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
        int n = matrix.Rows;
        Matrix<T> H = MatrixHelper<T>.ReduceToHessenbergFormat(matrix);
        Matrix<T> U = Matrix<T>.CreateIdentity(n);

        const int maxIterations = 100;
        T tolerance = _numOps.FromDouble(1e-10);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            for (int i = 0; i < n - 1; i++)
            {
                Matrix<T> Q = ComputeImplicitQStep(H, i);
                H = Q.Transpose().Multiply(H).Multiply(Q);
                U = U.Multiply(Q);
            }

            if (H.IsUpperTriangularMatrix(tolerance))
                break;
        }

        return (H, U);
    }

    /// <summary>
    /// Computes a single step of the implicit QR algorithm.
    /// </summary>
    /// <param name="H">The Hessenberg matrix.</param>
    /// <param name="start">The starting index for this step.</param>
    /// <returns>The transformation matrix for this step.</returns>
    private Matrix<T> ComputeImplicitQStep(Matrix<T> H, int start)
    {
        int n = H.Rows;
        T x = H[start, start];
        T y = H[start + 1, start];
        T z = start + 2 < n ? H[start + 2, start] : _numOps.Zero;

        for (int k = start; k < n - 1; k++)
        {
            Matrix<T> Q = ComputeHouseholderReflection(x, y, z);
            H = Q.Transpose().Multiply(H).Multiply(Q);

            if (k < n - 2)
            {
                x = H[k + 1, k];
                y = H[k + 2, k];
                z = k + 3 < n ? H[k + 3, k] : _numOps.Zero;
            }
        }

        return H;
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
        int n = _numOps.Equals(z ?? _numOps.Zero, _numOps.Zero) ? 2 : 3;
        Vector<T> v = new(n)
        {
            [0] = x,
            [1] = y
        };
        if (n == 3) v[2] = z ?? _numOps.Zero;

        T alpha = _numOps.Sqrt(v.DotProduct(v));
        if (_numOps.Equals(alpha, _numOps.Zero)) return Matrix<T>.CreateIdentity(n);

        v[0] = _numOps.Add(v[0], _numOps.Multiply(_numOps.SignOrZero(x), alpha));
        T beta = _numOps.Sqrt(_numOps.Multiply(_numOps.FromDouble(2), v.DotProduct(v)));

        if (_numOps.Equals(beta, _numOps.Zero)) return Matrix<T>.CreateIdentity(n);

        v = v.Divide(beta);
        return Matrix<T>.CreateIdentity(n).Subtract(v.OuterProduct(v).Multiply(_numOps.FromDouble(2)));
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
    public Vector<T> Solve(Vector<T> b)
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
    public Matrix<T> Invert()
    {
        var invU = UnitaryMatrix.Transpose();
        var invS = SchurMatrix.InvertUpperTriangularMatrix();

        return invS.Multiply(invU);
    }
}