namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements Singular Value Decomposition (SVD) for matrices.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SVD is a way to break down a matrix into three simpler matrices (U, S, and V^T).
/// Think of it like factoring a number, but for matrices. This decomposition is useful for data
/// compression, noise reduction, and solving systems of equations.
/// </para>
/// </remarks>
public class SvdDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the left singular vectors matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The U matrix represents the output space of the transformation.
    /// It contains information about patterns in the rows of your original data.
    /// </para>
    /// </remarks>
    public Matrix<T> U { get; private set; }

    /// <summary>
    /// Gets the singular values vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The S vector contains the "strength" of each pattern found in the data.
    /// Larger values indicate more important patterns. These values are always positive
    /// and are typically arranged in descending order.
    /// </para>
    /// </remarks>
    public Vector<T> S { get; private set; }

    /// <summary>
    /// Gets the transposed right singular vectors matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The V^T matrix represents the input space of the transformation.
    /// It contains information about patterns in the columns of your original data.
    /// </para>
    /// </remarks>
    public Matrix<T> Vt { get; private set; }

    private readonly SvdAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the SVD decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="svdAlgorithm">The algorithm to use for SVD computation.</param>
    public SvdDecomposition(Matrix<T> matrix, SvdAlgorithmType svdAlgorithm = SvdAlgorithmType.GolubReinsch)
        : base(matrix)
    {
        _algorithm = svdAlgorithm;
    }

    /// <summary>
    /// Performs the SVD decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (U, S, Vt) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Selects and applies the appropriate SVD algorithm based on the specified type.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The SVD algorithm to use.</param>
    /// <returns>A tuple containing the U, S, and V^T components of the decomposition.</returns>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeDecomposition(Matrix<T> matrix, SvdAlgorithmType algorithm)
    {
        return algorithm switch
        {
            SvdAlgorithmType.GolubReinsch => ComputeSvdGolubReinsch(matrix),
            SvdAlgorithmType.Jacobi => ComputeSvdJacobi(matrix),
            SvdAlgorithmType.Randomized => ComputeSvdRandomized(matrix),
            SvdAlgorithmType.PowerIteration => ComputeSvdPowerIteration(matrix),
            SvdAlgorithmType.TruncatedSVD => ComputeTruncatedSvd(matrix),
            SvdAlgorithmType.DividedAndConquer => ComputeSvdDividedAndConquer(matrix),
            _ => throw new ArgumentException("Unsupported SVD algorithm", nameof(algorithm)),
        };
    }

    /// <summary>
    /// Solves a linear system Ax = b using the SVD decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the values of x that best satisfy the equation Ax = b.
    /// SVD is particularly useful for solving systems that might not have an exact solution
    /// or when A is not a square matrix. It provides the best approximate solution in these cases.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        var x = new Vector<T>(Vt.Rows);
        for (int i = 0; i < S.Length; i++)
        {
            if (!NumOps.Equals(S[i], NumOps.Zero))
            {
                T r = NumOps.Zero;
                for (int j = 0; j < U.Rows; j++)
                {
                    r = NumOps.Add(r, NumOps.Multiply(U[j, i], b[j]));
                }
                r = NumOps.Divide(r, S[i]);
                for (int j = 0; j < Vt.Columns; j++)
                {
                    x[j] = NumOps.Add(x[j], NumOps.Multiply(Vt[i, j], r));
                }
            }
        }

        return x;
    }

    /// <summary>
    /// Computes the SVD using the Golub-Reinsch algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the U, S, and V^T components of the decomposition.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Golub-Reinsch algorithm is the standard approach for computing SVD.
    /// It works by first converting the matrix to a simpler form (bidiagonal) and then
    /// iteratively refining it until we get the diagonal form we need.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdGolubReinsch(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int l = Math.Min(m, n);

        Matrix<T> A = matrix.Clone();
        Matrix<T> U = new(m, m);
        Vector<T> S = new(l);
        Matrix<T> VT = new(n, n);

        // Bidiagonalization
        Bidiagonalize(A, U, VT);

        // Diagonalization
        DiagonalizeGolubReinsch(A, U, VT);

        // Extract singular values
        for (int i = 0; i < l; i++)
        {
            S[i] = A[i, i];
        }

        // Sort singular values in descending order
        SortSingularValues(S, U, VT);

        return (U, S, VT);
    }

    /// <summary>
    /// Performs bidiagonalization of a matrix as part of the SVD process.
    /// </summary>
    /// <param name="A">The matrix to bidiagonalize</param>
    /// <param name="U">The left orthogonal matrix</param>
    /// <param name="VT">The right orthogonal matrix transposed</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bidiagonalization transforms a matrix into a simpler form with non-zero values 
    /// only on the main diagonal and the diagonal above it. This makes it easier to compute the 
    /// singular values in the next steps of SVD.
    /// </para>
    /// </remarks>
    private void Bidiagonalize(Matrix<T> A, Matrix<T> U, Matrix<T> VT)
    {
        int m = A.Rows;
        int n = A.Columns;

        for (int k = 0; k < n; k++)
        {
            // Compute the Householder vector for the kth column
            Vector<T> x = A.GetColumn(k).Slice(k, m - k);
            T alpha = x.Norm();
            if (NumOps.LessThan(A[k, k], NumOps.Zero))
                alpha = NumOps.Negate(alpha);

            Vector<T> u = x.Clone();
            u[0] = NumOps.Add(u[0], alpha);
            T beta = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Divide(NumOps.One, u.DotProduct(u)));

            // Apply the Householder reflection to A and U
            for (int j = k; j < n; j++)
            {
                T sum = NumOps.Zero;
                for (int i = k; i < m; i++)
                    sum = NumOps.Add(sum, NumOps.Multiply(u[i - k], A[i, j]));
                sum = NumOps.Multiply(beta, sum);

                for (int i = k; i < m; i++)
                    A[i, j] = NumOps.Subtract(A[i, j], NumOps.Multiply(u[i - k], sum));
            }

            for (int j = 0; j < m; j++)
            {
                T sum = NumOps.Zero;
                for (int i = k; i < m; i++)
                    sum = NumOps.Add(sum, NumOps.Multiply(u[i - k], U[j, i]));
                sum = NumOps.Multiply(beta, sum);

                for (int i = k; i < m; i++)
                    U[j, i] = NumOps.Subtract(U[j, i], NumOps.Multiply(u[i - k], sum));
            }

            A[k, k] = NumOps.Negate(alpha);

            // If this is not the last column, compute the Householder vector for the kth row
            if (k < n - 2)
            {
                x = A.GetRow(k).Slice(k + 1, n - k - 1);
                alpha = x.Norm();
                if (NumOps.LessThan(A[k, k + 1], NumOps.Zero))
                    alpha = NumOps.Negate(alpha);

                Vector<T> v = x.Clone();
                v[0] = NumOps.Add(v[0], alpha);
                beta = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Divide(NumOps.One, v.DotProduct(v)));

                // Apply the Householder reflection to A and VT
                for (int i = k; i < m; i++)
                {
                    T sum = NumOps.Zero;
                    for (int j = k + 1; j < n; j++)
                        sum = NumOps.Add(sum, NumOps.Multiply(v[j - k - 1], A[i, j]));
                    sum = NumOps.Multiply(beta, sum);

                    for (int j = k + 1; j < n; j++)
                        A[i, j] = NumOps.Subtract(A[i, j], NumOps.Multiply(v[j - k - 1], sum));
                }

                for (int i = 0; i < n; i++)
                {
                    T sum = NumOps.Zero;
                    for (int j = k + 1; j < n; j++)
                        sum = NumOps.Add(sum, NumOps.Multiply(v[j - k - 1], VT[j, i]));
                    sum = NumOps.Multiply(beta, sum);

                    for (int j = k + 1; j < n; j++)
                        VT[j, i] = NumOps.Subtract(VT[j, i], NumOps.Multiply(v[j - k - 1], sum));
                }

                A[k, k + 1] = NumOps.Negate(alpha);
            }
        }
    }

    /// <summary>
    /// Diagonalizes a bidiagonal matrix using the Golub-Reinsch algorithm.
    /// </summary>
    /// <param name="B">The bidiagonal matrix to diagonalize</param>
    /// <param name="U">The left orthogonal matrix</param>
    /// <param name="VT">The right orthogonal matrix transposed</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a matrix that already has most elements set to zero
    /// (from the bidiagonalization step) and further simplifies it to have non-zero values 
    /// only on the main diagonal. These diagonal values become our singular values.
    /// </para>
    /// </remarks>
    private void DiagonalizeGolubReinsch(Matrix<T> B, Matrix<T> U, Matrix<T> VT)
    {
        int m = B.Rows;
        int n = B.Columns;
        int p = Math.Min(m, n);
        Vector<T> _e = new(n);  // Superdiagonal elements
        Vector<T> _d = new(p);  // Diagonal elements

        for (int i = 0; i < p; i++)
        {
            _d[i] = B[i, i];
            if (i < n - 1)
                _e[i] = B[i, i + 1];
        }

        for (int k = p - 1; k >= 0; k--)
        {
            int _maxIterations = 30;
            for (int iteration = 0; iteration < _maxIterations; iteration++)
            {
                bool flag = true;
                for (int l = k; l >= 0; l--)
                {
                    if (l == 0 || NumOps.LessThanOrEquals(NumOps.Abs(_e[l - 1]), NumOps.Multiply(NumOps.FromDouble(1e-12), NumOps.Add(NumOps.Abs(_d[l]), NumOps.Abs(_d[l - 1])))))
                    {
                        _e[l] = NumOps.Zero;
                        flag = false;
                        break;
                    }
                    if (NumOps.LessThanOrEquals(NumOps.Abs(_d[l - 1]), NumOps.Multiply(NumOps.FromDouble(1e-12), NumOps.Abs(_d[l]))))
                    {
                        GolubKahanStep(_d, _e, l, k, U, VT);
                        flag = false;
                        break;
                    }
                }
                if (flag)
                {
                    GolubKahanStep(_d, _e, 0, k, U, VT);
                }
            }
        }

        // Copy diagonal elements back to B
        for (int i = 0; i < p; i++)
        {
            B[i, i] = _d[i];
        }
    }

    /// <summary>
    /// Performs a Golub-Kahan SVD step on a bidiagonal matrix.
    /// </summary>
    /// <param name="d">The diagonal elements of the bidiagonal matrix</param>
    /// <param name="e">The superdiagonal elements of the bidiagonal matrix</param>
    /// <param name="l">The starting index for the computation</param>
    /// <param name="k">The ending index for the computation</param>
    /// <param name="U">The left orthogonal matrix to be updated</param>
    /// <param name="VT">The right orthogonal matrix (transposed) to be updated</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method implements a mathematical technique called the "Golub-Kahan SVD step" 
    /// which helps find the singular values of a matrix. Think of it as a specialized algorithm that 
    /// gradually refines our approximation of the matrix's important characteristics. The method works 
    /// on a simplified form of the matrix (called bidiagonal) where most values are zero except along 
    /// two diagonals.
    /// </para>
    /// </remarks>
    private void GolubKahanStep(Vector<T> d, Vector<T> e, int l, int k, Matrix<T> U, Matrix<T> VT)
    {
        T f = NumOps.Add(NumOps.Abs(d[l]), NumOps.Abs(d[l + 1]));
        T tst1 = NumOps.Add(f, NumOps.Abs(e[l]));
        if (NumOps.Equals(tst1, f))
        {
            e[l] = NumOps.Zero;
            return;
        }

        T g = NumOps.Divide(NumOps.Subtract(d[l + 1], d[l]), NumOps.Multiply(NumOps.FromDouble(2), e[l]));
        T r = NumOps.Sqrt(NumOps.Add(NumOps.One, NumOps.Multiply(g, g)));
        g = NumOps.Add(d[k], NumOps.Divide(e[l], NumOps.Add(g, NumOps.Multiply(NumOps.SignOrZero(g), r))));

        T s = NumOps.One;
        T c = NumOps.One;
        T p = NumOps.Zero;

        for (int i = k - 1; i >= l; i--)
        {
            T f2 = NumOps.Multiply(s, e[i]);
            T b = NumOps.Multiply(c, e[i]);
            r = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(f2, f2), NumOps.Multiply(g, g)));
            e[i + 1] = r;

            if (NumOps.Equals(r, NumOps.Zero))
            {
                d[i + 1] = NumOps.Subtract(d[i + 1], p);
                e[k] = NumOps.Zero;
                break;
            }

            s = NumOps.Divide(f2, r);
            c = NumOps.Divide(g, r);
            g = NumOps.Subtract(d[i + 1], p);
            r = NumOps.Divide(NumOps.Add(NumOps.Multiply(d[i], c), NumOps.Multiply(b, s)), g);
            p = NumOps.Multiply(r, s);
            d[i + 1] = NumOps.Add(g, p);
            g = NumOps.Subtract(NumOps.Multiply(c, r), b);

            // Accumulate transformation in U and VT
            for (int j = 0; j < U.Rows; j++)
            {
                T t = U[j, i + 1];
                U[j, i + 1] = NumOps.Add(NumOps.Multiply(s, U[j, i]), NumOps.Multiply(c, t));
                U[j, i] = NumOps.Subtract(NumOps.Multiply(c, U[j, i]), NumOps.Multiply(s, t));
            }

            for (int j = 0; j < VT.Columns; j++)
            {
                T t = VT[i, j];
                VT[i, j] = NumOps.Add(NumOps.Multiply(c, t), NumOps.Multiply(s, VT[i + 1, j]));
                VT[i + 1, j] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), t), NumOps.Multiply(c, VT[i + 1, j]));
            }
        }
    }

    /// <summary>
    /// Computes the Singular Value Decomposition using the Jacobi method.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Jacobi method is an iterative approach to find the SVD. It works by 
    /// repeatedly applying small rotations to eliminate off-diagonal elements. This is like 
    /// gradually untangling the relationships in your data until you find the core patterns 
    /// (singular values) and their associated directions (singular vectors).
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdJacobi(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int l = Math.Min(m, n);

        Matrix<T> A = matrix.Clone();
        Matrix<T> U = Matrix<T>.CreateIdentityMatrix(m);
        Vector<T> S = new(l);
        Matrix<T> VT = Matrix<T>.CreateIdentityMatrix(n);

        const int maxIterations = 100;
        int iteration = 0;

        while (iteration < maxIterations)
        {
            bool converged = true;
            for (int i = 0; i < n - 1; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    T alpha = NumOps.Zero;
                    T beta = NumOps.Zero;
                    T gamma = NumOps.Zero;

                    for (int k = 0; k < m; k++)
                    {
                        alpha = NumOps.Add(alpha, NumOps.Multiply(A[k, i], A[k, i]));
                        beta = NumOps.Add(beta, NumOps.Multiply(A[k, j], A[k, j]));
                        gamma = NumOps.Add(gamma, NumOps.Multiply(A[k, i], A[k, j]));
                    }

                    if (!NumOps.Equals(gamma, NumOps.Zero))
                    {
                        converged = false;
                        T zeta = NumOps.Divide(NumOps.Subtract(beta, alpha), NumOps.Multiply(NumOps.FromDouble(2), gamma));
                        T t = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.Abs(zeta), NumOps.Sqrt(NumOps.Add(NumOps.One, NumOps.Multiply(zeta, zeta)))));
                        if (NumOps.LessThan(zeta, NumOps.Zero))
                            t = NumOps.Negate(t);

                        T c = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(NumOps.One, NumOps.Multiply(t, t))));
                        T s = NumOps.Multiply(t, c);

                        for (int k = 0; k < m; k++)
                        {
                            T temp = A[k, i];
                            A[k, i] = NumOps.Add(NumOps.Multiply(c, temp), NumOps.Multiply(s, A[k, j]));
                            A[k, j] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp), NumOps.Multiply(c, A[k, j]));
                        }

                        for (int k = 0; k < n; k++)
                        {
                            T temp = VT[i, k];
                            VT[i, k] = NumOps.Add(NumOps.Multiply(c, temp), NumOps.Multiply(s, VT[j, k]));
                            VT[j, k] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp), NumOps.Multiply(c, VT[j, k]));
                        }
                    }
                }
            }

            if (converged)
                break;

            iteration++;
        }

        // Extract singular values
        for (int i = 0; i < l; i++)
        {
            S[i] = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(A[i, i], A[i, i]), NumOps.Multiply(A[i, i + 1], A[i, i + 1])));
        }

        // Sort singular values in descending order
        SortSingularValues(S, U, VT);

        return (U, S, VT);
    }

    /// <summary>
    /// Computes the Singular Value Decomposition using a randomized algorithm.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Randomized SVD is a faster approach for large matrices. Instead of processing 
    /// the entire matrix, it creates a smaller random projection of the data that preserves the most 
    /// important information. This is like creating a quick sketch of a large image that captures 
    /// the main features while ignoring minor details.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdRandomized(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int k = Math.Min(m, n) / 2; // Number of singular values to compute

        // Step 1: Random projection
        Matrix<T> Omega = GenerateRandomMatrix(n, k);
        Matrix<T> Y = matrix.Multiply(Omega);

        // Step 2: QR decomposition of Y
        var qr = new QrDecomposition<T>(Y);
        Matrix<T> Q = qr.Q;

        // Step 3: Form B = Q^T * A
        Matrix<T> B = Q.Transpose().Multiply(matrix);

        // Step 4: SVD of B
        var svd = new SvdDecomposition<T>(B);

        // Step 5: Compute U = Q * U_B
        Matrix<T> U = Q.Multiply(svd.U);

        return (U, svd.S, svd.Vt);
    }

    /// <summary>
    /// Generates a random matrix with values between -1 and 1.
    /// </summary>
    /// <param name="rows">Number of rows in the matrix</param>
    /// <param name="cols">Number of columns in the matrix</param>
    /// <returns>A randomly generated matrix</returns>
    private Matrix<T> GenerateRandomMatrix(int rows, int cols)
    {
        var random = new Random();
        var matrix = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = NumOps.FromDouble(random.NextDouble() * 2 - 1);
            }
        }

        return matrix;
    }

    /// <summary>
    /// Sorts singular values in descending order and rearranges the corresponding columns/rows in U and VT.
    /// </summary>
    /// <param name="S">Vector of singular values to sort</param>
    /// <param name="U">Left singular vectors matrix to rearrange</param>
    /// <param name="VT">Right singular vectors matrix (transposed) to rearrange</param>
    private void SortSingularValues(Vector<T> S, Matrix<T> U, Matrix<T> VT)
    {
        int l = S.Length;
        for (int i = 0; i < l - 1; i++)
        {
            for (int j = i + 1; j < l; j++)
            {
                if (NumOps.LessThan(S[i], S[j]))
                {
                    // Swap singular values
                    T temp = S[i];
                    S[i] = S[j];
                    S[j] = temp;

                    // Swap corresponding columns in U
                    for (int k = 0; k < U.Rows; k++)
                    {
                        temp = U[k, i];
                        U[k, i] = U[k, j];
                        U[k, j] = temp;
                    }

                    // Swap corresponding rows in VT
                    for (int k = 0; k < VT.Columns; k++)
                    {
                        temp = VT[i, k];
                        VT[i, k] = VT[j, k];
                        VT[j, k] = temp;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Computes the Singular Value Decomposition using the default algorithm.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the standard SVD algorithm that works in two main steps:
    /// 1. Bidiagonalization - transforms the matrix into a simpler form with non-zero values only on the main diagonal and one adjacent diagonal
    /// 2. Diagonalization - further simplifies the matrix to extract the singular values
    /// 
    /// Think of it as organizing a complex dataset into its most fundamental components.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdDefault(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int l = Math.Min(m, n);

        Matrix<T> A = matrix.Clone();
        Matrix<T> U = new(m, m);
        Vector<T> S = new(l);
        Matrix<T> VT = new(n, n);

        // Initialize U and VT as identity matrices
        for (int i = 0; i < m; i++) U[i, i] = NumOps.One;
        for (int i = 0; i < n; i++) VT[i, i] = NumOps.One;

        // Bidiagonalization
        for (int k = 0; k < l; k++)
        {
            // Householder transformation for columns
            T alpha = NumOps.Zero;
            for (int i = k; i < m; i++)
                alpha = NumOps.Add(alpha, NumOps.Multiply(A[i, k], A[i, k]));
            alpha = NumOps.Sqrt(alpha);
            if (NumOps.GreaterThan(A[k, k], NumOps.Zero))
                alpha = NumOps.Negate(alpha);

            T r = NumOps.Sqrt(NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(2), alpha), NumOps.Subtract(A[k, k], alpha)));
            Vector<T> u = new Vector<T>(m - k);
            u[0] = NumOps.Divide(NumOps.Subtract(A[k, k], alpha), r);
            for (int i = 1; i < m - k; i++)
                u[i] = NumOps.Divide(A[k + i, k], r);

            ApplyHouseholderLeft(A, u, k, m, k, n);
            ApplyHouseholderRight(U, u, 0, m, k, m);

            if (k < n - 1)
            {
                // Householder transformation for rows
                T beta = NumOps.Zero;
                for (int j = k + 1; j < n; j++)
                    beta = NumOps.Add(beta, NumOps.Multiply(A[k, j], A[k, j]));
                beta = NumOps.Sqrt(beta);
                if (NumOps.GreaterThan(A[k, k + 1], NumOps.Zero))
                    beta = NumOps.Negate(beta);

                r = NumOps.Sqrt(NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(2), beta), NumOps.Subtract(A[k, k + 1], beta)));
                Vector<T> v = new(n - k - 1)
                {
                    [0] = NumOps.Divide(NumOps.Subtract(A[k, k + 1], beta), r)
                };

                for (int j = 1; j < n - k - 1; j++)
                    v[j] = NumOps.Divide(A[k, k + j + 1], r);

                ApplyHouseholderRight(A, v, k, m, k + 1, n);
                ApplyHouseholderLeft(VT, v, k + 1, n, 0, n);
            }
        }

        // Diagonalization
        for (int i = 0; i < l; i++)
        {
            S[i] = A[i, i];
        }

        // Sort singular values in descending order
        for (int i = 0; i < l - 1; i++)
        {
            for (int j = i + 1; j < l; j++)
            {
                if (NumOps.LessThan(S[i], S[j]))
                {
                    T temp = S[i];
                    S[i] = S[j];
                    S[j] = temp;

                    for (int k = 0; k < m; k++)
                    {
                        temp = U[k, i];
                        U[k, i] = U[k, j];
                        U[k, j] = temp;
                    }

                    for (int k = 0; k < n; k++)
                    {
                        temp = VT[i, k];
                        VT[i, k] = VT[j, k];
                        VT[j, k] = temp;
                    }
                }
            }
        }

        return (U, S, VT);
    }

    /// <summary>
    /// Applies a Householder transformation from the left side to a matrix.
    /// </summary>
    /// <param name="A">The matrix to transform</param>
    /// <param name="u">The Householder vector</param>
    /// <param name="rowStart">Starting row index</param>
    /// <param name="rowEnd">Ending row index</param>
    /// <param name="colStart">Starting column index</param>
    /// <param name="colEnd">Ending column index</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Householder transformation is a mathematical operation that reflects vectors 
    /// across a plane. In SVD, we use it to systematically zero out elements in our matrix. This helps 
    /// simplify the matrix structure step by step until we can easily extract the singular values.
    /// </para>
    /// </remarks>
    private void ApplyHouseholderLeft(Matrix<T> A, Vector<T> u, int rowStart, int rowEnd, int colStart, int colEnd)
    {
        for (int j = colStart; j < colEnd; j++)
        {
            T s = NumOps.Zero;
            for (int i = 0; i < rowEnd - rowStart; i++)
                s = NumOps.Add(s, NumOps.Multiply(u[i], A[rowStart + i, j]));
            s = NumOps.Multiply(NumOps.FromDouble(2), s);

            for (int i = 0; i < rowEnd - rowStart; i++)
                A[rowStart + i, j] = NumOps.Subtract(A[rowStart + i, j], NumOps.Multiply(s, u[i]));
        }
    }

    /// <summary>
    /// Applies a Householder transformation from the right side to a matrix.
    /// </summary>
    /// <param name="A">The matrix to transform</param>
    /// <param name="u">The Householder vector</param>
    /// <param name="rowStart">Starting row index</param>
    /// <param name="rowEnd">Ending row index</param>
    /// <param name="colStart">Starting column index</param>
    /// <param name="colEnd">Ending column index</param>
    private void ApplyHouseholderRight(Matrix<T> A, Vector<T> u, int rowStart, int rowEnd, int colStart, int colEnd)
    {
        for (int i = rowStart; i < rowEnd; i++)
        {
            T s = NumOps.Zero;
            for (int j = 0; j < colEnd - colStart; j++)
                s = NumOps.Add(s, NumOps.Multiply(A[i, colStart + j], u[j]));
            s = NumOps.Multiply(NumOps.FromDouble(2), s);

            for (int j = 0; j < colEnd - colStart; j++)
                A[i, colStart + j] = NumOps.Subtract(A[i, colStart + j], NumOps.Multiply(s, u[j]));
        }
    }

    /// <summary>
    /// Computes the Singular Value Decomposition using the power iteration method.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Power iteration is like repeatedly applying a transformation to find the 
    /// dominant patterns in your data. Imagine shaking a box of mixed items - after enough shaking, 
    /// the heaviest items (dominant patterns) will settle to the bottom. This method finds singular 
    /// values one by one, from largest to smallest.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdPowerIteration(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int k = Math.Min(m, n); // Number of singular values to compute

        Matrix<T> U = new(m, k);
        Vector<T> S = new(k);
        Matrix<T> VT = new(k, n);

        Matrix<T> ATA = matrix.Transpose().Multiply(matrix);

        for (int i = 0; i < k; i++)
        {
            Vector<T> v = Vector<T>.CreateRandom(n);
            v = v.Normalize();

            for (int iter = 0; iter < 100; iter++) // You may need to adjust the number of iterations
            {
                v = ATA.Multiply(v);
                v = v.Normalize();
            }

            T sigma = NumOps.Sqrt(ATA.Multiply(v).DotProduct(v));
            Vector<T> u = matrix.Multiply(v).Divide(sigma);

            S[i] = sigma;
            for (int j = 0; j < m; j++) U[j, i] = u[j];
            for (int j = 0; j < n; j++) VT[i, j] = v[j];

            // Deflate ATA
            for (int r = 0; r < n; r++)
            {
                for (int c = 0; c < n; c++)
                {
                    ATA[r, c] = NumOps.Subtract(ATA[r, c], NumOps.Multiply(NumOps.Multiply(sigma, v[r]), v[c]));
                }
            }
        }

        return (U, S, VT);
    }

    /// <summary>
    /// Computes a truncated Singular Value Decomposition that keeps only the most significant singular values.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Truncated SVD is like creating a compressed version of your data that keeps 
    /// only the most important patterns. This is similar to image compression where you keep the 
    /// main features while discarding less noticeable details. It's useful for dimensionality 
    /// reduction and speeding up calculations.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeTruncatedSvd(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int k = Math.Min(m, n) / 2; // Truncate to half the singular values

        Matrix<T> Q = Matrix<T>.CreateRandom(m, k);
        Q = MatrixHelper<T>.OrthogonalizeColumns(Q);

        for (int iter = 0; iter < 5; iter++) // You may need to adjust the number of iterations
        {
            Q = matrix.Multiply(matrix.Transpose().Multiply(Q));
            Q = MatrixHelper<T>.OrthogonalizeColumns(Q);
        }

        Matrix<T> B = Q.Transpose().Multiply(matrix);
        var svd = new SvdDecomposition<T>(B);

        Matrix<T> U = Q.Multiply(svd.U);
        Vector<T> S = svd.S;
        Matrix<T> VT = svd.Vt;

        return (U, S, VT);
    }

    /// <summary>
    /// Computes the Singular Value Decomposition using a divide-and-conquer approach.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose</param>
    /// <returns>A tuple containing the U matrix, singular values S, and VT matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The divide-and-conquer approach breaks down a large problem into smaller, 
    /// more manageable pieces. It's like solving a large puzzle by first solving smaller sections 
    /// and then combining them. This method divides the matrix into four smaller matrices, computes 
    /// SVD for each, and then combines the results.
    /// </para>
    /// </remarks>
    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdDividedAndConquer(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        if (m <= 32 || n <= 32) // Base case: use another SVD method for small matrices
        {
            return ComputeSvdDefault(matrix);
        }

        // Divide the matrix into four submatrices
        int midM = m / 2;
        int midN = n / 2;

        Matrix<T> A11 = matrix.SubMatrix(0, midM, 0, midN);
        Matrix<T> A12 = matrix.SubMatrix(0, midM, midN, n);
        Matrix<T> A21 = matrix.SubMatrix(midM, m, 0, midN);
        Matrix<T> A22 = matrix.SubMatrix(midM, m, midN, n);

        // Recursively compute SVD for each submatrix
        var (U1, S1, VT1) = ComputeSvdDividedAndConquer(A11);
        var (U2, S2, VT2) = ComputeSvdDividedAndConquer(A12);
        var (U3, S3, VT3) = ComputeSvdDividedAndConquer(A21);
        var (U4, S4, VT4) = ComputeSvdDividedAndConquer(A22);

        // Combine the results
        Matrix<T> U = Matrix<T>.BlockDiagonal(U1, U2, U3, U4);
        Vector<T> S = Vector<T>.Concatenate(S1, S2, S3, S4);
        Matrix<T> VT = Matrix<T>.BlockDiagonal(VT1, VT2, VT3, VT4);

        // Perform a few iterations of the power method to refine the result
        for (int iter = 0; iter < 3; iter++)
        {
            Matrix<T> temp = matrix.Multiply(VT.Transpose());
            U = MatrixHelper<T>.OrthogonalizeColumns(temp);
            temp = U.Transpose().Multiply(matrix);
            VT = MatrixHelper<T>.OrthogonalizeColumns(temp.Transpose()).Transpose();
        }

        S = ComputeDiagonalElements(U.Transpose().Multiply(matrix).Multiply(VT.Transpose()));

        return (U, S, VT);
    }

    /// <summary>
    /// Extracts the diagonal elements from a matrix.
    /// </summary>
    /// <param name="matrix">The input matrix</param>
    /// <returns>A vector containing the diagonal elements</returns>
    private Vector<T> ComputeDiagonalElements(Matrix<T> matrix)
    {
        int minDim = Math.Min(matrix.Rows, matrix.Columns);
        Vector<T> diagonal = new Vector<T>(minDim);

        for (int i = 0; i < minDim; i++)
        {
            diagonal[i] = matrix[i, i];
        }

        return diagonal;
    }
}