using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class SvdDecomposition<T> : IMatrixDecomposition<T>
{
    public Matrix<T> U { get; private set; }
    public Vector<T> S { get; private set; }
    public Matrix<T> Vt { get; private set; }
    public Matrix<T> A { get; private set; }

    private readonly INumericOperations<T> NumOps;

    public SvdDecomposition(Matrix<T> matrix, SvdAlgorithmType svdAlgorithm = SvdAlgorithmType.GolubReinsch)
    {
        A = matrix;
        NumOps = MathHelper.GetNumericOperations<T>();
        (U, S, Vt) = Decompose(matrix, svdAlgorithm);
    }

    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) Decompose(Matrix<T> matrix, SvdAlgorithmType algorithm)
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

    public Vector<T> Solve(Vector<T> b)
    {
        var x = new Vector<T>(Vt.Rows, NumOps);
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

    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdGolubReinsch(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int l = Math.Min(m, n);

        Matrix<T> A = matrix.Copy();
        Matrix<T> U = new(m, m, NumOps);
        Vector<T> S = new(l, NumOps);
        Matrix<T> VT = new(n, n, NumOps);

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

            Vector<T> u = x.Copy();
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

                Vector<T> v = x.Copy();
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

    private void DiagonalizeGolubReinsch(Matrix<T> B, Matrix<T> U, Matrix<T> VT)
    {
        int m = B.Rows;
        int n = B.Columns;
        int p = Math.Min(m, n);
        Vector<T> e = new(n, NumOps);
        Vector<T> d = new(p, NumOps);

        for (int i = 0; i < p; i++)
        {
            d[i] = B[i, i];
            if (i < n - 1)
                e[i] = B[i, i + 1];
        }

        for (int k = p - 1; k >= 0; k--)
        {
            int maxIterations = 30;
            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                bool flag = true;
                for (int l = k; l >= 0; l--)
                {
                    if (l == 0 || NumOps.LessThanOrEquals(NumOps.Abs(e[l - 1]), NumOps.Multiply(NumOps.FromDouble(1e-12), NumOps.Add(NumOps.Abs(d[l]), NumOps.Abs(d[l - 1])))))
                    {
                        e[l] = NumOps.Zero;
                        flag = false;
                        break;
                    }
                    if (NumOps.LessThanOrEquals(NumOps.Abs(d[l - 1]), NumOps.Multiply(NumOps.FromDouble(1e-12), NumOps.Abs(d[l]))))
                    {
                        GolubKahanStep(d, e, l, k, U, VT);
                        flag = false;
                        break;
                    }
                }
                if (flag)
                {
                    GolubKahanStep(d, e, 0, k, U, VT);
                }
            }
        }

        // Copy diagonal elements back to B
        for (int i = 0; i < p; i++)
        {
            B[i, i] = d[i];
        }
    }

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

    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdJacobi(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int l = Math.Min(m, n);

        Matrix<T> A = matrix.Copy();
        Matrix<T> U = Matrix<T>.CreateIdentityMatrix<T>(m);
        Vector<T> S = new(l, NumOps);
        Matrix<T> VT = Matrix<T>.CreateIdentityMatrix<T>(n);

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

    private Matrix<T> GenerateRandomMatrix(int rows, int cols)
    {
        var random = new Random();
        var matrix = new Matrix<T>(rows, cols, NumOps);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = NumOps.FromDouble(random.NextDouble() * 2 - 1);
            }
        }

        return matrix;
    }

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

    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdDefault(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int l = Math.Min(m, n);

        Matrix<T> A = matrix.Copy();
        Matrix<T> U = new(m, m, NumOps);
        Vector<T> S = new(l, NumOps);
        Matrix<T> VT = new(n, n, NumOps);

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
            Vector<T> u = new Vector<T>(m - k, NumOps);
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
                Vector<T> v = new(n - k - 1, NumOps)
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

    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeSvdPowerIteration(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int k = Math.Min(m, n); // Number of singular values to compute

        Matrix<T> U = new(m, k, NumOps);
        Vector<T> S = new(k, NumOps);
        Matrix<T> VT = new(k, n, NumOps);

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

    private (Matrix<T> U, Vector<T> S, Matrix<T> VT) ComputeTruncatedSvd(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        int k = Math.Min(m, n) / 2; // Truncate to half the singular values

        Matrix<T> Q = Matrix<T>.CreateRandom(m, k, NumOps);
        Q = MatrixHelper.OrthogonalizeColumns(Q);

        for (int iter = 0; iter < 5; iter++) // You may need to adjust the number of iterations
        {
            Q = matrix.Multiply(matrix.Transpose().Multiply(Q));
            Q = MatrixHelper.OrthogonalizeColumns(Q);
        }

        Matrix<T> B = Q.Transpose().Multiply(matrix);
        var svd = new SvdDecomposition<T>(B);

        Matrix<T> U = Q.Multiply(svd.U);
        Vector<T> S = svd.S;
        Matrix<T> VT = svd.Vt;

        return (U, S, VT);
    }

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
            U = MatrixHelper.OrthogonalizeColumns(temp);
            temp = U.Transpose().Multiply(matrix);
            VT = MatrixHelper.OrthogonalizeColumns(temp.Transpose()).Transpose();
        }

        S = ComputeDiagonalElements(U.Transpose().Multiply(matrix).Multiply(VT.Transpose()));

        return (U, S, VT);
    }

    private Vector<T> ComputeDiagonalElements(Matrix<T> matrix)
    {
        int minDim = Math.Min(matrix.Rows, matrix.Columns);
        Vector<T> diagonal = new Vector<T>(minDim, NumOps);

        for (int i = 0; i < minDim; i++)
        {
            diagonal[i] = matrix[i, i];
        }

        return diagonal;
    }

    public Matrix<T> Invert()
    {
        return MatrixHelper.InvertUsingDecomposition(this);
    }
}