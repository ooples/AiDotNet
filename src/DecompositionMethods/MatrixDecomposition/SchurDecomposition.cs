namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class SchurDecomposition<T> : IMatrixDecomposition<T>
{
    public Matrix<T> SchurMatrix { get; private set; }
    public Matrix<T> UnitaryMatrix { get; private set; }
    public Matrix<T> A { get; private set; }

    private readonly INumericOperations<T> NumOps;

    public SchurDecomposition(Matrix<T> matrix, SchurAlgorithm algorithm = SchurAlgorithm.Francis)
    {
        A = matrix;
        NumOps = MathHelper.GetNumericOperations<T>();
        (SchurMatrix, UnitaryMatrix) = Decompose(matrix, algorithm);
    }

    private (Matrix<T> S, Matrix<T> U) Decompose(Matrix<T> matrix, SchurAlgorithm algorithm)
    {
        return algorithm switch
        {
            SchurAlgorithm.Francis => ComputeSchurFrancis(matrix),
            SchurAlgorithm.QR => ComputeSchurQR(matrix),
            SchurAlgorithm.Implicit => ComputeSchurImplicit(matrix),
            _ => throw new ArgumentException("Unsupported Schur decomposition algorithm.")
        };
    }

    private (Matrix<T> S, Matrix<T> U) ComputeSchurQR(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Matrix<T> H = MatrixHelper.ReduceToHessenbergFormat(matrix);
        Matrix<T> U = Matrix<T>.CreateIdentity(n, NumOps);
        Matrix<T> S = H.Copy();

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

    private (Matrix<T> S, Matrix<T> U) ComputeSchurFrancis(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Matrix<T> H = MatrixHelper.ReduceToHessenbergFormat(matrix);
        Matrix<T> U = Matrix<T>.CreateIdentity(n, NumOps);

        const int maxIterations = 100;
        T tolerance = NumOps.FromDouble(1e-10);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            for (int i = 0; i < n - 1; i++)
            {
                T s = NumOps.Add(H[n - 2, n - 2], H[n - 1, n - 1]);
                T t = NumOps.Subtract(NumOps.Multiply(H[n - 2, n - 2], H[n - 1, n - 1]), NumOps.Multiply(H[n - 2, n - 1], H[n - 1, n - 2]));

                Matrix<T> Q = ComputeFrancisQRStep(H, s, t);
                H = Q.Transpose().Multiply(H).Multiply(Q);
                U = U.Multiply(Q);
            }

            if (H.IsUpperTriangularMatrix(tolerance))
                break;
        }

        return (H, U);
    }

    private Matrix<T> ComputeFrancisQRStep(Matrix<T> H, T s, T t)
    {
        int n = H.Rows;
        T x = NumOps.Subtract(NumOps.Subtract(H[0, 0], s), NumOps.Divide(NumOps.Multiply(H[0, 1], H[1, 0]), NumOps.Subtract(H[1, 1], s)));
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

    private (Matrix<T> S, Matrix<T> U) ComputeSchurImplicit(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Matrix<T> H = MatrixHelper.ReduceToHessenbergFormat(matrix);
        Matrix<T> U = Matrix<T>.CreateIdentity(n, NumOps);

        const int maxIterations = 100;
        T tolerance = NumOps.FromDouble(1e-10);

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

    private Matrix<T> ComputeImplicitQStep(Matrix<T> H, int start)
    {
        int n = H.Rows;
        T x = H[start, start];
        T y = H[start + 1, start];
        T z = start + 2 < n ? H[start + 2, start] : NumOps.Zero;

        for (int k = start; k < n - 1; k++)
        {
            Matrix<T> Q = ComputeHouseholderReflection(x, y, z);
            H = Q.Transpose().Multiply(H).Multiply(Q);

            if (k < n - 2)
            {
                x = H[k + 1, k];
                y = H[k + 2, k];
                z = k + 3 < n ? H[k + 3, k] : NumOps.Zero;
            }
        }

        return H;
    }

    private Matrix<T> ComputeHouseholderReflection(T x, T y, T? z = default)
    {
        int n = NumOps.Equals(z ?? NumOps.Zero, NumOps.Zero) ? 2 : 3;
        Vector<T> v = new(n, NumOps)
        {
            [0] = x,
            [1] = y
        };
        if (n == 3) v[2] = z ?? NumOps.Zero;

        T alpha = NumOps.Sqrt(v.DotProduct(v));
        if (NumOps.Equals(alpha, NumOps.Zero)) return Matrix<T>.CreateIdentity(n, NumOps);

        v[0] = NumOps.Add(v[0], NumOps.Multiply(NumOps.SignOrZero(x), alpha));
        T beta = NumOps.Sqrt(NumOps.Multiply(NumOps.FromDouble(2), v.DotProduct(v)));

        if (NumOps.Equals(beta, NumOps.Zero)) return Matrix<T>.CreateIdentity(n, NumOps);

        v = v.Divide(beta);
        return Matrix<T>.CreateIdentity(n, NumOps).Subtract(v.OuterProduct(v).Multiply(NumOps.FromDouble(2)));
    }

    public Vector<T> Solve(Vector<T> b)
    {
        var y = UnitaryMatrix.ForwardSubstitution(b);
        return SchurMatrix.BackwardSubstitution(y);
    }

    public Matrix<T> Invert()
    {
        var invU = UnitaryMatrix.Transpose();
        var invS = SchurMatrix.InvertUpperTriangularMatrix();
        return invS.Multiply(invU);
    }
}