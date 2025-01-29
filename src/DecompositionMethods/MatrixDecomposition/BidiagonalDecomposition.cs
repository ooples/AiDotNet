using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class BidiagonalDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;

    public Matrix<T> A { get; }
    public Matrix<T> U { get; private set; }
    public Matrix<T> B { get; private set; }
    public Matrix<T> V { get; private set; }

    public BidiagonalDecomposition(Matrix<T> matrix, BidiagonalAlgorithmType algorithm = BidiagonalAlgorithmType.Householder)
    {
        A = matrix;
        NumOps = MathHelper.GetNumericOperations<T>();
        U = new Matrix<T>(matrix.Rows, matrix.Rows, NumOps);
        B = new Matrix<T>(matrix.Rows, matrix.Columns, NumOps);
        V = new Matrix<T>(matrix.Columns, matrix.Columns, NumOps);
        Decompose(algorithm);
    }

    public void Decompose(BidiagonalAlgorithmType algorithm = BidiagonalAlgorithmType.Householder)
    {
        switch (algorithm)
        {
            case BidiagonalAlgorithmType.Householder:
                DecomposeHouseholder();
                break;
            case BidiagonalAlgorithmType.Givens:
                DecomposeGivens();
                break;
            case BidiagonalAlgorithmType.Lanczos:
                DecomposeLanczos();
                break;
            default:
                throw new ArgumentException("Unsupported Bidiagonal decomposition algorithm.");
        }
    }

    private void DecomposeHouseholder()
    {
        int m = A.Rows;
        int n = A.Columns;
        B = A.Copy();
        U = Matrix<T>.CreateIdentity(m);
        V = Matrix<T>.CreateIdentity(n);

        for (int k = 0; k < Math.Min(m - 1, n); k++)
        {
            // Compute Householder vector for column
            Vector<T> x = B.GetColumnSegment(k, k, m - k);
            Vector<T> v = HouseholderVector(x);

            // Apply Householder reflection to B
            Matrix<T> P = Matrix<T>.CreateIdentity(m - k, NumOps).Subtract(v.OuterProduct(v).Multiply(NumOps.FromDouble(2)));
            Matrix<T> subB = B.GetSubMatrix(k, k, m - k, n - k);
            B.SetSubMatrix(k, k, P.Multiply(subB));

            // Update U
            Matrix<T> subU = U.GetSubMatrix(0, k, m, m - k);
            U.SetSubMatrix(0, k, subU.Multiply(P.Transpose()));

            if (k < n - 2)
            {
                // Compute Householder vector for row
                x = B.GetRowSegment(k, k + 1, n - k - 1);
                v = HouseholderVector(x);

                // Apply Householder reflection to B
                P = Matrix<T>.CreateIdentity(n - k - 1, NumOps).Subtract(v.OuterProduct(v).Multiply(NumOps.FromDouble(2)));
                subB = B.GetSubMatrix(k, k + 1, m - k, n - k - 1);
                B.SetSubMatrix(k, k + 1, subB.Multiply(P));

                // Update V
                Matrix<T> subV = V.GetSubMatrix(k + 1, 0, n - k - 1, n);
                V.SetSubMatrix(k + 1, 0, P.Multiply(subV));
            }
        }
    }

    private void DecomposeGivens()
    {
        int m = A.Rows;
        int n = A.Columns;
        B = A.Copy();
        U = Matrix<T>.CreateIdentity(m, NumOps);
        V = Matrix<T>.CreateIdentity(n, NumOps);

        for (int k = 0; k < Math.Min(m - 1, n); k++)
        {
            for (int i = m - 1; i > k; i--)
            {
                GivensRotation(B, U, i - 1, i, k, k, true);
            }

            if (k < n - 2)
            {
                for (int j = n - 1; j > k + 1; j--)
                {
                    GivensRotation(B, V, k, k, j - 1, j, false);
                }
            }
        }
    }

    private void DecomposeLanczos()
    {
        int m = A.Rows;
        int n = A.Columns;
        B = new Matrix<T>(m, n, NumOps);
        U = new Matrix<T>(m, m, NumOps);
        V = new Matrix<T>(n, n, NumOps);

        Vector<T> u = new Vector<T>(m, NumOps);
        Vector<T> v = new Vector<T>(n, NumOps);

        // Initialize with random unit vector
        Random rand = new();
        for (int i = 0; i < n; i++)
        {
            v[i] = NumOps.FromDouble(rand.NextDouble());
        }
        v = v.Divide(v.Norm());

        for (int j = 0; j < Math.Min(m, n); j++)
        {
            u = A.Multiply(v).Subtract(B.GetColumn(j - 1).Multiply(B[j - 1, j]));
            T alpha = u.Norm();
            u = u.Divide(alpha);

            v = A.Transpose().Multiply(u).Subtract(B.GetRow(j).Multiply(alpha));
            T beta = v.Norm();
            v = v.Divide(beta);

            B[j, j] = alpha;
            if (j < n - 1) B[j, j + 1] = beta;

            U.SetColumn(j, u);
            if (j < n) V.SetColumn(j, v);
        }
    }

    public Vector<T> Solve(Vector<T> b)
    {
        if (b.Length != A.Rows)
            throw new ArgumentException("Vector b must have the same length as the number of rows in matrix A.");

        // Solve Ax = b using U*B*V^T * x = b
        Vector<T> y = U.Transpose().Multiply(b);
        Vector<T> z = SolveBidiagonal(y);
        return V.Multiply(z);
    }

    public Matrix<T> Invert()
    {
        int n = A.Columns;
        Matrix<T> inverse = new Matrix<T>(n, n, NumOps);

        for (int i = 0; i < n; i++)
        {
            Vector<T> ei = new Vector<T>(n, NumOps);
            ei[i] = NumOps.One;
            inverse.SetColumn(i, Solve(ei));
        }

        return inverse;
    }

    private Vector<T> HouseholderVector(Vector<T> x)
    {
        T norm = x.Norm();
        Vector<T> v = x.Copy();
        v[0] = NumOps.Add(v[0], NumOps.Multiply(NumOps.SignOrZero(x[0]), norm));

        return v.Divide(v.Norm());
    }

    private void GivensRotation(Matrix<T> M, Matrix<T> Q, int i, int k, int j, int l, bool isLeft)
    {
        T a = M[i, j];
        T b = M[k, l];
        T r = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(a, a), NumOps.Multiply(b, b)));
        T c = NumOps.Divide(a, r);
        T s = NumOps.Divide(b, r);

        if (isLeft)
        {
            for (int j2 = 0; j2 < M.Columns; j2++)
            {
                T temp1 = M[i, j2];
                T temp2 = M[k, j2];
                M[i, j2] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                M[k, j2] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp1), NumOps.Multiply(c, temp2));
            }

            for (int i2 = 0; i2 < Q.Rows; i2++)
            {
                T temp1 = Q[i2, i];
                T temp2 = Q[i2, k];
                Q[i2, i] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                Q[i2, k] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp1), NumOps.Multiply(c, temp2));
            }
        }
        else
        {
            for (int i2 = 0; i2 < M.Rows; i2++)
            {
                T temp1 = M[i2, j];
                T temp2 = M[i2, l];
                M[i2, j] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                M[i2, l] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp1), NumOps.Multiply(c, temp2));
            }

            for (int j2 = 0; j2 < Q.Columns; j2++)
            {
                T temp1 = Q[j, j2];
                T temp2 = Q[l, j2];
                Q[j, j2] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                Q[l, j2] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp1), NumOps.Multiply(c, temp2));
            }
        }
    }

    private Vector<T> SolveBidiagonal(Vector<T> y)
    {
        int n = B.Columns;
        Vector<T> x = new(n, NumOps);

        for (int i = n - 1; i >= 0; i--)
        {
            T sum = y[i];
            if (i < n - 1)
                sum = NumOps.Subtract(sum, NumOps.Multiply(B[i, i + 1], x[i + 1]));
            x[i] = NumOps.Divide(sum, B[i, i]);
        }

        return x;
    }

    public (Matrix<T> U, Matrix<T> B, Matrix<T> V) GetFactors()
    {
        return (U, B, V);
    }
}