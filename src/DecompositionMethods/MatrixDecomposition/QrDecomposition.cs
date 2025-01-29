using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class QrDecomposition<T> : IMatrixDecomposition<T>
{
    public Matrix<T> Q { get; private set; }
    public Matrix<T> R { get; private set; }
    public Matrix<T> A { get; private set; }

    private readonly INumericOperations<T> NumOps;

    public QrDecomposition(Matrix<T> matrix, QrAlgorithmType qrAlgorithm = QrAlgorithmType.Householder)
    {
        A = matrix;
        NumOps = MathHelper.GetNumericOperations<T>();
        (Q, R) = Decompose(matrix, qrAlgorithm);
    }

    public Vector<T> Solve(Vector<T> b)
    {
        var y = Q.Transpose().Multiply(b);
        return BackSubstitution(R, y);
    }

    private (Matrix<T> Q, Matrix<T> R) Decompose(Matrix<T> matrix, QrAlgorithmType algorithm)
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

    private (Matrix<T> Q, Matrix<T> R) ComputeQrGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n, NumOps);
        Matrix<T> R = new(n, n, NumOps);

        for (int j = 0; j < n; j++)
        {
            Vector<T> v = matrix.GetColumn(j);
            for (int i = 0; i < j; i++)
            {
                R[i, j] = Q.GetColumn(i).DotProduct(v);
                v = v.Subtract(Q.GetColumn(i).Multiply(R[i, j]));
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

    private (Matrix<T> Q, Matrix<T> R) ComputeQrHouseholder(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = Matrix<T>.CreateIdentityMatrix<T>(m);
        Matrix<T> R = matrix.Copy();

        for (int k = 0; k < n; k++)
        {
            Vector<T> x = R.GetSubColumn(k, k, m - k);
            T normX = x.Norm();
            Vector<T> e = new(m - k, NumOps)
            {
                [0] = NumOps.One
            };

            Vector<T> u = x.Add(e.Multiply(normX));
            T normU = u.Norm();

            if (!NumOps.Equals(normU, NumOps.Zero))
            {
                u = u.Divide(normU);
                Matrix<T> H = Matrix<T>.CreateIdentityMatrix<T>(m - k)
                    .Subtract(u.OuterProduct(u).Multiply(NumOps.FromDouble(2)));

                Matrix<T> QkTranspose = Matrix<T>.CreateIdentityMatrix<T>(m);
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

    private (Matrix<T> Q, Matrix<T> R) ComputeQrGivens(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = Matrix<T>.CreateIdentityMatrix<T>(m);
        Matrix<T> R = matrix.Copy();

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

                    Matrix<T> G = Matrix<T>.CreateIdentityMatrix<T>(m);
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

    private (Matrix<T> Q, Matrix<T> R) ComputeQrModifiedGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n, NumOps);
        Matrix<T> R = new(n, n, NumOps);

        for (int k = 0; k < n; k++)
        {
            Vector<T> v = matrix.GetColumn(k);
            R[k, k] = v.Norm();
            Q.SetColumn(k, v.Divide(R[k, k]));

            for (int j = k + 1; j < n; j++)
            {
                R[k, j] = Q.GetColumn(k).DotProduct(matrix.GetColumn(j));
                matrix.SetColumn(j, matrix.GetColumn(j).Subtract(Q.GetColumn(k).Multiply(R[k, j])));
            }
        }

        return (Q, R);
    }

    private (Matrix<T> Q, Matrix<T> R) ComputeQrIterativeGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n, NumOps);
        Matrix<T> R = new(n, n, NumOps);

        for (int k = 0; k < n; k++)
        {
            Vector<T> v = matrix.GetColumn(k);
            for (int i = 0; i < 2; i++) // Perform two iterations
            {
                for (int j = 0; j < k; j++)
                {
                    T r = Q.GetColumn(j).DotProduct(v);
                    R[j, k] = NumOps.Add(R[j, k], r);
                    v = v.Subtract(Q.GetColumn(j).Multiply(r));
                }
            }
            R[k, k] = v.Norm();
            Q.SetColumn(k, v.Divide(R[k, k]));
        }

        return (Q, R);
    }

    private Vector<T> BackSubstitution(Matrix<T> R, Vector<T> y)
    {
        var x = new Vector<T>(R.Columns, NumOps);
        for (int i = R.Columns - 1; i >= 0; i--)
        {
            T sum = NumOps.Zero;
            for (int j = i + 1; j < R.Columns; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(R[i, j], x[j]));
            }
            x[i] = NumOps.Divide(NumOps.Subtract(y[i], sum), R[i, i]);
        }

        return x;
    }

    public Matrix<T> Invert()
    {
        return MatrixHelper.InvertUsingDecomposition(this);
    }
}