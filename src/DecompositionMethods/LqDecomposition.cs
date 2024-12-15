namespace AiDotNet.DecompositionMethods;

public class LqDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;

    public Matrix<T> L { get; private set; }
    public Matrix<T> Q { get; private set; }
    public Matrix<T> A { get; private set; }

    public LqDecomposition(Matrix<T> matrix, LqAlgorithm algorithm = LqAlgorithm.Householder)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        A = matrix;
        (L, Q) = Decompose(matrix, algorithm);
    }

    public Vector<T> Solve(Vector<T> b)
    {
        var y = ForwardSubstitution(L, b);
        return Q.Transpose().Multiply(y);
    }

    private (Matrix<T> L, Matrix<T> Q) Decompose(Matrix<T> matrix, LqAlgorithm algorithm)
    {
        return algorithm switch
        {
            LqAlgorithm.Householder => ComputeLqHouseholder(matrix),
            LqAlgorithm.GramSchmidt => ComputeLqGramSchmidt(matrix),
            LqAlgorithm.Givens => ComputeLqGivens(matrix),
            _ => throw new ArgumentException("Unsupported LQ decomposition algorithm."),
        };
    }

    private (Matrix<T> L, Matrix<T> Q) ComputeLqHouseholder(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var L = new Matrix<T>(m, n);
        var Q = Matrix<T>.CreateIdentityMatrix<T>(n);

        var a = matrix.Copy();

        for (int k = 0; k < Math.Min(m, n); k++)
        {
            var x = new Vector<T>(m - k);
            for (int i = k; i < m; i++)
            {
                x[i - k] = a[i, k];
            }

            T alpha = NumOps.Negate(NumOps.SignOrZero(x[0]));
            alpha = NumOps.Multiply(alpha, NumOps.Sqrt(x.DotProduct(x)));

            var u = new Vector<T>(x.Length);
            u[0] = NumOps.Subtract(x[0], alpha);
            for (int i = 1; i < x.Length; i++)
            {
                u[i] = x[i];
            }

            T norm_u = NumOps.Sqrt(u.DotProduct(u));
            for (int i = 0; i < u.Length; i++)
            {
                u[i] = NumOps.Divide(u[i], norm_u);
            }

            var uMatrix = new Matrix<T>(u.Length, 1);
            for (int i = 0; i < u.Length; i++)
            {
                uMatrix[i, 0] = u[i];
            }

            var uT = uMatrix.Transpose();
            var uTu = uMatrix.Multiply(uT);

            for (int i = 0; i < m - k; i++)
            {
                uTu[i, i] = NumOps.Subtract(uTu[i, i], NumOps.One);
            }

            var P = Matrix<T>.CreateIdentityMatrix<T>(m);
            for (int i = k; i < m; i++)
            {
                for (int j = k; j < m; j++)
                {
                    P[i, j] = NumOps.Add(P[i, j], NumOps.Multiply(NumOps.FromDouble(2), uTu[i - k, j - k]));
                }
            }

            a = P.Multiply(a);
            Q = Q.Multiply(P);
        }

        L = a;

        // Ensure Q is orthogonal
        for (int i = 0; i < Q.Rows; i++)
        {
            for (int j = 0; j < Q.Columns; j++)
            {
                if (i == j)
                {
                    Q[i, j] = NumOps.One;
                }
                else
                {
                    Q[i, j] = NumOps.Negate(Q[i, j]);
                }
            }
        }

        return (L, Q);
    }

    private (Matrix<T> L, Matrix<T> Q) ComputeLqGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var L = new Matrix<T>(m, n);
        var Q = Matrix<T>.CreateIdentityMatrix<T>(n);

        for (int i = 0; i < m; i++)
        {
            var v = matrix.GetRow(i);
            for (int j = 0; j < i; j++)
            {
                var q = Q.GetColumn(j);
                T proj = v.DotProduct(q);
                for (int k = 0; k < n; k++)
                {
                    v[k] = NumOps.Subtract(v[k], NumOps.Multiply(proj, q[k]));
                }
            }

            T norm = NumOps.Sqrt(v.DotProduct(v));
            for (int j = 0; j < n; j++)
            {
                Q[j, i] = NumOps.Divide(v[j], norm);
            }

            for (int j = 0; j <= i; j++)
            {
                L[i, j] = matrix.GetRow(i).DotProduct(Q.GetColumn(j));
            }
        }

        return (L, Q);
    }

    private (Matrix<T> L, Matrix<T> Q) ComputeLqGivens(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var L = matrix.Copy();
        var Q = Matrix<T>.CreateIdentityMatrix<T>(n);

        for (int i = m - 1; i >= 0; i--)
        {
            for (int j = n - 1; j > i; j--)
            {
                if (!NumOps.Equals(L[i, j], NumOps.Zero))
                {
                    T a = L[i, j - 1];
                    T b = L[i, j];
                    T r = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(a, a), NumOps.Multiply(b, b)));
                    T c = NumOps.Divide(a, r);
                    T s = NumOps.Divide(b, r);

                    for (int k = 0; k < m; k++)
                    {
                        T temp = L[k, j - 1];
                        L[k, j - 1] = NumOps.Add(NumOps.Multiply(c, temp), NumOps.Multiply(s, L[k, j]));
                        L[k, j] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp), NumOps.Multiply(c, L[k, j]));
                    }

                    for (int k = 0; k < n; k++)
                    {
                        T temp = Q[j - 1, k];
                        Q[j - 1, k] = NumOps.Add(NumOps.Multiply(c, temp), NumOps.Multiply(s, Q[j, k]));
                        Q[j, k] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp), NumOps.Multiply(c, Q[j, k]));
                    }
                }
            }
        }

        return (L, Q.Transpose());
    }

    private Vector<T> ForwardSubstitution(Matrix<T> L, Vector<T> b)
    {
        int n = L.Rows;
        var y = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < i; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(L[i, j], y[j]));
            }
            y[i] = NumOps.Divide(NumOps.Subtract(b[i], sum), L[i, i]);
        }

        return y;
    }

    public Matrix<T> Invert()
    {
        return MatrixHelper.InvertUsingDecomposition(this);
    }
}