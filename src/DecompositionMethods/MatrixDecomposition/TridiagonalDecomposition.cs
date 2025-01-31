using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class TridiagonalDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;

    public Matrix<T> A { get; }
    public Matrix<T> QMatrix { get; private set; }
    public Matrix<T> TMatrix { get; private set; }

    public TridiagonalDecomposition(Matrix<T> matrix, TridiagonalAlgorithmType algorithm = TridiagonalAlgorithmType.Householder)
    {
        A = matrix;
        NumOps = MathHelper.GetNumericOperations<T>();
        QMatrix = new Matrix<T>(matrix.Rows, matrix.Columns);
        TMatrix = new Matrix<T>(matrix.Rows, matrix.Columns);
        Decompose(algorithm);
    }

    public void Decompose(TridiagonalAlgorithmType algorithm = TridiagonalAlgorithmType.Householder)
    {
        switch (algorithm)
        {
            case TridiagonalAlgorithmType.Householder:
                DecomposeHouseholder();
                break;
            case TridiagonalAlgorithmType.Givens:
                DecomposeGivens();
                break;
            case TridiagonalAlgorithmType.Lanczos:
                DecomposeLanczos();
                break;
            default:
                throw new ArgumentException("Unsupported Tridiagonal decomposition algorithm.");
        }
    }

    private void DecomposeHouseholder()
    {
        int n = A.Rows;
        TMatrix = A.Copy();
        QMatrix = Matrix<T>.CreateIdentity(n);

        for (int k = 0; k < n - 2; k++)
        {
            Vector<T> x = TMatrix.GetColumn(k).GetSubVector(k + 1, n - k - 1);
            T alpha = NumOps.Multiply(NumOps.SignOrZero(x[0]), x.Norm());
            Vector<T> u = x.Subtract(Vector<T>.CreateDefault(x.Length, alpha).SetValue(0, x[0]));
            u = u.Divide(u.Norm());

            Matrix<T> P = Matrix<T>.CreateIdentity(n);
            for (int i = k + 1; i < n; i++)
            {
                for (int j = k + 1; j < n; j++)
                {
                    P[i, j] = NumOps.Subtract(P[i, j], NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(u[i - k - 1], u[j - k - 1])));
                }
            }

            TMatrix = P.Multiply(TMatrix).Multiply(P);
            QMatrix = QMatrix.Multiply(P);
        }
    }

    private void DecomposeGivens()
    {
        int n = A.Rows;
        QMatrix = Matrix<T>.CreateIdentity(n);
        TMatrix = A.Copy();

        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 2; j < n; j++)
            {
                if (!NumOps.Equals(TMatrix[j, i], NumOps.Zero))
                {
                    // Calculate Givens rotation
                    T a = TMatrix[i + 1, i];
                    T b = TMatrix[j, i];
                    T r = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(a, a), NumOps.Multiply(b, b)));
                    T c = NumOps.Divide(a, r);
                    T s = NumOps.Divide(b, r);

                    // Apply Givens rotation to TMatrix
                    for (int k = i; k < n; k++)
                    {
                        T temp1 = TMatrix[i + 1, k];
                        T temp2 = TMatrix[j, k];
                        TMatrix[i + 1, k] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                        TMatrix[j, k] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp1), NumOps.Multiply(c, temp2));
                    }

                    // Update QMatrix
                    for (int k = 0; k < n; k++)
                    {
                        T temp1 = QMatrix[k, i + 1];
                        T temp2 = QMatrix[k, j];
                        QMatrix[k, i + 1] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                        QMatrix[k, j] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp1), NumOps.Multiply(c, temp2));
                    }
                }
            }
        }

        // Ensure TMatrix is tridiagonal (set small values to zero)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (Math.Abs(i - j) > 1)
                {
                    TMatrix[i, j] = NumOps.Zero;
                }
            }
        }
    }

    private void DecomposeLanczos()
    {
        int n = A.Rows;
        QMatrix = new Matrix<T>(n, n);
        TMatrix = new Matrix<T>(n, n);

        Vector<T> v = new Vector<T>(n);
        v[0] = NumOps.One;
        Vector<T> w = A.Multiply(v);
        T alpha = w.DotProduct(v);
        w = w.Subtract(v.Multiply(alpha));
        T beta = w.Norm();

        QMatrix.SetColumn(0, v);
        TMatrix[0, 0] = alpha;

        for (int j = 1; j < n; j++)
        {
            if (NumOps.Equals(beta, NumOps.Zero))
            {
                break; // Early termination if beta becomes zero
            }

            v = w.Divide(beta);
            QMatrix.SetColumn(j, v);

            w = A.Multiply(v).Subtract(v.Multiply(beta));
            alpha = w.DotProduct(v);
            w = w.Subtract(v.Multiply(alpha));

            if (j < n - 1)
            {
                beta = w.Norm();
            }

            TMatrix[j, j] = alpha;
            TMatrix[j, j - 1] = beta;
            TMatrix[j - 1, j] = beta;
        }

        // Ensure QMatrix is orthogonal
        for (int i = 0; i < n; i++)
        {
            Vector<T> col = QMatrix.GetColumn(i);
            col = col.Divide(col.Norm());
            QMatrix.SetColumn(i, col);
        }
    }

    public Vector<T> Solve(Vector<T> b)
    {
        // Solve Tx = Q^T b
        Vector<T> y = QMatrix.Transpose().Multiply(b);
        Vector<T> x = SolveTridiagonal(y);

        return x;
    }

    private Vector<T> SolveTridiagonal(Vector<T> b)
    {
        int n = TMatrix.Rows;
        Vector<T> x = new(n);
        Vector<T> d = new(n);
        Vector<T> temp = new(n);

        // Forward elimination
        d[0] = TMatrix[0, 0];
        x[0] = b[0];
        for (int i = 1; i < n; i++)
        {
            temp[i] = NumOps.Divide(TMatrix[i, i - 1], d[i - 1]);
            d[i] = NumOps.Subtract(TMatrix[i, i], NumOps.Multiply(temp[i], TMatrix[i - 1, i]));
            x[i] = NumOps.Subtract(b[i], NumOps.Multiply(temp[i], x[i - 1]));
        }

        // Back substitution
        x[n - 1] = NumOps.Divide(x[n - 1], d[n - 1]);
        for (int i = n - 2; i >= 0; i--)
        {
            x[i] = NumOps.Divide(NumOps.Subtract(x[i], NumOps.Multiply(TMatrix[i, i + 1], x[i + 1])), d[i]);
        }

        return x;
    }

    public Matrix<T> Invert()
    {
        // Invert T
        Matrix<T> invT = InvertTridiagonal();

        // Compute Q * invT * Q^T
        return QMatrix.Multiply(invT).Multiply(QMatrix.Transpose());
    }

    private Matrix<T> InvertTridiagonal()
    {
        int n = TMatrix.Rows;
        Matrix<T> inv = new(n, n);

        for (int j = 0; j < n; j++)
        {
            Vector<T> e = Vector<T>.CreateStandardBasis(n, j);
            inv.SetColumn(j, SolveTridiagonal(e));
        }

        return inv;
    }

    public (Matrix<T> QMatrix, Matrix<T> TMatrix) GetFactors()
    {
        return (QMatrix, TMatrix);
    }
}