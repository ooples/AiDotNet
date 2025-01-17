namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class LdlDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;

    public Matrix<T> A { get; }
    public Matrix<T> L { get; private set; }
    public Vector<T> D { get; private set; }

    public LdlDecomposition(Matrix<T> matrix, LdlAlgorithm algorithm = LdlAlgorithm.Cholesky)
    {
        if (!matrix.IsSquareMatrix())
            throw new ArgumentException("Matrix must be square for LDL decomposition.");

        NumOps = MathHelper.GetNumericOperations<T>();
        A = matrix;
        int n = A.Rows;
        L = new Matrix<T>(n, n, NumOps);
        D = new Vector<T>(n, NumOps);
        Decompose(algorithm);
    }

    public void Decompose(LdlAlgorithm algorithm = LdlAlgorithm.Cholesky)
    {
        switch (algorithm)
        {
            case LdlAlgorithm.Cholesky:
                DecomposeCholesky();
                break;
            case LdlAlgorithm.Crout:
                DecomposeCrout();
                break;
            default:
                throw new ArgumentException("Unsupported LDL decomposition algorithm.");
        }
    }

    private void DecomposeCholesky()
    {
        int n = A.Rows;
        L = new Matrix<T>(n, n, NumOps);
        D = new Vector<T>(n, NumOps);

        for (int j = 0; j < n; j++)
        {
            T sum = NumOps.Zero;
            for (int k = 0; k < j; k++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(L[j, k], L[j, k]), D[k]));
            }
            D[j] = NumOps.Subtract(A[j, j], sum);

            L[j, j] = NumOps.One;

            for (int i = j + 1; i < n; i++)
            {
                sum = NumOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(L[i, k], L[j, k]), D[k]));
                }
                L[i, j] = NumOps.Divide(NumOps.Subtract(A[i, j], sum), D[j]);
            }
        }
    }

    private void DecomposeCrout()
    {
        int n = A.Rows;
        L = new Matrix<T>(n, n, NumOps);
        D = new Vector<T>(n, NumOps);

        for (int j = 0; j < n; j++)
        {
            T sum = NumOps.Zero;
            for (int k = 0; k < j; k++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(L[j, k], L[j, k]), D[k]));
            }
            D[j] = NumOps.Subtract(A[j, j], sum);

            L[j, j] = NumOps.One;

            for (int i = j + 1; i < n; i++)
            {
                sum = NumOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(L[i, k], L[j, k]), D[k]));
                }
                L[i, j] = NumOps.Divide(NumOps.Subtract(A[i, j], sum), D[j]);
            }
        }
    }

    public Vector<T> Solve(Vector<T> b)
    {
        if (b.Length != A.Rows)
            throw new ArgumentException("Vector b must have the same length as the number of rows in matrix A.");

        // Forward substitution
        Vector<T> y = new(b.Length, NumOps);
        for (int i = 0; i < b.Length; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < i; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(L[i, j], y[j]));
            }
            y[i] = NumOps.Subtract(b[i], sum);
        }

        // Diagonal scaling
        for (int i = 0; i < b.Length; i++)
        {
            y[i] = NumOps.Divide(y[i], D[i]);
        }

        // Backward substitution
        Vector<T> x = new Vector<T>(b.Length, NumOps);
        for (int i = b.Length - 1; i >= 0; i--)
        {
            T sum = NumOps.Zero;
            for (int j = i + 1; j < b.Length; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(L[j, i], x[j]));
            }
            x[i] = NumOps.Subtract(y[i], sum);
        }

        return x;
    }

    public Matrix<T> Invert()
    {
        int n = A.Rows;
        Matrix<T> inverse = new(n, n, NumOps);

        for (int i = 0; i < n; i++)
        {
            Vector<T> ei = new(n, NumOps)
            {
                [i] = NumOps.One
            };
            Vector<T> column = Solve(ei);
            for (int j = 0; j < n; j++)
            {
                inverse[j, i] = column[j];
            }
        }

        return inverse;
    }

    public (Matrix<T> L, Vector<T> D) GetFactors()
    {
        return (L, D);
    }
}