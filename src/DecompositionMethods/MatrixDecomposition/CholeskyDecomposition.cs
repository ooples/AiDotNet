using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class CholeskyDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;

    public Matrix<T> L { get; private set; }
    public Matrix<T> A { get; private set; }

    public CholeskyDecomposition(Matrix<T> matrix, CholeskyAlgorithmType algorithm = CholeskyAlgorithmType.Crout)
    {
        A = matrix;
        NumOps = MathHelper.GetNumericOperations<T>();
        L = Decompose(matrix, algorithm);
    }

    public Vector<T> Solve(Vector<T> b)
    {
        var y = ForwardSubstitution(L, b);
        return BackSubstitution(L.Transpose(), y);
    }

    private Matrix<T> Decompose(Matrix<T> matrix, CholeskyAlgorithmType algorithm)
    {
        return algorithm switch
        {
            CholeskyAlgorithmType.Crout => ComputeCholeskyCrout(matrix),
            CholeskyAlgorithmType.Banachiewicz => ComputeCholeskyBanachiewicz(matrix),
            CholeskyAlgorithmType.LDL => ComputeCholeskyLDL(matrix),
            CholeskyAlgorithmType.BlockCholesky => ComputeBlockCholesky(matrix),
            _ => throw new ArgumentException("Unsupported Cholesky decomposition algorithm.")
        };
    }

    private Matrix<T> ComputeCholeskyDefault(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        int n = matrix.Rows;
        var L = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                if (i != j && !NumOps.Equals(matrix[i, j], matrix[j, i]))
                {
                    throw new ArgumentException("Matrix must be symmetric for Cholesky decomposition.");
                }

                T sum = NumOps.Zero;

                if (j == i) // Diagonal elements
                {
                    for (int k = 0; k < j; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(L[j, k], L[j, k]));
                    }
                    T diagonalValue = NumOps.Subtract(matrix[j, j], sum);
                    if (NumOps.LessThanOrEquals(diagonalValue, NumOps.Zero))
                    {
                        throw new ArgumentException("Matrix is not positive definite.");
                    }
                    L[j, j] = NumOps.Sqrt(diagonalValue);
                }
                else // Lower triangular elements
                {
                    for (int k = 0; k < j; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(L[i, k], L[j, k]));
                    }
                    L[i, j] = NumOps.Divide(NumOps.Subtract(matrix[i, j], sum), L[j, j]);
                }
            }
        }

        return L;
    }

    private Matrix<T> ComputeCholeskyCrout(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        int n = matrix.Rows;
        var L = new Matrix<T>(n, n);

        for (int j = 0; j < n; j++)
        {
            T sum = NumOps.Zero;
            for (int k = 0; k < j; k++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(L[j, k], L[j, k]));
            }
            T diagonalValue = NumOps.Subtract(matrix[j, j], sum);
            if (NumOps.LessThanOrEquals(diagonalValue, NumOps.Zero))
            {
                throw new ArgumentException("Matrix is not positive definite.");
            }
            L[j, j] = NumOps.Sqrt(diagonalValue);

            for (int i = j + 1; i < n; i++)
            {
                sum = NumOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(L[i, k], L[j, k]));
                }
                L[i, j] = NumOps.Divide(NumOps.Subtract(matrix[i, j], sum), L[j, j]);
            }
        }

        return L;
    }

    private Matrix<T> ComputeCholeskyBanachiewicz(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        int n = matrix.Rows;
        var L = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(L[i, k], L[j, k]));
                }

                if (i == j)
                {
                    T diagonalValue = NumOps.Subtract(matrix[i, i], sum);
                    if (NumOps.LessThanOrEquals(diagonalValue, NumOps.Zero))
                    {
                        throw new ArgumentException("Matrix is not positive definite.");
                    }
                    L[i, j] = NumOps.Sqrt(diagonalValue);
                }
                else
                {
                    L[i, j] = NumOps.Divide(NumOps.Subtract(matrix[i, j], sum), L[j, j]);
                }
            }
        }

        return L;
    }

    private Matrix<T> ComputeCholeskyLDL(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        int n = matrix.Rows;
        var L = Matrix<T>.CreateIdentity(n);
        var D = new Vector<T>(n);

        for (int j = 0; j < n; j++)
        {
            T d = matrix[j, j];
            for (int k = 0; k < j; k++)
            {
                d = NumOps.Subtract(d, NumOps.Multiply(NumOps.Multiply(L[j, k], L[j, k]), D[k]));
            }
            D[j] = d;

            for (int i = j + 1; i < n; i++)
            {
                T sum = matrix[i, j];
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Subtract(sum, NumOps.Multiply(NumOps.Multiply(L[i, k], L[j, k]), D[k]));
                }
                L[i, j] = NumOps.Divide(sum, d);
            }
        }

        // Convert LDL' to LL'
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                L[i, j] = NumOps.Multiply(L[i, j], NumOps.Sqrt(D[j]));
            }
        }

        return L;
    }

    private Matrix<T> ComputeBlockCholesky(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        int n = matrix.Rows;
        int blockSize = 64; // Adjust this based on your needs
        var L = new Matrix<T>(n, n);

        for (int i = 0; i < n; i += blockSize)
        {
            int size = Math.Min(blockSize, n - i);
            var subMatrix = matrix.SubMatrix(i, i, size, size);
            var subL = ComputeCholeskyCrout(subMatrix);

            for (int r = 0; r < size; r++)
            {
                for (int c = 0; c <= r; c++)
                {
                    L[i + r, i + c] = subL[r, c];
                }
            }

            if (i + size < n)
            {
                var B = matrix.SubMatrix(i + size, i, n - i - size, size);
                var subLCholesky = new CholeskyDecomposition<T>(L.SubMatrix(i, i, size, size));
                var X = subLCholesky.SolveMatrix(B.Transpose());

                for (int r = 0; r < n - i - size; r++)
                {
                    for (int c = 0; c < size; c++)
                    {
                        L[i + size + r, i + c] = X[c, r];
                    }
                }

                var C = matrix.SubMatrix(i + size, i + size, n - i - size, n - i - size);
                for (int r = 0; r < n - i - size; r++)
                {
                    for (int c = 0; c <= r; c++)
                    {
                        T sum = NumOps.Zero;
                        for (int k = 0; k < size; k++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(L[i + size + r, i + k], L[i + size + c, i + k]));
                        }
                        C[r, c] = NumOps.Subtract(C[r, c], sum);
                        C[c, r] = C[r, c];
                    }
                }
            }
        }

        return L;
    }

    private Matrix<T> SolveMatrix(Matrix<T> B)
    {
        int columns = B.Columns;
        var X = new Matrix<T>(L.Columns, columns);

        for (int i = 0; i < columns; i++)
        {
            var columnVector = B.GetColumn(i);
            var solutionVector = Solve(columnVector);
            X.SetColumn(i, solutionVector);
        }

        return X;
    }

    private Vector<T> ForwardSubstitution(Matrix<T> L, Vector<T> b)
    {
        var y = new Vector<T>(L.Rows);
        for (int i = 0; i < L.Rows; i++)
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

    private Vector<T> BackSubstitution(Matrix<T> LT, Vector<T> y)
    {
        var x = new Vector<T>(LT.Columns);
        for (int i = LT.Columns - 1; i >= 0; i--)
        {
            T sum = NumOps.Zero;
            for (int j = i + 1; j < LT.Columns; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(LT[i, j], x[j]));
            }
            x[i] = NumOps.Divide(NumOps.Subtract(y[i], sum), LT[i, i]);
        }

        return x;
    }

    public Matrix<T> Invert()
    {
        return MatrixHelper<T>.InvertUsingDecomposition(this);
    }
}