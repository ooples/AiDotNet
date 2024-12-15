namespace AiDotNet.LinearAlgebra;

public class LuDecomposition<T> : IMatrixDecomposition<T>
{
    public Matrix<T> L { get; private set; }
    public Matrix<T> U { get; private set; }
    public Vector<int> P { get; private set; }
    public Matrix<T> A { get; private set; }

    private readonly INumericOperations<T> NumOps;

    public LuDecomposition(Matrix<T> matrix, LuAlgorithm luAlgorithm = LuAlgorithm.PartialPivoting)
    {
        A = matrix;
        NumOps = MathHelper.GetNumericOperations<T>();
        (L, U, P) = Decompose(matrix, luAlgorithm);
    }

    public Vector<T> Solve(Vector<T> b)
    {
        var pb = PermutateVector(b, P);
        var y = ForwardSubstitution(L, pb);

        return BackSubstitution(U, y);
    }

    private (Matrix<T> L, Matrix<T> U, Vector<int> P) Decompose(Matrix<T> matrix, LuAlgorithm algorithm)
    {
        return algorithm switch
        {
            LuAlgorithm.Doolittle => ComputeLuDoolittle(matrix),
            LuAlgorithm.Crout => ComputeLuCrout(matrix),
            LuAlgorithm.PartialPivoting => ComputeLuPartialPivoting(matrix),
            LuAlgorithm.CompletePivoting => ComputeLuCompletePivoting(matrix),
            LuAlgorithm.Cholesky => ComputeCholesky(matrix),
            _ => throw new ArgumentException("Unsupported LU decomposition algorithm."),
        };
    }

    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuPartialPivoting(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        Matrix<T> A = matrix.Copy();
        Matrix<T> L = new(n, n, NumOps);
        Vector<int> P = new(n, MathHelper.GetNumericOperations<int>());

        for (int i = 0; i < n; i++)
            P[i] = i;

        for (int k = 0; k < n - 1; k++)
        {
            int pivotRow = k;
            T pivotValue = NumOps.Abs(A[k, k]);
            for (int i = k + 1; i < n; i++)
            {
                T absValue = NumOps.Abs(A[i, k]);
                if (NumOps.GreaterThan(absValue, pivotValue))
                {
                    pivotRow = i;
                    pivotValue = absValue;
                }
            }

            if (pivotRow != k)
            {
                for (int j = 0; j < n; j++)
                {
                    T temp = A[k, j];
                    A[k, j] = A[pivotRow, j];
                    A[pivotRow, j] = temp;
                }

                (P[pivotRow], P[k]) = (P[k], P[pivotRow]);
            }

            for (int i = k + 1; i < n; i++)
            {
                T factor = NumOps.Divide(A[i, k], A[k, k]);
                L[i, k] = factor;
                for (int j = k; j < n; j++)
                {
                    A[i, j] = NumOps.Subtract(A[i, j], NumOps.Multiply(factor, A[k, j]));
                }
            }
        }

        Matrix<T> U = new(n, n, NumOps);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i > j)
                    L[i, j] = A[i, j];
                else if (i == j)
                    L[i, j] = NumOps.One;
                else
                    U[i, j] = A[i, j];
            }
        }

        return (L, U, P);
    }

    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuCompletePivoting(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        Matrix<T> A = matrix.Copy();
        Matrix<T> L = new(n, n, NumOps);
        Vector<int> P = new(n, MathHelper.GetNumericOperations<int>());
        Vector<int> Q = new(n, MathHelper.GetNumericOperations<int>());

        for (int i = 0; i < n; i++)
        {
            P[i] = i;
            Q[i] = i;
        }

        for (int k = 0; k < n - 1; k++)
        {
            int pivotRow = k, pivotCol = k;
            T pivotValue = NumOps.Abs(A[k, k]);

            for (int i = k; i < n; i++)
            {
                for (int j = k; j < n; j++)
                {
                    T absValue = NumOps.Abs(A[i, j]);
                    if (NumOps.GreaterThan(absValue, pivotValue))
                    {
                        pivotRow = i;
                        pivotCol = j;
                        pivotValue = absValue;
                    }
                }
            }

            if (pivotRow != k)
            {
                for (int j = 0; j < n; j++)
                {
                    T temp = A[k, j];
                    A[k, j] = A[pivotRow, j];
                    A[pivotRow, j] = temp;
                }
                (P[pivotRow], P[k]) = (P[k], P[pivotRow]);
            }

            if (pivotCol != k)
            {
                for (int i = 0; i < n; i++)
                {
                    T temp = A[i, k];
                    A[i, k] = A[i, pivotCol];
                    A[i, pivotCol] = temp;
                }
                (Q[pivotCol], Q[k]) = (Q[k], Q[pivotCol]);
            }

            for (int i = k + 1; i < n; i++)
            {
                T factor = NumOps.Divide(A[i, k], A[k, k]);
                L[i, k] = factor;
                for (int j = k; j < n; j++)
                {
                    A[i, j] = NumOps.Subtract(A[i, j], NumOps.Multiply(factor, A[k, j]));
                }
            }
        }

        Matrix<T> U = new(n, n, NumOps);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i > j)
                    L[i, j] = A[i, j];
                else if (i == j)
                    L[i, j] = NumOps.One;
                else
                    U[i, j] = A[i, j];
            }
        }

        // Adjust U and P for column permutations
        Matrix<T> adjustedU = new(n, n, NumOps);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                adjustedU[i, Q[j]] = U[i, j];
            }
        }

        return (L, adjustedU, P);
    }

    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeCholesky(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");

        Matrix<T> L = new(n, n, NumOps);
        Vector<int> P = new(n, MathHelper.GetNumericOperations<int>());

        for (int i = 0; i < n; i++)
            P[i] = i;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = NumOps.Zero;

                if (j == i)
                {
                    for (int k = 0; k < j; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(L[j, k], L[j, k]));
                    }
                    L[j, j] = NumOps.Sqrt(NumOps.Subtract(matrix[j, j], sum));
                }
                else
                {
                    for (int k = 0; k < j; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(L[i, k], L[j, k]));
                    }
                    L[i, j] = NumOps.Divide(NumOps.Subtract(matrix[i, j], sum), L[j, j]);
                }
            }
        }

        Matrix<T> U = L.Transpose();

        return (L, U, P);
    }

    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuDoolittle(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        Matrix<T> L = new(n, n, NumOps);
        Matrix<T> U = new(n, n, NumOps);
        Vector<int> P = new(n, MathHelper.GetNumericOperations<int>());

        // Initialize P as [0, 1, 2, ..., n-1]
        for (int i = 0; i < n; i++)
            P[i] = i;

        for (int i = 0; i < n; i++)
        {
            // Upper Triangular
            for (int k = i; k < n; k++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < i; j++)
                    sum = NumOps.Add(sum, NumOps.Multiply(L[i, j], U[j, k]));
                U[i, k] = NumOps.Subtract(matrix[i, k], sum);
            }

            // Lower Triangular
            for (int k = i; k < n; k++)
            {
                if (i == k)
                    L[i, i] = NumOps.One;
                else
                {
                    T sum = NumOps.Zero;
                    for (int j = 0; j < i; j++)
                        sum = NumOps.Add(sum, NumOps.Multiply(L[k, j], U[j, i]));
                    L[k, i] = NumOps.Divide(NumOps.Subtract(matrix[k, i], sum), U[i, i]);
                }
            }
        }

        return (L, U, P);
    }

    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuCrout(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        Matrix<T> L = new(n, n, NumOps);
        Matrix<T> U = new(n, n, NumOps);
        Vector<int> P = new(n, MathHelper.GetNumericOperations<int>());

        // Initialize P as [0, 1, 2, ..., n-1]
        for (int i = 0; i < n; i++)
            P[i] = i;

        for (int j = 0; j < n; j++)
        {
            U[j, j] = NumOps.One;
        }

        for (int j = 0; j < n; j++)
        {
            for (int i = j; i < n; i++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(L[i, k], U[k, j]));
                }
                L[i, j] = NumOps.Subtract(matrix[i, j], sum);
            }

            for (int i = j; i < n; i++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(L[j, k], U[k, i]));
                }
                if (!NumOps.Equals(L[j, j], NumOps.Zero))
                {
                    U[j, i] = NumOps.Divide(NumOps.Subtract(matrix[j, i], sum), L[j, j]);
                }
                else
                {
                    U[j, i] = NumOps.Zero;
                }
            }
        }

        return (L, U, P);
    }

    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuDefault(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        Matrix<T> A = matrix.Copy();
        Matrix<T> L = new(n, n, NumOps);
        Vector<int> P = new(n, MathHelper.GetNumericOperations<int>());

        // Initialize P as [0, 1, 2, ..., n-1]
        for (int i = 0; i < n; i++)
            P[i] = i;

        for (int k = 0; k < n - 1; k++)
        {
            // Find pivot
            int pivotRow = k;
            T pivotValue = NumOps.Abs(A[k, k]);
            for (int i = k + 1; i < n; i++)
            {
                T absValue = NumOps.Abs(A[i, k]);
                if (NumOps.GreaterThan(absValue, pivotValue))
                {
                    pivotRow = i;
                    pivotValue = absValue;
                }
            }

            // Swap rows if necessary
            if (pivotRow != k)
            {
                for (int j = 0; j < n; j++)
                {
                    T temp = A[k, j];
                    A[k, j] = A[pivotRow, j];
                    A[pivotRow, j] = temp;
                }

                (P[pivotRow], P[k]) = (P[k], P[pivotRow]);
            }

            // Perform elimination
            for (int i = k + 1; i < n; i++)
            {
                T factor = NumOps.Divide(A[i, k], A[k, k]);
                L[i, k] = factor;
                for (int j = k; j < n; j++)
                {
                    A[i, j] = NumOps.Subtract(A[i, j], NumOps.Multiply(factor, A[k, j]));
                }
            }
        }

        // Separate L and U
        Matrix<T> U = new(n, n, NumOps);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i > j)
                    L[i, j] = A[i, j];
                else if (i == j)
                    L[i, j] = NumOps.One;
                else
                    U[i, j] = A[i, j];
            }
        }

        return (L, U, P);
    }

    private Vector<T> PermutateVector(Vector<T> b, Vector<int> P)
    {
        var pb = new Vector<T>(b.Length, NumOps);
        for (int i = 0; i < b.Length; i++)
        {
            pb[i] = b[P[i]];
        }

        return pb;
    }

    private Vector<T> ForwardSubstitution(Matrix<T> L, Vector<T> b)
    {
        var y = new Vector<T>(L.Rows, NumOps);
        for (int i = 0; i < L.Rows; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < i; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(L[i, j], y[j]));
            }
            y[i] = NumOps.Subtract(b[i], sum);
        }

        return y;
    }

    private Vector<T> BackSubstitution(Matrix<T> U, Vector<T> y)
    {
        var x = new Vector<T>(U.Columns, NumOps);
        for (int i = U.Columns - 1; i >= 0; i--)
        {
            T sum = NumOps.Zero;
            for (int j = i + 1; j < U.Columns; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(U[i, j], x[j]));
            }
            x[i] = NumOps.Divide(NumOps.Subtract(y[i], sum), U[i, i]);
        }

        return x;
    }

    public Matrix<T> Invert()
    {
        return MatrixHelper.InvertUsingDecomposition(this);
    }
}