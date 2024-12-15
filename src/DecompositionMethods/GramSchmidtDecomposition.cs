namespace AiDotNet.LinearAlgebra;

public class GramSchmidtDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;

    public Matrix<T> Q { get; private set; }
    public Matrix<T> R { get; private set; }
    public Matrix<T> A { get; private set; }

    public GramSchmidtDecomposition(Matrix<T> matrix, GramSchmidtAlgorithm algorithm = GramSchmidtAlgorithm.Classical)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        A = matrix;
        (Q, R) = Decompose(matrix, algorithm);
    }

    private (Matrix<T> Q, Matrix<T> R) Decompose(Matrix<T> matrix, GramSchmidtAlgorithm algorithm)
    {
        return algorithm switch
        {
            GramSchmidtAlgorithm.Classical => ComputeClassicalGramSchmidt(matrix),
            GramSchmidtAlgorithm.Modified => ComputeModifiedGramSchmidt(matrix),
            _ => throw new ArgumentException("Unsupported Gram-Schmidt algorithm.")
        };
    }

    private (Matrix<T> Q, Matrix<T> R) ComputeClassicalGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var Q = new Matrix<T>(m, n);
        var R = new Matrix<T>(n, n);

        for (int j = 0; j < n; j++)
        {
            var v = matrix.GetColumn(j);

            for (int i = 0; i < j; i++)
            {
                R[i, j] = Q.GetColumn(i).DotProduct(matrix.GetColumn(j));
                v = v.Subtract(Q.GetColumn(i).Multiply(R[i, j]));
            }

            R[j, j] = v.Norm();
            Q.SetColumn(j, v.Divide(R[j, j]));
        }

        return (Q, R);
    }

    private (Matrix<T> Q, Matrix<T> R) ComputeModifiedGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var Q = new Matrix<T>(m, n);
        var R = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            var v = matrix.GetColumn(i);
                
            for (int j = 0; j < i; j++)
            {
                R[j, i] = Q.GetColumn(j).DotProduct(v);
                v = v.Subtract(Q.GetColumn(j).Multiply(R[j, i]));
            }

            R[i, i] = v.Norm();
            Q.SetColumn(i, v.Divide(R[i, i]));
        }

        return (Q, R);
    }

    public Vector<T> Solve(Vector<T> b)
    {
        // Solve Qy = b
        var y = Q.Transpose().Multiply(b);

        // Solve Rx = y using back-substitution
        return BackSubstitution(R, y);
    }

    private Vector<T> BackSubstitution(Matrix<T> R, Vector<T> y)
    {
        int n = R.Rows;
        var x = new Vector<T>(n);

        for (int i = n - 1; i >= 0; i--)
        {
            var sum = NumOps.Zero;
            for (int j = i + 1; j < n; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(R[i, j], x[j]));
            }
            x[i] = NumOps.Divide(NumOps.Subtract(y[i], sum), R[i, i]);
        }

        return x;
    }

    public Matrix<T> Invert()
    {
        int n = A.Rows;
        var identity = Matrix<T>.CreateIdentity(n);
        var inverse = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            inverse.SetColumn(i, Solve(identity.GetColumn(i)));
        }

        return inverse;
    }
}