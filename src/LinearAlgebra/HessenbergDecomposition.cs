namespace AiDotNet.LinearAlgebra;

public class HessenbergDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;

    public Matrix<T> HessenbergMatrix { get; private set; }
    public Matrix<T> A { get; private set; }

    public HessenbergDecomposition(Matrix<T> matrix, HessenbergAlgorithm algorithm = HessenbergAlgorithm.Householder)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        A = matrix;
        HessenbergMatrix = Decompose(matrix, algorithm);
    }

    private Matrix<T> Decompose(Matrix<T> matrix, HessenbergAlgorithm algorithm)
    {
        return algorithm switch
        {
            HessenbergAlgorithm.Householder => ComputeHessenbergHouseholder(matrix),
            HessenbergAlgorithm.Givens => ComputeHessenbergGivens(matrix),
            HessenbergAlgorithm.ElementaryTransformations => ComputeHessenbergElementaryTransformations(matrix),
            HessenbergAlgorithm.ImplicitQR => ComputeHessenbergImplicitQR(matrix),
            HessenbergAlgorithm.Lanczos => ComputeHessenbergLanczos(matrix),
            _ => throw new ArgumentException("Unsupported Hessenberg decomposition algorithm.")
        };
    }

    private Matrix<T> ComputeHessenbergHouseholder(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Copy();

        for (int k = 0; k < n - 2; k++)
        {
            var x = new Vector<T>(n - k - 1);
            for (int i = 0; i < n - k - 1; i++)
            {
                x[i] = H[k + 1 + i, k];
            }

            var v = MatrixHelper.CreateHouseholderVector(x);
            H = MatrixHelper.ApplyHouseholderTransformation(H, v, k);
        }

        return H;
    }

    private Matrix<T> ComputeHessenbergGivens(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Copy();

        for (int k = 0; k < n - 2; k++)
        {
            for (int i = n - 1; i > k + 1; i--)
            {
                var (c, s) = MatrixHelper.ComputeGivensRotation(H[i - 1, k], H[i, k]);
                MatrixHelper.ApplyGivensRotation(H, c, s, i - 1, i, k, n);
            }
        }

        return H;
    }

    private Matrix<T> ComputeHessenbergElementaryTransformations(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Copy();

        for (int k = 0; k < n - 2; k++)
        {
            for (int i = k + 2; i < n; i++)
            {
                if (!NumOps.Equals(H[i, k], NumOps.Zero))
                {
                    T factor = NumOps.Divide(H[i, k], H[k + 1, k]);
                    for (int j = k; j < n; j++)
                    {
                        H[i, j] = NumOps.Subtract(H[i, j], NumOps.Multiply(factor, H[k + 1, j]));
                    }
                    H[i, k] = NumOps.Zero;
                }
            }
        }

        return H;
    }

    private Matrix<T> ComputeHessenbergImplicitQR(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Copy();
        var Q = Matrix<T>.CreateIdentity(n, NumOps);

        for (int iter = 0; iter < 100; iter++) // Max iterations
        {
            for (int k = 0; k < n - 1; k++)
            {
                var (c, s) = MatrixHelper.ComputeGivensRotation(H[k, k], H[k + 1, k]);
                MatrixHelper.ApplyGivensRotation(H, c, s, k, k + 1, k, n);
                MatrixHelper.ApplyGivensRotation(Q, c, s, k, k + 1, 0, n);
            }

            if (MatrixHelper.IsUpperHessenberg(H, NumOps.FromDouble(1e-10)))
            {
                break;
            }
        }

        return H;
    }

    private Matrix<T> ComputeHessenbergLanczos(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = new Matrix<T>(n, n);
        var v = new Vector<T>(n)
        {
            [0] = NumOps.One
        };

        for (int j = 0; j < n; j++)
        {
            var w = matrix.Multiply(v);
            if (j > 0)
            {
                w = w.Subtract(v.Multiply(H[j - 1, j]));
            }
            H[j, j] = w.DotProduct(v);
            w = w.Subtract(v.Multiply(H[j, j]));
            if (j < n - 1)
            {
                H[j, j + 1] = H[j + 1, j] = w.Norm();
                v = w.Divide(H[j, j + 1]);
            }
        }

        return H;
    }

    public Vector<T> Solve(Vector<T> b)
    {
        var n = A.Rows;
        var y = new Vector<T>(n);

        // Forward substitution
        for (int i = 0; i < n; i++)
        {
            var sum = NumOps.Zero;
            for (int j = Math.Max(0, i - 1); j < i; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(HessenbergMatrix[i, j], y[j]));
            }
            y[i] = NumOps.Divide(NumOps.Subtract(b[i], sum), HessenbergMatrix[i, i]);
        }

        // Backward substitution
        var x = new Vector<T>(n);
        for (int i = n - 1; i >= 0; i--)
        {
            var sum = NumOps.Zero;
            for (int j = i + 1; j < n; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(HessenbergMatrix[i, j], x[j]));
            }
            x[i] = NumOps.Subtract(y[i], sum);
        }

        return x;
    }

    public Matrix<T> Invert()
    {
        return MatrixHelper.InvertUsingDecomposition(this);
    }
}