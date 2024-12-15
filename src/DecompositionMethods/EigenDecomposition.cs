namespace AiDotNet.DecompositionMethods;

public class EigenDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;

    public Matrix<T> EigenVectors { get; private set; }
    public Vector<T> EigenValues { get; private set; }

    public Matrix<T> A { get; private set; }

    public EigenDecomposition(Matrix<T> matrix, EigenAlgorithm algorithm = EigenAlgorithm.QR)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        A = matrix;
        (EigenValues, EigenVectors) = Decompose(matrix, algorithm);
    }

    private (Vector<T> eigenValues, Matrix<T> eigenVectors) Decompose(Matrix<T> matrix, EigenAlgorithm algorithm)
    {
        return algorithm switch
        {
            EigenAlgorithm.QR => ComputeEigenQR(matrix),
            EigenAlgorithm.PowerIteration => ComputeEigenPowerIteration(matrix),
            EigenAlgorithm.Jacobi => ComputeEigenJacobi(matrix),
            _ => throw new ArgumentException("Unsupported eigenvalue decomposition algorithm.")
        };
    }

    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenPowerIteration(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Vector<T> eigenValues = new(n, NumOps);
        Matrix<T> eigenVectors = Matrix<T>.CreateIdentity(n, NumOps);

        for (int i = 0; i < n; i++)
        {
            // Replace CreateRandom with a method to create a random vector
            Vector<T> v = Vector<T>.CreateRandom(n);
            for (int iter = 0; iter < 100; iter++)
            {
                Vector<T> w = matrix.Multiply(v);
                T eigenValue = NumOps.Divide(w.DotProduct(v), v.DotProduct(v));
                v = w.Divide(w.Norm());

                if (iter > 0 && NumOps.LessThan(NumOps.Abs(NumOps.Subtract(eigenValue, eigenValues[i])), NumOps.FromDouble(1e-10)))
                {
                    break;
                }
                eigenValues[i] = eigenValue;
            }
            eigenVectors.SetColumn(i, v);
            // Fix the Multiply operation
            matrix = matrix.Subtract(MatrixHelper.OuterProduct(v, v).Multiply(eigenValues[i]));
        }

        return (eigenValues, eigenVectors);
    }

    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenQR(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Matrix<T> A = matrix.Copy();
        Matrix<T> Q = Matrix<T>.CreateIdentity(n, NumOps);

        for (int iter = 0; iter < 100; iter++)
        {
            var qrDecomp = new QrDecomposition<T>(A);
            (var q, var r) = (qrDecomp.Q, qrDecomp.R);
            A = r.Multiply(q);
            Q = Q.Multiply(q);

            if (A.IsUpperTriangularMatrix(NumOps.FromDouble(1e-10)))
                break;
        }

        Vector<T> eigenValues = MatrixHelper.ExtractDiagonal(A);
        return (eigenValues, Q);
    }

    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenJacobi(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Matrix<T> A = matrix.Copy();
        Matrix<T> V = Matrix<T>.CreateIdentity(n, NumOps);

        for (int iter = 0; iter < 100; iter++)
        {
            T maxOffDiagonal = NumOps.Zero;
            int p = 0, q = 0;

            for (int i = 0; i < n - 1; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    T absValue = NumOps.Abs(A[i, j]);
                    if (NumOps.GreaterThan(absValue, maxOffDiagonal))
                    {
                        maxOffDiagonal = absValue;
                        p = i;
                        q = j;
                    }
                }
            }

            if (NumOps.LessThan(maxOffDiagonal, NumOps.FromDouble(1e-10)))
                break;

            T theta = NumOps.Divide(NumOps.Subtract(A[q, q], A[p, p]), NumOps.Multiply(NumOps.FromDouble(2), A[p, q]));
            T t = NumOps.Divide(NumOps.SignOrZero(theta), NumOps.Add(NumOps.Abs(theta), NumOps.Sqrt(NumOps.Add(NumOps.One, NumOps.Multiply(theta, theta)))));
            T c = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(NumOps.One, NumOps.Multiply(t, t))));
            T s = NumOps.Multiply(t, c);

            Matrix<T> J = Matrix<T>.CreateIdentity(n, NumOps);
            J[p, p] = c; J[q, q] = c;
            J[p, q] = s; J[q, p] = NumOps.Negate(s);

            A = J.Transpose().Multiply(A).Multiply(J);
            V = V.Multiply(J);
        }

        Vector<T> eigenValues = MatrixHelper.ExtractDiagonal(A);
        return (eigenValues, V);
    }

    public Vector<T> Solve(Vector<T> b)
    {
        Matrix<T> D = Matrix<T>.CreateDiagonal(EigenValues, NumOps);
        return EigenVectors.Multiply(D.InvertDiagonalMatrix()).Multiply(EigenVectors.Transpose()).Multiply(b);
    }

    public Matrix<T> Invert()
    {
        Matrix<T> D = Matrix<T>.CreateDiagonal(EigenValues, NumOps);
        return EigenVectors.Multiply(D.InvertDiagonalMatrix()).Multiply(EigenVectors.Transpose());
    }
}