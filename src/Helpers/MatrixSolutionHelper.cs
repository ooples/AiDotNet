namespace AiDotNet.Helpers;

public static class MatrixSolutionHelper
{
    public static Vector<T> SolveLinearSystem<T>(Matrix<T> A, Vector<T> b, MatrixDecompositionType decompositionType)
    {
        return decompositionType switch
        {
            MatrixDecompositionType.Lu => new LuDecomposition<T>(A).Solve(b),
            MatrixDecompositionType.Cholesky => new CholeskyDecomposition<T>(A).Solve(b),
            MatrixDecompositionType.Qr => new QrDecomposition<T>(A).Solve(b),
            MatrixDecompositionType.Svd => new SvdDecomposition<T>(A).Solve(b),
            MatrixDecompositionType.Cramer => SolveCramer(A, b),
            MatrixDecompositionType.GramSchmidt => SolveGramSchmidt(A, b),
            MatrixDecompositionType.Normal => SolveNormal(A, b),
            MatrixDecompositionType.Lq => new LqDecomposition<T>(A).Solve(b),
            MatrixDecompositionType.Takagi => throw new NotSupportedException("Takagi decomposition is not suitable for solving linear systems."),
            MatrixDecompositionType.Hessenberg => SolveHessenberg(A, b),
            MatrixDecompositionType.Schur => SolveSchur(A, b),
            MatrixDecompositionType.Eigen => SolveEigen(A, b),
            _ => throw new ArgumentException("Unsupported decomposition type", nameof(decompositionType))
        };
    }

    private static Vector<T> SolveCramer<T>(Matrix<T> A, Vector<T> b)
    {
        var det = A.Determinant();
        var x = new Vector<T>(b.Length);
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < b.Length; i++)
        {
            var Ai = A.Copy();
            for (int j = 0; j < b.Length; j++)
            {
                Ai[j, i] = b[j];
            }
            x[i] = numOps.Divide(Ai.Determinant(), det);
        }

        return x;
    }

    private static Vector<T> SolveGramSchmidt<T>(Matrix<T> A, Vector<T> b)
    {
        var gs = new GramSchmidtDecomposition<T>(A);
        return gs.Solve(b);
    }

    private static Vector<T> SolveNormal<T>(Matrix<T> A, Vector<T> b)
    {
        var ATA = A.Transpose().Multiply(A);
        var ATb = A.Transpose().Multiply(b);

        return new CholeskyDecomposition<T>(ATA).Solve(ATb);
    }

    private static Vector<T> SolveHessenberg<T>(Matrix<T> A, Vector<T> b)
    {
        var hessenberg = new HessenbergDecomposition<T>(A);
        return hessenberg.Solve(b);
    }

    private static Vector<T> SolveSchur<T>(Matrix<T> A, Vector<T> b)
    {
        var schur = new SchurDecomposition<T>(A);
        return schur.Solve(b);
    }

    private static Vector<T> SolveEigen<T>(Matrix<T> A, Vector<T> b)
    {
        var eigen = new EigenDecomposition<T>(A);
        return eigen.Solve(b);
    }
}