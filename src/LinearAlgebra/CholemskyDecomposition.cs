namespace AiDotNet.LinearAlgebra;

public class CholemskyDecomposition : IMatrixDecomposition<double>
{
    private Matrix<double> LMatrix { get; set; }
    private Matrix<double> AMatrix { get; set; }
    private Vector<double> BVector { get; set; }

    public Vector<double> SolutionVector { get; private set; }

    public CholemskyDecomposition(IEnumerable<IEnumerable<double>> expectedValues, IEnumerable<double> actualValues)
    {
        AMatrix = new Matrix<double>(expectedValues);
        BVector = new Vector<double>(actualValues);
        LMatrix = new Matrix<double>(AMatrix.RowCount, AMatrix.RowCount);
        Decompose(AMatrix);
        SolutionVector = Solve(LMatrix, BVector);
    }

    public void Decompose(Matrix<double> matrix)
    {
        int n = matrix.RowCount;
        LMatrix = new Matrix<double>(n, n);

        for (int i = 0; i < n; ++i)
        {
            for (int k = i; k < n; ++k)
            {
                double sum = 0;
                for (int j = 0; j < i; ++j)
                {
                    sum += LMatrix[i, j] * LMatrix[k, j];
                }

                var lValue = i == k ? Math.Sqrt(matrix[k, k] - sum) : (1.0 / LMatrix[i, i] * (matrix[k, i] - sum));

                if (lValue < 0)
                {
                    throw new InvalidOperationException("Matrix isn't positive-definite which means the decomposition created negative values.");
                }
                else
                {
                    LMatrix[k, i] = lValue;
                }
            }
        }
    }

    public Matrix<double> Invert()
    {
        var lMatrixInverse = Inverse(LMatrix);
        var lMatrixTransposed = lMatrixInverse.Transpose();

        return lMatrixInverse.DotProduct(lMatrixTransposed);
    }

    private Matrix<double> Inverse(Matrix<double> matrix)
    {
        int n = matrix.RowCount;
        var inv = new Matrix<double>(n, n);
        var eye = MatrixHelper.CreateIdentityMatrix<double>(n);

        for (int i = 0; i < n; i++)
        {
            var x = Solve(matrix, eye.GetColumn(i));
            for (int j = 0; j < n; ++j)
            {
                inv[j, i] = x[j];
            }
        }

        return inv;
    }

    public Vector<double> Solve(Matrix<double> lMatrix, Vector<double> bVector)
    {
        var y = ForwardSubstitution(lMatrix, bVector);

        return lMatrix.Transpose().BackwardSubstitution(y);
    }

    private static Vector<double> ForwardSubstitution(Matrix<double> lMatrix, Vector<double> bVector)
    {
        int n = lMatrix.RowCount;
        var yVector = new Vector<double>(n);
        for (int i = 0; i < n; ++i)
        {
            yVector[i] = bVector[i];
            for (int j = 0; j < i; ++j)
            {
                yVector[i] -= lMatrix[i, j] * yVector[j];
            }
            yVector[i] /= lMatrix[i, i];
        }

        return yVector;
    }
}