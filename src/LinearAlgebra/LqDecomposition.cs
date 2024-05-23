namespace AiDotNet.LinearAlgebra;

public class LqDecomposition : IMatrixDecomposition<double>
{
    private Matrix<double> LMatrix { get; set; }
    private Matrix<double> QMatrix { get; set; }
    private Matrix<double> AMatrix { get; set; }
    private Vector<double> BVector { get; set; }

    public Vector<double> SolutionVector { get; private set; }

    public LqDecomposition(IEnumerable<IEnumerable<double>> expectedValues, IEnumerable<double> actualValues)
    {
        AMatrix = new Matrix<double>(expectedValues);
        BVector = new Vector<double>(actualValues);
        LMatrix = new Matrix<double>(AMatrix.RowCount, AMatrix.RowCount);
        QMatrix = new Matrix<double>(AMatrix.RowCount, AMatrix.RowCount);
        Decompose(AMatrix);
        SolutionVector = Solve(AMatrix, BVector);
    }

    public void Decompose(Matrix<double> aMatrix)
    {
        // lq decomp using householder algo
        int m = aMatrix.RowCount;
        int n = aMatrix.ColumnCount;

        LMatrix = new Matrix<double>(m, n);
        QMatrix = MatrixHelper.CreateIdentityMatrix<double>(n);

        var a = aMatrix.Duplicate();

        // Perform Householder transformations
        for (int k = 0; k < Math.Min(m, n); k++)
        {
            var x = new Vector<double>(m - k);
            for (int i = k; i < m; i++)
            {
                x[i - k] = a[i, k];
            }

            double alpha = -Math.Sign(x[0]) * MatrixHelper.Hypotenuse(x.Values);
            var u = new Vector<double>(x.Count);
            for (int i = 0; i < x.Count; i++)
            {
                u[i] = (i == 0) ? x[i] - alpha : x[i];
            }

            double norm_u = MatrixHelper.Hypotenuse(u.Values);
            for (int i = 0; i < u.Count; i++)
            {
                u[i] /= norm_u;
            }

            var uMatrix = new Matrix<double>(u.Count, 1);
            for (int i = 0; i < u.Count; i++)
            {
                uMatrix[i, 0] = u[i];
            }

            var uT = uMatrix.Transpose();
            var uTu = uMatrix.DotProduct(uT);

            for (int i = 0; i < m - k; i++)
            {
                for (int j = 0; j < m - k; j++)
                {
                    if (i == j)
                    {
                        uTu[i, j] -= 1.0;
                    }
                }
            }

            var P = MatrixHelper.CreateIdentityMatrix<double>(m);
            for (int i = k; i < m; i++)
            {
                for (int j = k; j < m; j++)
                {
                    P[i, j] += 2.0 * uTu[i - k, j - k];
                }
            }

            a = P.DotProduct(a);
            QMatrix = QMatrix.DotProduct(P);
        }

        LMatrix = a;

        // Ensure QMatrix is orthogonal
        for (int i = 0; i < QMatrix.RowCount; i++)
        {
            for (int j = 0; j < QMatrix.ColumnCount; j++)
            {
                if (i == j)
                {
                    QMatrix[i, j] = 1.0;
                }
                else
                {
                    QMatrix[i, j] = -QMatrix[i, j];
                }
            }
        }
    }

    public Matrix<double> Invert()
    {
        int n = QMatrix.RowCount;

        var qMatrixTransposed = QMatrix.Transpose();
        var lMatrixInverted = LMatrix.InvertLowerTriangularMatrix();

        return qMatrixTransposed.DotProduct(lMatrixInverted);
    }

    public Vector<double> Solve(Matrix<double> aMatrix, Vector<double> bVector)
    {
        var y = MatrixHelper.ForwardSubstitution(LMatrix, bVector);

        return QMatrix.Transpose().DotProduct(y);
    }
}