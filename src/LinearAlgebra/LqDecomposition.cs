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
        LMatrix = new Matrix<double>(AMatrix.Rows, AMatrix.Rows);
        QMatrix = new Matrix<double>(AMatrix.Rows, AMatrix.Rows);
        Decompose(AMatrix);
        SolutionVector = Solve(AMatrix, BVector);
    }

    public void Decompose(Matrix<double> aMatrix)
    {
        int m = aMatrix.Rows;
        int n = aMatrix.Columns;

        LMatrix = new Matrix<double>(m, n);
        QMatrix = Matrix<double>.CreateIdentityMatrix<double>(n);

        var a = aMatrix.Copy();

        // Perform Householder transformations
        for (int k = 0; k < Math.Min(m, n); k++)
        {
            var x = VectorHelper.CreateVector<double>(m - k);
            for (int i = k; i < m; i++)
            {
                x[i - k] = a[i, k];
            }

            double alpha = -Math.Sign(x[0]) * MatrixHelper.Hypotenuse(x);
            var u = VectorHelper.CreateVector<double>(x.Length);
            for (int i = 0; i < x.Length; i++)
            {
                u[i] = (i == 0) ? x[i] - alpha : x[i];
            }

            double norm_u = MatrixHelper.Hypotenuse(u);
            for (int i = 0; i < u.Length; i++)
            {
                u[i] /= norm_u;
            }

            var uMatrix = new Matrix<double>(u.Length, 1);
            for (int i = 0; i < u.Length; i++)
            {
                uMatrix[i, 0] = u[i];
            }

            var uT = uMatrix.Transpose();
            var uTu = MatrixHelper.Multiply(uMatrix, uT);

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

            var P = Matrix<double>.CreateIdentityMatrix<double>(m);
            for (int i = k; i < m; i++)
            {
                for (int j = k; j < m; j++)
                {
                    P[i, j] += 2.0 * uTu[i - k, j - k];
                }
            }

            a = MatrixHelper.Multiply(P, a);
            QMatrix = MatrixHelper.Multiply(QMatrix, P);
        }

        LMatrix = a;

        // Ensure QMatrix is orthogonal
        for (int i = 0; i < QMatrix.Rows; i++)
        {
            for (int j = 0; j < QMatrix.Columns; j++)
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
        int n = QMatrix.Rows;

        var qMatrixTransposed = QMatrix.Transpose();
        var lMatrixInverted = LMatrix.InvertLowerTriangularMatrix();

        return MatrixHelper.Multiply(qMatrixTransposed, lMatrixInverted);
    }

    public Vector<double> Solve(Matrix<double> aMatrix, Vector<double> bVector)
    {
        var y = MatrixHelper.ForwardSubstitution(LMatrix, bVector);

        return QMatrix.Transpose().DotProduct(y);
    }
}