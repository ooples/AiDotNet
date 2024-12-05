namespace AiDotNet.LinearAlgebra;

public class HessenbergDecomposition : IMatrixDecomposition<double>
{
    private Matrix<double> HessenbergMatrix { get; set; }
    private Matrix<double> AMatrix { get; set; }
    private Vector<double> BVector { get; set; }

    public Vector<double> SolutionVector { get; private set; }

    public HessenbergDecomposition(IEnumerable<IEnumerable<double>> expectedValues, IEnumerable<double> actualValues)
    {
        AMatrix = new Matrix<double>(expectedValues);
        BVector = new Vector<double>(actualValues);
        HessenbergMatrix = new Matrix<double>(AMatrix.Rows, AMatrix.Rows);
        Decompose(AMatrix);
        SolutionVector = Solve(HessenbergMatrix, BVector);
    }

    public void Decompose(Matrix<double> aMatrix)
    {
        var rows = aMatrix.Rows;
        for (int k = 0; k < rows - 2; k++)
        {
            var xVector = new Vector<double>(rows - k - 1);
            for (int i = 0; i < rows - k - 1; i++)
            {
                xVector[i] = HessenbergMatrix[k + 1 + i, k];
            }

            var hVector = MatrixHelper.CreateHouseholderVector(xVector);
            HessenbergMatrix = MatrixHelper.ApplyHouseholderTransformation(HessenbergMatrix, hVector, k);
        }
    }

    public Matrix<double> Invert()
    {
        var rows = AMatrix.Rows;
        var iMatrix = MatrixHelper.CreateIdentityMatrix<double>(rows);
        var inv = new Matrix<double>(rows, rows);

        for (int i = 0; i < rows; i++)
        {
            var e = new Vector<double>(rows);
            e[i] = 1.0;
            var x = Solve(iMatrix, e);
            for (int j = 0; j < rows; j++)
            {
                inv[j, i] = x[j];
            }
        }

        return inv;
    }

    public Vector<double> Solve(Matrix<double> aMatrix, Vector<double> bVector)
    {
        var rows = aMatrix.Rows;
        var xVector = new Vector<double>(rows);
        var yVector = new Vector<double>(rows);

        // Forward substitution to solve Ly = Pb
        for (int i = 0; i < rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < i; j++)
            {
                sum += HessenbergMatrix[i, j] * yVector[j];
            }
            yVector[i] = (bVector[i] - sum) / HessenbergMatrix[i, i];
        }

        // Backward substitution to solve Ux = y
        for (int i = rows - 1; i >= 0; i--)
        {
            double sum = 0;
            for (int j = i + 1; j < rows; j++)
            {
                sum += HessenbergMatrix[i, j] * yVector[j];
            }
            xVector[i] = yVector[i] - sum;
        }

        return xVector;
    }
}