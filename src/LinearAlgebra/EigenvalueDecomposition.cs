namespace AiDotNet.LinearAlgebra;

public class EigenvalueDecomposition : IMatrixDecomposition<double>
{
    private Matrix<double> AMatrix { get; set; }
    private Vector<double> BVector { get; set; }

    public Vector<double> SolutionVector { get; private set; }
    public Matrix<double> EigenVectors { get; private set; }
    public Vector<double> EigenValues { get; private set; }

    public EigenvalueDecomposition(IEnumerable<Vector<double>> expectedValues, IEnumerable<double> actualValues)
    {
        AMatrix = Matrix.CreateDoubleMatrix(expectedValues);
        BVector = new Vector<double>(actualValues);
        EigenVectors = Matrix.CreateDoubleMatrix(AMatrix.Rows, AMatrix.Rows);
        EigenValues = new Vector<double>(AMatrix.Rows);
        Decompose(AMatrix);
        SolutionVector = Solve(EigenVectors, BVector);
    }

    public void Decompose(Matrix<double> aMatrix)
    {
        var rows = aMatrix.Rows;
        EigenValues = new Vector<double>(rows);
        EigenVectors = Matrix.CreateDoubleMatrix(rows, rows);
        var A = aMatrix.Copy();

        for (int i = 0; i < rows; i++)
        {
            var (eigenvalue, eigenvector) = MatrixHelper.PowerIteration(A, 1000, 1e-10);
            EigenValues[i] = eigenvalue;
            for (int j = 0; j < rows; j++)
            {
                EigenVectors[j, i] = eigenvector[j];
            }

            // Deflate the matrix
            for (int j = 0; j < rows; j++)
            {
                for (int k = 0; k < rows; k++)
                {
                    A[j, k] -= eigenvalue * eigenvector[j] * eigenvector[k];
                }
            }
        }
    }

    public Matrix<double> Invert()
    {
        var rows = AMatrix.Rows;
        var inv = Matrix.CreateDoubleMatrix(rows, rows);
        for (int i = 0; i < rows; i++)
        {
            double eigenvalue = EigenValues[i];
            var eigenvector = new Vector<double>(rows);
            for (int j = 0; j < rows; j++)
            {
                eigenvector[j] = EigenVectors[j, i];
            }

            for (int j = 0; j < rows; j++)
            {
                for (int k = 0; k < rows; k++)
                {
                    inv[j, k] += eigenvector[j] * eigenvector[k] / eigenvalue;
                }
            }
        }

        return inv;
    }

    public Vector<double> Solve(Matrix<double> eigenVectors, Vector<double> bVector)
    {
        var rows = eigenVectors.Rows;
        var x = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            double eigenvalue = EigenValues[i];
            double[] eigenvector = new double[rows];
            for (int j = 0; j < rows; j++)
            {
                eigenvector[j] = eigenVectors[j, i];
            }

            double dotProduct = 0;
            for (int j = 0; j < rows; j++)
            {
                dotProduct += eigenvector[j] * bVector[j];
            }

            for (int j = 0; j < rows; j++)
            {
                x[j] += dotProduct * eigenvector[j] / eigenvalue;
            }
        }

        return x;
    }
}