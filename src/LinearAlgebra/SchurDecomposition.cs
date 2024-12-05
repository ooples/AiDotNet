namespace AiDotNet.LinearAlgebra;

public class SchurDecomposition : IMatrixDecomposition<double>
{
    private Matrix<double> SchurMatrix { get; set; }
    private Matrix<double> UnitaryMatrix { get; set; }
    private Matrix<double> AMatrix { get; set; }
    private Vector<double> BVector { get; set; }

    public Vector<double> SolutionVector { get; private set; }

    public SchurDecomposition(IEnumerable<IEnumerable<double>> expectedValues, IEnumerable<double> actualValues)
    {
        AMatrix = Matrix.CreateDoubleMatrix(expectedValues);
        BVector = new Vector<double>(actualValues);
        SchurMatrix = Matrix.CreateDoubleMatrix(AMatrix.Rows, AMatrix.Rows);
        UnitaryMatrix = Matrix.CreateDoubleMatrix(AMatrix.Rows, AMatrix.Rows);
        Decompose(AMatrix);
        SolutionVector = Solve(SchurMatrix, BVector);
    }

    public void Decompose(Matrix<double> schurMatrix)
    {
        const int maxIterations = 100;
        double tolerance = 1e-10;
        schurMatrix = MatrixHelper.ReduceToHessenbergFormat(schurMatrix);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // QR decomposition of H
            var (qMatrix, rMatrix) = MatrixHelper.QRDecomposition(schurMatrix);

            // Update H = R * Q
            schurMatrix = rMatrix.DotProduct(qMatrix);

            // Accumulate the unitary matrix
            UnitaryMatrix = UnitaryMatrix.DotProduct(qMatrix);

            // Check for convergence
            if (schurMatrix.IsUpperTriangularMatrix(tolerance))
            {
                break;
            }
        }

        SchurMatrix = schurMatrix;
    }

    public Matrix<double> Invert()
    {
        var invU = UnitaryMatrix.Transpose();
        var invH = MatrixHelper.InvertUpperTriangularMatrix(SchurMatrix);

        return invH.DotProduct(invU);
    }

    public Vector<double> Solve(Matrix<double> schurMatrix, Vector<double> bVector)
    {
        var yVector = MatrixHelper.ForwardSubstitution(UnitaryMatrix, bVector);

        return MatrixHelper.BackwardSubstitution(schurMatrix, yVector);
    }
}