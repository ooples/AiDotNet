namespace AiDotNet.LinearAlgebra;

public class TakagiDecomposition : IMatrixDecomposition<double>
{
    private Matrix<double> SigmaMatrix { get; set; }
    private Matrix<Complex> UnitaryMatrix { get; set; }
    private Vector<Complex> EigenValues { get; set; }
    private Matrix<double> AMatrix { get; set; }
    private Vector<double> BVector { get; set; }

    public Vector<double> SolutionVector { get; private set; }

    public TakagiDecomposition(IEnumerable<IEnumerable<double>> expectedValues, IEnumerable<double> actualValues)
    {
        AMatrix = new Matrix<double>(expectedValues);
        BVector = new Vector<double>(actualValues);
        SigmaMatrix = new Matrix<double>(AMatrix.RowCount, AMatrix.RowCount);
        var eigenDecomposition = new EigenvalueDecomposition(expectedValues, actualValues);
        UnitaryMatrix = eigenDecomposition.EigenVectors.ToComplexMatrix();
        EigenValues = eigenDecomposition.EigenValues.ToComplexVector();
        Decompose(AMatrix);
        SolutionVector = Solve(SigmaMatrix, BVector);
    }

    public void Decompose(Matrix<double> aMatrix)
    {
        var rows = aMatrix.RowCount;
        SigmaMatrix = new Matrix<double>(rows, rows);
        for (int i = 0; i < rows; i++)
        {
            SigmaMatrix[i, i] = Math.Sqrt(EigenValues[i].Magnitude);
        }
    }

    public Matrix<double> Invert()
    {
        var invSigma = SigmaMatrix.InvertDiagonalMatrix();
        var invU = UnitaryMatrix.InvertUnitaryMatrix();
        var inv = invU.DotProduct(invSigma.ToComplexMatrix()).DotProduct(invU.Transpose());

        return inv.ToRealMatrix();
    }

    public Vector<double> Solve(Matrix<double> sigmaMatrix, Vector<double> bVector)
    {
        var bComplex = new Vector<Complex>(bVector.Count);
        for (int i = 0; i < bVector.Count; i++)
        {
            bComplex[i] = new Complex(bVector[i], 0);
        }
        var yVector = MatrixHelper.ForwardSubstitution(UnitaryMatrix, bComplex);

        return MatrixHelper.BackwardSubstitution(sigmaMatrix, yVector).ToRealVector();
    }
}