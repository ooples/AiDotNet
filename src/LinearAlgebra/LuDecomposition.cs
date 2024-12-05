namespace AiDotNet.LinearAlgebra;

public class LuDecomposition : IMatrixDecomposition<double>
{
    private Matrix<double> LMatrix { get; set; }
    private Matrix<double> UMatrix { get; set; }
    private Matrix<double> AMatrix { get; set; }
    private Vector<double> BVector { get; set; }
    private Vector<int> PVector { get; set; }

    public Vector<double> SolutionVector { get; private set; }

    public LuDecomposition(IEnumerable<IEnumerable<double>> expectedValues, IEnumerable<double> actualValues)
    {
        AMatrix = new Matrix<double>(expectedValues);
        BVector = new Vector<double>(actualValues);
        PVector = new Vector<int>(BVector.Count);
        LMatrix = new Matrix<double>(AMatrix.Rows, AMatrix.Rows);
        UMatrix = new Matrix<double>(AMatrix.Rows, AMatrix.Rows);
        Decompose(AMatrix);
        SolutionVector = Solve(LMatrix, BVector);
    }

    public void Decompose(Matrix<double> aMatrix)
    {
        var rows = aMatrix.Rows;
        LMatrix = new Matrix<double>(rows, rows);
        UMatrix = aMatrix.Duplicate();
        PVector = new Vector<int>(Enumerable.Range(0, rows));

        for (int k = 0; k < rows; k++)
        {
            // Pivoting
            int max = k;
            for (int i = k + 1; i < rows; i++)
            {
                if (Math.Abs(UMatrix[i, k]) > Math.Abs(UMatrix[max, k]))
                {
                    max = i;
                }
            }
            if (k != max)
            {
                MatrixHelper.SwapRows(UMatrix, k, max);
                int temp = PVector[k];
                PVector[k] = PVector[max];
                PVector[max] = temp;
            }

            // Decompose
            for (int i = k + 1; i < rows; i++)
            {
                UMatrix[i, k] /= UMatrix[k, k];
                for (int j = k + 1; j < rows; j++)
                {
                    UMatrix[i, j] -= UMatrix[i, k] * UMatrix[k, j];
                }
            }
        }

        // Form L and U matrices
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                if (i > j)
                {
                    LMatrix[i, j] = UMatrix[i, j];
                    UMatrix[i, j] = 0;
                }
                else if (i == j)
                {
                    LMatrix[i, j] = 1;
                }
                else
                {
                    LMatrix[i, j] = 0;
                }
            }
        }
    }

    public Matrix<double> Invert()
    {
        var rows = AMatrix.Rows;
        var inv = new Matrix<double>(rows, rows);
        var eVector = new Vector<double>(rows);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                eVector[j] = (i == j) ? 1 : 0;
            }
            var x = Solve(AMatrix, eVector);
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

        // Apply permutation matrix to b
        for (int i = 0; i < rows; i++)
        {
            yVector[i] = bVector[PVector[i]];
        }

        // Forward substitution to solve Ly = Pb
        for (int i = 0; i < rows; i++)
        {
            xVector[i] = yVector[i];
            for (int j = 0; j < i; j++)
            {
                xVector[i] -= LMatrix[i, j] * xVector[j];
            }
        }

        // Backward substitution to solve Ux = y
        for (int i = rows - 1; i >= 0; i--)
        {
            for (int j = i + 1; j < rows; j++)
            {
                xVector[i] -= UMatrix[i, j] * xVector[j];
            }
            xVector[i] /= UMatrix[i, i];
        }

        return xVector;
    }
}