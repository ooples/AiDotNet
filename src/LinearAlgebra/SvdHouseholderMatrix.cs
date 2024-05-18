namespace AiDotNet.LinearAlgebra;

public class SvdHouseholderMatrix : ISvdDecomposition<double>
{
    private readonly Matrix<double> _matrix;
    private readonly Vector<double> _vector;
    private readonly Vector<int> _indexVector;
    private readonly Vector<double> _solutionVector;

    public SvdHouseholderMatrix(IEnumerable<IEnumerable<double>> expectedValues, IEnumerable<double> actualValues)
    {
        _matrix = new Matrix<double>(expectedValues);
        _vector = new Vector<double>(actualValues);
    }

    public void Decompose(out Matrix<double> matrix, out Vector<int> indexVector)
    {
        throw new NotImplementedException();
    }

    public double GetDeterminant()
    {
        throw new NotImplementedException();
    }

    public Matrix<double> GetInverse()
    {
        throw new NotImplementedException();
    }

    public int Nullity(double threshold)
    {
        var rows = _matrix.RowCount;
        var columns = _matrix.ColumnCount;
        var weightsVector = new Vector<double>(columns);
        var thresh = threshold >= 0 ? threshold : 0.5 * Math.Sqrt(rows + columns + 1) * weightsVector[0] * double.Epsilon;
        int nullity = 0;

        for (int i = 0; i < columns; i++)
        {
            if (weightsVector[i] <= thresh)
            {
                nullity++;
            }
        }

        return nullity;
    }

    public double Nullspace(double threshold)
    {
        throw new NotImplementedException();
    }

    public Matrix<double> Range(double threshold)
    {
        int rows = _matrix.RowCount, columns = _matrix.ColumnCount, rank = 0;
        var weightsVector = new Vector<double>(columns);
        var rangeMatrix = new Matrix<double>(rows, Rank(threshold));
        var uMatrix = _matrix.Duplicate();

        for (int i = 0; i < columns; i++)
        {
            if (weightsVector[i] > threshold)
            {
                for (int j = 0; j < rows; j++)
                {
                    rangeMatrix[j][rank] = uMatrix[j][i];
                }
                rank++;
            }
        }

        return rangeMatrix;
    }

    public int Rank(double threshold = -1)
    {
        var rows = _matrix.RowCount;
        var columns = _matrix.ColumnCount;
        var weightsVector = new Vector<double>(columns);
        var thresh = threshold >= 0 ? threshold : 0.5 * Math.Sqrt(rows + columns + 1) * weightsVector[0] * double.Epsilon;
        int rank = 0;

        for (int i = 0; i < columns; i++)
        {
            if (weightsVector[i] > thresh)
            {
                rank++;
            }
        }

        return rank;
    }

    public void Solve(Vector<double> vector, out Vector<double> solutionVector)
    {
        throw new NotImplementedException();
    }

    public void Solve(Matrix<double> vector, out Matrix<double> solutionMatrix)
    {
        throw new NotImplementedException();
    }
}