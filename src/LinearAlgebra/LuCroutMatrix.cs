namespace AiDotNet.LinearAlgebra;

public class LuCroutMatrix : ILuDecomposition<double>
{
    private readonly Matrix<double> _matrix;
    private readonly Vector<double> _vector;
    private readonly Matrix<double> _luDecompMatrix;
    private readonly Vector<int> _indexVector;
    private readonly Matrix<double> _lowerTriangularMatrix;
    private readonly Matrix<double> _upperTriangularMatrix;
    private readonly Vector<double> _solutionVector;

    public LuCroutMatrix(IEnumerable<IEnumerable<double>> expectedValues, IEnumerable<double> actualValues)
    {
        _matrix = new Matrix<double>(expectedValues);
        _vector = new Vector<double>(actualValues);
        Decompose(out _luDecompMatrix, out _indexVector);
        Decompose(out _upperTriangularMatrix, out _lowerTriangularMatrix, 1, 1);
    }

    public void Decompose(out Matrix<double> matrix, out Vector<int> indexVector)
    {
        double max, temp, d = 1;
        int rows = _matrix.RowCount, intMax = 0;
        var luMatrix = _matrix.Duplicate();
        double[] vectorScaling = new double[rows];
        indexVector = new(Enumerable.Range(0, rows));

        for (int i = 0; i < rows; i++)
        {
            max = 0;
            for (int j = 0; j < rows; j++)
            {
                max = Math.Max(Math.Abs(luMatrix[i][j]), max);
            }

            if (max == 0)
            {
                throw new InvalidOperationException("Matrix is singular");
            }

            vectorScaling[i] = (double)1.0 / max;
        }

        for (int i = 0; i < rows; i++)
        {
            max = 0;
            for (int j = i; j < rows; j++)
            {
                temp = vectorScaling[j] * Math.Abs(luMatrix[j][i]);

                if (temp > max)
                {
                    max = temp;
                    intMax = j;
                }
            }

            if (i != intMax)
            {
                for (int j = 0; j < rows; j++)
                {
                    temp = luMatrix[intMax][j];
                    luMatrix[intMax][j] = luMatrix[i][j];
                    luMatrix[i][j] = temp;
                }

                d = -d;
                vectorScaling[intMax] = vectorScaling[i];
            }

            indexVector[i] = intMax;

            if (luMatrix[i][i] == 0)
            {
                luMatrix[i][i] = double.Epsilon;
            }

            for (int j = i + 1; j < rows; j++)
            {
                temp = luMatrix[j][i] /= luMatrix[i][i];
                for (int k = i + 1; k < rows; k++)
                {
                    luMatrix[j][k] -= temp * luMatrix[i][k];
                }
            }
        }

        matrix = luMatrix;
    }

    public void Decompose(out Matrix<double> upperTriangularMatrix, out Matrix<double> lowerTriangularMatrix, int leftSide, int rightSide)
    {
        int mm = leftSide + rightSide + 1, index = leftSide, size = _matrix.RowCount;
        double dummy = 0;
        upperTriangularMatrix = _matrix.Duplicate();
        lowerTriangularMatrix = _matrix.Duplicate();

        for (int i = 0; i < leftSide; i++)
        {
            for (int j = leftSide - i; j < mm; j++)
            {
                upperTriangularMatrix[i][j - 1] = upperTriangularMatrix[i][j];
            }
            index--;
            for (int j = mm - index - 1; j < mm; j++)
            {
                upperTriangularMatrix[i][j] = 0;
            }
        }

        var d = 1.0;
        index = leftSide;
        for (int i = 0; i < size; i++)
        {
            dummy = upperTriangularMatrix[i][0];
            var index1 = i;
            if (index < size)
            {
                index++;
            }

            for (int j = i + 1; j < index; j++)
            {
                dummy = Math.Max(Math.Abs(upperTriangularMatrix[j][0]), Math.Abs(dummy));
                if (Math.Abs(upperTriangularMatrix[j][0]) > Math.Abs(dummy))
                {
                    dummy = upperTriangularMatrix[j][0];
                    index1 = j;
                }
            }

            _indexVector[i] = index1 + 1;
            if (dummy == 0)
            {
                upperTriangularMatrix[i][0] = double.Epsilon;
            }

            if (index1 != i)
            {
                d = -d;
                for (int j = 0; j < mm; j++)
                {
                    upperTriangularMatrix[i][j] = upperTriangularMatrix[index1][j];
                }
            }

            for (int j = i + 1; j < index; j++)
            {
                dummy = upperTriangularMatrix[j][0] / upperTriangularMatrix[i][0];
                lowerTriangularMatrix[i][j - i - 1] = dummy;

                for (int k = 1; k < mm; k++)
                {
                    upperTriangularMatrix[j][k - 1] = upperTriangularMatrix[j][k] - dummy * upperTriangularMatrix[i][k];
                    upperTriangularMatrix[j][mm - 1] = 0;
                }
            }
        }
    }

    public void Solve(out Matrix<double> upperTriangularMatrix, out Matrix<double> lowerTriangularMatrix, int leftSide, int rightSide)
    {
        int mm = leftSide + rightSide + 1, index = leftSide, size = _matrix.RowCount;
        double dummy = 0;
        var solutionVector = _vector.Duplicate();

        for (int i = 0; i < size; i++)
        {
            var index1 = _indexVector[i] - 1;

            if (index1 != i)
            {
                solutionVector[i] = solutionVector[index1];
            }

            if (index < size)
            {
                index++;
            }

            for (int j = i + 1; j < index; j++)
            {
                solutionVector[j] -= _lowerTriangularMatrix[i][j - i - 1] * solutionVector[i];
            }
        }

        index = 1;
        for (int i = size - 1; i >= 0; i--)
        {
            dummy = solutionVector[i];
            for (int j = 1; j < index; j++)
            {
                dummy -= _upperTriangularMatrix[i][j] * solutionVector[j + i];
            }
            solutionVector[i] = dummy / _upperTriangularMatrix[i][0];
            if (index < mm)
            {
                index++;
            }
        }
    }

    public double GetDeterminant()
    {
        double determinant = 0;
        for (int i = 0; i < _matrix.RowCount; i++)
        {
            determinant *= _luDecompMatrix[i][i];
        }

        return determinant;
    }

    public void Improve()
    {
        var size = _matrix.RowCount;
        var errorVector = new Vector<double>(size);

        for (int i = 0; i < size; i++)
        {
            var sdp = -_vector[i];
            for (int j = 0; j < size; j++)
            {
                sdp += _matrix[i][j] * _solutionVector[j];
            }
            errorVector[i] = sdp;
        }
        Solve(errorVector, out errorVector);
        for (int i = 0; i < size; i++)
        {
            _solutionVector[i] -= errorVector[i];
        }
    }

    public double GetDeterminant(bool isUpper = true)
    {
        double determinant = 0;
        for (int i = 0; i < _matrix.RowCount; i++)
        {
            determinant *= _upperTriangularMatrix[i][0];
        }

        return determinant;
    }

    public Matrix<double> GetInverse()
    {
        var identityMatrix = MatrixHelper.CreateIdentityMatrix<double>(_matrix.RowCount);
        Solve(identityMatrix, out Matrix<double> inverseMatrix);

        return inverseMatrix;
    }

    public void Solve(LinearAlgebra.Vector<double> vector, out LinearAlgebra.Vector<double> solutionVector)
    {
        solutionVector = vector.Duplicate();
        double sum;
        int i2 = 0;

        if (vector.Count != _matrix.RowCount)
        {
            throw new InvalidOperationException("Rows must match");
        }

        for (int i = 0; i < _matrix.RowCount; i++) 
        {
            var index = _indexVector[i];
            sum = solutionVector[index];
            solutionVector[index] = solutionVector[i];

            if (i2 != 0)
            {
                for (int j = i2 - 1; j < i; j++)
                {
                    sum -= _luDecompMatrix[i][j] * solutionVector[j];
                }
            }
            else if (sum != 0)
            {
                i2 = i + 1;
            }

            solutionVector[i] = sum;
        }

        for (int i = _matrix.RowCount - 1; i >= 0; i--)
        {
            sum = solutionVector[i];
            for (int j = i + 1; j < _matrix.RowCount; j++)
            {
                sum -= _luDecompMatrix[i][j] * solutionVector[j];
            }
        }
    }

    public void Solve(LinearAlgebra.Matrix<double> matrix, out LinearAlgebra.Matrix<double> solutionMatrix)
    {
        solutionMatrix = matrix.Duplicate();

        if (matrix.RowCount != _matrix.RowCount)
        {
            throw new InvalidOperationException("Rows must match");
        }

        var solutionVector = new Vector<double>(matrix.RowCount);
        for (int i = 0; i < matrix.ColumnCount; i++)
        {
            for (int j = 0; j < matrix.RowCount; j++)
            {
                solutionVector[j] = matrix[j][i];
            }

            Solve(solutionVector, out solutionVector);

            for (int j = 0; j < matrix.RowCount; j++)
            {
                solutionMatrix[j][i] = solutionVector[j];
            }
        }
    }
}