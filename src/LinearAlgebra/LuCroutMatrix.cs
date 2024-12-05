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

    public LuCroutMatrix(IEnumerable<Vector<double>> expectedValues, IEnumerable<double> actualValues)
    {
        var operations = MatrixHelper.GetNumericOperations<double>();
        _matrix = new Matrix<double>(expectedValues, operations);
        _vector = new Vector<double>(actualValues, operations);
        Decompose(out _luDecompMatrix, out _indexVector);
        Decompose(out _upperTriangularMatrix, out _lowerTriangularMatrix, 1, 1);
        _solutionVector = new Vector<double>(_matrix.Rows, operations);
    }

    public void Decompose(out Matrix<double> matrix, out Vector<int> indexVector)
    {
        double max, temp, d = 1;
        int rows = _matrix.Rows, intMax = 0;
        var luMatrix = _matrix.Copy();
        double[] vectorScaling = new double[rows];
        var operations = MatrixHelper.GetNumericOperations<int>();
        indexVector = new Vector<int>(Enumerable.Range(0, rows), operations);

        for (int i = 0; i < rows; i++)
        {
            max = 0;
            for (int j = 0; j < rows; j++)
            {
                max = Math.Max(Math.Abs(luMatrix[i, j]), max);
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
                temp = vectorScaling[j] * Math.Abs(luMatrix[j, i]);

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
                    temp = luMatrix[intMax, j];
                    luMatrix[intMax, j] = luMatrix[i, j];
                    luMatrix[i, j] = temp;
                }

                d = -d;
                vectorScaling[intMax] = vectorScaling[i];
            }

            indexVector[i] = intMax;

            if (luMatrix[i, i] == 0)
            {
                luMatrix[i, i] = double.Epsilon;
            }

            for (int j = i + 1; j < rows; j++)
            {
                temp = luMatrix[j, i] /= luMatrix[i, i];
                for (int k = i + 1; k < rows; k++)
                {
                    luMatrix[j, k] -= temp * luMatrix[i, k];
                }
            }
        }

        matrix = luMatrix;
    }

    public void Decompose(out Matrix<double> upperTriangularMatrix, out Matrix<double> lowerTriangularMatrix, int leftSide, int rightSide)
    {
        int mm = leftSide + rightSide + 1, index = leftSide, size = _matrix.Rows;
        double dummy = 0;
        upperTriangularMatrix = _matrix.Copy();
        lowerTriangularMatrix = _matrix.Copy();

        for (int i = 0; i < leftSide; i++)
        {
            for (int j = leftSide - i; j < mm; j++)
            {
                upperTriangularMatrix[i, j - 1] = upperTriangularMatrix[i, j];
            }
            index--;
            for (int j = mm - index - 1; j < mm; j++)
            {
                upperTriangularMatrix[i, j] = 0;
            }
        }

        var d = 1.0;
        index = leftSide;
        for (int i = 0; i < size; i++)
        {
            dummy = upperTriangularMatrix[i, 0];
            var index1 = i;
            if (index < size)
            {
                index++;
            }

            for (int j = i + 1; j < index; j++)
            {
                dummy = Math.Max(Math.Abs(upperTriangularMatrix[j, 0]), Math.Abs(dummy));
                if (Math.Abs(upperTriangularMatrix[j, 0]) > Math.Abs(dummy))
                {
                    dummy = upperTriangularMatrix[j, 0];
                    index1 = j;
                }
            }

            _indexVector[i] = index1 + 1;
            if (dummy == 0)
            {
                upperTriangularMatrix[i, 0] = double.Epsilon;
            }

            if (index1 != i)
            {
                d = -d;
                for (int j = 0; j < mm; j++)
                {
                    upperTriangularMatrix[i, j] = upperTriangularMatrix[index1, j];
                }
            }

            for (int j = i + 1; j < index; j++)
            {
                dummy = upperTriangularMatrix[j, 0] / upperTriangularMatrix[i, 0];
                lowerTriangularMatrix[i, j - i - 1] = dummy;

                for (int k = 1; k < mm; k++)
                {
                    upperTriangularMatrix[j, k - 1] = upperTriangularMatrix[j, k] - dummy * upperTriangularMatrix[i, k];
                    upperTriangularMatrix[j, mm - 1] = 0;
                }
            }
        }
    }

    public void Solve(out Matrix<double> upperTriangularMatrix, out Matrix<double> lowerTriangularMatrix, int leftSide, int rightSide)
    {
        int mm = leftSide + rightSide + 1, index = leftSide, size = _matrix.Rows;
        double dummy = 0;
        var solutionVector = _vector.Copy();

        // Initialize the out parameters
        upperTriangularMatrix = _upperTriangularMatrix.Copy();
        lowerTriangularMatrix = _lowerTriangularMatrix.Copy();

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
                solutionVector[j] -= lowerTriangularMatrix[i, j - i - 1] * solutionVector[i];
            }
        }

        index = 1;
        for (int i = size - 1; i >= 0; i--)
        {
            dummy = solutionVector[i];
            for (int j = 1; j < index; j++)
            {
                dummy -= upperTriangularMatrix[i, j] * solutionVector[j + i];
            }
            solutionVector[i] = dummy / upperTriangularMatrix[i, 0];
            if (index < mm)
            {
                index++;
            }
        }

        // If you want to update the solution vector in the class
        for (int i = 0; i < size; i++)
        {
            _solutionVector[i] = solutionVector[i];
        }
    }

    public double GetDeterminant()
    {
        double determinant = 0;
        for (int i = 0; i < _matrix.Rows; i++)
        {
            determinant *= _luDecompMatrix[i, i];
        }

        return determinant;
    }

    public void Improve()
    {
        var size = _matrix.Rows;
        var operations = MatrixHelper.GetNumericOperations<double>();
        var errorVector = new Vector<double>(size, operations);

        for (int i = 0; i < size; i++)
        {
            var sdp = -_vector[i];
            for (int j = 0; j < size; j++)
            {
                sdp += _matrix[i, j] * _solutionVector[j];
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
        for (int i = 0; i < _matrix.Rows; i++)
        {
            determinant *= _upperTriangularMatrix[i, 0];
        }

        return determinant;
    }

    public Matrix<double> GetInverse()
    {
        var identityMatrix = MatrixHelper.CreateIdentityMatrix<double>(_matrix.Rows);
        Solve(identityMatrix, out Matrix<double> inverseMatrix);

        return inverseMatrix;
    }

    public void Solve(Vector<double> vector, out Vector<double> solutionVector)
    {
        solutionVector = vector.Copy();
        double sum;
        int i2 = 0;

        if (vector.Length != _matrix.Rows)
        {
            throw new InvalidOperationException("Rows must match");
        }

        for (int i = 0; i < _matrix.Rows; i++) 
        {
            var index = _indexVector[i];
            sum = solutionVector[index];
            solutionVector[index] = solutionVector[i];

            if (i2 != 0)
            {
                for (int j = i2 - 1; j < i; j++)
                {
                    sum -= _luDecompMatrix[i, j] * solutionVector[j];
                }
            }
            else if (sum != 0)
            {
                i2 = i + 1;
            }

            solutionVector[i] = sum;
        }

        for (int i = _matrix.Rows - 1; i >= 0; i--)
        {
            sum = solutionVector[i];
            for (int j = i + 1; j < _matrix.Rows; j++)
            {
                sum -= _luDecompMatrix[i, j] * solutionVector[j];
            }
        }
    }

    public void Solve(Matrix<double> matrix, out Matrix<double> solutionMatrix)
    {
        solutionMatrix = matrix.Copy();

        if (matrix.Rows != _matrix.Rows)
        {
            throw new InvalidOperationException("Rows must match");
        }

        var operations = MatrixHelper.GetNumericOperations<double>();
        var solutionVector = new Vector<double>(matrix.Rows, operations);
        for (int i = 0; i < matrix.Columns; i++)
        {
            for (int j = 0; j < matrix.Rows; j++)
            {
                solutionVector[j] = matrix[j, i];
            }

            Solve(solutionVector, out solutionVector);

            for (int j = 0; j < matrix.Rows; j++)
            {
                solutionMatrix[j, i] = solutionVector[j];
            }
        }
    }

    public void Decompose(out Matrix<double> upperTriangularMatrix, out Matrix<double> lowerTriangularMatrix)
    {
        throw new NotImplementedException();
    }

    public void Decompose(Matrix<double> aMatrix)
    {
        throw new NotImplementedException();
    }

    public Vector<double> Solve(Matrix<double> aMatrix, Vector<double> bVector)
    {
        throw new NotImplementedException();
    }

    public Matrix<double> Invert()
    {
        throw new NotImplementedException();
    }
}