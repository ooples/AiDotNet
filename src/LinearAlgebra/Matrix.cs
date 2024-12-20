namespace AiDotNet.LinearAlgebra;

public class Matrix<T> : MatrixBase<T>
{
    public Matrix(int rows, int columns, INumericOperations<T>? numericOperations = null) 
        : base(rows, columns, numericOperations ?? MathHelper.GetNumericOperations<T>())
    {
    }

    public Matrix(IEnumerable<IEnumerable<T>> values, INumericOperations<T>? numericOperations = null)
        : base(values, numericOperations ?? MathHelper.GetNumericOperations<T>())
    {
    }

    protected override MatrixBase<T> CreateInstance(int rows, int cols)
    {
        return new Matrix<T>(rows, cols, ops);
    }

    public static Matrix<T> CreateMatrix<T2>(int rows, int columns)
    {
        return new Matrix<T>(rows, columns);
    }

    public static Matrix<T> CreateIdentityMatrix<T2>(int size)
    {
        if (size <= 1)
        {
            throw new ArgumentException($"{nameof(size)} has to be a minimum of 2", nameof(size));
        }

        var operations = MathHelper.GetNumericOperations<T>();
        var identityMatrix = new Matrix<T>(size, size, operations);
        for (int i = 0; i < size; i++)
        {
            identityMatrix[i, i] = operations.One;
        }

        return identityMatrix;
    }

    public new Vector<T> GetColumn(int col)
    {
        return (Vector<T>)base.GetColumn(col);
    }

    public new Matrix<T> Copy()
    {
        return (Matrix<T>)base.Copy();
    }

    public new Matrix<T> Add(MatrixBase<T> other)
    {
        return (Matrix<T>)base.Add(other);
    }

    public new Matrix<T> Subtract(MatrixBase<T> other)
    {
        return (Matrix<T>)base.Subtract(other);
    }

    public new Matrix<T> Multiply(MatrixBase<T> other)
    {
        return (Matrix<T>)base.Multiply(other);
    }

    public new Vector<T> Multiply(Vector<T> vector)
    {
        return (Vector<T>)base.Multiply(vector);
    }

    public new Matrix<T> Multiply(T scalar)
    {
        return (Matrix<T>)base.Multiply(scalar);
    }

    public new Matrix<T> Transpose()
    {
        return (Matrix<T>)base.Transpose();
    }

    public static Matrix<T> operator +(Matrix<T> left, Matrix<T> right)
    {
        return left.Add(right);
    }

    public static Matrix<T> operator -(Matrix<T> left, Matrix<T> right)
    {
        return left.Subtract(right);
    }

    public static Matrix<T> operator *(Matrix<T> left, Matrix<T> right)
    {
        return left.Multiply(right);
    }

    public static Vector<T> operator *(Matrix<T> matrix, Vector<T> vector)
    {
        return matrix.Multiply(vector);
    }

    public static Matrix<T> operator *(Matrix<T> matrix, T scalar)
    {
        return matrix.Multiply(scalar);
    }

    public static Matrix<T> CreateFromVector(Vector<T> vector)
    {
        return new Matrix<T>(new[] { vector.AsEnumerable() });
    }

    public new Matrix<T> Ones(int rows, int cols)
    {
        return (Matrix<T>)base.Ones(rows, cols);
    }

    public new Matrix<T> Zeros(int rows, int cols)
    {
        return (Matrix<T>)base.Zeros(rows, cols);
    }

    public static Matrix<T> CreateOnes(int rows, int cols, INumericOperations<T>? numericOperations = null)
    {
        var matrix = new Matrix<T>(rows, cols, numericOperations);
        return matrix.Ones(rows, cols);
    }

    public static Matrix<T> CreateZeros(int rows, int cols, INumericOperations<T>? numericOperations = null)
    {
        var matrix = new Matrix<T>(rows, cols, numericOperations);
        return matrix.Zeros(rows, cols);
    }

    public static Matrix<T> CreateDiagonal(Vector<T> diagonal, INumericOperations<T>? numericOperations = null)
    {
        var matrix = new Matrix<T>(diagonal.Length, diagonal.Length, numericOperations);
        for (int i = 0; i < diagonal.Length; i++)
        {
            matrix[i, i] = diagonal[i];
        }

        return matrix;
    }

    public static Matrix<T> CreateIdentity(int size, INumericOperations<T>? numericOperations = null)
    {
        var numOp = numericOperations ?? MathHelper.GetNumericOperations<T>();
        var identity = new Matrix<T>(size, size, numOp);
        for (int i = 0; i < size; i++)
        {
            identity[i, i] = numOp.One;
        }

        return identity;
    }

    public static Matrix<T> CreateRandom(int rows, int columns, INumericOperations<T>? numericOperations = null)
    {
        var numOp = numericOperations ?? MathHelper.GetNumericOperations<T>();
        Matrix<T> matrix = new(rows, columns, numOp);
        Random random = new();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = numOp.FromDouble(random.NextDouble());
            }
        }

        return matrix;
    }

    public static Matrix<T> BlockDiagonal(params Matrix<T>[] matrices)
    {
        int totalRows = matrices.Sum(m => m.Rows);
        int totalCols = matrices.Sum(m => m.Columns);
        Matrix<T> result = new(totalRows, totalCols);

        int rowOffset = 0;
        int colOffset = 0;
        foreach (var matrix in matrices)
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    result[rowOffset + i, colOffset + j] = matrix[i, j];
                }
            }
            rowOffset += matrix.Rows;
            colOffset += matrix.Columns;
        }

        return result;
    }

    public new static Matrix<T> Empty()
    {
        return new Matrix<T>(0, 0);
    }

    public static Matrix<T> FromColumnVectors(IEnumerable<IEnumerable<T>> vectors)
    {
        if (vectors == null)
            throw new ArgumentNullException(nameof(vectors));
        var vectorList = vectors.Select(v => v.ToList()).ToList();
        if (vectorList.Count == 0)
            throw new ArgumentException("Vector list cannot be empty");
        int rows = vectorList[0].Count;
        if (vectorList.Any(v => v.Count != rows))
            throw new ArgumentException("All vectors must have the same length");

        var matrix = new Matrix<T>(rows, vectorList.Count);
        
        for (int j = 0; j < vectorList.Count; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                matrix[i, j] = vectorList[j][i];
            }
        }

        return matrix;
    }

    public static Matrix<T> FromRows(params IEnumerable<T>[] vectors)
    {
        return FromRowVectors(vectors);
    }

    public static Matrix<T> FromColumns(params IEnumerable<T>[] vectors)
    {
        return FromColumnVectors(vectors);
    }

    public static Matrix<T> FromRowVectors(IEnumerable<IEnumerable<T>> vectors)
    {
        if (vectors == null)
            throw new ArgumentNullException(nameof(vectors));
        var vectorList = vectors.Select(v => v.ToList()).ToList();
        if (vectorList.Count == 0)
            throw new ArgumentException("Vector list cannot be empty");
        int cols = vectorList[0].Count;
        if (vectorList.Any(v => v.Count != cols))
            throw new ArgumentException("All vectors must have the same length");

        var matrix = new Matrix<T>(vectorList.Count, cols);
        
        for (int i = 0; i < vectorList.Count; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = vectorList[i][j];
            }
        }

        return matrix;
    }

    public IEnumerable<Vector<T>> EnumerateColumns()
    {
        for (var i = 0; i < Columns; i++)
        {
            yield return GetColumn(i);
        }
    }

    public Matrix<T> RemoveRow(int rowIndex)
    {
        if (rowIndex < 0 || rowIndex >= Rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));

        var newMatrix = new Matrix<T>(Rows - 1, Columns);
        int newRow = 0;

        for (int i = 0; i < Rows; i++)
        {
            if (i == rowIndex) continue;

            for (int j = 0; j < Columns; j++)
            {
                newMatrix[newRow, j] = this[i, j];
            }
            newRow++;
        }

        return newMatrix;
    }

    public Matrix<T> RemoveColumn(int columnIndex)
    {
        if (columnIndex < 0 || columnIndex >= Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));

        var newMatrix = new Matrix<T>(Rows, Columns - 1);

        for (int i = 0; i < Rows; i++)
        {
            int newColumn = 0;
            for (int j = 0; j < Columns; j++)
            {
                if (j == columnIndex) continue;
                newMatrix[i, newColumn] = this[i, j];
                newColumn++;
            }
        }

        return newMatrix;
    }
}

public static class Matrix
{
    public static Matrix<double> CreateDoubleMatrix(int rows, int columns)
    {
        return new Matrix<double>(rows, columns, new DoubleOperations());
    }

    public static Matrix<Complex<T>> CreateComplexMatrix<T>(int rows, int columns)
    {
        return new Matrix<Complex<T>>(rows, columns);
    }

    public static Matrix<double> CreateDoubleMatrix(IEnumerable<IEnumerable<double>> values)
    {
        return new Matrix<double>(values, new DoubleOperations());
    }

    public static Matrix<Complex<T>> CreateComplexMatrix<T>(IEnumerable<IEnumerable<Complex<T>>> values)
    {
        return new Matrix<Complex<T>>(values);
    }
}