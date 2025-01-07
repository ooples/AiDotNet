namespace AiDotNet.LinearAlgebra;

public class Matrix<T> : MatrixBase<T>
{
    public Matrix(int rows, int columns, INumericOperations<T>? numericOperations = null) 
        : base(rows, columns, numericOperations)
    {
    }

    public Matrix(IEnumerable<IEnumerable<T>> values, INumericOperations<T>? numericOperations = null)
        : base(values, numericOperations)
    {
    }

    public Matrix(T[,] data, INumericOperations<T>? numericOperations = null)
        : base(data, numericOperations)
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

    public static Matrix<T> operator /(Matrix<T> matrix, T scalar)
    {
        return matrix.Divide(scalar);
    }

     public static Matrix<T> operator /(Matrix<T> left, Matrix<T> right)
    {
        return left.Divide(right);
    }

    public static Matrix<T> CreateFromVector(Vector<T> vector)
    {
        return new Matrix<T>([vector.AsEnumerable()]);
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

    public Matrix<T> Subtract(Matrix<T> other)
    {
        if (this.Rows != other.Rows || this.Columns != other.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions for subtraction.");
        }

        var numOp = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(Rows, Columns);
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                result[i, j] = numOp.Subtract(this[i, j], other[i, j]);
            }
        }

        return result;
    }

    public Vector<T> GetColumnSegment(int columnIndex, int startRow, int length)
    {
        return new Vector<T>(Enumerable.Range(startRow, length).Select(i => this[i, columnIndex]));
    }

    public Vector<T> GetRowSegment(int rowIndex, int startColumn, int length)
    {
        return new Vector<T>(Enumerable.Range(startColumn, length).Select(j => this[rowIndex, j]));
    }

    public Matrix<T> GetSubMatrix(int startRow, int startColumn, int rowCount, int columnCount)
    {
        Matrix<T> subMatrix = new(rowCount, columnCount);
        for (int i = 0; i < rowCount; i++)
        {
            for (int j = 0; j < columnCount; j++)
            {
                subMatrix[i, j] = this[startRow + i, startColumn + j];
            }
        }

        return subMatrix;
    }

    public void SetSubMatrix(int startRow, int startColumn, Matrix<T> subMatrix)
    {
        for (int i = 0; i < subMatrix.Rows; i++)
        {
            for (int j = 0; j < subMatrix.Columns; j++)
            {
                this[startRow + i, startColumn + j] = subMatrix[i, j];
            }
        }
    }

    public static Matrix<T> FromRows(params IEnumerable<T>[] vectors)
    {
        return FromRowVectors(vectors);
    }

    public static Matrix<T> FromColumns(params IEnumerable<T>[] vectors)
    {
        return FromColumnVectors(vectors);
    }

    public static Matrix<T> FromVector(Vector<T> vector)
    {
        var matrix = new Matrix<T>(vector.Length, 1);
        for (int i = 0; i < vector.Length; i++)
        {
            matrix[i, 0] = vector[i];
        }

        return matrix;
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

    public Vector<T> RowWiseMax()
    {
        Vector<T> result = new(Rows);
        for (int i = 0; i < Rows; i++)
        {
            T max = this[i, 0];
            for (int j = 1; j < Columns; j++)
            {
                if (ops.GreaterThan(this[i, j], max))
                    max = this[i, j];
            }

            result[i] = max;
        }

        return result;
    }

    public Matrix<T> Transform(Func<T, int, int, T> transformer)
    {
        Matrix<T> result = new(Rows, Columns);

        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                result[i, j] = transformer(this[i, j], i, j);
            }
        }

        return result;
    }

    public Vector<T> RowWiseSum()
    {
        Vector<T> result = new(Rows);

        for (int i = 0; i < Rows; i++)
        {
            T sum = ops.Zero;

            for (int j = 0; j < Columns; j++)
            {
                sum = ops.Add(sum, this[i, j]);
            }

            result[i] = sum;
        }

        return result;
    }

    public Matrix<T> PointwiseDivide(Matrix<T> other)
    {
        if (Rows != other.Rows || Columns != other.Columns)
            throw new ArgumentException("Matrices must have the same dimensions for pointwise division.");

        Matrix<T> result = new(Rows, Columns);

        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                result[i, j] = ops.Divide(this[i, j], other[i, j]);
            }
        }

        return result;
    }

    public Matrix<T> Divide(T scalar)
    {
        var numOp = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(Rows, Columns);
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                result[i, j] = numOp.Divide(this[i, j], scalar);
            }
        }

        return result;
    }

    public Matrix<T> Divide(Matrix<T> other)
    {
        if (this.Rows != other.Rows || this.Columns != other.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions for division.");
        }

        var numOp = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(Rows, Columns);
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                result[i, j] = numOp.Divide(this[i, j], other[i, j]);
            }
        }

        return result;
    }

    public static Matrix<T> OuterProduct(Vector<T> a, Vector<T> b)
    {
        if (a == null || b == null)
        {
            throw new ArgumentNullException(a == null ? nameof(a) : nameof(b), "Vectors cannot be null.");
        }

        int rows = a.Length;
        int cols = b.Length;
        var result = new Matrix<T>(rows, cols);
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = numOps.Multiply(a[i], b[j]);
            }
        }

        return result;
    }

    public byte[] Serialize()
    {
        return SerializationHelper<T>.SerializeMatrix(this);
    }

    public static Matrix<T> Deserialize(byte[] data)
    {
        return SerializationHelper<T>.DeserializeMatrix(data);
    }

    public IEnumerable<Vector<T>> GetColumns()
    {
        for (var i = 0; i < Columns; i++)
        {
            yield return GetColumn(i);
        }
    }

    public IEnumerable<Vector<T>> GetRows()
    {
        for (var i = 0; i < Rows; i++)
        {
            yield return GetRow(i);
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

    public Matrix<T> GetRows(IEnumerable<int> indices)
    {
        var indexArray = indices.ToArray();
        var newRows = indexArray.Length;
        var newMatrix = new T[newRows, Columns];
        for (int i = 0; i < newRows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                newMatrix[i, j] = this[indexArray[i], j];
            }
        }

        return new Matrix<T>(newMatrix);
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