namespace AiDotNet.Extensions;

public static class MatrixExtensions
{
    public static Matrix<T> AddConstantColumn<T>(this Matrix<T> matrix, T value)
    {
        var newMatrix = new Matrix<T>(matrix.Rows, matrix.Columns + 1);
        for (int i = 0; i < matrix.Rows; i++)
        {
            newMatrix[i, 0] = value;
            for (int j = 0; j < matrix.Columns; j++)
            {
                newMatrix[i, j + 1] = matrix[i, j];
            }
        }

        return newMatrix;
    }

    public static Vector<T> GetSubColumn<T>(this Matrix<T> matrix, int columnIndex, int startRow, int length)
    {
        if (columnIndex < 0 || columnIndex >= matrix.Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));
        if (startRow < 0 || startRow >= matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(startRow));
        if (length < 0 || startRow + length > matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(length));

        var result = new Vector<T>(length, MathHelper.GetNumericOperations<T>());
        for (int i = 0; i < length; i++)
        {
            result[i] = matrix[startRow + i, columnIndex];
        }

        return result;
    }

    public static T LogDeterminant<T>(this Matrix<T> matrix)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
            
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square to calculate determinant.");
        }

        // Use LU decomposition to calculate log determinant
        var lu = new LuDecomposition<T>(matrix);
        var U = lu.U;

        T logDet = numOps.Zero;
        for (int i = 0; i < U.Rows; i++)
        {
            logDet = numOps.Add(logDet, numOps.Log(numOps.Abs(U[i, i])));
        }

        return logDet;
    }

    public static Matrix<T> PointwiseMultiply<T>(this Matrix<T> matrix, Matrix<T> other)
    {
        if (matrix.Rows != other.Rows || matrix.Columns != other.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions for pointwise multiplication.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = numOps.Multiply(matrix[i, j], other[i, j]);
            }
        }

        return result;
    }

    public static Matrix<T> PointwiseMultiply<T>(this Matrix<T> matrix, Vector<T> vector)
    {
        if (matrix.Rows != vector.Length)
        {
            throw new ArgumentException("The number of rows in the matrix must match the length of the vector.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(matrix.Rows, matrix.Columns);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = numOps.Multiply(matrix[i, j], vector[i]);
            }
        }

        return result;
    }

    public static Matrix<T> AddColumn<T>(this Matrix<T> matrix, Vector<T> column)
    {
        if (matrix.Rows != column.Length)
        {
            throw new ArgumentException("Column length must match matrix row count.");
        }

        Matrix<T> newMatrix = new Matrix<T>(matrix.Rows, matrix.Columns + 1);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                newMatrix[i, j] = matrix[i, j];
            }
            newMatrix[i, matrix.Columns] = column[i];
        }

        return newMatrix;
    }

    public static Matrix<T> Submatrix<T>(this Matrix<T> matrix, int startRow, int startCol, int numRows, int numCols)
    {
        if (startRow < 0 || startCol < 0 || startRow + numRows > matrix.Rows || startCol + numCols > matrix.Columns)
        {
            throw new ArgumentOutOfRangeException("Invalid submatrix dimensions");
        }

        Matrix<T> submatrix = new Matrix<T>(numRows, numCols);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                submatrix[i, j] = matrix[startRow + i, startCol + j];
            }
        }

        return submatrix;
    }

    public static Matrix<T> GetColumns<T>(this Matrix<T> matrix, IEnumerable<int> columnIndices)
    {
        return new Matrix<T>(GetColumnVectors(matrix, columnIndices));
    }

    public static List<Vector<T>> GetColumnVectors<T>(this Matrix<T> matrix, IEnumerable<int> columnIndices)
    {
        var selectedColumns = new List<Vector<T>>();
        foreach (int index in columnIndices)
        {
            if (index < 0 || index >= matrix.Columns)
            {
                throw new ArgumentOutOfRangeException(nameof(columnIndices), $"Column index {index} is out of range.");
            }
            selectedColumns.Add(matrix.GetColumn(index));
        }

        return selectedColumns;
    }

    public static T Max<T>(this Matrix<T> matrix, Func<T, T> selector)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T max = selector(matrix[0, 0]);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                T value = selector(matrix[i, j]);
                if (numOps.GreaterThan(value, max))
                {
                    max = value;
                }
            }
        }

        return max;
    }

    public static Matrix<T> GetRowRange<T>(this Matrix<T> matrix, int startRow, int rowCount)
    {
        Matrix<T> result = new(rowCount, matrix.Columns);
        for (int i = 0; i < rowCount; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = matrix[startRow + i, j];
            }
        }

        return result;
    }

    public static Vector<T> RowWiseArgmax<T>(this Matrix<T> matrix)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        Vector<T> result = new(matrix.Rows);

        for (int i = 0; i < matrix.Rows; i++)
        {
            T max = matrix[i, 0];
            int maxIndex = 0;
            for (int j = 1; j < matrix.Columns; j++)
            {
                if (numOps.GreaterThan(matrix[i, j], max))
                {
                    max = matrix[i, j];
                    maxIndex = j;
                }
            }
            result[i] = numOps.FromDouble(maxIndex);
        }

        return result;
    }

    public static Matrix<T> KroneckerProduct<T>(this Matrix<T> a, Matrix<T> b)
    {
        int m = a.Rows;
        int n = a.Columns;
        int p = b.Rows;
        int q = b.Columns;

        var numOps = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(m * p, n * q);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    for (int l = 0; l < q; l++)
                    {
                        result[i * p + k, j * q + l] = numOps.Multiply(a[i, j], b[k, l]);
                    }
                }
            }
        }

        return result;
    }

    public static Vector<T> Flatten<T>(this Matrix<T> matrix)
    {
        var vector = new Vector<T>(matrix.Rows * matrix.Columns);
        int index = 0;

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                vector[index++] = matrix[i, j];
            }
        }

        return vector;
    }

    public static Matrix<T> Reshape<T>(this Matrix<T> matrix, int newRows, int newColumns)
    {
        if (matrix.Rows * matrix.Columns != newRows * newColumns)
        {
            throw new ArgumentException("The total number of elements must remain the same after reshaping.");
        }

        var reshaped = new Matrix<T>(newRows, newColumns);
        int index = 0;
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                int newRow = index / newColumns;
                int newCol = index % newColumns;
                reshaped[newRow, newCol] = matrix[i, j];
                index++;
            }
        }

        return reshaped;
    }
}