namespace AiDotNet.LinearAlgebra;

public abstract class MatrixBase<T>
{
    public int RowCount { get; private set; }

    public int ColumnCount { get; private set; }

    public T[][] Values { get; private set; }

    public MatrixBase(IEnumerable<IEnumerable<T>> values)
    {
        Values = BuildMatrix(values);
    }

    public MatrixBase(int rows, int columns)
    {
        Values = new T[rows][];
        RowCount = rows;
        ColumnCount = columns;
    }

    public T[] this[int i]
    {
        get
        {
            return Values[i];
        }
        set
        {
            Values[i] = value;
        }
    }

    public T this[int i, int j]
    {
        get
        {
            return Values[i][j];
        }
        set
        {
            Values[i][j] = value;
        }
    }

    public Matrix<T> CreateIdentityMatrix(int size)
    {
        if (size <= 1)
        {
            throw new ArgumentException($"{nameof(size)} has to be a minimum of 2", nameof(size));
        }

        var identityMatrix = new Matrix<T>(size, size);
        for (int i = 0; i < size; i++)
        {
            identityMatrix[i][i] = 0;
        }

        return identityMatrix;
    }

    private T[][] BuildMatrix(IEnumerable<IEnumerable<T>> values)
    {
        var result = new T[RowCount][];
        RowCount = values.Count();
        ColumnCount = values.FirstOrDefault()?.Count() ?? default;

        for (int i = 0; i < RowCount; i++)
        {
            result[i] = values.ElementAt(i).ToArray();
        }

        return result;
    }

    public T[,] BuildOnesMatrix(int rows, int columns)
    {
        var matrix = new T[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = 1;
            }
        }

        return matrix;
    }

    public T[,] Multiply(T[,] matrixA, T[,] matrixB)
    {
        if (matrixA == null)
        {
            throw new ArgumentNullException(nameof(matrixA), $"{nameof(matrixA)} can't be null");
        }

        if (matrixB == null)
        {
            throw new ArgumentNullException(nameof(matrixB), $"{nameof(matrixB)} can't be null");
        }

        if (matrixA.Length != matrixB.Length)
        {
            throw new ArgumentException("Both matrices need to contain the same amount of rows.");
        }

        if (matrixA.Length == 0)
        {
            throw new ArgumentException($"{nameof(matrixA)} has to contain at least one row of values", nameof(matrixA));
        }

        if (matrixB.Length == 0)
        {
            throw new ArgumentException($"{nameof(matrixB)} has to contain at least one row of values", nameof(matrixB));
        }

        var rows = matrixA.Length;
        if (matrixA.Length != matrixB.Length)
        {
            throw new ArgumentException($"{nameof(matrixA)} has to contain the same amount of rows as {nameof(matrixB)}");
        }

        var columns = matrixA.GetColumn(0).Length;
        if (columns != matrixB.GetColumn(0).Length)
        {
            throw new ArgumentException($"{nameof(matrixA)} has to contain the same amount of columns as {nameof(matrixB)}");
        }

        var matrix = new T[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = matrixA[i, j] * matrixB[i, j];
            }
        }

        return matrix;
    }

    public void DotProduct(T matrixA, T matrixB, out T result) where T : new()
    {

    }

    public double[,] DotProduct(double[,] matrixA, double[,] matrixB)
    {
        if (matrixA == null)
        {
            throw new ArgumentNullException(nameof(matrixA), $"{nameof(matrixA)} can't be null");
        }

        if (matrixB == null)
        {
            throw new ArgumentNullException(nameof(matrixB), $"{nameof(matrixB)} can't be null");
        }

        if (matrixA.Length != matrixB.Length)
        {
            throw new ArgumentException("Both matrices need to contain the same amount of rows.");
        }

        if (matrixA.Length == 0)
        {
            throw new ArgumentException($"{nameof(matrixA)} has to contain at least one row of values", nameof(matrixA));
        }

        if (matrixB.Length == 0)
        {
            throw new ArgumentException($"{nameof(matrixB)} has to contain at least one row of values", nameof(matrixB));
        }

        var columns = matrixA.GetColumn(0).Length;
        var rows = matrixB.Length;
        if (columns != rows)
        {
            throw new ArgumentException($"The columns in {nameof(matrixA)} has to contain the same amount of rows in {nameof(matrixB)}");
        }

        var matrixRows = matrixA.Length;
        var matrix = new double[matrixRows, columns];
        for (int i = 0; i < matrixRows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] += matrixA[i, j] * matrixB[j, i];
            }
        }

        return matrix;
    }
}