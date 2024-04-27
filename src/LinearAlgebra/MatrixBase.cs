namespace AiDotNet.LinearAlgebra;

public abstract class MatrixBase<T, T2>
{
    public int ColumnCount { get; private set; }
    public int RowCount { get; private set; }
    private readonly List<Vector<T>> _values;

    public MatrixBase(IEnumerable<Vector<T>> values)
    {
        _values = new List<Vector<T>>(values);
        ColumnCount = _values[0].Count;
        RowCount = _values.Count;
    }

    public virtual (T[,] mainMatrix, T[,] subMatrix, T[,] yTerms) BuildMatrix(T[] inputs, T[] outputs, T2 order)
    {
        throw new NotImplementedException();
    }

    public virtual T[] Solve(T[,] matrix1, T[,] matrix2, T[,] matrix3)
    {
        throw new NotImplementedException();
    }

    public T ValueAt(int x, int y)
    {
        return _values[x].ValueAt(y);
    }

    public float[,] BuildOnesMatrix(int rows, int columns)
    {
        var matrix = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = 1;
            }
        }

        return matrix;
    }

    public double[,] MultiplyMatrices(double[,] matrixA, double[,] matrixB)
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

        var matrix = new double[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = matrixA[i, j] * matrixB[i, j];
            }
        }

        return matrix;
    }

    public double[,] DotProductMatrices(double[,] matrixA, double[,] matrixB)
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