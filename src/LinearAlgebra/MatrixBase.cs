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

}