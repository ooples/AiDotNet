namespace AiDotNet.LinearAlgebra;

public class ColumnVector<T> : IVector<T> where T : class
{
    public T[] Value { get; private set; }

    public ColumnVector(int length)
    {
        Value = new T[length];
    }

    public ColumnVector<T> BuildVector(T[] values)
    {
        Value = values;

        return this;
    }
}