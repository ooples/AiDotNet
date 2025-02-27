namespace AiDotNet.Interfaces;

public interface IVector<T> where T : class
{
    public ColumnVector<T> BuildVector(T[] values);
}