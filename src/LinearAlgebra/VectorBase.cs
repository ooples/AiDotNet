namespace AiDotNet.LinearAlgebra;

public abstract class VectorBase<T>
{
    public int Count { get; private set; }
    private readonly List<T> _values;

    public VectorBase(IEnumerable<T> values)
    {
        _values = new List<T>(values);
        Count = _values.Count;
    }

    public T ValueAt(int index)
    {
        return _values[index];
    }
}