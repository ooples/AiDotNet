namespace AiDotNet.LinearAlgebra;

public abstract class VectorBase<T>
{
    public T[] Values { get; private set; }

    public int Count { get; private set; }

    public VectorBase(IEnumerable<T> values)
    {
        Values = values.ToArray();
        Count = Values.Length;
    }

    public VectorBase(int count)
    {
        Values = new T[count];
        Count = count;
    }

    public T this[int i]
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
}