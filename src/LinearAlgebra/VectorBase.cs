namespace AiDotNet.LinearAlgebra;

public abstract class VectorBase<T>
{
    protected readonly T[] data;
    protected readonly INumericOperations<T> ops;

    protected VectorBase(int length, INumericOperations<T> operations)
    {
        if (length <= 0)
            throw new ArgumentException("Length must be positive", nameof(length));
    
        data = new T[length];
        ops = operations;
    }

    protected VectorBase(IEnumerable<T> values, INumericOperations<T> operations)
    {
        data = [.. values];
        ops = operations;
    }

    public int Length => data.Length;

    public virtual T this[int index]
    {
        get
        {
            ValidateIndex(index);
            return data[index];
        }
        set
        {
            ValidateIndex(index);
            data[index] = value;
        }
    }

    protected void ValidateIndex(int index)
    {
        if (index < 0 || index >= Length)
            throw new ArgumentOutOfRangeException(nameof(index));
    }

    public virtual T[] ToArray()
    {
        return (T[])data.Clone();
    }

    public virtual VectorBase<T> Copy()
    {
        return CreateInstance(data);
    }

    public static VectorBase<T> Empty()
    {
        return new Vector<T>(0);
    }

    public virtual VectorBase<T> Zeros(int size)
    {
        var result = CreateInstance(size);
        for (int i = 0; i < size; i++)
        {
            result[i] = ops.Zero;
        }

        return result;
    }

    public VectorBase<T> GetSubVector(int startIndex, int length)
    {
        if (startIndex < 0 || startIndex >= this.Length)
            throw new ArgumentOutOfRangeException(nameof(startIndex));
        if (length < 0 || startIndex + length > this.Length)
            throw new ArgumentOutOfRangeException(nameof(length));

        VectorBase<T> subVector = CreateInstance(length);
        for (int i = 0; i < length; i++)
        {
            subVector[i] = this[startIndex + i];
        }

        return subVector;
    }

    public VectorBase<T> SetValue(int index, T value)
    {
        if (index < 0 || index >= this.Length)
            throw new ArgumentOutOfRangeException(nameof(index));

        VectorBase<T> newVector = this.Copy();
        newVector[index] = value;

        return newVector;
    }

    public static VectorBase<T> CreateDefault(int length, T value)
    {
        Vector<T> vector = new(length);
        for (int i = 0; i < length; i++)
        {
            vector[i] = value;
        }

        return vector;
    }

    public virtual T Mean()
    {
        if (Length == 0) throw new InvalidOperationException("Cannot calculate mean of an empty vector.");
        return ops.Divide(this.Sum(), ops.FromDouble(Length));
    }

    public virtual T Sum()
    {
        T sum = ops.Zero;
        for (int i = 0; i < Length; i++)
        {
            sum = ops.Add(sum, data[i]);
        }
        return sum;
    }

    public virtual T L2Norm()
    {
        T sum = ops.Zero;
        for (int i = 0; i < Length; i++)
        {
            T value = data[i];
            sum = ops.Add(sum, ops.Multiply(value, value));
        }
        return ops.Sqrt(sum);
    }

    public virtual VectorBase<TResult> Transform<TResult>(Func<T, TResult> function)
    {
        var result = CreateInstance<TResult>(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = function(data[i]);
        }

        return result;
    }

    public virtual VectorBase<TResult> Transform<TResult>(Func<T, int, TResult> function)
    {
        var result = CreateInstance<TResult>(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = function(data[i], i);
        }

        return result;
    }

    public virtual VectorBase<T> Ones(int size)
    {
        var result = CreateInstance(size);
        for (int i = 0; i < size; i++)
        {
            result[i] = ops.One;
        }

        return result;
    }

    public virtual VectorBase<T> Default(int size, T defaultValue)
    {
        var result = CreateInstance(size);
        for (int i = 0; i < size; i++)
        {
            result[i] = defaultValue;
        }

        return result;
    }

    protected abstract VectorBase<T> CreateInstance(int size);
    protected abstract VectorBase<T> CreateInstance(T[] data);
    protected abstract VectorBase<TResult> CreateInstance<TResult>(int size);

    public virtual VectorBase<T> Add(VectorBase<T> other)
    {
        if (Length != other.Length)
            throw new ArgumentException("Vectors must have the same length");

        var result = CreateInstance(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = ops.Add(this[i], other[i]);
        }
        return result;
    }

    public virtual VectorBase<T> Subtract(VectorBase<T> other)
    {
        if (Length != other.Length)
            throw new ArgumentException("Vectors must have the same length");

        var result = CreateInstance(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = ops.Subtract(this[i], other[i]);
        }
        return result;
    }

    public virtual VectorBase<T> Multiply(T scalar)
    {
        var result = CreateInstance(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = ops.Multiply(this[i], scalar);
        }
        return result;
    }

    public virtual VectorBase<T> Divide(T scalar)
    {
        var result = CreateInstance(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = ops.Divide(this[i], scalar);
        }
        return result;
    }

    public override string ToString()
    {
        return $"[{string.Join(", ", data)}]";
    }
}