global using System.Collections;

namespace AiDotNet.LinearAlgebra;

public class Vector<T> : VectorBase<T>, IEnumerable<T>
{
    public Vector(int length, INumericOperations<T>? numericOperations = null)
        : base(length, numericOperations ?? MathHelper.GetNumericOperations<T>())
    {
    }

    public Vector(IEnumerable<T> values, INumericOperations<T>? numericOperations = null)
        : base(values, numericOperations ?? MathHelper.GetNumericOperations<T>())
    {
    }

    public IEnumerator<T> GetEnumerator()
    {
        return ((IEnumerable<T>)data).GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    public T Variance()
    {
        T mean = Mean();
        return this.Select(x => ops.Square(ops.Subtract(x, mean))).Mean();
    }

    public Vector<T> Where(Func<T, bool> predicate)
    {
        return new Vector<T>(data.Where(predicate), ops);
    }

    public Vector<TResult> Select<TResult>(Func<T, TResult> selector)
    {
        return new Vector<TResult>(data.Select(selector), MathHelper.GetNumericOperations<TResult>());
    }

    public new Vector<T> Copy()
    {
        return new Vector<T>([.. this], ops);
    }

    public override VectorBase<T> Zeros(int size)
    {
        return new Vector<T>(size, ops);
    }

    public override VectorBase<T> Default(int size, T defaultValue)
    {
        return base.Default(size, defaultValue);
    }


    public new Vector<TResult> Transform<TResult>(Func<T, TResult> function)
    {
        return new Vector<TResult>(base.Transform(function).ToArray(), MathHelper.GetNumericOperations<TResult>());
    }

    public new Vector<TResult> Transform<TResult>(Func<T, int, TResult> function)
    {
        return new Vector<TResult>(base.Transform(function).ToArray(), MathHelper.GetNumericOperations<TResult>());
    }

    public override VectorBase<T> Ones(int size)
    {
        return new Vector<T>(Enumerable.Repeat(ops.One, size), ops);
    }

    public new static Vector<T> Empty()
    {
        return new Vector<T>(0);
    }

    public static Vector<T> CreateDefault(int length, T value)
    {
        var vector = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            vector[i] = value;
        }

        return vector;
    }

    public static Vector<T> CreateRandom(int size)
    {
        Vector<T> vector = new(size);
        var ops = MathHelper.GetNumericOperations<T>();
        Random random = new();
        for (int i = 0; i < size; i++)
        {
            vector[i] = ops.FromDouble(random.NextDouble());
        }

        return vector;
    }

    public static Vector<T> CreateStandardBasis(int size, int index)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var vector = new Vector<T>(size, ops)
        {
            [index] = ops.One
        };

        return vector;
    }

    public Vector<T> Normalize()
    {
        var NumOps = MathHelper.GetNumericOperations<T>();
        T norm = this.Norm();
        if (NumOps.Equals(norm, NumOps.Zero))
        {
            throw new InvalidOperationException("Cannot normalize a zero vector.");
        }

        return this.Divide(norm);
    }

    public static Vector<T> Concatenate(params Vector<T>[] vectors)
    {
        int totalSize = vectors.Sum(v => v.Length);
        Vector<T> result = new(totalSize);

        int offset = 0;
        foreach (var vector in vectors)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                result[offset + i] = vector[i];
            }
            offset += vector.Length;
        }

        return result;
    }

    protected override VectorBase<T> CreateInstance(int size)
    {
        return new Vector<T>(size, ops);
    }

    protected override VectorBase<T> CreateInstance(T[] data)
    {
        return new Vector<T>(data, ops);
    }

    protected override VectorBase<TResult> CreateInstance<TResult>(int size)
    {
        return new Vector<TResult>(size, MathHelper.GetNumericOperations<TResult>());
    }

    public new Vector<T> Add(VectorBase<T> other)
    {
        return new Vector<T>(base.Add(other).ToArray(), ops);
    }

    public new Vector<T> Subtract(VectorBase<T> other)
    {
        return new Vector<T>(base.Subtract(other).ToArray(), ops);
    }

    public new Vector<T> Multiply(T scalar)
    {
        return new Vector<T>(base.Multiply(scalar).ToArray(), ops);
    }

    public new Vector<T> Divide(T scalar)
    {
        return new Vector<T>(base.Divide(scalar).ToArray(), ops);
    }

    public static Vector<T> operator +(Vector<T> left, Vector<T> right)
    {
        return left.Add(right);
    }

    public static Vector<T> operator +(Vector<T> vector, T scalar)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        return vector.Add(scalar);
    }

    public static Vector<T> operator -(Vector<T> vector, T scalar)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        return vector.Subtract(scalar);
    }

    public static Vector<T> operator -(Vector<T> left, Vector<T> right)
    {
        return left.Subtract(right);
    }

    public static Vector<T> operator *(Vector<T> vector, T scalar)
    {
        return vector.Multiply(scalar);
    }

    public static Vector<T> operator *(T scalar, Vector<T> vector)
    {
        return vector * scalar;
    }

    public static Vector<T> operator /(Vector<T> vector, T scalar)
    {
        return vector.Divide(scalar);
    }

    public static implicit operator T[](Vector<T> vector)
    {
        return vector.ToArray();
    }

    public static Vector<T> FromArray(T[] array)
    {
        return new Vector<T>(array);
    }

    public static Vector<T> FromEnumerable(IEnumerable<T> enumerable)
    {
        if (enumerable == null)
            throw new ArgumentNullException(nameof(enumerable));
        if (enumerable is T[] arr)
            return FromArray(arr);
        if (enumerable is List<T> list)
            return FromList(list);
        var tempList = enumerable.ToList();
        return FromList(tempList);
    }

    public static Vector<T> FromList(List<T> list)
    {
        if (list == null)
            throw new ArgumentNullException(nameof(list));
        var vector = new Vector<T>(list.Count);
        list.CopyTo(vector.data);
        return vector;
    }
}