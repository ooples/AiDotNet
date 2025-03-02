global using System.Collections;

namespace AiDotNet.LinearAlgebra;

public class Vector<T> : VectorBase<T>, IEnumerable<T>
{
    public Vector(int length) : base(length)
    {
    }

    public Vector(IEnumerable<T> values) : base(values)
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

    public Vector<T> ElementwiseDivide(Vector<T> other)
    {
        if (this.Length != other.Length)
        {
            throw new ArgumentException("Vectors must have the same length for element-wise division.");
        }

        Vector<T> result = new Vector<T>(this.Length);
        for (int i = 0; i < this.Length; i++)
        {
            result[i] = NumOps.Divide(this[i], other[i]);
        }

        return result;
    }

    public T Variance()
    {
        T mean = Mean();
        return this.Select(x => NumOps.Square(NumOps.Subtract(x, mean))).Mean();
    }

    public Vector<T> Where(Func<T, bool> predicate)
    {
        return new Vector<T>(data.Where(predicate));
    }

    public Vector<TResult> Select<TResult>(Func<T, TResult> selector)
    {
        return new Vector<TResult>(data.Select(selector));
    }

    public new Vector<T> Copy()
    {
        return new Vector<T>([.. this]);
    }

    public override VectorBase<T> Zeros(int size)
    {
        return new Vector<T>(size);
    }

    public override VectorBase<T> Default(int size, T defaultValue)
    {
        return base.Default(size, defaultValue);
    }


    public new Vector<TResult> Transform<TResult>(Func<T, TResult> function)
    {
        return new Vector<TResult>(base.Transform(function).ToArray());
    }

    public new Vector<TResult> Transform<TResult>(Func<T, int, TResult> function)
    {
        return new Vector<TResult>(base.Transform(function).ToArray());
    }

    public override VectorBase<T> Ones(int size)
    {
        return new Vector<T>(Enumerable.Repeat(NumOps.One, size));
    }

    public new static Vector<T> Empty()
    {
        return new Vector<T>(0);
    }

    public new Vector<T> GetSubVector(int startIndex, int length)
    {
        if (startIndex < 0 || startIndex >= this.Length)
            throw new ArgumentOutOfRangeException(nameof(startIndex));
        if (length < 0 || startIndex + length > this.Length)
            throw new ArgumentOutOfRangeException(nameof(length));

        Vector<T> subVector = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            subVector[i] = this[startIndex + i];
        }

        return subVector;
    }

    public new Vector<T> SetValue(int index, T value)
    {
        if (index < 0 || index >= this.Length)
            throw new ArgumentOutOfRangeException(nameof(index));

        Vector<T> newVector = new([.. this])
        {
            [index] = value
        };

        return newVector;
    }

    public T Norm()
    {
        return NumOps.Sqrt(this.DotProduct(this));
    }

    public new Vector<T> Divide(T scalar)
    {
        return new Vector<T>(this.Select(x => NumOps.Divide(x, scalar)));
    }

    protected override VectorBase<T> CreateInstance(int size)
    {
        return new Vector<T>(size);
    }

    public byte[] Serialize()
    {
        return SerializationHelper<T>.SerializeVector(this);
    }

    public static Vector<T> Deserialize(byte[] data)
    {
        return SerializationHelper<T>.DeserializeVector(data);
    }

    public Vector<T> ElementwiseMultiply(Vector<T> other)
    {
        if (this.Length != other.Length)
            throw new ArgumentException("Vectors must have the same length for element-wise multiplication.");

        var result = new Vector<T>(this.Length);
        for (int i = 0; i < this.Length; i++)
        {
            result[i] = NumOps.Multiply(this[i], other[i]);
        }

        return result;
    }

    public static Vector<T> Range(int start, int count)
    {
        Vector<T> result = new Vector<T>(count);
        for (int i = 0; i < count; i++)
        {
            result[i] = NumOps.FromDouble(start + i);
        }

        return result;
    }

    public Vector<T> Subvector(int startIndex, int length)
    {
        if (startIndex < 0 || startIndex + length > Length)
            throw new ArgumentOutOfRangeException(nameof(startIndex));

        Vector<T> result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = this[startIndex + i];
        }

        return result;
    }

    public int BinarySearch(T value)
    {
        IComparer<T> comparer = Comparer<T>.Default;
        int low = 0;
        int high = Length - 1;

        while (low <= high)
        {
            int mid = low + ((high - low) >> 1);
            int comparison = comparer.Compare(this[mid], value);

            if (comparison == 0)
                return mid;
            else if (comparison < 0)
                low = mid + 1;
            else
                high = mid - 1;
        }

        return ~low;
    }

    public Vector<T> GetRange(int startIndex, int count)
    {
        if (startIndex < 0 || startIndex >= Length)
            throw new ArgumentOutOfRangeException(nameof(startIndex), "Start index is out of range.");
        if (count < 0 || startIndex + count > Length)
            throw new ArgumentOutOfRangeException(nameof(count), "Count is out of range.");

        T[] newData = new T[count];
        Array.Copy(data, startIndex, newData, 0, count);

        return new Vector<T>(newData);
    }

    public int IndexOfMax()
    {
        if (this.Length == 0)
            throw new InvalidOperationException("Vector is empty");

        int maxIndex = 0;
        T maxValue = this[0];
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 1; i < this.Length; i++)
        {
            if (NumOps.GreaterThan(this[i], maxValue))
            {
                maxValue = this[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public Matrix<T> OuterProduct(Vector<T> other)
    {
        int m = this.Length;
        int n = other.Length;
        var numOps = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(m, n);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = NumOps.Multiply(this[i], other[j]);
            }
        }

        return result;
    }

    public Vector<T> GetSegment(int startIndex, int length)
    {
        return new Vector<T>(this.Skip(startIndex).Take(length));
    }

    public static new Vector<T> CreateDefault(int length, T value)
    {
        Vector<T> vector = new(length);
        for (int i = 0; i < length; i++)
        {
            vector[i] = value;
        }

        return vector;
    }

    protected override VectorBase<T> CreateInstance(T[] data)
    {
        return new Vector<T>(data);
    }

    protected override VectorBase<TResult> CreateInstance<TResult>(int size)
    {
        return new Vector<TResult>(size);
    }

    public static Vector<T> CreateRandom(int size)
    {
        Vector<T> vector = new(size);
        Random random = new();
        for (int i = 0; i < size; i++)
        {
            vector[i] = NumOps.FromDouble(random.NextDouble());
        }

        return vector;
    }

    public static Vector<T> CreateRandom(int size, double min = -1.0, double max = 1.0)
    {
        if (min >= max)
            throw new ArgumentException("Minimum value must be less than maximum value");
        
        var random = new Random();
        var vector = new Vector<T>(size);
    
        for (int i = 0; i < size; i++)
        {
            // Generate random value between min and max
            double randomValue = random.NextDouble() * (max - min) + min;
            vector[i] = NumOps.FromDouble(randomValue);
        }
    
        return vector;
    }

    public static Vector<T> CreateStandardBasis(int size, int index)
    {
        var vector = new Vector<T>(size)
        {
            [index] = NumOps.One
        };

        return vector;
    }

    public Vector<T> Normalize()
    {
        T norm = this.Norm();
        if (NumOps.Equals(norm, NumOps.Zero))
        {
            throw new InvalidOperationException("Cannot normalize a zero vector.");
        }

        return this.Divide(norm);
    }

    public IEnumerable<int> NonZeroIndices()
    {
        var NumOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < Length; i++)
        {
            if (!NumOps.Equals(this[i], NumOps.Zero))
            {
                yield return i;
            }
        }
    }

    public Matrix<T> Transpose()
    {
        var result = new Matrix<T>(1, this.Length);
        for (int i = 0; i < this.Length; i++)
        {
            result[0, i] = this[i];
        }

        return result;
    }

    public Matrix<T> AppendAsMatrix(T value)
    {
        var result = new Matrix<T>(this.Length, 2);
        for (int i = 0; i < this.Length; i++)
        {
            result[i, 0] = this[i];
            result[i, 1] = value;
        }

        return result;
    }

    public Vector<T> GetElements(IEnumerable<int> indices)
    {
        var indexList = indices.ToList();
        var newVector = new T[indexList.Count];
        for (int i = 0; i < indexList.Count; i++)
        {
            newVector[i] = this[indexList[i]];
        }

        return new Vector<T>(newVector);
    }

    public Vector<T> RemoveAt(int index)
    {
        if (index < 0 || index >= Length)
            throw new ArgumentOutOfRangeException(nameof(index));

        var newData = new T[Length - 1];
        Array.Copy(data, 0, newData, 0, index);
        Array.Copy(data, index + 1, newData, index, Length - index - 1);

        return new Vector<T>(newData);
    }

    public int NonZeroCount()
    {
        return NonZeroIndices().Count();
    }

    public void Fill(T value)
    {
        for (int i = 0; i < Length; i++)
        {
            this[i] = value;
        }
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

    public static Vector<T> Concatenate(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0)
            return new Vector<T>(0);
            
        Vector<T> result = vectors[0];
        for (int i = 1; i < vectors.Count; i++)
        {
            result = Vector<T>.Concatenate(result, vectors[i]);
        }

        return result;
    }

    public new Vector<T> Add(VectorBase<T> other)
    {
        return new Vector<T>(base.Add(other).ToArray());
    }

    public new Vector<T> Subtract(VectorBase<T> other)
    {
        return new Vector<T>(base.Subtract(other).ToArray());
    }

    public new Vector<T> Multiply(T scalar)
    {
        return new Vector<T>(base.Multiply(scalar).ToArray());
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