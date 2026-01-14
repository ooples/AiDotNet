using System.Buffers;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// An abstract base class that represents a mathematical vector with elements of type T.
/// </summary>
/// <typeparam name="T">The type of elements in the vector (typically numeric types like double, float, etc.)</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A vector is like a list of numbers that can be used in mathematical operations.
/// Think of it as a row or column of values that you can add, subtract, multiply, etc. Vectors are fundamental
/// building blocks in machine learning for representing data points and model parameters.</para>
/// </remarks>
public abstract class VectorBase<T>
{
    /// <summary>
    /// The internal memory that stores the vector's elements.
    /// </summary>
    /// <remarks>
    /// <para><b>Migration Note:</b> This field replaces the previous T[] _data field.
    /// Memory&lt;T&gt; provides zero-copy slicing, better Span&lt;T&gt; interop, and integration with memory pooling.</para>
    /// </remarks>
    protected readonly Memory<T> _memory;

    /// <summary>
    /// Gets the underlying array from the memory backing store.
    /// </summary>
    /// <remarks>
    /// <para>This property provides backward compatibility for code that accessed the old T[] _data field.
    /// When the memory is backed by an array (the common case), this returns the array directly.
    /// Otherwise, it creates a copy.</para>
    /// </remarks>
    protected T[] _data => MemoryMarshal.TryGetArray<T>(_memory, out var segment) && segment.Offset == 0 && segment.Count == segment.Array!.Length
        ? segment.Array
        : _memory.ToArray();

    /// <summary>
    /// Provides operations for numeric types (addition, subtraction, etc.).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This helper allows the vector to work with different number types
    /// (like int, double, float) by providing a common way to perform math operations on them.</para>
    /// </remarks>
    protected static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Creates a new vector with the specified length.
    /// </summary>
    /// <param name="length">The number of elements in the vector.</param>
    /// <exception cref="ArgumentException">Thrown when length is not positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an empty vector with a specific size.
    /// For example, creating a vector with length 3 gives you a vector with 3 elements,
    /// but all elements start with the default value (usually 0).</para>
    /// </remarks>
    protected VectorBase(int length)
    {
        if (length < 0)
            throw new ArgumentException("Length must be non-negative", nameof(length));

        _memory = new T[length];
    }

    /// <summary>
    /// Creates a new vector from an existing Memory&lt;T&gt; backing store.
    /// </summary>
    /// <param name="memory">The memory to use as the vector's backing store.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector that uses existing memory without copying.
    /// This is useful for zero-copy operations and integration with memory pooling.</para>
    /// </remarks>
    protected VectorBase(Memory<T> memory)
    {
        _memory = memory;
    }

    /// <summary>
    /// Creates a new vector from a collection of values.
    /// </summary>
    /// <param name="values">The values to initialize the vector with.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector using existing values.
    /// For example, you can create a vector from a list or array of numbers.</para>
    /// </remarks>
    protected VectorBase(IEnumerable<T> values)
    {
        _memory = values.ToArray();
    }

    /// <summary>
    /// Gets the number of elements in the vector.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many numbers are in your vector.
    /// For example, the vector [1,2,3] has a Length of 3.</para>
    /// </remarks>
    public int Length => _memory.Length;

    /// <summary>
    /// Gets the underlying array that stores the vector's elements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property exists for performance-sensitive internal code paths (e.g., kernels and SIMD routines) that require direct array access.
    /// External consumers should use ToArray() to get a copy of the data.
    /// </para>
    /// <para><b>For Beginners:</b> This exposes the raw list of numbers inside the vector for internal use only.</para>
    /// </remarks>
    internal T[] Data => _data;

    /// <summary>
    /// Gets a value indicating whether the vector contains no elements.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you if your vector is empty (has no elements).
    /// It returns true if the vector has no elements, and false if it has at least one element.</para>
    /// </remarks>
    public bool IsEmpty => Length == 0;

    /// <summary>
    /// Gets or sets the element at the specified index in the vector.
    /// </summary>
    /// <param name="index">The zero-based index of the element to get or set.</param>
    /// <returns>The element at the specified index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is outside the valid range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This allows you to access or change individual elements in the vector.
    /// For example, if your vector is [10,20,30], then vector[1] would give you 20 (the second element,
    /// since counting starts at 0).</para>
    /// </remarks>
    public virtual T this[int index]
    {
        get
        {
            ValidateIndex(index);
            return _data[index];
        }
        set
        {
            ValidateIndex(index);
            _data[index] = value;
        }
    }

    /// <summary>
    /// Checks if the given index is valid for this vector.
    /// </summary>
    /// <param name="index">The index to validate.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the index is outside the valid range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a helper method that makes sure you're not trying to access
    /// a position in the vector that doesn't exist. For example, trying to access the 5th element
    /// of a 3-element vector would cause an error.</para>
    /// </remarks>
    protected void ValidateIndex(int index)
    {
        if (index < 0 || index >= Length)
            throw new ArgumentOutOfRangeException(nameof(index));
    }

    /// <summary>
    /// Creates a new array containing a copy of the vector's elements.
    /// </summary>
    /// <returns>A new array containing the vector's elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a regular array from your vector.
    /// The array will contain all the same values as your vector, but it will be
    /// a separate copy, so changes to the array won't affect the original vector.</para>
    /// </remarks>
    public virtual T[] ToArray()
    {
        return _memory.ToArray();
    }

    /// <summary>
    /// Gets a read-only span view of the vector's data without copying.
    /// </summary>
    /// <returns>A read-only span over the vector's elements.</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-003 - Zero-Copy Operations</b></para>
    /// <para>
    /// This method provides direct memory access to the vector's internal storage
    /// without creating a copy. It's used by GPU operations to eliminate the overhead
    /// of array allocation and copying (2-5x speedup for large vectors).
    /// </para>
    /// <para><b>For Beginners:</b> This gives you a window into the vector's data
    /// without making a copy. Think of it like looking at the original data through
    /// a glass window instead of making a photocopy.</para>
    /// </remarks>
    public ReadOnlySpan<T> AsSpan()
    {
        return _memory.Span;
    }

    /// <summary>
    /// Gets a writable span view of the vector's data without copying.
    /// </summary>
    /// <returns>A writable span over the vector's elements.</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-003 - Zero-Copy Operations</b></para>
    /// <para>
    /// This method provides direct writable access to the vector's internal storage.
    /// Used by GPU operations to write results directly without intermediate copies.
    /// </para>
    /// <para><b>Warning:</b> Use with caution - modifications affect the vector directly.</para>
    /// </remarks>
    internal Span<T> AsWritableSpan()
    {
        return _memory.Span;
    }

    /// <summary>
    /// Gets a read-only memory view of the vector's data without copying.
    /// </summary>
    /// <returns>A read-only memory over the vector's elements.</returns>
    /// <remarks>
    /// <para><b>Issue #693: Memory&lt;T&gt; Migration</b></para>
    /// <para>
    /// This method provides access to the underlying Memory&lt;T&gt; backing store.
    /// Unlike Span&lt;T&gt;, Memory&lt;T&gt; can be stored in fields and passed across async boundaries.
    /// </para>
    /// <para><b>For Beginners:</b> This gives you access to the vector's data in a format
    /// that can be stored and passed around, unlike Span which must be used immediately.</para>
    /// </remarks>
    public ReadOnlyMemory<T> AsMemory()
    {
        return _memory;
    }

    /// <summary>
    /// Gets a writable memory view of the vector's data without copying.
    /// </summary>
    /// <returns>A writable memory over the vector's elements.</returns>
    /// <remarks>
    /// <para><b>Issue #693: Memory&lt;T&gt; Migration</b></para>
    /// <para>
    /// This method provides direct writable access to the underlying Memory&lt;T&gt; backing store.
    /// </para>
    /// <para><b>Warning:</b> Use with caution - modifications affect the vector directly.</para>
    /// </remarks>
    internal Memory<T> AsWritableMemory()
    {
        return _memory;
    }

    /// <summary>
    /// Creates a new vector that is a copy of this vector.
    /// </summary>
    /// <returns>A new vector containing the same elements as this vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a complete duplicate of your vector.
    /// The new vector will have the same values, but changes to one won't affect the other.</para>
    /// </remarks>
    public virtual VectorBase<T> Clone()
    {
        return CreateInstance(_data);
    }

    /// <summary>
    /// Creates a new empty vector with zero elements.
    /// </summary>
    /// <returns>A new empty vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector with no elements.
    /// It's like an empty list or array.</para>
    /// </remarks>
    public static VectorBase<T> Empty()
    {
        return new Vector<T>(0);
    }

    /// <summary>
    /// Creates a new vector of the specified size with all elements set to zero.
    /// </summary>
    /// <param name="size">The size of the vector to create.</param>
    /// <returns>A new vector with all elements set to zero.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector filled with zeros.
    /// For example, Zeros(3) would create the vector [0,0,0].</para>
    /// </remarks>
    public virtual VectorBase<T> Zeros(int size)
    {
        var result = CreateInstance(size);
        // Use vectorized Fill for SIMD acceleration
        _numOps.Fill(result.AsWritableSpan(), _numOps.Zero);

        return result;
    }

    /// <summary>
    /// Creates a new vector containing a portion of this vector.
    /// </summary>
    /// <param name="startIndex">The zero-based starting index of the subvector.</param>
    /// <param name="length">The number of elements in the subvector.</param>
    /// <returns>A new vector containing the specified portion of this vector.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when startIndex is negative or greater than or equal to the vector's length,
    /// or when length is negative or would extend beyond the end of the vector.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This extracts a smaller part of your vector.
    /// For example, if your vector is [10,20,30,40,50], then GetSubVector(1,3) would
    /// give you [20,30,40] - starting at position 1 (the second element) and taking 3 elements.</para>
    /// </remarks>
    public VectorBase<T> GetSubVector(int startIndex, int length)
    {
        if (startIndex < 0 || startIndex > this.Length)
            throw new ArgumentOutOfRangeException(nameof(startIndex), "Start index must be between 0 and the vector length (inclusive).");
        if (length < 0 || startIndex + length > this.Length)
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be non-negative and the range must not exceed the vector bounds.");

        VectorBase<T> subVector = CreateInstance(length);
        // Use vectorized Copy for efficient memory transfer
        _numOps.Copy(new ReadOnlySpan<T>(_data, startIndex, length), subVector.AsWritableSpan());

        return subVector;
    }

    /// <summary>
    /// Finds the index of the first occurrence of the specified value in the vector.
    /// </summary>
    /// <param name="item">The value to locate in the vector.</param>
    /// <returns>The zero-based index of the first occurrence of the value, or -1 if not found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This searches for a specific value in your vector and tells you
    /// its position. For example, in the vector [5,10,15,10], IndexOf(10) would return 1
    /// because 10 first appears at position 1 (the second element). If the value isn't found,
    /// it returns -1.</para>
    /// </remarks>
    public virtual int IndexOf(T item)
    {
        for (int i = 0; i < Length; i++)
        {
            if (_numOps.Equals(this[i], item))
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>
    /// Creates a new vector that is a copy of this vector with one element changed.
    /// </summary>
    /// <param name="index">The zero-based index of the element to change.</param>
    /// <param name="value">The new value for the element.</param>
    /// <returns>A new vector with the specified element changed.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is outside the valid range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a copy of your vector but changes one specific value.
    /// For example, if your vector is [1,2,3] and you call SetValue(1, 9), you'll get a new vector [1,9,3]
    /// where the element at position 1 (the second element) has been changed to 9.</para>
    /// </remarks>
    public VectorBase<T> SetValue(int index, T value)
    {
        if (index < 0 || index >= this.Length)
            throw new ArgumentOutOfRangeException(nameof(index));

        VectorBase<T> newVector = this.Clone();
        newVector[index] = value;

        return newVector;
    }

    /// <summary>
    /// Creates a new vector with all elements set to the specified value.
    /// </summary>
    /// <param name="length">The length of the vector to create.</param>
    /// <param name="value">The value to assign to all elements.</param>
    /// <returns>A new vector with all elements set to the specified value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector where every position contains the same value.
    /// For example, CreateDefault(3, 5) would create the vector [5,5,5] - a vector of length 3
    /// where each element is 5.</para>
    /// </remarks>
    public static Vector<T> CreateDefault(int length, T value)
    {
        Vector<T> vector = new(length);
        // Use vectorized Fill for SIMD acceleration
        _numOps.Fill(vector.AsWritableSpan(), value);

        return vector;
    }

    /// <summary>
    /// Calculates the arithmetic mean (average) of all elements in the vector.
    /// </summary>
    /// <returns>The mean value of the vector's elements.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the vector is empty.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates the average of all numbers in your vector.
    /// For example, the mean of [2,4,6] is (2+4+6)/3 = 4. This is a common operation in data analysis
    /// to find the "center" of your data.</para>
    /// </remarks>
    public virtual T Mean()
    {
        if (Length == 0) throw new InvalidOperationException("Cannot calculate mean of an empty vector.");
        return _numOps.Divide(this.Sum(), _numOps.FromDouble(Length));
    }

    /// <summary>
    /// Calculates the sum of all elements in the vector.
    /// </summary>
    /// <returns>The sum of all elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adds up all the numbers in your vector.
    /// For example, the sum of [1,2,3] is 1+2+3 = 6. Summing is a basic operation
    /// used in many statistical calculations.</para>
    /// <para><b>Performance:</b> This method uses SIMD-accelerated operations for float/double types
    /// via TensorPrimitives, providing 8-12x speedup for large vectors.</para>
    /// </remarks>
    public virtual T Sum()
    {
        return _numOps.Sum(new ReadOnlySpan<T>(_data));
    }

    /// <summary>
    /// Calculates the L2 norm (Euclidean norm) of the vector.
    /// </summary>
    /// <returns>The L2 norm of the vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The L2 norm is the "length" or "magnitude" of a vector in a mathematical sense.
    /// It's calculated by taking the square root of the sum of squares of all elements.
    /// For example, the L2 norm of [3,4] is sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5.
    /// This is commonly used in machine learning to measure the "size" of vectors or the distance between points.</para>
    /// </remarks>
    public virtual T L2Norm()
    {
        // Use vectorized Dot product (sum of squares) then Sqrt - 10-15x faster with AVX2
        T sumOfSquares = _numOps.Dot(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(_data));

        return _numOps.Sqrt(sumOfSquares);
    }

    /// <summary>
    /// Creates a new vector by applying a function to each element of this vector.
    /// </summary>
    /// <typeparam name="TResult">The type of elements in the resulting vector.</typeparam>
    /// <param name="function">The function to apply to each element.</param>
    /// <returns>A new vector containing the transformed elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This lets you change every element in your vector using a formula.
    /// For example, if you have a vector [1,2,3] and you apply a function that doubles each number,
    /// you'll get [2,4,6]. This is useful for operations like scaling data or applying mathematical
    /// transformations to your values.</para>
    /// </remarks>
    public virtual VectorBase<TResult> Transform<TResult>(Func<T, TResult> function)
    {
        var result = CreateInstance<TResult>(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = function(_data[i]);
        }

        return result;
    }

    /// <summary>
    /// Creates a new vector by applying a function to each element and its index in this vector.
    /// </summary>
    /// <typeparam name="TResult">The type of elements in the resulting vector.</typeparam>
    /// <param name="function">The function to apply to each element and its index.</param>
    /// <returns>A new vector containing the transformed elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Similar to the other Transform method, but this one also gives you
    /// the position (index) of each element as you transform it. This is useful when the transformation
    /// depends on where the element is located in the vector. For example, you might want to multiply
    /// each element by its position: [1,2,3] would become [1*0, 2*1, 3*2] = [0,2,6].</para>
    /// </remarks>
    public virtual VectorBase<TResult> Transform<TResult>(Func<T, int, TResult> function)
    {
        var result = CreateInstance<TResult>(Length);
        for (int i = 0; i < Length; i++)
        {
            result[i] = function(_data[i], i);
        }

        return result;
    }

    /// <summary>
    /// Creates a new vector of the specified size with all elements set to one.
    /// </summary>
    /// <param name="size">The size of the vector to create.</param>
    /// <returns>A new vector with all elements set to one.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector filled with ones.
    /// For example, Ones(3) would create the vector [1,1,1]. Vectors of ones are often used
    /// in machine learning algorithms, particularly when working with bias terms in models.</para>
    /// </remarks>
    public virtual VectorBase<T> Ones(int size)
    {
        var result = CreateInstance(size);
        // Use vectorized Fill for SIMD acceleration
        _numOps.Fill(result.AsWritableSpan(), _numOps.One);

        return result;
    }

    /// <summary>
    /// Creates a new vector of the specified size with all elements set to the default value.
    /// </summary>
    /// <param name="size">The size of the vector to create.</param>
    /// <param name="defaultValue">The value to assign to all elements.</param>
    /// <returns>A new vector with all elements set to the default value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector where every position contains the same value.
    /// For example, Default(3, 5) would create the vector [5,5,5] - a vector of length 3
    /// where each element is 5. This is useful when you need a starting point for algorithms
    /// that require vectors with specific initial values.</para>
    /// </remarks>
    public virtual VectorBase<T> Default(int size, T defaultValue)
    {
        var result = CreateInstance(size);
        // Use vectorized Fill for SIMD acceleration
        _numOps.Fill(result.AsWritableSpan(), defaultValue);

        return result;
    }

    /// <summary>
    /// Creates a new empty vector of the specified size.
    /// </summary>
    /// <param name="size">The size of the vector to create.</param>
    /// <returns>A new empty vector of the specified size.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is an internal method that creates a new vector of a specific size.
    /// Derived classes must implement this to create the correct type of vector.</para>
    /// </remarks>
    protected abstract VectorBase<T> CreateInstance(int size);

    /// <summary>
    /// Creates a new vector with the specified data.
    /// </summary>
    /// <param name="data">The data to initialize the vector with.</param>
    /// <returns>A new vector containing the specified data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is an internal method that creates a new vector from existing data.
    /// Derived classes must implement this to create the correct type of vector.</para>
    /// </remarks>
    protected abstract VectorBase<T> CreateInstance(T[] data);

    /// <summary>
    /// Creates a new vector of a different type with the specified size.
    /// </summary>
    /// <typeparam name="TResult">The type of elements in the resulting vector.</typeparam>
    /// <param name="size">The size of the vector to create.</param>
    /// <returns>A new empty vector of the specified type and size.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is an internal method that creates a new vector with a different
    /// element type. This is used when transforming vectors from one type to another, such as
    /// converting a vector of integers to a vector of doubles.</para>
    /// </remarks>
    protected abstract VectorBase<TResult> CreateInstance<TResult>(int size);

    /// <summary>
    /// Adds another vector to this vector, element by element.
    /// </summary>
    /// <param name="other">The vector to add to this vector.</param>
    /// <returns>A new vector containing the sum of the two vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adds two vectors together by adding their corresponding elements.
    /// For example, [1,2,3] + [4,5,6] = [5,7,9]. Vector addition is a fundamental operation in
    /// linear algebra and is used extensively in machine learning algorithms.</para>
    /// <para><b>Performance:</b> This method uses SIMD-accelerated operations for float/double types
    /// via TensorPrimitives, providing 5-15x speedup for large vectors.</para>
    /// </remarks>
    public virtual VectorBase<T> Add(VectorBase<T> other)
    {
        if (Length != other.Length)
            throw new ArgumentException("Vectors must have the same length");

        var result = CreateInstance(Length);
        _numOps.Add(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(other._data), result.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// Adds another vector to this vector in-place, modifying this vector.
    /// </summary>
    /// <param name="other">The vector to add to this vector.</param>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adds another vector to this vector, modifying the current vector
    /// instead of creating a new one. This is more memory-efficient when you don't need to preserve
    /// the original vector.</para>
    /// <para><b>Performance:</b> This method performs zero allocations and uses SIMD-accelerated operations,
    /// making it significantly faster than the regular Add method for repeated operations.</para>
    /// </remarks>
    public virtual void AddInPlace(VectorBase<T> other)
    {
        if (Length != other.Length)
            throw new ArgumentException("Vectors must have the same length");

        _numOps.Add(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(other._data), new Span<T>(_data));
    }

    /// <summary>
    /// Adds another vector to this vector, storing the result in the destination span.
    /// </summary>
    /// <param name="other">The vector to add to this vector.</param>
    /// <param name="destination">The span to store the result in.</param>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths or destination is too small.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adds two vectors and stores the result in a pre-allocated buffer,
    /// avoiding any memory allocation. This is the fastest way to add vectors when you can reuse buffers.</para>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated addition - matches TensorPrimitives performance.</para>
    /// </remarks>
    public virtual void Add(VectorBase<T> other, Span<T> destination)
    {
        if (Length != other.Length)
            throw new ArgumentException("Vectors must have the same length");
        if (destination.Length < Length)
            throw new ArgumentException("Destination span is too small", nameof(destination));

        _numOps.Add(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(other._data), destination);
    }

    /// <summary>
    /// Subtracts another vector from this vector, element by element.
    /// </summary>
    /// <param name="other">The vector to subtract from this vector.</param>
    /// <returns>A new vector containing the difference of the two vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This subtracts one vector from another by subtracting their corresponding elements.
    /// For example, [5,7,9] - [1,2,3] = [4,5,6]. Vector subtraction is commonly used in machine learning
    /// to calculate differences between data points or to measure how far predictions are from actual values.</para>
    /// <para><b>Performance:</b> This method uses SIMD-accelerated operations for float/double types
    /// via TensorPrimitives, providing 5-15x speedup for large vectors.</para>
    /// </remarks>
    public virtual VectorBase<T> Subtract(VectorBase<T> other)
    {
        if (Length != other.Length)
            throw new ArgumentException("Vectors must have the same length");

        var result = CreateInstance(Length);
        _numOps.Subtract(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(other._data), result.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// Subtracts another vector from this vector in-place, modifying this vector.
    /// </summary>
    /// <param name="other">The vector to subtract from this vector.</param>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This subtracts another vector from this vector, modifying the current vector
    /// instead of creating a new one. This is more memory-efficient when you don't need to preserve
    /// the original vector.</para>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated subtraction.</para>
    /// </remarks>
    public virtual void SubtractInPlace(VectorBase<T> other)
    {
        if (Length != other.Length)
            throw new ArgumentException("Vectors must have the same length");

        _numOps.Subtract(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(other._data), new Span<T>(_data));
    }

    /// <summary>
    /// Subtracts another vector from this vector, storing the result in the destination span.
    /// </summary>
    /// <param name="other">The vector to subtract from this vector.</param>
    /// <param name="destination">The span to store the result in.</param>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths or destination is too small.</exception>
    /// <remarks>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated subtraction.</para>
    /// </remarks>
    public virtual void Subtract(VectorBase<T> other, Span<T> destination)
    {
        if (Length != other.Length)
            throw new ArgumentException("Vectors must have the same length");
        if (destination.Length < Length)
            throw new ArgumentException("Destination span is too small", nameof(destination));

        _numOps.Subtract(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(other._data), destination);
    }

    /// <summary>
    /// Multiplies each element of this vector by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <returns>A new vector with each element multiplied by the scalar.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This multiplies every element in your vector by the same number (scalar).
    /// For example, if you multiply [1,2,3] by 2, you get [2,4,6]. Scalar multiplication is used to
    /// scale vectors, which is useful in many AI algorithms like gradient descent where you need to
    /// adjust values by a learning rate.</para>
    /// <para><b>Performance:</b> Uses SIMD-accelerated operations (5-15x faster with AVX2).</para>
    /// </remarks>
    public virtual VectorBase<T> Multiply(T scalar)
    {
        var result = CreateInstance(Length);
        _numOps.MultiplyScalar(new ReadOnlySpan<T>(_data), scalar, result.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// Multiplies each element of this vector by a scalar value in-place.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This multiplies every element in your vector by the same number,
    /// modifying the current vector instead of creating a new one.</para>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated multiplication.</para>
    /// </remarks>
    public virtual void MultiplyInPlace(T scalar)
    {
        _numOps.MultiplyScalar(new ReadOnlySpan<T>(_data), scalar, new Span<T>(_data));
    }

    /// <summary>
    /// Multiplies each element of this vector by a scalar value, storing the result in the destination span.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <param name="destination">The span to store the result in.</param>
    /// <exception cref="ArgumentException">Thrown when destination is too small.</exception>
    /// <remarks>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated multiplication.</para>
    /// </remarks>
    public virtual void Multiply(T scalar, Span<T> destination)
    {
        if (destination.Length < Length)
            throw new ArgumentException("Destination span is too small", nameof(destination));

        _numOps.MultiplyScalar(new ReadOnlySpan<T>(_data), scalar, destination);
    }

    /// <summary>
    /// Divides each element of this vector by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <returns>A new vector with each element divided by the scalar.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This divides every element in your vector by the same number (scalar).
    /// For example, if you divide [2,4,6] by 2, you get [1,2,3]. Division is often used in
    /// normalization, where you might divide by the sum or maximum value to scale your data
    /// to a specific range.</para>
    /// <para><b>Performance:</b> Uses SIMD-accelerated operations (5-15x faster with AVX2).</para>
    /// </remarks>
    public virtual VectorBase<T> Divide(T scalar)
    {
        var result = CreateInstance(Length);
        _numOps.DivideScalar(new ReadOnlySpan<T>(_data), scalar, result.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// Divides each element of this vector by a scalar value in-place.
    /// </summary>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <remarks>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated division.</para>
    /// </remarks>
    public virtual void DivideInPlace(T scalar)
    {
        _numOps.DivideScalar(new ReadOnlySpan<T>(_data), scalar, new Span<T>(_data));
    }

    /// <summary>
    /// Divides each element of this vector by a scalar value, storing the result in the destination span.
    /// </summary>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <param name="destination">The span to store the result in.</param>
    /// <exception cref="ArgumentException">Thrown when destination is too small.</exception>
    /// <remarks>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated division.</para>
    /// </remarks>
    public virtual void Divide(T scalar, Span<T> destination)
    {
        if (destination.Length < Length)
            throw new ArgumentException("Destination span is too small", nameof(destination));

        _numOps.DivideScalar(new ReadOnlySpan<T>(_data), scalar, destination);
    }

    /// <summary>
    /// Returns a string representation of the vector.
    /// </summary>
    /// <returns>A string showing the vector's elements in square brackets, separated by commas.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts your vector to a readable text format.
    /// For example, a vector containing the values 1, 2, and 3 would be displayed as "[1, 2, 3]".
    /// This is helpful for debugging or displaying results to users.</para>
    /// </remarks>
    public override string ToString()
    {
        return $"[{string.Join(", ", _data)}]";
    }
}
