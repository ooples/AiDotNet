global using System.Collections;

#if NET6_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
#endif
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Represents a mathematical vector with generic type elements.
/// </summary>
/// <typeparam name="T">The type of elements in the vector.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A vector is a list of numbers arranged in a specific order.
/// Think of it as a one-dimensional array or a list of values. In machine learning,
/// vectors are commonly used to represent features or _data points.</para>
/// </remarks>
public class Vector<T> : VectorBase<T>, IEnumerable<T>
{
    /// <summary>
    /// Gets whether CPU SIMD acceleration is available for vector operations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> SIMD (Single Instruction Multiple Data) allows the CPU
    /// to perform the same operation on multiple values simultaneously.
    ///
    /// When IsCpuAccelerated is true, operations like Add, Multiply, etc. can be
    /// hardware-accelerated using instructions like SSE, AVX, or NEON, making them
    /// significantly faster.
    /// </para>
    /// </remarks>
    public static bool IsCpuAccelerated => DetectCpuAcceleration();

    /// <summary>
    /// Gets whether GPU acceleration is available for vector operations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GPU acceleration uses the graphics card to perform
    /// many calculations in parallel, which can be much faster than CPU for large datasets.
    ///
    /// When IsGpuAccelerated is true, large vector operations may be offloaded to the GPU.
    /// </para>
    /// </remarks>
    public static bool IsGpuAccelerated => MathHelper.SupportsGpuAcceleration<T>();

    /// <summary>
    /// Gets the number of elements that fit in a SIMD register for the current type.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many numbers can be processed
    /// at once using SIMD instructions. For example, with AVX and float, this is typically 8
    /// (256 bits / 32 bits per float = 8 floats).
    /// </para>
    /// </remarks>
    public static int SimdVectorCount => GetSimdVectorCount();

    /// <summary>
    /// Detects whether CPU SIMD acceleration is available.
    /// </summary>
    private static bool DetectCpuAcceleration()
    {
        // Check for type-specific CPU acceleration support
        if (!MathHelper.SupportsCpuAcceleration<T>())
            return false;

#if NET6_0_OR_GREATER
        // Check for actual hardware SIMD support
        return Sse.IsSupported || AdvSimd.IsSupported;
#else
        // .NET Framework doesn't have hardware intrinsics
        return false;
#endif
    }

    /// <summary>
    /// Gets the SIMD vector count based on hardware capabilities and type size.
    /// </summary>
    private static int GetSimdVectorCount()
    {
        if (!IsCpuAccelerated)
            return 1;

        var typeSize = GetTypeSizeInBytes();
        if (typeSize == 0)
            return 1;

#if NET6_0_OR_GREATER
        // Determine max vector width in bytes based on hardware
        int maxVectorWidth;
        if (Avx512F.IsSupported)
            maxVectorWidth = 64; // 512 bits
        else if (Avx.IsSupported)
            maxVectorWidth = 32; // 256 bits
        else if (Sse.IsSupported || AdvSimd.IsSupported)
            maxVectorWidth = 16; // 128 bits
        else
            return 1;

        return maxVectorWidth / typeSize;
#else
        // .NET Framework doesn't have hardware intrinsics
        return 1;
#endif
    }

    /// <summary>
    /// Gets the size in bytes of the element type T.
    /// </summary>
    private static int GetTypeSizeInBytes()
    {
        return typeof(T) switch
        {
            var t when t == typeof(float) => sizeof(float),
            var t when t == typeof(double) => sizeof(double),
            var t when t == typeof(int) => sizeof(int),
            var t when t == typeof(long) => sizeof(long),
            var t when t == typeof(short) => sizeof(short),
            var t when t == typeof(byte) => sizeof(byte),
            var t when t == typeof(Half) => 2,
            _ => 0
        };
    }

    /// <summary>
    /// Initializes a new instance of the Vector class with the specified length.
    /// </summary>
    /// <param name="length">The length of the vector.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an empty vector with the given size.
    /// All elements will be initialized to their default values (0 for numeric types).</para>
    /// </remarks>
    public Vector(int length) : base(length)
    {
    }

    /// <summary>
    /// Initializes a new instance of the Vector class with the specified values.
    /// </summary>
    /// <param name="values">The collection of values to initialize the vector with.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector containing the values you provide.
    /// For example, new Vector&lt;double&gt;(new[] {1.0, 2.0, 3.0}) creates a vector with three elements.</para>
    /// </remarks>
    public Vector(IEnumerable<T> values) : base(values)
    {
    }

    /// <summary>
    /// Returns an enumerator that iterates through the vector.
    /// </summary>
    /// <returns>An enumerator that can be used to iterate through the vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This allows you to use the vector in foreach loops,
    /// making it easy to process each element one by one.</para>
    /// </remarks>
    public IEnumerator<T> GetEnumerator()
    {
        return ((IEnumerable<T>)_data).GetEnumerator();
    }

    /// <summary>
    /// Returns an enumerator that iterates through the vector.
    /// </summary>
    /// <returns>An enumerator that can be used to iterate through the vector.</returns>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    /// <summary>
    /// Performs element-wise division of this vector by another vector.
    /// </summary>
    /// <param name="other">The vector to divide by.</param>
    /// <returns>A new vector containing the results of dividing each element of this vector by the corresponding element in the other vector.</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This divides each element in your vector by the corresponding element
    /// in another vector. For example, [10, 20, 30] divided by [2, 4, 5] gives [5, 5, 6].</para>
    /// <para><b>Performance:</b> This method uses SIMD-accelerated operations for float/double types
    /// via TensorPrimitives, providing 5-10x speedup for large vectors.</para>
    /// </remarks>
    public Vector<T> ElementwiseDivide(Vector<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        if (this.Length != other.Length)
        {
            throw new ArgumentException("Vectors must have the same length for element-wise division.", nameof(other));
        }

        var resultArray = new T[this.Length];
        _numOps.Divide(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(other._data), new Span<T>(resultArray));

        return new Vector<T>(resultArray);
    }

    /// <summary>
    /// Calculates the variance of the vector elements.
    /// </summary>
    /// <returns>The variance of the vector elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Variance measures how spread out the values in your vector are.
    /// A high variance means the values are widely spread; a low variance means they're clustered together.
    /// It's calculated by finding the average of the squared differences from the mean.</para>
    /// </remarks>
    public T Variance()
    {
        T mean = Mean();
        return this.Select(x => _numOps.Square(_numOps.Subtract(x, mean))).Mean();
    }

    /// <summary>
    /// Filters the vector elements based on a condition.
    /// </summary>
    /// <param name="predicate">A function that determines whether an element should be included.</param>
    /// <returns>A new vector containing only the elements that satisfy the condition.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This lets you keep only certain elements from your vector
    /// that meet a condition you specify. For example, you could keep only positive numbers
    /// or only values above a certain threshold.</para>
    /// </remarks>
    public Vector<T> Where(Func<T, bool> predicate)
    {
        return new Vector<T>(_data.Where(predicate));
    }

    /// <summary>
    /// Projects each element of the vector into a new form.
    /// </summary>
    /// <typeparam name="TResult">The type of the elements in the resulting vector.</typeparam>
    /// <param name="selector">A function that transforms each element.</param>
    /// <returns>A new vector containing the transformed elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This transforms each element in your vector using a function you provide.
    /// For example, you could multiply each element by 2, or convert each number to its absolute value.</para>
    /// </remarks>
    public Vector<TResult> Select<TResult>(Func<T, TResult> selector)
    {
        return new Vector<TResult>(_data.Select(selector));
    }

    /// <summary>
    /// Creates a deep copy of this vector.
    /// </summary>
    /// <returns>A new vector with the same values as this vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an exact duplicate of your vector.
    /// Changes to the copy won't affect the original vector, and vice versa.</para>
    /// </remarks>
    public new Vector<T> Clone()
    {
        return new Vector<T>([.. this]);
    }

    /// <summary>
    /// Creates a vector of the specified size with all elements set to zero.
    /// </summary>
    /// <param name="size">The size of the vector.</param>
    /// <returns>A new vector with all elements set to zero.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector filled with zeros.
    /// It's often used as a starting point for calculations.</para>
    /// </remarks>
    public override VectorBase<T> Zeros(int size)
    {
        return new Vector<T>(size);
    }

    /// <summary>
    /// Creates a vector of the specified size with all elements set to the default value.
    /// </summary>
    /// <param name="size">The size of the vector.</param>
    /// <param name="defaultValue">The default value for all elements.</param>
    /// <returns>A new vector with all elements set to the default value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector where every element has the same value
    /// that you specify.</para>
    /// </remarks>
    public override VectorBase<T> Default(int size, T defaultValue)
    {
        return base.Default(size, defaultValue);
    }

    /// <summary>
    /// Applies a transformation function to each element of the vector.
    /// </summary>
    /// <typeparam name="TResult">The type of the elements in the resulting vector.</typeparam>
    /// <param name="function">The transformation function to apply to each element.</param>
    /// <returns>A new vector containing the transformed elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is similar to Select, but specifically designed for
    /// mathematical transformations. It applies a function to each element in your vector.</para>
    /// </remarks>
    public new Vector<TResult> Transform<TResult>(Func<T, TResult> function)
    {
        return new Vector<TResult>(base.Transform(function).ToArray());
    }

    /// <summary>
    /// Applies a transformation function to each element of the vector, also providing the element's index.
    /// </summary>
    /// <typeparam name="TResult">The type of the elements in the resulting vector.</typeparam>
    /// <param name="function">The transformation function to apply to each element and its index.</param>
    /// <returns>A new vector containing the transformed elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like Transform, but your function also receives the position (index)
    /// of each element. This is useful when the transformation depends on where the element is located
    /// in the vector.</para>
    /// </remarks>
    public new Vector<TResult> Transform<TResult>(Func<T, int, TResult> function)
    {
        return new Vector<TResult>(base.Transform(function).ToArray());
    }

    /// <summary>
    /// Creates a vector of the specified size with all elements set to one.
    /// </summary>
    /// <param name="size">The size of the vector.</param>
    /// <returns>A new vector with all elements set to one.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector filled with ones.
    /// It's commonly used in various mathematical operations and algorithms.</para>
    /// </remarks>
    public override VectorBase<T> Ones(int size)
    {
        return new Vector<T>(Enumerable.Repeat(_numOps.One, size));
    }

    /// <summary>
    /// Creates an empty vector with zero elements.
    /// </summary>
    /// <returns>A new empty vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector with no elements (length of 0).
    /// It's useful as a starting point when you need to build a vector by adding elements.</para>
    /// </remarks>
    public new static Vector<T> Empty()
    {
        return new Vector<T>(0);
    }

    /// <summary>
    /// Extracts a portion of the vector as a new vector.
    /// </summary>
    /// <param name="startIndex">The zero-based index at which to start extraction.</param>
    /// <param name="length">The number of elements to extract.</param>
    /// <returns>A new vector containing the extracted elements.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when startIndex is negative or greater than or equal to the vector's length,
    /// or when length is negative or would extend beyond the end of the vector.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you take a slice of your vector.
    /// For example, if you have a vector [1,2,3,4,5] and call GetSubVector(1,3),
    /// you'll get a new vector [2,3,4].</para>
    /// </remarks>
    public new Vector<T> GetSubVector(int startIndex, int length)
    {
        if (startIndex < 0 || startIndex >= this.Length)
            throw new ArgumentOutOfRangeException(nameof(startIndex), "Start index must be within the bounds of the vector.");
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length cannot be negative.");
        if (startIndex + length > this.Length)
            throw new ArgumentOutOfRangeException(nameof(length), "The subvector would extend beyond the end of the vector.");

        Vector<T> subVector = new Vector<T>(length);
        // Use vectorized copy via Span slicing
        _numOps.Copy(new ReadOnlySpan<T>(_data, startIndex, length), subVector.AsWritableSpan());
        return subVector;
    }

    /// <summary>
    /// Creates a new vector with a single value changed.
    /// </summary>
    /// <param name="index">The zero-based index of the element to change.</param>
    /// <param name="value">The new value to set.</param>
    /// <returns>A new vector with the specified element changed.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when index is negative or greater than or equal to the vector's length.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a copy of your vector with just one value changed.
    /// It's useful when you want to keep your original vector unchanged while working with a modified version.</para>
    /// </remarks>
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

    /// <summary>
    /// Calculates the Euclidean norm (magnitude) of the vector.
    /// </summary>
    /// <returns>The Euclidean norm of the vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The norm is the "length" of a vector in multi-dimensional space.
    /// For a 2D vector [x,y], the norm is sqrt(x^2 + y^2), which is the same as the Pythagorean theorem.
    /// For higher dimensions, it's the square root of the sum of all squared components.</para>
    /// <para><b>Performance:</b> This method uses SIMD-accelerated dot product for float/double types
    /// via TensorPrimitives, providing 5-10x speedup for large vectors.</para>
    /// </remarks>
    public T Norm()
    {
        // Use vectorized dot product: ||x|| = sqrt(x . x)
        T sumOfSquares = _numOps.Dot(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(_data));
        return _numOps.Sqrt(sumOfSquares);
    }

    /// <summary>
    /// Divides each element of the vector by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <returns>A new vector with each element divided by the scalar.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This divides every number in your vector by the same value.
    /// For example, [10,20,30] divided by 10 gives [1,2,3].</para>
    /// <para><b>Performance:</b> This method uses SIMD-accelerated operations for float/double types
    /// via TensorPrimitives, providing 5-10x speedup for large vectors.</para>
    /// </remarks>
    public new Vector<T> Divide(T scalar)
    {
        var resultArray = new T[this.Length];
        _numOps.DivideScalar(new ReadOnlySpan<T>(_data), scalar, new Span<T>(resultArray));
        return new Vector<T>(resultArray);
    }

    /// <summary>
    /// Creates a new instance of the vector type with the specified size.
    /// </summary>
    /// <param name="size">The size of the new vector.</param>
    /// <returns>A new vector instance of the specified size.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is an internal method that helps create new vectors
    /// of the right type when performing operations.</para>
    /// </remarks>
    protected override VectorBase<T> CreateInstance(int size)
    {
        return new Vector<T>(size);
    }

    /// <summary>
    /// Converts the vector to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array representing the serialized vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Serialization converts your vector into a format that can be
    /// saved to a file or sent over a network. This is useful when you want to save your
    /// trained model for later use.</para>
    /// </remarks>
    public byte[] Serialize()
    {
        throw new NotImplementedException("Serialization requires AI-specific SerializationHelper class");
    }

    /// <summary>
    /// Creates a vector from a previously serialized byte array.
    /// </summary>
    /// <param name="_data">The byte array containing the serialized vector _data.</param>
    /// <returns>A new vector created from the serialized _data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts a previously serialized vector back into
    /// a usable vector object. Use this when loading a saved model from a file.</para>
    /// </remarks>
    public static Vector<T> Deserialize(byte[] _data)
    {
        throw new NotImplementedException("Deserialization requires AI-specific SerializationHelper class");
    }

    /// <summary>
    /// Multiplies each element of this vector with the corresponding element of another vector.
    /// </summary>
    /// <param name="other">The vector to multiply with.</param>
    /// <returns>A new vector containing the element-wise product.</returns>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the vectors have different lengths.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This multiplies corresponding elements together.
    /// For example, [1,2,3] element-wise multiplied by [4,5,6] gives [4,10,18].
    /// This is different from dot product, which would give a single number (1*4 + 2*5 + 3*6 = 32).</para>
    /// <para><b>Performance:</b> This method uses SIMD-accelerated operations for float/double types
    /// via TensorPrimitives, providing 5-10x speedup for large vectors.</para>
    /// </remarks>
    public Vector<T> ElementwiseMultiply(Vector<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        if (this.Length != other.Length)
            throw new ArgumentException("Vectors must have the same length for element-wise multiplication.", nameof(other));

        var resultArray = new T[this.Length];
        _numOps.Multiply(new ReadOnlySpan<T>(_data), new ReadOnlySpan<T>(other._data), new Span<T>(resultArray));

        return new Vector<T>(resultArray);
    }

    /// <summary>
    /// Creates a vector with sequential values starting from a specified value.
    /// </summary>
    /// <param name="start">The starting value.</param>
    /// <param name="count">The number of elements in the vector.</param>
    /// <returns>A new vector with sequential values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a vector with evenly spaced values.
    /// For example, Range(1, 5) creates [1,2,3,4,5]. This is useful for creating
    /// indices or x-values for plotting.</para>
    /// </remarks>
    public static Vector<T> Range(int start, int count)
    {
        Vector<T> result = new Vector<T>(count);
        for (int i = 0; i < count; i++)
        {
            result[i] = _numOps.FromDouble(start + i);
        }

        return result;
    }

    /// <summary>
    /// Extracts a portion of the vector as a new vector.
    /// </summary>
    /// <param name="startIndex">The zero-based index at which to start extraction.</param>
    /// <param name="length">The number of elements to extract.</param>
    /// <returns>A new vector containing the extracted elements.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when startIndex is negative, length is negative, or the subvector would extend beyond the end of the vector.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method delegates to GetSubVector and provides the same functionality.
    /// It lets you extract a portion of your vector as a new vector.</para>
    /// </remarks>
    public Vector<T> Subvector(int startIndex, int length)
    {
        return GetSubVector(startIndex, length);
    }

    /// <summary>
    /// Searches for a specified value in a sorted vector.
    /// </summary>
    /// <param name="value">The value to search for.</param>
    /// <returns>
    /// The index of the specified value if found; otherwise, a negative number that is the bitwise
    /// complement of the index of the next element that is larger than value or the bitwise complement
    /// of the vector's length if there is no larger element.
    /// </returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method quickly finds a value in a sorted vector.
    /// It returns the position if found. If not found, it returns a negative number that tells
    /// you where the value would be if it were in the vector. The vector must be sorted
    /// for this to work correctly.</para>
    /// </remarks>
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

    /// <summary>
    /// Extracts a portion of the vector as a new vector.
    /// </summary>
    /// <param name="startIndex">The zero-based index at which to start extraction.</param>
    /// <param name="count">The number of elements to extract.</param>
    /// <returns>A new vector containing the extracted elements.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when startIndex is negative or beyond the vector's bounds,
    /// or when count is negative or would extend beyond the end of the vector.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method delegates to GetSubVector and provides the same functionality.
    /// For example, if you have a vector [1,2,3,4,5] and call GetRange(1,3), you'll get a new vector [2,3,4].</para>
    /// </remarks>
    public Vector<T> GetRange(int startIndex, int count)
    {
        return GetSubVector(startIndex, count);
    }

    /// <summary>
    /// Finds the index of the maximum value in the vector.
    /// </summary>
    /// <returns>The zero-based index of the maximum value in the vector.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the vector is empty.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method tells you the position of the largest number in your vector.
    /// For example, in the vector [3,8,2,5], the maximum value is 8, which is at index 1.</para>
    /// </remarks>
    public int IndexOfMax()
    {
        if (this.Length == 0)
            throw new InvalidOperationException("Vector is empty");

        int maxIndex = 0;
        T maxValue = this[0];
        var _numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 1; i < this.Length; i++)
        {
            if (_numOps.GreaterThan(this[i], maxValue))
            {
                maxValue = this[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /// <summary>
    /// Computes the outer product of two vectors.
    /// </summary>
    /// <param name="other">The second vector for the outer product.</param>
    /// <returns>A matrix representing the outer product of the two vectors.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The outer product creates a matrix by multiplying each element of the first vector
    /// with every element of the second vector. For example, if you have vectors [1,2] and [3,4,5], 
    /// the result will be a 2ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¿Ãƒâ€šÃ‚Â½3 matrix:
    /// [1ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¿Ãƒâ€šÃ‚Â½3, 1ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¿Ãƒâ€šÃ‚Â½4, 1ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¿Ãƒâ€šÃ‚Â½5]
    /// [2ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¿Ãƒâ€šÃ‚Â½3, 2ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¿Ãƒâ€šÃ‚Â½4, 2ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¿Ãƒâ€šÃ‚Â½5]
    /// which equals:
    /// [3, 4, 5]
    /// [6, 8, 10]</para>
    /// </remarks>
    public Matrix<T> OuterProduct(Vector<T> other)
    {
        int m = this.Length;
        int n = other.Length;
        var _numOps = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(m, n);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = _numOps.Multiply(this[i], other[j]);
            }
        }

        return result;
    }

    /// <summary>
    /// Extracts a segment of the vector using LINQ operations.
    /// </summary>
    /// <param name="startIndex">The zero-based index at which to start extraction.</param>
    /// <param name="length">The number of elements to extract.</param>
    /// <returns>A new vector containing the extracted elements.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when startIndex is negative or beyond the vector's bounds,
    /// or when length is negative or would extend beyond the end of the vector.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method delegates to GetSubVector and provides the same functionality.
    /// It extracts a portion of your vector starting at a specific position and taking a certain number of elements.</para>
    /// </remarks>
    public Vector<T> GetSegment(int startIndex, int length)
    {
        return GetSubVector(startIndex, length);
    }

    /// <summary>
    /// Creates a new instance of the vector class with the specified _data.
    /// </summary>
    /// <param name="_data">The array of _data to initialize the vector with.</param>
    /// <returns>A new vector containing the specified _data.</returns>
    protected override VectorBase<T> CreateInstance(T[] _data)
    {
        return new Vector<T>(_data);
    }

    /// <summary>
    /// Creates a new instance of the vector class with the specified size and default values.
    /// </summary>
    /// <typeparam name="TResult">The type of elements in the new vector.</typeparam>
    /// <param name="size">The size of the new vector.</param>
    /// <returns>A new vector of the specified size.</returns>
    protected override VectorBase<TResult> CreateInstance<TResult>(int size)
    {
        return new Vector<TResult>(size);
    }

    /// <summary>
    /// Creates a new vector of the specified size filled with random values between 0 and 1.
    /// </summary>
    /// <param name="size">The size of the vector to create.</param>
    /// <returns>A new vector filled with random values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a vector of a specific size where each element
    /// is a random number between 0 and 1. This is useful for initializing vectors for machine learning algorithms.</para>
    /// </remarks>
    public static Vector<T> CreateRandom(int size)
    {
        Vector<T> vector = new(size);
        Random random = new();
        for (int i = 0; i < size; i++)
        {
            vector[i] = _numOps.FromDouble(random.NextDouble());
        }

        return vector;
    }

    /// <summary>
    /// Creates a new vector of the specified size filled with random values between the specified minimum and maximum values.
    /// </summary>
    /// <param name="size">The size of the vector to create.</param>
    /// <param name="min">The minimum value for the random numbers (default is -1.0).</param>
    /// <param name="max">The maximum value for the random numbers (default is 1.0).</param>
    /// <returns>A new vector filled with random values within the specified range.</returns>
    /// <exception cref="ArgumentException">Thrown when min is greater than or equal to max.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a vector of a specific size where each element
    /// is a random number between the minimum and maximum values you specify. For example, CreateRandom(3, 0, 10)
    /// might create a vector like [2.7, 9.1, 4.3] with random values between 0 and 10.</para>
    /// </remarks>
    public static Vector<T> CreateRandom(int size, double min = -1.0, double max = 1.0)
    {
        if (min >= max)
            throw new ArgumentException("Minimum value must be less than maximum value");

        var random = RandomHelper.CreateSecureRandom();
        var vector = new Vector<T>(size);

        for (int i = 0; i < size; i++)
        {
            // Generate random value between min and max
            double randomValue = random.NextDouble() * (max - min) + min;
            vector[i] = _numOps.FromDouble(randomValue);
        }

        return vector;
    }

    /// <summary>
    /// Creates a standard basis vector of the specified size with a 1 at the specified index and 0s elsewhere.
    /// </summary>
    /// <param name="size">The size of the vector to create.</param>
    /// <param name="index">The index at which to place the value 1.</param>
    /// <returns>A new standard basis vector.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when size is negative or when index is negative or greater than or equal to size.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> A standard basis vector has a 1 at one position and 0s everywhere else.
    /// For example, CreateStandardBasis(3, 1) creates the vector [0,1,0]. These vectors are important in
    /// linear algebra and are used to represent directions in space.</para>
    /// </remarks>
    public static Vector<T> CreateStandardBasis(int size, int index)
    {
        if (size < 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be non-negative.");
        if (index < 0 || index >= size)
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be within the bounds of the vector.");

        var vector = new Vector<T>(size)
        {
            [index] = _numOps.One
        };

        return vector;
    }

    /// <summary>
    /// Creates a unit vector in the same direction as this vector.
    /// </summary>
    /// <returns>A new vector with length 1 in the same direction as this vector.</returns>
    /// <exception cref="InvalidOperationException">Thrown when trying to normalize a zero vector.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Normalizing a vector means changing its length to 1 while keeping its direction.
    /// This is useful in many algorithms where only the direction matters, not the magnitude.
    /// For example, normalizing [3,4] gives [0.6,0.8] because 0.6ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¿Ãƒâ€šÃ‚Â½ + 0.8ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¿Ãƒâ€šÃ‚Â½ = 1.</para>
    /// </remarks>
    public Vector<T> Normalize()
    {
        T norm = this.Norm();
        if (_numOps.Equals(norm, _numOps.Zero))
        {
            throw new InvalidOperationException("Cannot normalize a zero vector.");
        }

        return this.Divide(norm);
    }

    /// <summary>
    /// Returns the indices of all non-zero elements in the vector.
    /// </summary>
    /// <returns>An enumerable collection of indices where the vector has non-zero values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method finds the positions of all elements in your vector that are not zero.
    /// For example, in the vector [0,5,0,3,0], this would return the indices 1 and 3, since those are the positions
    /// of the non-zero values (5 and 3).</para>
    /// </remarks>
    public IEnumerable<int> NonZeroIndices()
    {
        var _numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < Length; i++)
        {
            if (!_numOps.Equals(this[i], _numOps.Zero))
            {
                yield return i;
            }
        }
    }

    /// <summary>
    /// Converts this vector into a 1xn matrix (a row vector).
    /// </summary>
    /// <returns>A matrix with 1 row and n columns, where n is the length of this vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method transforms your vector into a matrix with just one row.
    /// For example, the vector [1,2,3] becomes the matrix [[1,2,3]]. This is useful when you need to
    /// perform matrix operations with your vector _data.</para>
    /// </remarks>
    public Matrix<T> Transpose()
    {
        // Create matrix directly from vector data - Matrix constructor accepts IEnumerable<T[]>
        // For a 1xn matrix, pass the data as a single row
        return new Matrix<T>([_data]);
    }

    /// <summary>
    /// Creates a matrix by appending a constant value as a second column to this vector.
    /// </summary>
    /// <param name="value">The value to append to each element of the vector.</param>
    /// <returns>A matrix where the first column contains this vector's values and the second column contains the specified value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a matrix with two columns. The first column contains
    /// your original vector values, and the second column has the same value repeated for each row.
    /// For example, if your vector is [1,2,3] and the value is 5, the result will be:
    /// [[1,5],
    ///  [2,5],
    ///  [3,5]]
    /// This is particularly useful in machine learning when adding a bias term to feature vectors.</para>
    /// </remarks>
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

    /// <summary>
    /// Creates a new vector containing only the elements at the specified indices.
    /// </summary>
    /// <param name="indices">The indices of elements to include in the new vector.</param>
    /// <returns>A new vector containing only the elements at the specified indices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you pick specific elements from your vector by their positions.
    /// For example, if your vector is [10,20,30,40,50] and you specify indices [1,3], the result will be [20,40].</para>
    /// </remarks>
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

    /// <summary>
    /// Creates a new vector with one element removed at the specified index.
    /// </summary>
    /// <param name="index">The zero-based index of the element to remove.</param>
    /// <returns>A new vector with the element at the specified index removed.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the index is negative or greater than or equal to the vector's length.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a new vector that's identical to your original vector,
    /// but with one element removed. For example, if your vector is [1,2,3,4] and you remove the element at index 1,
    /// the result will be [1,3,4].</para>
    /// </remarks>
    public Vector<T> RemoveAt(int index)
    {
        if (index < 0 || index >= Length)
            throw new ArgumentOutOfRangeException(nameof(index));

        var newData = new T[Length - 1];
        Array.Copy(_data, 0, newData, 0, index);
        Array.Copy(_data, index + 1, newData, index, Length - index - 1);

        return new Vector<T>(newData);
    }

    /// <summary>
    /// Counts the number of non-zero elements in the vector.
    /// </summary>
    /// <returns>The count of non-zero elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method tells you how many elements in your vector are not zero.
    /// For example, in the vector [0,5,0,3,0], there are 2 non-zero elements (5 and 3).</para>
    /// </remarks>
    public int NonZeroCount()
    {
        return NonZeroIndices().Count();
    }

    /// <summary>
    /// Sets all elements of the vector to the specified value.
    /// </summary>
    /// <param name="value">The value to set for all elements.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method changes every element in your vector to the same value.
    /// For example, if your vector is [1,2,3] and you call Fill(5), your vector will become [5,5,5].</para>
    /// <para><b>Performance:</b> This method uses SIMD-accelerated fill operation for float/double types,
    /// providing 5-10x speedup for large vectors.</para>
    /// </remarks>
    public void Fill(T value)
    {
        _numOps.Fill(AsWritableSpan(), value);
    }

    /// <summary>
    /// Combines multiple vectors into a single vector by placing them one after another.
    /// </summary>
    /// <param name="vectors">The vectors to concatenate.</param>
    /// <returns>A new vector containing all elements from the input vectors in sequence.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method joins multiple vectors together end-to-end.
    /// For example, if you concatenate [1,2] and [3,4,5], the result will be [1,2,3,4,5].</para>
    /// </remarks>
    public static Vector<T> Concatenate(params Vector<T>[] vectors)
    {
        int totalSize = vectors.Sum(v => v.Length);
        Vector<T> result = new(totalSize);

        int offset = 0;
        foreach (var vector in vectors)
        {
            // Use vectorized copy for each vector segment
            _numOps.Copy(new ReadOnlySpan<T>(vector._data), new Span<T>(result._data, offset, vector.Length));
            offset += vector.Length;
        }

        return result;
    }

    /// <summary>
    /// Combines a list of vectors into a single vector by placing them one after another.
    /// </summary>
    /// <param name="vectors">The list of vectors to concatenate.</param>
    /// <returns>A new vector containing all elements from the input vectors in sequence.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is similar to the other Concatenate method but accepts
    /// a list of vectors instead of individual parameters. It joins all vectors in the list together end-to-end.</para>
    /// </remarks>
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

    /// <summary>
    /// Adds another vector to this vector.
    /// </summary>
    /// <param name="other">The vector to add.</param>
    /// <returns>A new vector that is the sum of this vector and the other vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adds two vectors together element by element.
    /// For example, adding [1,2,3] and [4,5,6] gives [5,7,9].</para>
    /// </remarks>
    public new Vector<T> Add(VectorBase<T> other)
    {
        return new Vector<T>(base.Add(other).ToArray());
    }

    /// <summary>
    /// Subtracts another vector from this vector.
    /// </summary>
    /// <param name="other">The vector to subtract.</param>
    /// <returns>A new vector that is the difference of this vector and the other vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method subtracts one vector from another element by element.
    /// For example, subtracting [4,5,6] from [10,10,10] gives [6,5,4].</para>
    /// </remarks>
    public new Vector<T> Subtract(VectorBase<T> other)
    {
        return new Vector<T>(base.Subtract(other).ToArray());
    }

    /// <summary>
    /// Multiplies this vector by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <returns>A new vector with each element multiplied by the scalar.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method multiplies every element in your vector by the same number.
    /// For example, multiplying [1,2,3] by 2 gives [2,4,6].</para>
    /// </remarks>
    public new Vector<T> Multiply(T scalar)
    {
        return new Vector<T>(base.Multiply(scalar).ToArray());
    }

    /// <summary>
    /// Adds two vectors together.
    /// </summary>
    /// <param name="left">The first vector.</param>
    /// <param name="right">The second vector.</param>
    /// <returns>A new vector that is the sum of the two vectors.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator lets you use the + symbol to add two vectors together.
    /// For example, you can write "result = vectorA + vectorB" instead of "result = vectorA.Add(vectorB)".</para>
    /// </remarks>
    public static Vector<T> operator +(Vector<T> left, Vector<T> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Adds a scalar value to each element of the vector.
    /// </summary>
    /// <param name="vector">The vector to add the scalar to.</param>
    /// <param name="scalar">The scalar value to add to each element.</param>
    /// <returns>A new vector with the scalar added to each element.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the vector is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator lets you add a single number to every element in your vector.
    /// For example, if your vector is [1,2,3] and you add 5, the result will be [6,7,8].</para>
    /// <para><b>Performance:</b> This operator uses SIMD-accelerated operations for float/double types
    /// via TensorPrimitives, providing 5-10x speedup for large vectors.</para>
    /// </remarks>
    public static Vector<T> operator +(Vector<T> vector, T scalar)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        var resultArray = new T[vector.Length];
        _numOps.AddScalar(new ReadOnlySpan<T>(vector._data), scalar, new Span<T>(resultArray));
        return new Vector<T>(resultArray);
    }

    /// <summary>
    /// Subtracts a scalar value from each element of the vector.
    /// </summary>
    /// <param name="vector">The vector to subtract the scalar from.</param>
    /// <param name="scalar">The scalar value to subtract from each element.</param>
    /// <returns>A new vector with the scalar subtracted from each element.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the vector is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator lets you subtract a single number from every element in your vector.
    /// For example, if your vector is [5,7,9] and you subtract 2, the result will be [3,5,7].</para>
    /// <para><b>Performance:</b> This operator uses SIMD-accelerated operations for float/double types
    /// via TensorPrimitives, providing 5-10x speedup for large vectors.</para>
    /// </remarks>
    public static Vector<T> operator -(Vector<T> vector, T scalar)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        var resultArray = new T[vector.Length];
        _numOps.SubtractScalar(new ReadOnlySpan<T>(vector._data), scalar, new Span<T>(resultArray));
        return new Vector<T>(resultArray);
    }

    /// <summary>
    /// Subtracts one vector from another.
    /// </summary>
    /// <param name="left">The vector to subtract from (minuend).</param>
    /// <param name="right">The vector to subtract (subtrahend).</param>
    /// <returns>A new vector that is the difference of the two vectors.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator lets you use the - symbol to subtract one vector from another.
    /// For example, you can write "result = vectorA - vectorB" instead of "result = vectorA.Subtract(vectorB)".
    /// The subtraction happens element by element, so [10,20,30] - [1,2,3] gives [9,18,27].</para>
    /// </remarks>
    public static Vector<T> operator -(Vector<T> left, Vector<T> right)
    {
        return left.Subtract(right);
    }

    /// <summary>
    /// Multiplies each element of the vector by a scalar value.
    /// </summary>
    /// <param name="vector">The vector to multiply.</param>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <returns>A new vector with each element multiplied by the scalar.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator lets you use the * symbol to multiply every element in your vector by a number.
    /// For example, you can write "result = vector * 2" to double every value in your vector.
    /// So [1,2,3] * 2 gives [2,4,6].</para>
    /// </remarks>
    public static Vector<T> operator *(Vector<T> vector, T scalar)
    {
        return vector.Multiply(scalar);
    }

    /// <summary>
    /// Multiplies a scalar value by each element of the vector.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <param name="vector">The vector to multiply.</param>
    /// <returns>A new vector with each element multiplied by the scalar.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator is the same as the previous one, but allows you to write the scalar first.
    /// For example, you can write "result = 2 * vector" instead of "result = vector * 2".
    /// Both will give the same result, like [2,4,6] for a vector [1,2,3].</para>
    /// </remarks>
    public static Vector<T> operator *(T scalar, Vector<T> vector)
    {
        return vector * scalar;
    }

    /// <summary>
    /// Divides each element of the vector by a scalar value.
    /// </summary>
    /// <param name="vector">The vector to divide.</param>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <returns>A new vector with each element divided by the scalar.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator lets you divide every element in your vector by a number.
    /// For example, if your vector is [10,20,30] and you divide by 10, the result will be [1,2,3].</para>
    /// </remarks>
    public static Vector<T> operator /(Vector<T> vector, T scalar)
    {
        return vector.Divide(scalar);
    }

    /// <summary>
    /// Implicitly converts a vector to an array of its elements.
    /// </summary>
    /// <param name="vector">The vector to convert.</param>
    /// <returns>An array containing the vector's elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator allows a vector to be used anywhere an array is expected.
    /// For example, if you have a method that takes an array as a parameter, you can pass a vector directly
    /// without having to manually convert it to an array first.</para>
    /// </remarks>
    public static implicit operator T[](Vector<T> vector)
    {
        return vector.ToArray();
    }

    /// <summary>
    /// Creates a new vector from an array of values.
    /// </summary>
    /// <param name="array">The array of values to create the vector from.</param>
    /// <returns>A new vector containing the values from the array.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a vector from an existing array of numbers.
    /// For example, if you have an array [1,2,3], you can create a vector with the same values
    /// by calling Vector.FromArray([1,2,3]).</para>
    /// </remarks>
    public static Vector<T> FromArray(T[] array)
    {
        return new Vector<T>(array);
    }

    /// <summary>
    /// Creates a new vector from any collection of values.
    /// </summary>
    /// <param name="enumerable">The collection of values to create the vector from.</param>
    /// <returns>A new vector containing the values from the collection.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the collection is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a vector from any collection of numbers.
    /// It's more flexible than FromArray or FromList because it works with any type of collection.
    /// For example, you could use it with a Queue, Stack, or any other collection type in C#.</para>
    /// <para>The method is smart enough to use the most efficient approach based on what type of collection you provide.</para>
    /// </remarks>
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

    /// <summary>
    /// Creates a new vector from a list of values.
    /// </summary>
    /// <param name="list">The list of values to create the vector from.</param>
    /// <returns>A new vector containing the values from the list.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the list is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a vector from a List collection.
    /// Lists are one of the most common collection types in C#, so this method provides
    /// a convenient and efficient way to convert your list _data into a vector for mathematical operations.</para>
    /// </remarks>
    public static Vector<T> FromList(List<T> list)
    {
        if (list == null)
            throw new ArgumentNullException(nameof(list));
        var vector = new Vector<T>(list.Count);
        list.CopyTo(vector._data);
        return vector;
    }
}
