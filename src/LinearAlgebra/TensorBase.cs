namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Represents a base class for multi-dimensional arrays of numeric values used in machine learning and AI computations.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor elements (e.g., float, double, int).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> TensorBase is an abstract class that provides the foundation for working with tensors.
/// It defines common properties and methods that all tensor implementations should have, regardless of their specific type or dimensionality.
/// </para>
/// </remarks>
public abstract class TensorBase<T>
{
    /// <summary>
    /// The underlying data storage for the tensor elements.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This field stores all the values in the tensor in a one-dimensional array.
    /// Even though a tensor can have multiple dimensions, we store its data in a flat structure for efficiency.
    /// The class provides methods to convert between multi-dimensional indices and this flat storage.</para>
    /// </remarks>
    protected readonly Vector<T> _data;

    /// <summary>
    /// Provides numeric operations for the tensor's element type.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This field holds a set of mathematical operations (like addition, multiplication, etc.)
    /// that work with the specific numeric type of this tensor. It allows the tensor to perform calculations
    /// regardless of whether it contains integers, floating-point numbers, or other numeric types.</para>
    /// </remarks>
    protected static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the shape (dimensions) of the tensor.
    /// </summary>
    public int[] Shape { get; }

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    public int Length => _data.Length;

    /// <summary>
    /// Gets the rank (number of dimensions) of the tensor.
    /// </summary>
    public int Rank => Shape.Length;

    /// <summary>
    /// Gets the underlying data vector of the tensor.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This gives you direct access to the internal vector storing the tensor's values.
    /// The data is stored in a flattened format.</para>
    /// </remarks>
    public Vector<T> Data => _data;

    /// <summary>
    /// Gets or sets metadata associated with the tensor.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Metadata is additional information about the tensor,
    /// such as labels, descriptions, or other contextual data.</para>
    /// </remarks>
    public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();

    /// <summary>
    /// Initializes a new instance of the TensorBase class with the specified shape.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    protected TensorBase(params int[] shape)
    {
        Shape = shape;
        int totalSize = shape.Aggregate(1, (acc, dim) => acc * dim);
        _data = new Vector<T>(totalSize);
    }

    /// <summary>
    /// Initializes a new instance of the TensorBase class with the specified data and shape.
    /// </summary>
    /// <param name="data">The data to populate the tensor with.</param>
    /// <param name="shape">The shape of the tensor.</param>
    protected TensorBase(IEnumerable<T> data, params int[] shape)
    {
        Shape = shape;
        _data = new Vector<T>(data);
        if (_data.Length != shape.Aggregate(1, (acc, dim) => acc * dim))
        {
            throw new ArgumentException("The number of values does not match the specified shape.");
        }
    }

    /// <summary>
    /// Gets or sets the value at the specified indices.
    /// </summary>
    /// <param name="indices">The indices of the element.</param>
    /// <returns>The value at the specified indices.</returns>
    public virtual T this[params int[] indices]
    {
        get
        {
            ValidateIndices(indices);
            return _data[GetFlatIndex(indices)];
        }
        set
        {
            ValidateIndices(indices);
            _data[GetFlatIndex(indices)] = value;
        }
    }

    /// <summary>
    /// Validates the provided indices against the tensor's shape.
    /// </summary>
    /// <param name="indices">The indices to validate.</param>
    protected void ValidateIndices(int[] indices)
    {
        if (indices.Length != Shape.Length)
            throw new ArgumentException("Number of indices must match the tensor's rank.");

        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] < 0 || indices[i] >= Shape[i])
                throw new ArgumentOutOfRangeException(nameof(indices), $"Index {i} is out of range.");
        }
    }

    /// <summary>
    /// Converts multi-dimensional indices to a flat index.
    /// </summary>
    /// <param name="indices">The multi-dimensional indices.</param>
    /// <returns>The corresponding flat index.</returns>
    protected int GetFlatIndex(int[] indices)
    {
        int flatIndex = 0;
        int multiplier = 1;

        for (int i = indices.Length - 1; i >= 0; i--)
        {
            flatIndex += indices[i] * multiplier;
            multiplier *= Shape[i];
        }

        return flatIndex;
    }

    /// <summary>
    /// Creates a deep copy of this tensor.
    /// </summary>
    /// <returns>A new tensor with the same shape and values as this tensor.</returns>
    public virtual TensorBase<T> Clone()
    {
        var result = CreateInstance(Shape);
        for (int i = 0; i < Length; i++)
        {
            result._data[i] = _data[i];
        }

        return result;
    }

    /// <summary>
    /// Creates a new instance of the tensor with the specified shape.
    /// </summary>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>A new tensor with the specified shape.</returns>
    protected abstract TensorBase<T> CreateInstance(int[] shape);

    /// <summary>
    /// Creates a new instance of the tensor with the specified data and shape.
    /// </summary>
    /// <param name="data">The data to populate the new tensor with.</param>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>A new tensor with the specified data and shape.</returns>
    protected abstract TensorBase<T> CreateInstance(T[] data, int[] shape);

    /// <summary>
    /// Creates a new instance of the tensor with the specified shape and a different element type.
    /// </summary>
    /// <typeparam name="TResult">The type of elements in the new tensor.</typeparam>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>A new tensor with the specified shape and element type.</returns>
    protected abstract TensorBase<TResult> CreateInstance<TResult>(params int[] shape);

    /// <summary>
    /// Applies a function to each element of the tensor.
    /// </summary>
    /// <typeparam name="TResult">The type of elements in the resulting tensor.</typeparam>
    /// <param name="func">The function to apply to each element.</param>
    /// <returns>A new tensor with the function applied to each element.</returns>
    public TensorBase<TResult> Transform<TResult>(Func<T, TResult> func)
    {
        var result = CreateInstance<TResult>(Shape);
        for (int i = 0; i < Length; i++)
        {
            result._data[i] = func(_data[i]);
        }

        return result;
    }

    /// <summary>
    /// Applies a function to each element of the tensor, providing the element's indices.
    /// </summary>
    /// <typeparam name="TResult">The type of elements in the resulting tensor.</typeparam>
    /// <param name="func">The function to apply to each element, which takes the element value and its indices as parameters.</param>
    /// <returns>A new tensor with the function applied to each element.</returns>
    public TensorBase<TResult> Transform<TResult>(Func<T, int[], TResult> func)
    {
        var result = CreateInstance<TResult>(Shape);
        var indices = new int[Rank];
        for (int i = 0; i < Length; i++)
        {
            GetIndices(i, indices);
            result._data[i] = func(_data[i], indices);
        }

        return result;
    }

    /// <summary>
    /// Converts a flat index to multi-dimensional indices.
    /// </summary>
    /// <param name="flatIndex">The flat index to convert.</param>
    /// <param name="indices">An array to store the resulting indices.</param>
    protected void GetIndices(int flatIndex, int[] indices)
    {
        int remainder = flatIndex;
        for (int i = Rank - 1; i >= 0; i--)
        {
            indices[i] = remainder % Shape[i];
            remainder /= Shape[i];
        }
    }

    /// <summary>
    /// Converts a flat index to multi-dimensional indices and returns them as a new array.
    /// </summary>
    /// <param name="flatIndex">The flat index to convert.</param>
    /// <returns>An array containing the multi-dimensional indices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts a single number (flat index) that represents
    /// a position in the tensor's internal storage back to the multi-dimensional coordinates.
    /// For example, in a 3x3 matrix stored as a flat array, flat index 4 would convert to [1, 1]
    /// (row 1, column 1 in 0-based indexing).</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when flatIndex is negative or exceeds the tensor's length.</exception>
    public int[] GetIndexFromFlat(int flatIndex)
    {
        if (flatIndex < 0 || flatIndex >= Length)
        {
            throw new ArgumentOutOfRangeException(nameof(flatIndex), 
                $"Flat index {flatIndex} is out of range for tensor with {Length} elements.");
        }

        var indices = new int[Rank];
        GetIndices(flatIndex, indices);
        return indices;
    }

    /// <summary>
    /// Returns a string representation of the tensor.
    /// </summary>
    /// <returns>A string representation of the tensor.</returns>
    public override string ToString()
    {
        return $"Tensor<{typeof(T).Name}> with shape [{string.Join(", ", Shape)}]";
    }
}