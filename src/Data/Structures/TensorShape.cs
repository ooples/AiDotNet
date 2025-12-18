namespace AiDotNet.Data.Structures;

/// <summary>
/// Represents the shape of a tensor with dimension information.
/// </summary>
/// <remarks>
/// <para>
/// TensorShape provides a convenient way to work with multi-dimensional
/// tensor dimensions without manually managing integer arrays. It includes
/// utilities for common shape operations and validations.
/// </para>
/// <para><b>For Beginners:</b> A tensor's shape tells you its dimensions:
///
/// Examples:
/// - Scalar: [] (no dimensions)
/// - Vector: [1000] (1000 elements)
/// - Matrix: [28, 28] (28 rows, 28 columns)
/// - RGB Image: [224, 224, 3] (224×224 pixels, 3 color channels)
/// - Batch of images: [32, 224, 224, 3] (32 images)
/// - Sequence: [50, 512] (50 time steps, 512 features)
/// </para>
/// </remarks>
public class TensorShape
{
    private readonly int[] _dimensions;

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    /// <value>
    /// Product of all dimensions. Returns 1 for empty (scalar) tensor.
    /// </value>
    /// <example>
    /// <code>
    /// var shape = new TensorShape(2, 3, 4);
    /// Console.WriteLine(shape.TotalElements); // Output: 24
    /// </code>
    /// </example>
    public int TotalElements { get; private set; }

    /// <summary>
    /// Gets the number of dimensions (rank) of the tensor.
    /// </summary>
    /// <value>
    /// Number of dimensions. 0 for scalar, 1 for vector, 2 for matrix, etc.
    /// </value>
    public int Rank => _dimensions.Length;

    /// <summary>
    /// Gets a read-only view of the dimensions.
    /// </summary>
    /// <value>
    /// Array containing the size of each dimension.
    /// </value>
    public IReadOnlyList<int> Dimensions => Array.AsReadOnly(_dimensions);

    /// <summary>
    /// Initializes a new instance of the TensorShape class.
    /// </summary>
    /// <param name="dimensions">The dimensions of the tensor.</param>
    /// <exception cref="ArgumentException">Thrown when any dimension is negative.</exception>
    /// <example>
    /// <code>
    /// // Create shape for a 28x28 image
    /// var imageShape = new TensorShape(28, 28);
    ///
    /// // Create shape for batch of RGB images
    /// var batchShape = new TensorShape(32, 224, 224, 3);
    /// </code>
    /// </example>
    public TensorShape(params int[] dimensions)
    {
        if (dimensions == null)
            throw new ArgumentNullException(nameof(dimensions));

        _dimensions = new int[dimensions.Length];
        TotalElements = 1;

        for (int i = 0; i < dimensions.Length; i++)
        {
            if (dimensions[i] < 0)
                throw new ArgumentException($"Dimension at index {i} cannot be negative: {dimensions[i]}");

            _dimensions[i] = dimensions[i];
            TotalElements *= dimensions[i];
        }
    }

    
    /// <summary>
    /// Gets the size of a specific dimension.
    /// </summary>
    /// <param name="index">Zero-based index of the dimension.</param>
    /// <returns>The size of the dimension at the specified index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range.</exception>
    public int this[int index]
    {
        get
        {
            if (index < 0 || index >= _dimensions.Length)
                throw new ArgumentOutOfRangeException(nameof(index), $"Dimension index out of range: {index}");
            return _dimensions[index];
        }
    }

    /// <summary>
    /// Determines if two tensor shapes are equal.
    /// </summary>
    /// <param name="other">The other TensorShape to compare.</param>
    /// <returns>True if shapes have identical dimensions, false otherwise.</returns>
    public bool Equals(TensorShape? other)
    {
        if (ReferenceEquals(null, other))
            return false;
        if (ReferenceEquals(this, other))
            return true;

        if (_dimensions.Length != other._dimensions.Length)
            return false;

        for (int i = 0; i < _dimensions.Length; i++)
        {
            if (_dimensions[i] != other._dimensions[i])
                return false;
        }

        return true;
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return Equals(obj as TensorShape);
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        int hash = 17;
        foreach (int dim in _dimensions)
        {
            hash = hash * 31 + dim;
        }
        return hash;
    }

    /// <summary>
    /// Returns a string representation of the shape.
    /// </summary>
    /// <returns>
    /// String representation in format [dim1, dim2, ...].
    /// Empty brackets [] for scalar.
    /// </returns>
    /// <example>
    /// <code>
    /// var shape = new TensorShape(2, 3, 4);
    /// Console.WriteLine(shape.ToString()); // Output: [2, 3, 4]
    /// </code>
    /// </example>
    public override string ToString()
    {
        if (_dimensions.Length == 0)
            return "[]";

        return "[" + string.Join(", ", _dimensions) + "]";
    }

    /// <summary>
    /// Reshapes the tensor if the total number of elements matches.
    /// </summary>
    /// <param name="newDimensions">The new dimensions.</param>
    /// <returns>A new TensorShape with the specified dimensions.</returns>
    /// <exception cref="ArgumentException">Thrown when total elements don't match.</exception>
    /// <example>
    /// <code>
    /// var shape1 = new TensorShape(2, 6);
    /// var shape2 = shape1.Reshape(3, 4); // Valid: 2*6 = 3*4 = 12
    ///
    /// var shape3 = shape1.Reshape(2, 5); // Throws exception: 2*6 != 2*5
    /// </code>
    /// </example>
    public TensorShape Reshape(params int[] newDimensions)
    {
        var newShape = new TensorShape(newDimensions);
        if (newShape.TotalElements != TotalElements)
        {
            throw new ArgumentException(
                $"Cannot reshape: total elements must match. " +
                $"Current: {TotalElements}, Requested: {newShape.TotalElements}");
        }
        return newShape;
    }

    /// <summary>
    /// Checks if the shape is compatible for broadcasting with another shape.
    /// </summary>
    /// <param name="other">The other shape to check compatibility.</param>
    /// <returns>True if shapes can be broadcast together, false otherwise.</returns>
    /// <remarks>
    /// <para><b>Broadcasting rules:</b></para>
    /// 1. Shapes are compatible from right to left
    /// 2. Dimensions are compatible if they are equal or one of them is 1
    /// 3. A shape can be prepended with 1s to match another shape
    ///
    /// <para><b>Examples:</b></para>
    /// - [3, 4] and [3, 1] → broadcast to [3, 4]
    /// - [3, 4] and [1, 4] → broadcast to [3, 4]
    /// - [3, 4] and [4] → broadcast to [3, 4]
    /// - [3, 4] and [5, 4] → incompatible
    /// </remarks>
    public bool IsBroadcastCompatibleWith(TensorShape other)
    {
        if (other == null)
            return false;

        int rankDiff = Math.Abs(Rank - other.Rank);
        int maxRank = Math.Max(Rank, other.Rank);

        for (int i = 1; i <= maxRank; i++)
        {
            int thisDim = i <= Rank ? this[Rank - i] : 1;
            int otherDim = i <= other.Rank ? other[other.Rank - i] : 1;

            if (thisDim != otherDim && thisDim != 1 && otherDim != 1)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Computes the broadcast result shape with another shape.
    /// </summary>
    /// <param name="other">The other shape to broadcast with.</param>
    /// <returns>The broadcast result shape.</returns>
    /// <exception cref="ArgumentException">Thrown when shapes are not broadcast compatible.</exception>
    public TensorShape BroadcastTo(TensorShape other)
    {
        if (!IsBroadcastCompatibleWith(other))
            throw new ArgumentException($"Shapes are not broadcast compatible: {this} and {other}");

        int maxRank = Math.Max(Rank, other.Rank);
        var resultDims = new int[maxRank];

        for (int i = 0; i < maxRank; i++)
        {
            int thisIdx = i - (maxRank - Rank);
            int otherIdx = i - (maxRank - other.Rank);

            int thisDim = thisIdx >= 0 ? this[thisIdx] : 1;
            int otherDim = otherIdx >= 0 ? other[otherIdx] : 1;

            resultDims[i] = Math.Max(thisDim, otherDim);
        }

        return new TensorShape(resultDims);
    }

    /// <summary>
    /// Implicit conversion from int array to TensorShape.
    /// </summary>
    /// <param name="dimensions">Array of dimensions.</param>
    public static implicit operator TensorShape(int[] dimensions)
    {
        return new TensorShape(dimensions);
    }

    /// <summary>
    /// Equality operator.
    /// </summary>
    public static bool operator ==(TensorShape? left, TensorShape? right)
    {
        return Equals(left, right);
    }

    /// <summary>
    /// Inequality operator.
    /// </summary>
    public static bool operator !=(TensorShape? left, TensorShape? right)
    {
        return !Equals(left, right);
    }

    /// <summary>
    /// Creates a scalar shape (no dimensions).
    /// </summary>
    /// <returns>A scalar TensorShape.</returns>
    public static TensorShape Scalar => new TensorShape();

    /// <summary>
    /// Creates a vector shape with the specified length.
    /// </summary>
    /// <param name="length">The vector length.</param>
    /// <returns>A vector TensorShape.</returns>
    public static TensorShape Vector(int length) => new TensorShape(length);

    /// <summary>
    /// Creates a matrix shape with the specified dimensions.
    /// </summary>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <returns>A matrix TensorShape.</returns>
    public static TensorShape Matrix(int rows, int cols) => new TensorShape(rows, cols);

    /// <summary>
    /// Creates a 3D tensor shape.
    /// </summary>
    /// <param name="dim0">First dimension.</param>
    /// <param name="dim1">Second dimension.</param>
    /// <param name="dim2">Third dimension.</param>
    /// <returns>A 3D TensorShape.</returns>
    public static TensorShape Tensor3D(int dim0, int dim1, int dim2) => new TensorShape(dim0, dim1, dim2);

    /// <summary>
    /// Creates a 4D tensor shape (common for image batches).
    /// </summary>
    /// <param name="batch">Batch size.</param>
    /// <param name="height">Height.</param>
    /// <param name="width">Width.</param>
    /// <param name="channels">Number of channels.</param>
    /// <returns>A 4D TensorShape.</returns>
    public static TensorShape ImageBatch(int batch, int height, int width, int channels)
        => new TensorShape(batch, height, width, channels);
}