namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Represents a multi-dimensional array of numeric values used in machine learning and AI computations.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor elements (e.g., float, double, int).</typeparam>
/// <remarks>
/// A tensor is a mathematical object that can represent data in multiple dimensions. 
/// Think of it as a container that can hold numbers in an organized way:
/// - A 1D tensor is like a list of numbers (a vector)
/// - A 2D tensor is like a table of numbers (a matrix)
/// - A 3D tensor is like a cube of numbers
/// - And so on for higher dimensions
/// 
/// Tensors are fundamental building blocks for many AI algorithms, especially in neural networks.
/// </remarks>
public class Tensor<T> : IEnumerable<T>
{
    private readonly Vector<T> _data;
    private readonly int[] _dimensions;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the dimensions of the tensor as an array of integers.
    /// </summary>
    /// <remarks>
    /// The shape describes how many elements exist in each dimension.
    /// For example, a shape of [2, 3] means a 2×3 matrix (2 rows, 3 columns).
    /// </remarks>
    public int[] Shape => _dimensions;

    /// <summary>
    /// Gets the number of dimensions (axes) in the tensor.
    /// </summary>
    /// <remarks>
    /// For example:
    /// - A vector has rank 1
    /// - A matrix has rank 2
    /// - A 3D array has rank 3
    /// </remarks>
    public int Rank => _dimensions.Length;

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    /// <remarks>
    /// This is the product of all dimension sizes. For example, a 2×3×4 tensor has 24 elements.
    /// </remarks>
    public int Length => _data.Length;

    /// <summary>
    /// Creates a new tensor with the specified dimensions, initialized with default values.
    /// </summary>
    /// <param name="dimensions">An array specifying the size of each dimension.</param>
    /// <remarks>
    /// For example, new Tensor&lt;float&gt;([2, 3]) creates a 2×3 matrix of zeros.
    /// </remarks>
    public Tensor(int[] dimensions)
    {
        _dimensions = dimensions;
        int totalSize = dimensions.Aggregate(1, (a, b) => a * b);
        _data = new Vector<T>(totalSize);
    }

    /// <summary>
    /// Creates a new tensor with the specified dimensions and pre-populated data.
    /// </summary>
    /// <param name="dimensions">An array specifying the size of each dimension.</param>
    /// <param name="data">A vector containing the data to populate the tensor with.</param>
    /// <exception cref="VectorLengthMismatchException">Thrown when the data length doesn't match the product of dimensions.</exception>
    /// <remarks>
    /// The data is stored in row-major order, meaning that the rightmost indices vary fastest.
    /// </remarks>
    public Tensor(int[] dimensions, Vector<T> data)
    {
        _dimensions = dimensions;
        int totalSize = dimensions.Aggregate(1, (a, b) => a * b);
        VectorValidator.ValidateLength(data, totalSize);
        _data = data;
    }

    /// <summary>
    /// Returns an enumerator that iterates through all elements in the tensor.
    /// </summary>
    /// <returns>An enumerator for the tensor's elements.</returns>
    public IEnumerator<T> GetEnumerator()
    {
        return _data.GetEnumerator();
    }

    /// <summary>
    /// Returns an enumerator that iterates through all elements in the tensor.
    /// </summary>
    /// <returns>An enumerator for the tensor's elements.</returns>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    /// <summary>
    /// Gets or sets the element at the specified indices.
    /// </summary>
    /// <param name="indices">The indices specifying the position of the element.</param>
    /// <returns>The element at the specified position.</returns>
    /// <remarks>
    /// For example, in a 2D tensor (matrix), you can access elements using: tensor[row, column]
    /// </remarks>
    public T this[params int[] indices]
    {
        get => _data[GetFlatIndex(indices)];
        set => _data[GetFlatIndex(indices)] = value;
    }

    /// <summary>
    /// Gets a slice of the tensor's data as a vector.
    /// </summary>
    /// <param name="start">The starting index in the flattened data.</param>
    /// <param name="length">The number of elements to include in the slice.</param>
    /// <returns>A vector containing the requested slice of data.</returns>
    /// <remarks>
    /// This method accesses the underlying flat storage of the tensor directly.
    /// </remarks>
    public Vector<T> GetSlice(int start, int length)
    {
        return _data.Slice(start, length);
    }

    /// <summary>
    /// Sets a slice of the tensor's data from a vector.
    /// </summary>
    /// <param name="start">The starting index in the flattened data.</param>
    /// <param name="slice">The vector containing the data to set.</param>
    /// <remarks>
    /// This method modifies the underlying flat storage of the tensor directly.
    /// </remarks>
    public void SetSlice(int start, Vector<T> slice)
    {
        for (int i = 0; i < slice.Length; i++)
        {
            _data[start + i] = slice[i];
        }
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    /// <param name="left">The first tensor.</param>
    /// <param name="right">The second tensor.</param>
    /// <returns>A new tensor containing the sum of the two tensors.</returns>
    /// <remarks>
    /// Both tensors must have the same shape for this operation to work.
    /// </remarks>
    public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Adds a vector to a tensor element-wise.
    /// </summary>
    /// <param name="left">The tensor.</param>
    /// <param name="right">The vector to add.</param>
    /// <returns>A new tensor containing the result of the addition.</returns>
    /// <remarks>
    /// The vector is typically broadcast to match the tensor's dimensions.
    /// </remarks>
    public static Tensor<T> operator +(Tensor<T> left, Vector<T> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Multiplies two tensors.
    /// </summary>
    /// <param name="left">The first tensor.</param>
    /// <param name="right">The second tensor.</param>
    /// <returns>A new tensor containing the result of the multiplication.</returns>
    /// <remarks>
    /// This performs tensor multiplication according to the rules of linear algebra.
    /// </remarks>
    public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Sets a slice of the tensor at the specified index along the first dimension.
    /// </summary>
    /// <param name="index">The index along the first dimension.</param>
    /// <param name="slice">The tensor slice to set.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the index is out of range.</exception>
    /// <exception cref="TensorShapeMismatchException">Thrown when the slice shape doesn't match the expected shape.</exception>
    /// <remarks>
    /// For example, in a 3D tensor of shape [4, 5, 6], setting a slice at index 2 would replace
    /// the 2D slice at that position with a tensor of shape [5, 6].
    /// </remarks>
    public void SetSlice(int index, Tensor<T> slice)
    {
        if (index < 0 || index >= Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        TensorValidator.ValidateShape(slice, [..Shape.Skip(1)]);

        int sliceSize = slice.Length;
        int offset = index * sliceSize;

        for (int i = 0; i < sliceSize; i++)
        {
            _data[offset + i] = slice._data[i];
        }
    }

    /// <summary>
    /// Sets a slice of the tensor at the specified index along the specified dimension.
    /// </summary>
    /// <param name="dimension">The dimension along which to set the slice.</param>
    /// <param name="index">The index along the specified dimension.</param>
    /// <param name="slice">The tensor slice to set.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the dimension or index is out of range.</exception>
    /// <exception cref="ArgumentException">Thrown when the slice shape doesn't match the expected shape.</exception>
    /// <remarks>
    /// This is a more general version of the SetSlice method that allows setting slices along any dimension.
    /// </remarks>
    public void SetSlice(int dimension, int index, Tensor<T> slice)
    {
        if (dimension < 0 || dimension >= Rank)
            throw new ArgumentOutOfRangeException(nameof(dimension), "Dimension is out of range.");

        if (index < 0 || index >= Shape[dimension])
            throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range for the specified dimension.");

        // Check if the slice shape matches the expected shape
        int[] expectedSliceShape = new int[Rank - 1];
        for (int i = 0, j = 0; i < Rank; i++)
        {
            if (i != dimension)
                expectedSliceShape[j++] = Shape[i];
        }

        TensorValidator.ValidateShape(slice, expectedSliceShape);

        // Calculate the stride for the specified dimension
        int stride = 1;
        for (int i = dimension + 1; i < Rank; i++)
            stride *= Shape[i];

        // Calculate the starting index in the flat array
        int startIndex = index * stride;
        for (int i = 0; i < dimension; i++)
            startIndex *= Shape[i];

        // Copy the slice data into the tensor
        for (int i = 0; i < slice.Length; i++)
        {
            int targetIndex = startIndex + (i % stride) + i / stride * stride * Shape[dimension];
            _data[targetIndex] = slice._data[i];
        }
    }

    /// <summary>
    /// Creates a deep copy of this tensor.
    /// </summary>
    /// <returns>A new tensor with the same shape and values as this tensor.</returns>
    /// <remarks>
    /// Changes to the returned tensor will not affect the original tensor.
    /// </remarks>
    public Tensor<T> Copy()
    {
        var newData = new Vector<T>(_data.Length);
        Array.Copy(_data, newData, _data.Length);

        return new Tensor<T>(Shape, newData);
    }

    /// <summary>
    /// Performs element-wise multiplication of two tensors.
    /// </summary>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise product of the input tensors.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    /// <remarks>
    /// This operation multiplies corresponding elements from both tensors and returns a new tensor
    /// with the same shape containing the results.
    /// </remarks>
    public static Tensor<T> ElementwiseMultiply(Tensor<T> a, Tensor<T> b)
    {
        TensorValidator.ValidateShape(a, b.Shape);

        Tensor<T> result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result._data[i] = NumOps.Multiply(a._data[i], b._data[i]);
        }

        return result;
    }

    /// <summary>
    /// Fills the entire tensor with a specified value.
    /// </summary>
    /// <param name="value">The value to fill the tensor with.</param>
    /// <remarks>
    /// This method replaces all elements in the tensor with the specified value.
    /// </remarks>
    public void Fill(T value)
    {
        for (int i = 0; i < _data.Length; i++)
        {
            _data[i] = value;
        }
    }

    /// <summary>
    /// Sets the value at the specified flat index in the tensor.
    /// </summary>
    /// <param name="flatIndex">The flat (linear) index into the tensor's data.</param>
    /// <param name="value">The value to set.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the flat index is out of range.</exception>
    /// <remarks>
    /// The flat index treats the tensor as a one-dimensional array, regardless of its actual shape.
    /// </remarks>
    public void SetFlatIndex(int flatIndex, T value)
    {
        if (flatIndex < 0 || flatIndex >= _data.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(flatIndex), "Flat index is out of range.");
        }

        _data[flatIndex] = value;
    }

    /// <summary>
    /// Gets the complex value from a tensor at the specified index.
    /// </summary>
    /// <param name="tensor">The tensor to retrieve the value from.</param>
    /// <param name="index">The index of the value to retrieve.</param>
    /// <returns>The value at the specified index as a complex number.</returns>
    /// <remarks>
    /// If the value is already a Complex&lt;T&gt;, it is returned directly.
    /// Otherwise, a new Complex&lt;T&gt; is created with the value as the real part
    /// and zero as the imaginary part.
    /// </remarks>
    public static Complex<T> GetComplex(Tensor<T> tensor, int index)
    {
        var value = tensor[index];
        
        // If the value is already a Complex<T>, return it
        if (value is Complex<T> complex)
        {
            return complex;
        }
        
        // Otherwise, create a new Complex<T> with the value as the real part
        // and zero as the imaginary part
        return new Complex<T>(value, NumOps.Zero);
    }

    /// <summary>
    /// Creates a tensor with random values of the specified dimensions.
    /// </summary>
    /// <param name="dimensions">The dimensions of the tensor to create.</param>
    /// <returns>A new tensor filled with random values between 0 and 1.</returns>
    /// <exception cref="ArgumentException">Thrown when dimensions are null or empty.</exception>
    /// <remarks>
    /// The random values are generated using the System.Random class and converted to type T.
    /// </remarks>
    public static Tensor<T> CreateRandom(params int[] dimensions)
    {
        if (dimensions == null || dimensions.Length == 0)
            throw new ArgumentException("Dimensions cannot be null or empty.", nameof(dimensions));

        var tensor = new Tensor<T>(dimensions);
        var random = new Random();
        var numOps = MathHelper.GetNumericOperations<T>();

        // Flatten the tensor into a 1D array for easier iteration
        var flattenedSize = dimensions.Aggregate(1, (a, b) => a * b);
        for (int i = 0; i < flattenedSize; i++)
        {
            // Generate a random value between 0 and 1
            var randomValue = numOps.FromDouble(random.NextDouble());
    
            // Calculate the multi-dimensional index
            var index = new int[dimensions.Length];
            var remaining = i;
            for (int j = dimensions.Length - 1; j >= 0; j--)
            {
                index[j] = remaining % dimensions[j];
                remaining /= dimensions[j];
            }

            // Set the random value in the tensor using the indexer
            tensor[index] = randomValue;
        }

        return tensor;
    }

    /// <summary>
    /// Extracts a sub-tensor by fixing the first N dimensions to specific indices.
    /// </summary>
    /// <param name="indices">The indices to fix for the first dimensions.</param>
    /// <returns>A tensor with reduced dimensionality.</returns>
    /// <exception cref="ArgumentException">Thrown when the number of indices exceeds tensor dimensions.</exception>
    /// <remarks>
    /// This method creates a view into the tensor with fewer dimensions by fixing specific indices
    /// for the first dimensions. For example, if you have a 3D tensor and provide one index,
    /// you'll get a 2D tensor (matrix) at that index.
    /// </remarks>
    public Tensor<T> SubTensor(params int[] indices)
    {
        if (indices.Length > Shape.Length)
            throw new ArgumentException("Number of indices exceeds tensor dimensions.");

        int[] newShape = new int[Shape.Length - indices.Length];
        for (int i = 0; i < newShape.Length; i++)
        {
            newShape[i] = Shape[indices.Length + i];
        }

        Tensor<T> subTensor = new Tensor<T>(newShape);

        int[] currentIndices = new int[Shape.Length];
        Array.Copy(indices, currentIndices, indices.Length);

        CopySubTensorData(this, subTensor, currentIndices, indices.Length);

        return subTensor;
    }

    /// <summary>
    /// Helper method to recursively copy data from a source tensor to a destination sub-tensor.
    /// </summary>
    /// <param name="source">The source tensor to copy from.</param>
    /// <param name="destination">The destination tensor to copy to.</param>
    /// <param name="currentIndices">The current indices being processed.</param>
    /// <param name="fixedDimensions">The number of dimensions that are fixed.</param>
    private static void CopySubTensorData(Tensor<T> source, Tensor<T> destination, int[] currentIndices, int fixedDimensions)
    {
        if (fixedDimensions == source.Shape.Length)
        {
            destination[[]] = source[currentIndices];
            return;
        }

        for (int i = 0; i < source.Shape[fixedDimensions]; i++)
        {
            currentIndices[fixedDimensions] = i;
            CopySubTensorData(source, destination, currentIndices, fixedDimensions + 1);
        }
    }

    /// <summary>
    /// Creates an empty tensor with no dimensions.
    /// </summary>
    /// <returns>An empty tensor.</returns>
    public static Tensor<T> Empty()
    {
        return new Tensor<T>([]);
    }

    /// <summary>
    /// Extracts a 2D slice from a 2D tensor.
    /// </summary>
    /// <param name="startRow">The starting row index (inclusive).</param>
    /// <param name="startCol">The starting column index (inclusive).</param>
    /// <param name="endRow">The ending row index (exclusive).</param>
    /// <param name="endCol">The ending column index (exclusive).</param>
    /// <returns>A new tensor containing the specified slice.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the tensor is not 2D.</exception>
    /// <exception cref="ArgumentException">Thrown when slice parameters are invalid.</exception>
    /// <remarks>
    /// This method works only on 2D tensors (matrices) and extracts a rectangular region.
    /// </remarks>
    public Tensor<T> Slice(int startRow, int startCol, int endRow, int endCol)
    {
        if (this.Rank != 2)
        {
            throw new InvalidOperationException("This Slice method is only applicable for 2D tensors.");
        }

        if (startRow < 0 || startCol < 0 || endRow > this.Shape[0] || endCol > this.Shape[1] || startRow >= endRow || startCol >= endCol)
        {
            throw new ArgumentException("Invalid slice parameters.");
        }

        int newRows = endRow - startRow;
        int newCols = endCol - startCol;
        int[] newShape = [newRows, newCols];

        Tensor<T> result = new Tensor<T>(newShape);

        for (int i = 0; i < newRows; i++)
        {
            for (int j = 0; j < newCols; j++)
            {
                result[i, j] = this[startRow + i, startCol + j];
            }
        }

        return result;
    }

    /// <summary>
    /// Extracts a slice along the first dimension of the tensor.
    /// </summary>
    /// <param name="index">The index to slice at.</param>
    /// <returns>A tensor with one fewer dimension than the original.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the index is out of range.</exception>
    /// <remarks>
    /// This method reduces the dimensionality of the tensor by fixing the first dimension
    /// to the specified index. For example, slicing a 3D tensor gives a 2D tensor.
    /// </remarks>
    public Tensor<T> Slice(int index)
    {
        if (index < 0 || index >= Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        int[] newShape = Shape.Skip(1).ToArray();
        int sliceSize = newShape.Aggregate(1, (a, b) => a * b);
        int offset = index * sliceSize;

        var sliceData = new Vector<T>(sliceSize);
        Array.Copy(_data, offset, sliceData, 0, sliceSize);

        return new Tensor<T>(newShape, sliceData);
    }

    /// <summary>
    /// Extracts a slice of the tensor along the specified axis.
    /// </summary>
    /// <param name="axis">The axis along which to slice.</param>
    /// <param name="start">The starting index of the slice.</param>
    /// <param name="end">The ending index of the slice (exclusive). If null, slices to the end of the axis.</param>
    /// <returns>A new tensor containing the specified slice.</returns>
    /// <exception cref="ArgumentException">Thrown when axis or indices are invalid.</exception>
    /// <remarks>
    /// This method creates a new tensor that is a subset of the original tensor along the specified axis.
    /// </remarks>
    public Tensor<T> Slice(int axis, int start, int? end = null)
    {
        if (axis < 0 || axis >= Rank)
            throw new ArgumentException($"Invalid axis. Must be between 0 and {Rank - 1}.");

        int axisSize = Shape[axis];
        int actualEnd = end ?? axisSize;
        if (start < 0 || start >= axisSize || actualEnd <= start || actualEnd > axisSize)
            throw new ArgumentException("Invalid start or end index for slicing.");

        int sliceSize = actualEnd - start;
        int[] newShape = new int[Rank];
        Array.Copy(Shape, newShape, Rank);
        newShape[axis] = sliceSize;

        Tensor<T> result = new Tensor<T>(newShape);

        int[] sourceIndices = new int[Rank];
        int[] destIndices = new int[Rank];

        void SliceRecursive(int depth)
        {
            if (depth == Rank)
            {
                result[destIndices] = this[sourceIndices];
                return;
            }

            int limit = depth == axis ? sliceSize : Shape[depth];
            for (int i = 0; i < limit; i++)
            {
                sourceIndices[depth] = depth == axis ? i + start : i;
                destIndices[depth] = i;
                SliceRecursive(depth + 1);
            }
        }

        SliceRecursive(0);
        return result;
    }

    /// <summary>
    /// Computes the mean values along the specified axis.
    /// </summary>
    /// <param name="axis">The axis along which to compute the mean.</param>
    /// <returns>A new tensor with the specified axis removed, containing mean values.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the axis is invalid.</exception>
    /// <remarks>
    /// The resulting tensor has one fewer dimension than the original tensor.
    /// </remarks>
    public Tensor<T> MeanOverAxis(int axis)
    {
        if (axis < 0 || axis >= Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        var newShape = Shape.ToList();
        newShape.RemoveAt(axis);
        var result = new Tensor<T>([.. newShape]);
        int axisSize = Shape[axis];

        // Iterate over all elements, grouping by the non-axis dimensions
        for (int i = 0; i < _data.Length; i += axisSize)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < axisSize; j++)
            {
                sum = NumOps.Add(sum, _data[i + j]);
            }

            result._data[i / axisSize] = NumOps.Divide(sum, NumOps.FromDouble(axisSize));
        }

        return result;
    }

    /// <summary>
    /// Finds the maximum values along the specified axis.
    /// </summary>
    /// <param name="axis">The axis along which to find maximum values.</param>
    /// <returns>A new tensor with the specified axis removed, containing maximum values.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the axis is invalid.</exception>
    /// <remarks>
    /// The resulting tensor has one fewer dimension than the original tensor.
    /// </remarks>
    public Tensor<T> MaxOverAxis(int axis)
    {
        if (axis < 0 || axis >= Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        var newShape = Shape.ToList();
        newShape.RemoveAt(axis);
        var result = new Tensor<T>([.. newShape]);
        int axisSize = Shape[axis];

        // Iterate over all elements, grouping by the non-axis dimensions
        for (int i = 0; i < _data.Length; i += axisSize)
        {
            T max = _data[i];
            for (int j = 1; j < axisSize; j++)
            {
                if (NumOps.GreaterThan(_data[i + j], max))
                    max = _data[i + j];
            }

            result._data[i / axisSize] = max;
        }

        return result;
    }

    /// <summary>
    /// Computes the sum along the specified axis.
    /// </summary>
    /// <param name="axis">The axis along which to compute the sum.</param>
    /// <returns>A new tensor with the specified axis removed, containing sum values.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the axis is invalid.</exception>
    /// <remarks>
    /// The resulting tensor has one fewer dimension than the original tensor.
    /// </remarks>
    public Tensor<T> SumOverAxis(int axis)
    {
        if (axis < 0 || axis >= Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        var newShape = Shape.ToList();
        newShape.RemoveAt(axis);
        var result = new Tensor<T>([.. newShape]);
        int axisSize = Shape[axis];

        // Iterate over all elements, grouping by the non-axis dimensions
        for (int i = 0; i < _data.Length; i += axisSize)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < axisSize; j++)
            {
                sum = NumOps.Add(sum, _data[i + j]);
            }

            result._data[i / axisSize] = sum;
        }

        return result;
    }

    /// <summary>
    /// Sets a sub-tensor at the specified indices.
    /// </summary>
    /// <param name="indices">The starting indices for the sub-tensor.</param>
    /// <param name="subTensor">The sub-tensor to insert.</param>
    /// <exception cref="ArgumentException">Thrown when indices length doesn't match sub-tensor rank.</exception>
    /// <remarks>
    /// This method replaces a portion of the tensor with the provided sub-tensor.
    /// </remarks>
    public void SetSubTensor(int[] indices, Tensor<T> subTensor)
    {
        if (indices.Length != subTensor.Rank)
            throw new ArgumentException("Number of indices must match the rank of the sub-tensor.");

        int[] currentIndices = new int[Rank];
        Array.Copy(indices, currentIndices, indices.Length);

        SetSubTensorRecursive(subTensor, currentIndices, 0);
    }

        /// <summary>
    /// Recursively sets values from a sub-tensor into this tensor at the specified position.
    /// </summary>
    /// <param name="subTensor">The smaller tensor whose values will be copied into this tensor.</param>
    /// <param name="indices">The current position in this tensor where values should be placed.</param>
    /// <param name="dimension">The current dimension being processed in the recursion.</param>
    /// <remarks>
    /// For beginners: This is a helper method that works through each dimension of the sub-tensor
    /// one by one, copying its values to the correct positions in the larger tensor.
    /// 
    /// Think of it like placing a small sticker (sub-tensor) onto the correct position of a larger
    /// sheet of paper (the main tensor). The indices tell us where to start placing the sticker,
    /// and this method makes sure each part of the sticker goes to the right spot.
    /// 
    /// The recursion works by:
    /// 1. If we've processed all dimensions, copy the single value
    /// 2. Otherwise, loop through the current dimension and recursively process the next dimension
    /// </remarks>
    private void SetSubTensorRecursive(Tensor<T> subTensor, int[] indices, int dimension)
    {
        if (dimension == subTensor.Rank)
        {
            this[indices] = subTensor._data[0];
            return;
        }

        for (int i = 0; i < subTensor.Shape[dimension]; i++)
        {
            indices[indices.Length - subTensor.Rank + dimension] = i;
            SetSubTensorRecursive(subTensor, indices, dimension + 1);
        }
    }

    /// <summary>
    /// Computes the dot product of this tensor with another tensor.
    /// </summary>
    /// <param name="other">The other tensor.</param>
    /// <returns>The scalar dot product result.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    /// <remarks>
    /// Both tensors must have identical shapes for this operation.
    /// </remarks>
    public T DotProduct(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for dot product.");

        T result = NumOps.Zero;
        for (int i = 0; i < _data.Length; i++)
        {
            result = NumOps.Add(result, NumOps.Multiply(_data[i], other._data[i]));
        }

        return result;
    }

    /// <summary>
    /// Scales the tensor by multiplying each element by a factor.
    /// </summary>
    /// <param name="factor">The scaling factor.</param>
    /// <returns>A new tensor with scaled values.</returns>
    /// <remarks>
    /// This method creates a new tensor and does not modify the original.
    /// </remarks>
    public Tensor<T> Scale(T factor)
    {
        var result = new Tensor<T>(this.Shape);
    
        // Apply scaling to each element in the tensor
        for (int i = 0; i < this.Length; i++)
        {
            result[i] = NumOps.Multiply(this[i], factor);
        }
    
        return result;
    }

    /// <summary>
    /// Scales the tensor in-place by multiplying each element by a factor.
    /// </summary>
    /// <param name="factor">The scaling factor.</param>
    /// <remarks>
    /// This method modifies the original tensor directly.
    /// </remarks>
    public void ScaleInPlace(T factor)
    {
        for (int i = 0; i < this.Length; i++)
        {
            this[i] = NumOps.Multiply(this[i], factor);
        }
    }

    /// <summary>
    /// Stacks multiple tensors along the specified axis.
    /// </summary>
    /// <param name="tensors">The array of tensors to stack.</param>
    /// <param name="axis">The axis along which to stack the tensors.</param>
    /// <returns>A new tensor with an additional dimension.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes or the axis is invalid.</exception>
    /// <remarks>
    /// All input tensors must have the same shape. The resulting tensor will have rank+1 dimensions.
    /// </remarks>
    public static Tensor<T> Stack(Tensor<T>[] tensors, int axis = 0)
    {
        if (tensors == null || tensors.Length == 0)
            throw new ArgumentException("At least one tensor must be provided for stacking.");

        int rank = tensors[0].Rank;
        if (axis < 0 || axis > rank)
            throw new ArgumentException($"Invalid axis. Must be between 0 and {rank}.");

        // Validate that all tensors have the same shape
        for (int i = 1; i < tensors.Length; i++)
        {
            if (!tensors[i].Shape.SequenceEqual(tensors[0].Shape))
                throw new ArgumentException("All tensors must have the same shape for stacking.");
        }

        // Calculate the new shape
        int[] newShape = new int[rank + 1];
        int shapeIndex = 0;
        for (int i = 0; i <= rank; i++)
        {
            if (i == axis)
            {
                newShape[i] = tensors.Length;
            }
            else
            {
                newShape[i] = tensors[0].Shape[shapeIndex];
                shapeIndex++;
            }
        }

        // Create the new tensor
        Tensor<T> result = new Tensor<T>(newShape);

        // Copy data from input tensors to the result tensor
        int[] indices = new int[rank + 1];
        for (int i = 0; i < tensors.Length; i++)
        {
            indices[axis] = i;
            CopyTensorToStack(tensors[i], result, indices, axis);
        }

        return result;
    }

        /// <summary>
    /// Copies data from a source tensor into a destination tensor that is being constructed by the Stack operation.
    /// </summary>
    /// <param name="source">The tensor to copy data from.</param>
    /// <param name="destination">The tensor to copy data to.</param>
    /// <param name="destIndices">The current indices in the destination tensor where data should be placed.</param>
    /// <param name="stackAxis">The axis along which tensors are being stacked.</param>
    /// <remarks>
    /// For beginners: This helper method is used when combining multiple tensors into a single larger tensor.
    /// 
    /// When stacking tensors (like stacking sheets of paper), we need to carefully copy each value from 
    /// the original tensors to the correct position in the new combined tensor. This method handles that
    /// copying process by recursively traversing through all dimensions of the tensors.
    /// 
    /// For example, when stacking 3 images of size [28x28] along a new first dimension, 
    /// the result will be a tensor of shape [3x28x28].
    /// </remarks>
    private static void CopyTensorToStack(Tensor<T> source, Tensor<T> destination, int[] destIndices, int stackAxis)
    {
        int[] _sourceIndices = new int[source.Rank];

        void CopyRecursive(int depth)
        {
            if (depth == source.Rank)
            {
                destination[destIndices] = source[_sourceIndices];
                return;
            }

            int destDepth = depth < stackAxis ? depth : depth + 1;
            for (int i = 0; i < source.Shape[depth]; i++)
            {
                _sourceIndices[depth] = i;
                destIndices[destDepth] = i;
                CopyRecursive(depth + 1);
            }
        }

        CopyRecursive(0);
    }

    /// <summary>
    /// Concatenates multiple tensors along the specified axis.
    /// </summary>
    /// <param name="tensors">The array of tensors to concatenate.</param>
    /// <param name="axis">The axis along which to concatenate the tensors.</param>
    /// <returns>A new tensor with the same rank as the input tensors.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have incompatible shapes or the axis is invalid.</exception>
    /// <remarks>
    /// All input tensors must have the same shape except along the concatenation axis.
    /// </remarks>
    public static Tensor<T> Concatenate(Tensor<T>[] tensors, int axis)
    {
        if (tensors == null || tensors.Length == 0)
            throw new ArgumentException("At least one tensor must be provided for concatenation.");

        int rank = tensors[0].Rank;
        if (axis < 0 || axis >= rank)
            throw new ArgumentException($"Invalid axis. Must be between 0 and {rank - 1}.");

        // Validate that all tensors have the same shape except for the concatenation axis
        for (int i = 1; i < tensors.Length; i++)
        {
            if (tensors[i].Rank != rank)
                throw new ArgumentException("All tensors must have the same rank.");

            for (int j = 0; j < rank; j++)
            {
                if (j != axis && tensors[i].Shape[j] != tensors[0].Shape[j])
                    throw new ArgumentException("All tensors must have the same shape except for the concatenation axis.");
            }
        }

        // Calculate the new shape
        int[] newShape = new int[rank];
        Array.Copy(tensors[0].Shape, newShape, rank);
        for (int i = 1; i < tensors.Length; i++)
        {
            newShape[axis] += tensors[i].Shape[axis];
        }

        // Create the new tensor
        Tensor<T> result = new Tensor<T>(newShape);

        // Copy data from input tensors to the result tensor
        int offset = 0;
        for (int i = 0; i < tensors.Length; i++)
        {
            CopyTensorSlice(tensors[i], result, axis, offset);
            offset += tensors[i].Shape[axis];
        }

        return result;
    }

        /// <summary>
    /// Copies a slice from a source tensor to a destination tensor along a specified axis.
    /// </summary>
    /// <param name="source">The tensor to copy data from.</param>
    /// <param name="destination">The tensor to copy data to.</param>
    /// <param name="axis">The axis along which to copy the slice.</param>
    /// <param name="destinationOffset">The offset in the destination tensor where the slice should be placed.</param>
    /// <remarks>
    /// This is a helper method used by the Concatenate method to combine multiple tensors.
    /// It recursively traverses the tensors to copy values from the source to the correct position in the destination.
    /// </remarks>
    private static void CopyTensorSlice(Tensor<T> source, Tensor<T> destination, int axis, int destinationOffset)
    {
        int[] sourceIndices = new int[source.Rank];
        int[] destIndices = new int[destination.Rank];

        void CopyRecursive(int depth)
        {
            if (depth == source.Rank)
            {
                destination[destIndices] = source[sourceIndices];
                return;
            }

            int limit = depth == axis ? source.Shape[depth] : destination.Shape[depth];
            for (int i = 0; i < limit; i++)
            {
                sourceIndices[depth] = i;
                destIndices[depth] = depth == axis ? i + destinationOffset : i;
                CopyRecursive(depth + 1);
            }
        }

        CopyRecursive(0);
    }

    /// <summary>
    /// Extracts a slice from the tensor at the specified batch index.
    /// </summary>
    /// <param name="batchIndex">The index of the batch to extract.</param>
    /// <returns>A new tensor containing the extracted slice.</returns>
    /// <remarks>
    /// For beginners: In machine learning, data is often organized in batches.
    /// This method extracts a single item (slice) from a batch of data.
    /// 
    /// For example, if you have a tensor with shape [32, 784] representing 32 images
    /// with 784 features each, GetSlice(5) would return the 6th image (index 5)
    /// as a tensor with shape [784].
    /// 
    /// This method assumes the first dimension is the batch dimension.
    /// </remarks>
    public Tensor<T> GetSlice(int batchIndex)
    {
        int[] newShape = new int[Shape.Length - 1];
        Array.Copy(Shape, 1, newShape, 0, Shape.Length - 1);
    
        Tensor<T> slice = new Tensor<T>(newShape);
    
        int sliceSize = slice.Length;
        Array.Copy(_data, batchIndex * sliceSize, slice._data, 0, sliceSize);
    
        return slice;
    }

    /// <summary>
    /// Creates a tensor with all elements initialized to the specified value.
    /// </summary>
    /// <param name="shape">The shape of the tensor to create.</param>
    /// <param name="value">The value to fill the tensor with.</param>
    /// <returns>A new tensor filled with the specified value.</returns>
    /// <remarks>
    /// This is a convenience method for creating pre-initialized tensors.
    /// </remarks>
    public static Tensor<T> CreateDefault(int[] shape, T value)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor._data[i] = value;
        }

        return tensor;
    }

    /// <summary>
    /// Subtracts another tensor from this tensor element-wise.
    /// </summary>
    /// <param name="other">The tensor to subtract.</param>
    /// <returns>A new tensor containing the result of the subtraction.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    /// <remarks>
    /// Both tensors must have identical shapes for this operation.
    /// </remarks>
    public Tensor<T> Subtract(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for subtraction.");

        var result = new Tensor<T>(Shape);
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = ops.Subtract(_data[i], other._data[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs element-wise multiplication with another tensor.
    /// </summary>
    /// <param name="other">The tensor to multiply with.</param>
    /// <returns>A new tensor containing the element-wise product.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have different dimensions.</exception>
    /// <remarks>
    /// Both tensors must have identical shapes for this operation.
    /// </remarks>
    public Tensor<T> ElementwiseMultiply(Tensor<T> other)
    {
        if (!_dimensions.SequenceEqual(other._dimensions))
            throw new ArgumentException("Tensors must have the same dimensions for element-wise multiplication.");

        Vector<T> result = _data.PointwiseMultiply(other._data);
        return new Tensor<T>(_dimensions, result);
    }

        /// <summary>
    /// Performs element-wise multiplication with broadcasting support.
    /// </summary>
    /// <param name="other">The tensor to multiply with.</param>
    /// <returns>A new tensor containing the element-wise product.</returns>
    /// <remarks>
    /// For beginners: This method multiplies each element in this tensor with the corresponding element in the other tensor.
    /// 
    /// Broadcasting allows tensors of different shapes to be multiplied together by automatically expanding
    /// smaller dimensions to match larger ones. For example, you can multiply a 3×4 tensor with a 1×4 tensor
    /// (which will be treated as if it were repeated 3 times).
    /// 
    /// This is particularly useful in machine learning when applying the same operation across multiple
    /// data points or features.
    /// </remarks>
    public Tensor<T> PointwiseMultiply(Tensor<T> other)
    {
        if (this.Shape.SequenceEqual(other.Shape))
        {
            // Simple case: tensors have the same shape
            var result = new Tensor<T>(this.Shape);
            for (int i = 0; i < this.Length; i++)
            {
                result._data[i] = NumOps.Multiply(this._data[i], other._data[i]);
            }
            return result;
        }
        else
        {
            // Handle broadcasting
            return BroadcastPointwiseMultiply(other);
        }
    }

    /// <summary>
    /// Performs element-wise multiplication with broadcasting support for tensors of different shapes.
    /// </summary>
    /// <param name="other">The tensor to multiply with.</param>
    /// <returns>A new tensor containing the element-wise product.</returns>
    /// <remarks>
    /// This is a private helper method that implements the broadcasting logic for PointwiseMultiply.
    /// </remarks>
    private Tensor<T> BroadcastPointwiseMultiply(Tensor<T> other)
    {
        int[] broadcastShape = GetBroadcastShape(this.Shape, other.Shape);
        var result = new Tensor<T>(broadcastShape);

        // Create index arrays for both tensors
        int[] thisIndices = new int[this.Rank];
        int[] otherIndices = new int[other.Rank];

        // Iterate over the result tensor
        foreach (var index in result.GetIndices())
        {
            // Map result index to this tensor's index
            for (int i = 0; i < this.Rank; i++)
            {
                thisIndices[i] = this.Shape[i] == 1 ? 0 : index[i];
            }

            // Map result index to other tensor's index
            for (int i = 0; i < other.Rank; i++)
            {
                otherIndices[i] = other.Shape[i] == 1 ? 0 : index[i];
            }

            // Perform multiplication
            result[index] = NumOps.Multiply(this[thisIndices], other[otherIndices]);
        }

        return result;
    }

    /// <summary>
    /// Calculates the shape that results from broadcasting two tensors together.
    /// </summary>
    /// <param name="shape1">The shape of the first tensor.</param>
    /// <param name="shape2">The shape of the second tensor.</param>
    /// <returns>The resulting broadcast shape as an array of integers.</returns>
    /// <exception cref="ArgumentException">Thrown when the shapes cannot be broadcast together.</exception>
    /// <remarks>
    /// This method implements the standard broadcasting rules used in numerical computing:
    /// 1. Dimensions are aligned from right to left
    /// 2. Each dimension pair must either be equal, or one of them must be 1
    /// 3. The output dimension is the maximum of the two input dimensions
    /// </remarks>
    private static int[] GetBroadcastShape(int[] shape1, int[] shape2)
    {
        int maxRank = Math.Max(shape1.Length, shape2.Length);
        var broadcastShape = new int[maxRank];

        for (int i = 0; i < maxRank; i++)
        {
            int dim1 = i < shape1.Length ? shape1[shape1.Length - 1 - i] : 1;
            int dim2 = i < shape2.Length ? shape2[shape2.Length - 1 - i] : 1;

            if (dim1 == dim2 || dim1 == 1 || dim2 == 1)
            {
                broadcastShape[maxRank - 1 - i] = Math.Max(dim1, dim2);
            }
            else
            {
                throw new ArgumentException("Tensors cannot be broadcast to a single shape.");
            }
        }

        return broadcastShape;
    }

    /// <summary>
    /// Generates all possible index combinations for iterating through a tensor.
    /// </summary>
    /// <returns>An enumerable sequence of index arrays, each representing a position in the tensor.</returns>
    /// <remarks>
    /// This method efficiently generates all indices without allocating unnecessary arrays.
    /// It uses a yield-based approach to return indices one at a time.
    /// </remarks>
    private IEnumerable<int[]> GetIndices()
    {
        int[] index = new int[this.Rank];
        int totalElements = this.Length;

        for (int i = 0; i < totalElements; i++)
        {
            yield return index;

            // Update index
            for (int j = this.Rank - 1; j >= 0; j--)
            {
                if (++index[j] < this.Shape[j])
                    break;
                index[j] = 0;
            }
        }
    }

    /// <summary>
    /// Performs element-wise subtraction with another tensor.
    /// </summary>
    /// <param name="other">The tensor to subtract.</param>
    /// <returns>A new tensor containing the element-wise difference.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    /// <remarks>
    /// Both tensors must have identical shapes for this operation.
    /// </remarks>
    public Tensor<T> ElementwiseSubtract(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for elementwise subtraction.");

        var result = new Tensor<T>(Shape);
        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = NumOps.Subtract(_data[i], other._data[i]);
        }

        return result;
    }

    /// <summary>
    /// Subtracts a scalar from each element of the tensor.
    /// </summary>
    /// <param name="scalar">The scalar value to subtract.</param>
    /// <returns>A new tensor containing the result of the subtraction.</returns>
    /// <remarks>
    /// This method creates a new tensor and does not modify the original.
    /// </remarks>
    public Tensor<T> ElementwiseSubtract(T scalar)
    {
        var result = new Tensor<T>(Shape);
        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = NumOps.Subtract(_data[i], scalar);
        }

        return result;
    }

    /// <summary>
    /// Applies a transformation function to each element of the tensor.
    /// </summary>
    /// <param name="transformer">A function that takes an element value and its index and returns a new value.</param>
    /// <returns>A new tensor containing the transformed values.</returns>
    /// <remarks>
    /// This method creates a new tensor and does not modify the original.
    /// </remarks>
    public Tensor<T> Transform(Func<T, int, T> transformer)
    {
        var result = new Vector<T>(_data.Length);
        for (int i = 0; i < _data.Length; i++)
        {
            result[i] = transformer(_data[i], i);
        }

        return new Tensor<T>(Shape, result);
    }

    /// <summary>
    /// Adds a vector to the last dimension of a 3D tensor.
    /// </summary>
    /// <param name="vector">The vector to add.</param>
    /// <returns>A new tensor containing the result of the addition.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor rank is not 3 or vector length doesn't match the last dimension.</exception>
    /// <remarks>
    /// This method is specifically designed for 3D tensors and adds the vector to each slice along the last dimension.
    /// </remarks>
    public Tensor<T> Add(Vector<T> vector)
    {
        if (this.Rank != 3 || this.Shape[2] != vector.Length)
            throw new ArgumentException("Vector length must match the last dimension of the tensor.");

        var result = new Tensor<T>(this.Shape);
        for (int i = 0; i < this.Shape[0]; i++)
        {
            for (int j = 0; j < this.Shape[1]; j++)
            {
                for (int k = 0; k < this.Shape[2]; k++)
                {
                    result[i, j, k] = NumOps.Add(this[i, j, k], vector[k]);
                }
            }
        }

        return result;
    }

        /// <summary>
    /// Multiplies a 3D tensor with a matrix along the last dimension.
    /// </summary>
    /// <param name="matrix">The matrix to multiply with the tensor.</param>
    /// <returns>A new tensor containing the result of the multiplication.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the tensor is not 3D or when the matrix rows don't match the last dimension of the tensor.
    /// </exception>
    /// <remarks>
    /// For beginners: This operation performs matrix multiplication between each 2D slice of the 3D tensor
    /// and the provided matrix. Think of it as applying the same transformation (represented by the matrix)
    /// to each 2D slice of your 3D data.
    /// 
    /// The resulting tensor will have the same first two dimensions as the original tensor,
    /// but the third dimension will match the number of columns in the matrix.
    /// </remarks>
    public Tensor<T> Multiply(Matrix<T> matrix)
    {
        if (this.Rank != 3 || this.Shape[2] != matrix.Rows)
            throw new ArgumentException("Matrix rows must match the last dimension of the tensor.");

        var result = new Tensor<T>([this.Shape[0], this.Shape[1], matrix.Columns]);
        for (int i = 0; i < this.Shape[0]; i++)
        {
            for (int j = 0; j < this.Shape[1]; j++)
            {
                for (int k = 0; k < matrix.Columns; k++)
                {
                    T sum = NumOps.Zero;
                    for (int l = 0; l < this.Shape[2]; l++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(this[i, j, l], matrix[l, k]));
                    }

                    result[i, j, k] = sum;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Performs matrix multiplication between two 2D tensors (matrices).
    /// </summary>
    /// <param name="other">The second tensor to multiply with.</param>
    /// <returns>A new tensor containing the result of the matrix multiplication.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when either tensor is not 2D or when the inner dimensions don't match.
    /// </exception>
    /// <remarks>
    /// For beginners: Matrix multiplication is a fundamental operation in linear algebra and machine learning.
    /// 
    /// For two matrices A and B to be multiplied:
    /// - The number of columns in A must equal the number of rows in B
    /// - The result will have dimensions: (rows of A) × (columns of B)
    /// 
    /// This is different from element-wise multiplication where corresponding elements are simply multiplied together.
    /// </remarks>
    public Tensor<T> MatrixMultiply(Tensor<T> other)
    {
        if (this.Rank != 2 || other.Rank != 2)
        {
            throw new ArgumentException("MatMul is only defined for 2D tensors (matrices).");
        }

        if (this.Shape[1] != other.Shape[0])
        {
            throw new ArgumentException("Incompatible matrix dimensions for multiplication.");
        }

        return this.Multiply(other);
    }

    /// <summary>
    /// Performs tensor multiplication with broadcasting support.
    /// </summary>
    /// <param name="other">The tensor to multiply with.</param>
    /// <returns>A new tensor containing the result of the multiplication.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when tensor shapes are not compatible for multiplication with broadcasting.
    /// </exception>
    /// <remarks>
    /// For beginners: Tensor multiplication with broadcasting allows multiplication between tensors of different shapes.
    /// 
    /// Broadcasting is a powerful concept that automatically expands smaller tensors to match the shape of larger ones
    /// when possible. This allows operations between tensors of different shapes without manually replicating data.
    /// 
    /// For example, you can multiply a 3×4 tensor with a 4×2 tensor to get a 3×2 result, or even multiply
    /// a 2×3×4 tensor with a 4×5 tensor to get a 2×3×5 result.
    /// </remarks>
    public Tensor<T> Multiply(Tensor<T> other)
    {
        // Check if shapes are compatible for multiplication
        if (!AreShapesMultiplicationCompatible(_dimensions, other._dimensions))
            throw new ArgumentException("Tensor shapes are not compatible for multiplication.");

        // Determine the output shape
        int[] outputShape = GetOutputShape(_dimensions, other._dimensions);

        // Create the result tensor
        Tensor<T> result = new Tensor<T>(outputShape);

        // Perform the multiplication
        MultiplyTensors(this, other, result);

        return result;
    }

    /// <summary>
    /// Transposes the tensor by rearranging its dimensions according to the specified permutation.
    /// </summary>
    /// <param name="permutation">An array specifying the new order of dimensions.</param>
    /// <returns>A new tensor with rearranged dimensions.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the permutation array length doesn't match the tensor rank or contains invalid values.
    /// </exception>
    /// <remarks>
    /// For beginners: Transposing a tensor means rearranging its dimensions.
    /// 
    /// For example, with a 2D tensor (matrix), transposing swaps rows and columns.
    /// For higher-dimensional tensors, you can specify exactly how you want to rearrange the dimensions.
    /// 
    /// The permutation array indicates the new positions of each dimension:
    /// - For a 3D tensor with shape [2,3,4], a permutation [2,0,1] would result in a tensor with shape [4,2,3]
    /// - The value at position i in the permutation array indicates which dimension of the original tensor
    ///   should be placed at position i in the result
    /// </remarks>
    public Tensor<T> Transpose(int[] permutation)
    {
        if (permutation.Length != Rank)
            throw new ArgumentException("Permutation array length must match tensor rank.");

        if (!permutation.OrderBy(x => x).SequenceEqual(Enumerable.Range(0, Rank)))
            throw new ArgumentException("Invalid permutation array.");

        int[] newShape = new int[Rank];
        for (int i = 0; i < Rank; i++)
        {
            newShape[i] = Shape[permutation[i]];
        }

        Tensor<T> result = new Tensor<T>(newShape);

        int[] oldIndices = new int[Rank];
        int[] newIndices = new int[Rank];

        for (int i = 0; i < Length; i++)
        {
            GetIndicesFromFlatIndex(i, Shape, oldIndices);
            for (int j = 0; j < Rank; j++)
            {
                newIndices[j] = oldIndices[permutation[j]];
            }

            result[newIndices] = this[oldIndices];
        }

        return result;
    }

    /// <summary>
    /// Converts a flat index to multi-dimensional indices based on the tensor's shape.
    /// </summary>
    /// <param name="flatIndex">The flat (linear) index to convert.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="indices">Array to store the resulting multi-dimensional indices.</param>
    /// <remarks>
    /// This is a helper method used internally for tensor operations.
    /// </remarks>
    private void GetIndicesFromFlatIndex(int flatIndex, int[] shape, int[] indices)
    {
        for (int i = Rank - 1; i >= 0; i--)
        {
            indices[i] = flatIndex % shape[i];
            flatIndex /= shape[i];
        }
    }

    /// <summary>
    /// Converts a 2D tensor to a Matrix object.
    /// </summary>
    /// <returns>A Matrix object containing the same data as the tensor.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the tensor is not 2-dimensional.
    /// </exception>
    /// <remarks>
    /// For beginners: This method allows you to convert a 2D tensor to a Matrix object,
    /// which might have specialized methods for matrix operations.
    /// 
    /// A 2D tensor and a matrix are conceptually the same thing - a rectangular grid of numbers.
    /// This method simply changes the representation from one class to another.
    /// </remarks>
    public Matrix<T> ToMatrix()
    {
        if (Rank != 2)
        {
            throw new InvalidOperationException("Tensor must be 2-dimensional to convert to Matrix.");
        }

        var matrix = new Matrix<T>(Shape[0], Shape[1]);
        for (int i = 0; i < Shape[0]; i++)
        {
            for (int j = 0; j < Shape[1]; j++)
            {
                matrix[i, j] = this[i, j];
            }
        }

        return matrix;
    }

        /// <summary>
    /// Computes the sum of tensor elements along specified axes.
    /// </summary>
    /// <param name="axes">The axes along which to sum. If null or empty, sums all elements.</param>
    /// <returns>A new tensor containing the sum results.</returns>
    /// <remarks>
    /// For beginners: This method reduces the tensor by adding up values along specified dimensions.
    /// 
    /// For example, if you have a 3D tensor representing multiple 2D images (like a stack of photos),
    /// summing along axis 0 would give you the total pixel values across all images at each position.
    /// 
    /// If no axes are specified, it returns a single value representing the sum of all elements.
    /// </remarks>
    public Tensor<T> Sum(int[]? axes = null)
    {
        if (axes == null || axes.Length == 0)
        {
            // Sum all elements
            T sum = NumOps.Zero;
            for (int i = 0; i < Length; i++)
            {
                sum = NumOps.Add(sum, _data[i]);
            }
            return new Tensor<T>([1], new Vector<T>(new[] { sum }));
        }

        axes = axes.OrderBy(x => x).ToArray();
        int[] newShape = new int[Rank - axes.Length];
        int newIndex = 0;

        for (int i = 0; i < Rank; i++)
        {
            if (!axes.Contains(i))
            {
                newShape[newIndex++] = Shape[i];
            }
        }

        Tensor<T> result = new Tensor<T>(newShape);
        int[] indices = new int[Rank];
        SumRecursive(this, result, axes, indices, 0, NumOps.Zero);

        return result;
    }

    /// <summary>
    /// Helper method for recursively computing sums along specified axes.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="result">The result tensor.</param>
    /// <param name="axes">The axes along which to sum.</param>
    /// <param name="indices">Current indices being processed.</param>
    /// <param name="depth">Current recursion depth.</param>
    /// <param name="currentSum">Running sum at the current position.</param>
    private void SumRecursive(Tensor<T> input, Tensor<T> result, int[] axes, int[] indices, int depth, T currentSum)
    {
        if (depth == Rank)
        {
            int[] resultIndices = new int[result.Rank];
            int resultIndex = 0;
            for (int i = 0; i < Rank; i++)
            {
                if (!axes.Contains(i))
                {
                    resultIndices[resultIndex++] = indices[i];
                }
            }
            result[resultIndices] = NumOps.Add(result[resultIndices], currentSum);
            return;
        }

        if (axes.Contains(depth))
        {
            for (int i = 0; i < Shape[depth]; i++)
            {
                indices[depth] = i;
                SumRecursive(input, result, axes, indices, depth + 1, NumOps.Add(currentSum, this[indices]));
            }
        }
        else
        {
            for (int i = 0; i < Shape[depth]; i++)
            {
                indices[depth] = i;
                SumRecursive(input, result, axes, indices, depth + 1, currentSum);
            }
        }
    }

    /// <summary>
    /// Checks if two tensor shapes are compatible for element-wise multiplication with broadcasting.
    /// </summary>
    /// <param name="shape1">First tensor shape.</param>
    /// <param name="shape2">Second tensor shape.</param>
    /// <returns>True if shapes are compatible, false otherwise.</returns>
    private static bool AreShapesMultiplicationCompatible(int[] shape1, int[] shape2)
    {
        int rank1 = shape1.Length;
        int rank2 = shape2.Length;
        int maxRank = Math.Max(rank1, rank2);

        for (int i = 0; i < maxRank; i++)
        {
            int dim1 = i < rank1 ? shape1[rank1 - 1 - i] : 1;
            int dim2 = i < rank2 ? shape2[rank2 - 1 - i] : 1;

            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Calculates the output shape for a broadcasting operation between two tensors.
    /// </summary>
    /// <param name="shape1">First tensor shape.</param>
    /// <param name="shape2">Second tensor shape.</param>
    /// <returns>The resulting shape after broadcasting.</returns>
    private static int[] GetOutputShape(int[] shape1, int[] shape2)
    {
        int maxRank = Math.Max(shape1.Length, shape2.Length);
        int[] outputShape = new int[maxRank];

        for (int i = 0; i < maxRank; i++)
        {
            int dim1 = i < shape1.Length ? shape1[shape1.Length - 1 - i] : 1;
            int dim2 = i < shape2.Length ? shape2[shape2.Length - 1 - i] : 1;
            outputShape[maxRank - 1 - i] = Math.Max(dim1, dim2);
        }

        return outputShape;
    }

    /// <summary>
    /// Extracts a vector from the tensor at the specified index.
    /// </summary>
    /// <param name="index">The index of the vector to extract.</param>
    /// <returns>A vector containing the data at the specified index.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the tensor has fewer than 2 dimensions.
    /// </exception>
    /// <remarks>
    /// For beginners: This method extracts a row from a 2D tensor (or higher dimensions).
    /// Think of it as getting a single row from a table of data.
    /// 
    /// For example, in a tensor representing multiple data samples (where each row is a sample),
    /// this method would extract a single sample at the specified index.
    /// </remarks>
    public Vector<T> GetVector(int index)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to get a vector.");

        int vectorSize = Shape[1];
        var vector = new Vector<T>(vectorSize);
        for (int i = 0; i < vectorSize; i++)
        {
            vector[i] = this[index, i];
        }

        return vector;
    }

    /// <summary>
    /// Sets the values of a row in the tensor.
    /// </summary>
    /// <param name="rowIndex">The index of the row to set.</param>
    /// <param name="vector">The vector containing the values to set.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the tensor has fewer than 2 dimensions.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the vector length doesn't match the second dimension of the tensor.
    /// </exception>
    /// <remarks>
    /// For beginners: This method replaces an entire row in your tensor with new values.
    /// 
    /// For example, if your tensor represents a dataset where each row is a data sample,
    /// this method would replace one sample with new data.
    /// </remarks>
    public void SetRow(int rowIndex, Vector<T> vector)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to set a row.");

        if (vector.Length != Shape[1])
            throw new ArgumentException("Vector length must match the second dimension of the tensor.");

        for (int i = 0; i < vector.Length; i++)
        {
            this[rowIndex, i] = vector[i];
        }
    }

    /// <summary>
    /// Sets the values of a column in the tensor.
    /// </summary>
    /// <param name="columnIndex">The index of the column to set.</param>
    /// <param name="vector">The vector containing the values to set.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the tensor has fewer than 2 dimensions.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the vector length doesn't match the first dimension of the tensor.
    /// </exception>
    /// <remarks>
    /// For beginners: This method replaces an entire column in your tensor with new values.
    /// 
    /// For example, if your tensor represents a dataset where each column is a feature,
    /// this method would replace one feature with new values across all samples.
    /// </remarks>
    public void SetColumn(int columnIndex, Vector<T> vector)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to set a column.");

        if (vector.Length != Shape[0])
            throw new ArgumentException("Vector length must match the first dimension of the tensor.");

        for (int i = 0; i < vector.Length; i++)
        {
            this[i, columnIndex] = vector[i];
        }
    }

        /// <summary>
    /// Retrieves a column vector from the tensor at the specified column index.
    /// </summary>
    /// <param name="columnIndex">The index of the column to retrieve.</param>
    /// <returns>A vector containing the values from the specified column.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the tensor has fewer than 2 dimensions.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when the column index is outside the valid range.
    /// </exception>
    /// <remarks>
    /// For beginners: This method extracts a single column from your tensor.
    /// 
    /// Think of a tensor as a multi-dimensional table. If it's a 2D tensor (like a spreadsheet),
    /// this method would extract an entire column of data.
    /// 
    /// For example, in a dataset where each column represents a feature (like height, weight, etc.),
    /// this method would extract all values for a single feature.
    /// </remarks>
    public Vector<T> GetColumn(int columnIndex)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to get a column.");

        if (columnIndex < 0 || columnIndex >= Shape[1])
            throw new ArgumentOutOfRangeException(nameof(columnIndex), "Column index is out of range.");

        int columnLength = Shape[0];
        Vector<T> column = new Vector<T>(columnLength);

        int stride = Shape[1];
        for (int i = 0; i < columnLength; i++)
        {
            column[i] = _data[i * stride + columnIndex];
        }

        return column;
    }

    /// <summary>
    /// Retrieves a row vector from the tensor at the specified row index.
    /// </summary>
    /// <param name="rowIndex">The index of the row to retrieve.</param>
    /// <returns>A vector containing the values from the specified row.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when the row index is outside the valid range.
    /// </exception>
    /// <remarks>
    /// For beginners: This method extracts a single row from your tensor.
    /// 
    /// In a 2D tensor (like a table or spreadsheet), this would extract an entire row of data.
    /// 
    /// For example, in a dataset where each row represents a sample or observation,
    /// this method would extract all features for a single sample.
    /// </remarks>
    public Vector<T> GetRow(int rowIndex)
    {
        if (rowIndex < 0 || rowIndex >= Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(rowIndex), "Row index is out of range.");
        }

        int rowLength = 1;
        for (int i = 1; i < Shape.Length; i++)
        {
            rowLength *= Shape[i];
        }

        Vector<T> row = new Vector<T>(rowLength);
        int startIndex = rowIndex * rowLength;

        for (int i = 0; i < rowLength; i++)
        {
            row[i] = _data[startIndex + i];
        }

        return row;
    }

    /// <summary>
    /// Sets the values of a vector in the tensor at the specified index.
    /// </summary>
    /// <param name="index">The index where to set the vector.</param>
    /// <param name="vector">The vector containing the values to set.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the tensor has fewer than 2 dimensions.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the vector length doesn't match the second dimension of the tensor.
    /// </exception>
    /// <remarks>
    /// For beginners: This method replaces an entire row in your tensor with new values.
    /// 
    /// For example, if your tensor represents a dataset where each row is a data sample,
    /// this method would replace one sample with new data.
    /// </remarks>
    public void SetVector(int index, Vector<T> vector)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to set a vector.");

        if (vector.Length != Shape[1])
            throw new ArgumentException("Vector length must match the second dimension of the tensor.");

        for (int i = 0; i < vector.Length; i++)
        {
            this[index, i] = vector[i];
        }
    }

    /// <summary>
    /// Creates a new tensor with the same data but a different shape.
    /// </summary>
    /// <param name="newShape">The new shape for the tensor.</param>
    /// <returns>A new tensor with the specified shape containing the same data.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the total number of elements in the new shape doesn't match the original tensor.
    /// </exception>
    /// <remarks>
    /// For beginners: This method changes how your data is organized without changing the actual values.
    /// 
    /// Think of it like rearranging items in a container - the items stay the same, but their organization changes.
    /// 
    /// For example, you could reshape a 4×3 tensor (4 rows, 3 columns) into a 2×6 tensor (2 rows, 6 columns).
    /// The total number of elements (12) must remain the same.
    /// </remarks>
    public Tensor<T> Reshape(params int[] newShape)
    {
        if (newShape.Aggregate(1, (a, b) => a * b) != Length)
            throw new ArgumentException("New shape must have the same total number of elements as the original tensor.");

        var reshaped = new Tensor<T>(newShape);
        for (int i = 0; i < Length; i++)
        {
            reshaped._data[i] = _data[i];
        }

        return reshaped;
    }

    /// <summary>
    /// Helper method for element-wise multiplication of tensors.
    /// </summary>
    /// <param name="a">First tensor operand.</param>
    /// <param name="b">Second tensor operand.</param>
    /// <param name="result">Tensor to store the multiplication result.</param>
    private static void MultiplyTensors(Tensor<T> a, Tensor<T> b, Tensor<T> result)
    {
        int[] indices = new int[result.Rank];
        MultiplyTensorsRecursive(a, b, result, indices, 0);
    }

    /// <summary>
    /// Recursive helper method for element-wise tensor multiplication.
    /// </summary>
    /// <param name="a">First tensor operand.</param>
    /// <param name="b">Second tensor operand.</param>
    /// <param name="result">Tensor to store the multiplication result.</param>
    /// <param name="indices">Current indices being processed.</param>
    /// <param name="depth">Current recursion depth.</param>
    private static void MultiplyTensorsRecursive(Tensor<T> a, Tensor<T> b, Tensor<T> result, int[] indices, int depth)
    {
        if (depth == result.Rank)
        {
            result[indices] = NumOps.Multiply(a[indices], b[indices]);
            return;
        }

        for (int i = 0; i < result.Shape[depth]; i++)
        {
            indices[depth] = i;
            MultiplyTensorsRecursive(a, b, result, indices, depth + 1);
        }
    }

    /// <summary>
    /// Extracts a sub-tensor from a 4D tensor (typically used for image data).
    /// </summary>
    /// <param name="batch">The batch index.</param>
    /// <param name="channel">The channel index.</param>
    /// <param name="startHeight">The starting height position.</param>
    /// <param name="startWidth">The starting width position.</param>
    /// <param name="height">The height of the sub-tensor to extract.</param>
    /// <param name="width">The width of the sub-tensor to extract.</param>
    /// <returns>A new tensor containing the extracted sub-region.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when any of the indices or dimensions are outside the valid range.
    /// </exception>
    /// <remarks>
    /// For beginners: This method extracts a smaller region from a 4D tensor.
    /// 
    /// 4D tensors are commonly used for image data, where:
    /// - The first dimension (batch) represents multiple images
    /// - The second dimension (channel) represents color channels (like R,G,B)
    /// - The third and fourth dimensions (height, width) represent the image dimensions
    /// 
    /// This method lets you extract a "window" or region from an image at specific coordinates.
    /// </remarks>
    public Tensor<T> GetSubTensor(int batch, int channel, int startHeight, int startWidth, int height, int width)
    {
        if (batch < 0 || batch >= Shape[0]) throw new ArgumentOutOfRangeException(nameof(batch));
        if (channel < 0 || channel >= Shape[1]) throw new ArgumentOutOfRangeException(nameof(channel));
        if (startHeight < 0 || startHeight + height > Shape[2]) throw new ArgumentOutOfRangeException(nameof(startHeight));
        if (startWidth < 0 || startWidth + width > Shape[3]) throw new ArgumentOutOfRangeException(nameof(startWidth));

        var subTensor = new Tensor<T>([1, 1, height, width]);

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                subTensor[0, 0, h, w] = this[batch, channel, startHeight + h, startWidth + w];
            }
        }

        return subTensor;
    }

        /// <summary>
    /// Finds the maximum value in the tensor and its corresponding index.
    /// </summary>
    /// <returns>
    /// A tuple containing the maximum value and its index in the flattened tensor.
    /// </returns>
    /// <remarks>
    /// For beginners: This method scans through all values in your tensor and finds the largest one.
    /// 
    /// It returns both the maximum value and its position (index) in the flattened tensor.
    /// The flattened index treats the tensor as if it were a single long array of values.
    /// 
    /// For example, in a tensor representing image pixel values, this could help you
    /// find the brightest pixel and its location.
    /// </remarks>
    public (T maxVal, int maxIndex) Max()
    {
        T maxVal = _data[0];
        int maxIndex = 0;

        for (int i = 1; i < _data.Length; i++)
        {
            if (NumOps.GreaterThan(_data[i], maxVal))
            {
                maxVal = _data[i];
                maxIndex = i;
            }
        }

        return (maxVal, maxIndex);
    }

    /// <summary>
    /// Calculates the arithmetic mean (average) of all values in the tensor.
    /// </summary>
    /// <returns>The mean value of all elements in the tensor.</returns>
    /// <remarks>
    /// For beginners: This method adds up all the values in your tensor and divides by the total number of values.
    /// 
    /// The mean is a common statistical measure that represents the "center" or "average" of your data.
    /// 
    /// For example, if your tensor contains test scores, the mean would give you the average score.
    /// </remarks>
    public T Mean()
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < _data.Length; i++)
        {
            sum = NumOps.Add(sum, _data[i]);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(_data.Length));
    }

    /// <summary>
    /// Retrieves the value at a specific position in the flattened tensor.
    /// </summary>
    /// <param name="flatIndex">The index in the flattened (1D) representation of the tensor.</param>
    /// <returns>The value at the specified flat index.</returns>
    /// <remarks>
    /// For beginners: This method lets you access a value using a single index number,
    /// even if your tensor has multiple dimensions.
    /// 
    /// Think of it as if all the values in your tensor were laid out in a single line,
    /// and you're picking one value from that line.
    /// 
    /// For example, in a 2×3 tensor, the flat indices would map like this:
    /// [0,0]=0, [0,1]=1, [0,2]=2, [1,0]=3, [1,1]=4, [1,2]=5
    /// </remarks>
    public T GetFlatIndexValue(int flatIndex)
    {
        int[] indices = new int[Rank];
        GetIndicesFromFlatIndex(flatIndex, _dimensions, indices);
        return this[indices];
    }

    /// <summary>
    /// Sets the value at a specific position in the flattened tensor.
    /// </summary>
    /// <param name="flatIndex">The index in the flattened (1D) representation of the tensor.</param>
    /// <param name="value">The value to set at the specified position.</param>
    /// <remarks>
    /// For beginners: This method lets you change a value using a single index number,
    /// even if your tensor has multiple dimensions.
    /// 
    /// Think of it as if all the values in your tensor were laid out in a single line,
    /// and you're changing one value in that line.
    /// 
    /// For example, in a 2×3 tensor, the flat indices would map like this:
    /// [0,0]=0, [0,1]=1, [0,2]=2, [1,0]=3, [1,1]=4, [1,2]=5
    /// </remarks>
    public void SetFlatIndexValue(int flatIndex, T value)
    {
        int[] indices = new int[Rank];
        GetIndicesFromFlatIndex(flatIndex, _dimensions, indices);
        this[indices] = value;
    }

    /// <summary>
    /// Converts multi-dimensional indices to a flat (1D) index.
    /// </summary>
    /// <param name="indices">The multi-dimensional indices to convert.</param>
    /// <returns>The corresponding flat index.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the number of indices doesn't match the tensor's rank.
    /// </exception>
    /// <exception cref="IndexOutOfRangeException">
    /// Thrown when any index is outside the valid range for its dimension.
    /// </exception>
    /// <remarks>
    /// For beginners: This method converts a position specified by multiple coordinates
    /// (one for each dimension) into a single index number.
    /// 
    /// For example, if you have a 2D tensor (like a grid), you might refer to a position
    /// using row and column coordinates [2,3]. This method converts those coordinates
    /// to a single number that represents the same position in a flattened version of the tensor.
    /// </remarks>
    public int GetFlatIndex(int[] indices)
    {
        if (indices.Length != Rank)
            throw new ArgumentException("Number of indices must match tensor rank.");

        int flatIndex = 0;
        int stride = 1;

        for (int i = Rank - 1; i >= 0; i--)
        {
            if (indices[i] < 0 || indices[i] >= _dimensions[i])
                throw new IndexOutOfRangeException($"Index {indices[i]} is out of range for dimension {i}.");

            flatIndex += indices[i] * stride;
            stride *= _dimensions[i];
        }

        return flatIndex;
    }

    /// <summary>
    /// Adds another tensor to this tensor element-wise.
    /// </summary>
    /// <param name="other">The tensor to add to this tensor.</param>
    /// <returns>A new tensor containing the sum of the two tensors.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the tensors have different shapes.
    /// </exception>
    /// <remarks>
    /// For beginners: This method adds two tensors together by adding their corresponding elements.
    /// 
    /// For example, if you have two tensors representing measurements from two experiments,
    /// adding them would give you the combined measurements at each point.
    /// 
    /// Both tensors must have exactly the same shape (dimensions) for this operation to work.
    /// </remarks>
    public Tensor<T> Add(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for addition.");

        return new Tensor<T>(Shape, _data.Add(other._data));
    }

    /// <summary>
    /// Multiplies all elements in the tensor by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <returns>A new tensor with all elements multiplied by the scalar.</returns>
    /// <remarks>
    /// For beginners: This method multiplies every value in your tensor by a single number.
    /// 
    /// For example, if you have a tensor of measurements in inches and want to convert to centimeters,
    /// you could multiply by 2.54 (since 1 inch = 2.54 cm).
    /// 
    /// This is useful for scaling, normalizing, or converting units in your data.
    /// </remarks>
    public Tensor<T> Multiply(T scalar)
    {
        return new Tensor<T>(Shape, _data.Multiply(scalar));
    }

    /// <summary>
    /// Converts a rank-1 tensor to a vector.
    /// </summary>
    /// <returns>A vector containing the same data as the tensor.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the tensor's rank is not 1.
    /// </exception>
    /// <remarks>
    /// For beginners: This method converts a one-dimensional tensor into a vector.
    /// 
    /// A vector is a simpler data structure that represents a sequence of values.
    /// This conversion is only possible if your tensor has exactly one dimension
    /// (like a single row or column of data).
    /// 
    /// For example, if you have a 1D tensor with 5 elements, you can convert it to
    /// a vector with 5 elements containing the same values.
    /// </remarks>
    public Vector<T> ToVector()
    {
        if (Rank != 1)
            throw new InvalidOperationException("Can only convert rank-1 tensors to vectors.");

        return _data;
    }

    /// <summary>
    /// Creates a tensor from a vector.
    /// </summary>
    /// <param name="vector">The source vector.</param>
    /// <returns>A new tensor with shape [vector.Length] containing the vector's data.</returns>
    /// <remarks>
    /// This is a convenience method for converting a Vector to a rank-1 Tensor.
    /// </remarks>
    public static Tensor<T> FromVector(Vector<T> vector)
    {
        return new Tensor<T>([vector.Length], vector);
    }

    /// <summary>
    /// Creates a tensor from a matrix.
    /// </summary>
    /// <param name="matrix">The source matrix.</param>
    /// <returns>A new tensor with shape [matrix.Rows, matrix.Columns] containing the matrix's data.</returns>
    /// <remarks>
    /// This is a convenience method for converting a Matrix to a rank-2 Tensor.
    /// The matrix is converted to a column vector internally before creating the tensor.
    /// </remarks>
    public static Tensor<T> FromMatrix(Matrix<T> matrix)
    {
        return new Tensor<T>([matrix.Rows, matrix.Columns], matrix.ToColumnVector());
    }
}