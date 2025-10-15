using AiDotNet.Extensions;
using System.Collections.Generic;

namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Represents a multi-dimensional array of numeric values used in machine learning and AI computations.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor elements (e.g., float, double, int).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A tensor is a mathematical object that can represent data in multiple dimensions. 
/// Think of it as a container that can hold numbers in an organized way:
/// - A 1D tensor is like a list of numbers (a vector)
/// - A 2D tensor is like a table of numbers (a matrix)
/// - A 3D tensor is like a cube of numbers
/// - And so on for higher dimensions
/// 
/// Tensors are fundamental building blocks for many AI algorithms, especially in neural networks.
/// For example, in image processing, a color image can be represented as a 3D tensor:
/// - First dimension: height (rows of pixels)
/// - Second dimension: width (columns of pixels)
/// - Third dimension: color channels (red, green, blue)
/// </para>
/// </remarks>
public class Tensor<T> : TensorBase<T>, IEnumerable<T>
{
    /// <summary>
    /// Creates an empty tensor with zero elements.
    /// </summary>
    /// <returns>A new empty tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a tensor with no elements.
    /// It's useful as a placeholder or when you need to represent the absence of data.</para>
    /// </remarks>
    public static Tensor<T> Empty()
    {
        // Create a singleton empty tensor instance if it doesn't exist yet
        if (_emptyTensor == null)
        {
            _emptyTensor = new Tensor<T>([])
            {
                IsEmpty = true
            };
        }

        return _emptyTensor;
    }

    // Static field to hold the singleton empty tensor instance
    private static Tensor<T>? _emptyTensor;

    /// <summary>
    /// Creates a tensor with the specified dimensions, initialized with zeros.
    /// </summary>
    /// <param name="dimensions">An array specifying the size of each dimension.</param>
    /// <returns>A new tensor filled with zeros.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a tensor of the specified shape with all elements set to zero.
    /// It's useful for initializing tensors before filling them with actual values or as a starting point
    /// for computations.</para>
    /// <para>
    /// For example:
    /// - Tensor&lt;double&gt;.Zeros([3]) creates a vector [0, 0, 0]
    /// - Tensor&lt;double&gt;.Zeros([2, 3]) creates a 2x3 matrix of zeros
    /// </para>
    /// </remarks>
    public static Tensor<T> Zeros(params int[] dimensions)
    {
        return new Tensor<T>(dimensions);
    }

    /// <summary>
    /// Gets a value indicating whether this tensor is empty.
    /// </summary>
    public bool IsEmpty { get; private set; }

    /// <summary>
    /// Gets or sets the metadata associated with this tensor.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();

    /// <summary>
    /// Creates a detached copy of this tensor that does not share memory with the original.
    /// </summary>
    /// <returns>A new tensor with the same data but independent memory.</returns>
    public Tensor<T> Detach()
    {
        // Create a new array with the tensor's data
        T[] data = new T[Length];
        for (int i = 0; i < Length; i++)
        {
            data[i] = _data[i];
        }
        
        var detachedData = new Vector<T>(data);
        var detached = new Tensor<T>(Shape.ToArray(), detachedData);
        
        // Copy metadata
        foreach (var kvp in Metadata)
        {
            detached.Metadata[kvp.Key] = kvp.Value;
        }
        
        return detached;
    }

    /// <summary>
    /// Creates a new tensor with the specified dimensions, initialized with default values.
    /// </summary>
    /// <param name="dimensions">An array specifying the size of each dimension.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor creates an empty tensor with the shape you specify.
    /// All elements will be initialized to their default values (usually 0).
    /// 
    /// For example:
    /// - new Tensor&lt;float&gt;([5]) creates a vector with 5 zeros
    /// - new Tensor&lt;float&gt;([2, 3]) creates a 2�3 matrix of zeros
    /// </para>
    /// </remarks>
    public Tensor(int[] dimensions) : base(dimensions != null && dimensions.Length > 0 ? dimensions : [1])
    {
        // Mark as empty if shape is empty or contains a zero dimension
        IsEmpty = dimensions == null || dimensions.Length == 0 || dimensions.Any(dim => dim == 0);
    }

    /// <summary>
    /// Creates a new tensor with the specified dimensions and pre-populated data.
    /// </summary>
    /// <param name="dimensions">An array specifying the size of each dimension.</param>
    /// <param name="data">A vector containing the data to populate the tensor with.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor creates a tensor with a specific shape and fills it with
    /// the values you provide in the data parameter.
    /// 
    /// The data is stored in "row-major order," which means we fill the tensor one row at a time.
    /// For a 2�3 matrix, the data would be arranged as:
    /// [row1-col1, row1-col2, row1-col3, row2-col1, row2-col2, row2-col3]
    /// 
    /// The length of your data must match the total number of elements needed for the tensor's shape.
    /// </para>
    /// </remarks>
    public Tensor(int[] dimensions, Vector<T> data) : base(data, dimensions)
    {
    }

    /// <summary>
    /// Creates a new tensor with the specified dimensions using data from a matrix.
    /// </summary>
    /// <param name="dimensions">An array specifying the size of each dimension.</param>
    /// <param name="matrix">A matrix containing the data to populate the tensor with.</param>
    /// <exception cref="ArgumentException">Thrown when the matrix dimensions don't match the specified tensor dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor creates a tensor using a matrix as the data source.
    /// 
    /// This is especially useful when:
    /// - You already have your data organized in a matrix format
    /// - You're converting between matrix operations and tensor operations
    /// - You're building higher-dimensional tensors from multiple matrices
    /// 
    /// The matrix's dimensions must be compatible with the tensor dimensions you specify.
    /// For a rank-2 tensor (a matrix), the dimensions should match exactly.
    /// For higher-rank tensors, the matrix is "reshaped" to fit the specified dimensions.
    /// </para>
    /// </remarks>
    public Tensor(int[] dimensions, Matrix<T> matrix) : base(dimensions)
    {
        int totalSize = dimensions.Aggregate(1, (a, b) => a * b);
    
        if (matrix.Rows * matrix.Columns != totalSize)
        {
            throw new ArgumentException($"Matrix size ({matrix.Rows}�{matrix.Columns} = {matrix.Rows * matrix.Columns}) " +
                                        $"does not match the specified tensor dimensions (total elements: {totalSize})");
        }
    
        if (dimensions.Length == 2 && (dimensions[0] != matrix.Rows || dimensions[1] != matrix.Columns))
        {
            throw new ArgumentException($"For a 2D tensor, matrix dimensions must match exactly. " +
                                        $"Expected: [{dimensions[0]}, {dimensions[1]}], " +
                                        $"Got: [{matrix.Rows}, {matrix.Columns}]");
        }
    
        int index = 0;
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                _data[index++] = matrix[i, j];
            }
        }
    }

    /// <summary>
    /// Returns an enumerator that iterates through all elements in the tensor.
    /// </summary>
    /// <returns>An enumerator for the tensor's elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to loop through all values in the tensor
    /// one by one, regardless of its shape. This is useful when you want to process each element
    /// without worrying about the tensor's dimensions.
    /// 
    /// For example, you can use it in a foreach loop:
    /// <code>
    /// foreach (var value in myTensor)
    /// {
    ///     // Process each value
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public IEnumerator<T> GetEnumerator()
    {
        return ((IEnumerable<T>)_data).GetEnumerator();
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
    /// Converts a multi-dimensional tensor into a one-dimensional vector by placing all elements in a single row.
    /// </summary>
    /// <returns>A vector containing all elements of the tensor in row-major order.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Flattening a tensor means converting it from a multi-dimensional structure
    /// into a 1D structure (a single line of values). This method takes all the values from your tensor
    /// and puts them into a vector (a one-dimensional array), reading in row-major order.
    /// </para>
    /// <para>
    /// For example, if you have a 2�2�2 tensor:
    /// [[[1, 2], [3, 4]],
    ///  [[5, 6], [7, 8]]]
    /// The flattened vector would be: [1, 2, 3, 4, 5, 6, 7, 8]
    /// </para>
    /// <para>
    /// This is commonly used in machine learning when you need to feed a multi-dimensional structure 
    /// (like an image or a 3D volume) into an algorithm that only accepts 1D inputs.
    /// </para>
    /// </remarks>
    public Vector<T> ToVector()
    {
        var vector = new Vector<T>(this.Length);
        int index = 0;

        // Use a recursive helper method to traverse all dimensions
        FlattenHelper(new int[Shape.Length], 0, ref index, vector);

        return vector;
    }

    /// <summary>
    /// Extracts a sub-tensor by fixing the first N dimensions to specific indices.
    /// </summary>
    /// <param name="indices">The indices to fix for the first dimensions.</param>
    /// <returns>A tensor with reduced dimensionality.</returns>
    /// <exception cref="ArgumentException">Thrown when the number of indices exceeds tensor dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Think of a tensor as a multi-dimensional array. This method allows you to "slice" the tensor
    /// by fixing some of its dimensions to specific values. For example, if you have a 3D tensor representing a video
    /// (width x height x time), fixing the time dimension to a specific value would give you a single 2D frame from that video.
    /// The indices parameter specifies which values to fix for each dimension, starting from the first dimension.</para>
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
    /// Sets a sub-tensor at the specified indices.
    /// </summary>
    /// <param name="indices">The starting indices for the sub-tensor.</param>
    /// <param name="subTensor">The sub-tensor to insert.</param>
    /// <exception cref="ArgumentException">Thrown when indices length doesn't match sub-tensor rank.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you replace a portion of your tensor with another smaller tensor.
    /// Think of it like pasting a small image into a specific location of a larger image. The indices parameter
    /// specifies where in the larger tensor you want to place the smaller one.</para>
    /// <para>This method replaces a portion of the tensor with the provided sub-tensor.</para>
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
    /// Creates a tensor with random values of the specified dimensions.
    /// </summary>
    /// <param name="dimensions">The dimensions of the tensor to create.</param>
    /// <returns>A new tensor filled with random values between 0 and 1.</returns>
    /// <exception cref="ArgumentException">Thrown when dimensions are null or empty.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a new tensor with the specified shape and fills it with random values
    /// between 0 and 1. Random initialization is a common practice in machine learning to give the algorithm
    /// a starting point before training. The dimensions parameter determines the shape of the tensor - for example,
    /// [3,4] would create a 3x4 matrix (2D tensor), while [2,3,4] would create a 3D tensor.</para>
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
    /// Gets the complex value from a tensor at the specified index.
    /// </summary>
    /// <param name="tensor">The tensor to retrieve the value from.</param>
    /// <param name="index">The index of the value to retrieve.</param>
    /// <returns>The value at the specified index as a complex number.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Complex numbers have two parts: a real part and an imaginary part.
    /// This method retrieves a value from the tensor and converts it to a complex number.
    /// If the value is already a complex number, it's returned as is.
    /// If not, it creates a complex number where the real part is the original value and the imaginary part is zero.
    /// Complex numbers are used in many advanced AI algorithms, especially those involving signal processing or quantum computing.</para>
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
        return new Complex<T>(value, _numOps.Zero);
    }

    /// <summary>
    /// Performs element-wise subtraction with another tensor.
    /// </summary>
    /// <param name="other">The tensor to subtract.</param>
    /// <returns>A new tensor containing the element-wise difference.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method subtracts each element in the second tensor from the corresponding 
    /// element in the first tensor. Both tensors must have exactly the same shape.</para>
    /// 
    /// <para>For example, if you have tensor A with values [[1,2],[3,4]] and tensor B with values [[5,6],[7,8]],
    /// the result will be [[-4,-4],[-4,-4]] (each element of A minus the corresponding element of B).</para>
    /// 
    /// <para>This creates a new tensor and doesn't modify the original tensors.</para>
    /// </remarks>
    public Tensor<T> ElementwiseSubtract(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for elementwise subtraction.");

        var result = new Tensor<T>(Shape);
        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = _numOps.Subtract(_data[i], other._data[i]);
        }

        return result;
    }

    /// <summary>
    /// Adds a vector to the tensor along the last dimension.
    /// </summary>
    /// <param name="vector">The vector to add. Its length must match the size of the tensor's last dimension.</param>
    /// <returns>A new tensor with the same shape as the original, where each element has the corresponding vector element added to it.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector length does not match the last dimension of the tensor.</exception>
    /// <exception cref="NotSupportedException">Thrown when the tensor rank is not supported (currently supports ranks 2 and 3).</exception>
    /// <remarks>
    /// For a rank-2 tensor with shape [m,n], the vector of length n is added to each row.
    /// For a rank-3 tensor with shape [m,n,p], the vector of length p is added to each element along the third dimension.
    /// </remarks>
    public Tensor<T> Add(Vector<T> vector)
    {
        // Check that vector length matches the last dimension of the tensor
        int lastDimIndex = this.Rank - 1;
        int lastDimSize = this.Shape[lastDimIndex];

        if (lastDimSize != vector.Length)
            throw new ArgumentException($"Vector length must match the last dimension of the tensor. Expected {lastDimSize}, got {vector.Length}.");

        var result = new Tensor<T>(this.Shape);

        // Handle based on tensor rank
        if (this.Rank == 2)
        {
            for (int i = 0; i < this.Shape[0]; i++)
            {
                for (int j = 0; j < this.Shape[1]; j++)
                {
                    result[i, j] = _numOps.Add(this[i, j], vector[j]);
                }
            }
        }
        else if (this.Rank == 3)
        {
            for (int i = 0; i < this.Shape[0]; i++)
            {
                for (int j = 0; j < this.Shape[1]; j++)
                {
                    for (int k = 0; k < this.Shape[2]; k++)
                    {
                        result[i, j, k] = _numOps.Add(this[i, j, k], vector[k]);
                    }
                }
            }
        }
        else
        {
            throw new NotSupportedException($"Adding vector to tensor of rank {this.Rank} is not currently supported.");
        }

        return result;
    }

    /// <summary>
    /// Sets a slice of the tensor at the specified index along the first dimension.
    /// </summary>
    /// <param name="index">The index along the first dimension.</param>
    /// <param name="slice">The tensor slice to set.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the index is out of range.</exception>
    /// <exception cref="TensorShapeMismatchException">Thrown when the slice shape doesn't match the expected shape.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you replace a portion of your tensor with new values.
    /// 
    /// Imagine a 3D tensor as a stack of 2D matrices. Using SetSlice(2, newMatrix) would replace
    /// the 3rd matrix in the stack with your new matrix. The new matrix must have the same shape
    /// as the slice you're replacing.
    /// 
    /// For example, if you have a tensor with shape [4, 5, 6]:
    /// - It contains 4 slices, each with shape [5, 6]
    /// - SetSlice(2, newSlice) would replace the 3rd slice (index 2)
    /// - The newSlice must have shape [5, 6] to fit correctly
    /// </para>
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
    /// Computes the dot product of this tensor with another tensor.
    /// </summary>
    /// <param name="other">The other tensor.</param>
    /// <returns>The scalar dot product result.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> The dot product is a way to multiply two tensors together to get a single number.
    /// It works by multiplying corresponding elements and then adding all those products together.</para>
    /// 
    /// <para>For example, if you have two tensors [1,2,3] and [4,5,6], the dot product would be:
    /// (1�4) + (2�5) + (3�6) = 4 + 10 + 18 = 32</para>
    /// 
    /// <para>Both tensors must have identical shapes for this operation.</para>
    /// </remarks>
    public T DotProduct(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for dot product.");

        T result = _numOps.Zero;
        for (int i = 0; i < _data.Length; i++)
        {
            result = _numOps.Add(result, _numOps.Multiply(_data[i], other._data[i]));
        }

        return result;
    }

    /// <summary>
    /// Fills the entire tensor with a specified value.
    /// </summary>
    /// <param name="value">The value to fill the tensor with.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method replaces all elements in the tensor with the same value.
    /// It's like painting all cells in a spreadsheet with the same color.</para>
    /// </remarks>
    public void Fill(T value)
    {
        for (int i = 0; i < _data.Length; i++)
        {
            _data[i] = value;
        }
    }

    /// <summary>
    /// Recursively sets values from a sub-tensor into this tensor at the specified position.
    /// </summary>
    /// <param name="subTensor">The smaller tensor whose values will be copied into this tensor.</param>
    /// <param name="indices">The current position in this tensor where values should be placed.</param>
    /// <param name="dimension">The current dimension being processed in the recursion.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a helper method that works through each dimension of the sub-tensor
    /// one by one, copying its values to the correct positions in the larger tensor.</para>
    /// 
    /// <para>Think of it like placing a small sticker (sub-tensor) onto the correct position of a larger
    /// sheet of paper (the main tensor). The indices tell us where to start placing the sticker,
    /// and this method makes sure each part of the sticker goes to the right spot.</para>
    /// 
    /// <para>The recursion works by:
    /// 1. If we've processed all dimensions, copy the single value
    /// 2. Otherwise, loop through the current dimension and recursively process the next dimension</para>
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
    /// Extracts a slice along the first dimension of the tensor.
    /// </summary>
    /// <param name="index">The index to slice at.</param>
    /// <returns>A tensor with one fewer dimension than the original.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the index is out of range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Think of a tensor as a multi-dimensional array. If you have a 3D tensor 
    /// (like a cube of numbers), slicing it at a specific index along the first dimension gives you a 2D tensor 
    /// (like a single sheet from that cube). This method lets you extract that sheet.</para>
    /// <para>This method reduces the dimensionality of the tensor by fixing the first dimension
    /// to the specified index. For example, slicing a 3D tensor gives a 2D tensor.</para>
    /// </remarks>
    public Tensor<T> Slice(int index)
    {
        if (index < 0 || index >= Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        int[] newShape = [.. Shape.Skip(1)];
        int sliceSize = newShape.Aggregate(1, (a, b) => a * b);
        int offset = index * sliceSize;

        var sliceData = new Vector<T>(sliceSize);
        Array.Copy(_data, offset, sliceData, 0, sliceSize);

        return new Tensor<T>(newShape, sliceData);
    }

    /// <summary>
    /// Scales the tensor by multiplying each element by a factor.
    /// </summary>
    /// <param name="factor">The scaling factor.</param>
    /// <returns>A new tensor with scaled values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Scaling a tensor means multiplying every number in it by the same value.
    /// For example, scaling [1,2,3] by 2 gives you [2,4,6].</para>
    /// 
    /// <para>This method creates a new tensor and does not modify the original.</para>
    /// </remarks>
    public Tensor<T> Scale(T factor)
    {
        var result = new Tensor<T>(this.Shape);

        // Apply scaling to each element in the tensor
        for (int i = 0; i < this.Length; i++)
        {
            result[i] = _numOps.Multiply(this[i], factor);
        }

        return result;
    }

    /// <summary>
    /// Helper method for flattening a multi-dimensional tensor into a one-dimensional vector.
    /// </summary>
    /// <param name="indices">An array to keep track of the current position in the tensor.</param>
    /// <param name="dimension">The current dimension being processed.</param>
    /// <param name="index">A reference to the current index in the output vector.</param>
    /// <param name="vector">The output vector to store the flattened tensor.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method uses recursion to navigate through all dimensions of the tensor.
    /// Recursion means the method calls itself, each time moving deeper into the tensor's structure.
    /// </para>
    /// <para>
    /// Here's how it works:
    /// 1. If we've reached the deepest level (all dimensions processed), we add the current element to the vector.
    /// 2. If not, we loop through the current dimension and recursively process the next dimension.
    /// 3. This continues until all elements have been added to the vector in the correct order.
    /// </para>
    /// <para>
    /// This approach allows us to flatten tensors of any number of dimensions, making it very flexible.
    /// </para>
    /// </remarks>
    private void FlattenHelper(int[] indices, int dimension, ref int index, Vector<T> vector)
    {
        if (dimension == Shape.Length)
        {
            // We've reached the deepest level, add the element to the vector
            vector[index++] = this[indices];
        }
        else
        {
            // Recursively traverse the current dimension
            for (int i = 0; i < Shape[dimension]; i++)
            {
                indices[dimension] = i;
                FlattenHelper(indices, dimension + 1, ref index, vector);
            }
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
    /// <para><b>For Beginners:</b> Stacking tensors is like putting sheets of paper on top of each other to make a stack.
    /// The "axis" parameter tells the method which direction to stack them.</para>
    /// 
    /// <para>For example:
    /// - If you have three 2�3 tensors (like three rectangular sheets of paper) and stack them with axis=0,
    ///   you'll get a 3�2�3 tensor (like a stack of three sheets).
    /// - If you stack them with axis=1, you'll get a 2�3�3 tensor (like sheets arranged side by side).
    /// - If you stack them with axis=2, you'll get a 2�3�3 tensor (like sheets arranged in a grid).</para>
    /// 
    /// <para>All input tensors must have the same shape. The resulting tensor will have rank+1 dimensions.</para>
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
    /// Transposes the tensor according to the specified permutation of dimensions.
    /// </summary>
    /// <param name="permutation">An array specifying the new order of dimensions. Must contain each dimension index exactly once.</param>
    /// <returns>A new tensor with dimensions rearranged according to the specified permutation.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when:
    /// - Permutation array length doesn't match tensor rank
    /// - Permutation doesn't contain exactly one occurrence of each dimension index
    /// - For rank-1 tensors, if a non-trivial permutation is attempted
    /// </exception>
    /// <remarks>
    /// For a rank-2 tensor, transpose([1,0]) swaps rows and columns.
    /// For rank-3+ tensors, dimensions are rearranged according to the permutation.
    /// For rank-1 tensors, only the trivial permutation [0] is valid.
    /// </remarks>
    public Tensor<T> Transpose(int[] permutation)
    {
        // Handle the case where the tensor is rank-1 but a longer permutation is provided
        if (Rank == 1 && permutation.Length > 1)
        {
            throw new ArgumentException($"Cannot transpose a rank-1 tensor with a permutation of length {permutation.Length}. " +
                                       "Consider reshaping the tensor first if you want to change its dimensionality.");
        }

        // Standard validation
        if (permutation.Length != Rank)
            throw new ArgumentException($"Permutation array length ({permutation.Length}) must match tensor rank ({Rank}).");

        if (!permutation.OrderBy(x => x).SequenceEqual(Enumerable.Range(0, Rank)))
            throw new ArgumentException("Invalid permutation array. Must contain exactly one occurrence of each dimension index.");

        // For rank-1, check if it's the trivial permutation [0]
        if (Rank == 1)
        {
            // This is a no-op for rank-1, just return a copy
            return this;
        }

        // Continue with the original implementation for rank >= 2
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
            GetIndicesFromFlatIndex(i, oldIndices);
            for (int j = 0; j < Rank; j++)
            {
                newIndices[j] = oldIndices[permutation[j]];
            }

            result[newIndices] = this[oldIndices];
        }

        return result;
    }

    /// <summary>
    /// Computes the sum of tensor elements along specified axes.
    /// </summary>
    /// <param name="axes">The axes along which to sum. If null or empty, sums all elements.</param>
    /// <returns>A new tensor containing the sum results.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adds up values along specific dimensions of your tensor.</para>
    /// 
    /// <para>Think of a tensor as a multi-dimensional array. For example, a 2D tensor is like a table with rows and columns:</para>
    /// <para>- Summing along axis 0 (rows) would give you the total for each column</para>
    /// <para>- Summing along axis 1 (columns) would give you the total for each row</para>
    /// 
    /// <para>If you don't specify any axes, it will simply add up all numbers in the tensor and return a single value.</para>
    /// 
    /// <para>This is useful for calculating totals or averages across specific dimensions of your data.</para>
    /// </remarks>
    public Tensor<T> Sum(int[]? axes = null)
    {
        if (axes == null || axes.Length == 0)
        {
            // Sum all elements
            T sum = _numOps.Zero;
            for (int i = 0; i < Length; i++)
            {
                sum = _numOps.Add(sum, _data[i]);
            }

            return new Tensor<T>([1], new Vector<T>([sum]));
        }

        axes = [.. axes.OrderBy(x => x)];
        int[] newShape = new int[Rank - axes.Length];
        int newIndex = 0;

        for (int i = 0; i < Rank; i++)
        {
            if (!axes.Contains(i))
            {
                newShape[newIndex++] = Shape[i];
            }
        }

        var result = new Tensor<T>(newShape);
        int[] indices = new int[Rank];
        SumRecursive(this, result, axes, indices, 0, _numOps.Zero);

        return result;
    }

    /// <summary>
    /// Gets a slice of the tensor's data as a vector.
    /// </summary>
    /// <param name="start">The starting index in the flattened data.</param>
    /// <param name="length">The number of elements to include in the slice.</param>
    /// <returns>A vector containing the requested slice of data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a portion of the tensor's data as a simple vector.
    /// It works directly with the flattened (one-dimensional) representation of the tensor.
    /// 
    /// Think of it like cutting out a section from the tensor's internal storage.
    /// This is useful when you need to access a continuous segment of the tensor's data
    /// without worrying about its multi-dimensional structure.
    /// </para>
    /// </remarks>
    public Vector<T> GetSlice(int start, int length)
    {
        return _data.Slice(start, length);
    }

    /// <summary>
    /// Finds the maximum value in the tensor and its corresponding index.
    /// </summary>
    /// <returns>
    /// A tuple containing the maximum value and its index in the flattened tensor.
    /// </returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method finds the largest value in your tensor and tells you where it is.</para>
    /// 
    /// <para>It returns two pieces of information:
    /// 1. The maximum value itself
    /// 2. The position (index) where that value is located</para>
    /// 
    /// <para>The position is given as a "flat index," which means the tensor is treated as if it were
    /// a single long list of values, regardless of its original dimensions.</para>
    /// 
    /// <para>For example, in a tensor of test scores, this could help you find the highest score
    /// and which test it was from.</para>
    /// </remarks>
    public (T maxVal, int maxIndex) Max()
    {
        T maxVal = _data[0];
        int maxIndex = 0;

        for (int i = 1; i < _data.Length; i++)
        {
            if (_numOps.GreaterThan(_data[i], maxVal))
            {
                maxVal = _data[i];
                maxIndex = i;
            }
        }

        return (maxVal, maxIndex);
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
    /// <para><b>For Beginners:</b> This method changes how your data is organized without changing the actual values.</para>
    /// 
    /// <para>Think of it like rearranging items in a container - the items stay the same, but their organization changes.
    /// The total number of elements must remain the same.</para>
    /// 
    /// <para>For example, you could reshape a 4�3 tensor (4 rows, 3 columns) into a 2�6 tensor (2 rows, 6 columns).
    /// Both shapes contain exactly 12 elements.</para>
    /// 
    /// <para>This is useful when you need to transform your data to fit a specific algorithm's requirements
    /// or to view the same data from a different perspective.</para>
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
    /// Helper method for recursively computing sums along specified axes.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="result">The result tensor.</param>
    /// <param name="axes">The axes along which to sum.</param>
    /// <param name="indices">Current indices being processed.</param>
    /// <param name="depth">Current recursion depth.</param>
    /// <param name="currentSum">Running sum at the current position.</param>
    /// <remarks>
    /// <para>This is an internal helper method used by the Sum method to perform the actual summation.</para>
    /// 
    /// <para><b>For Beginners:</b> This method uses recursion (a technique where a function calls itself) 
    /// to navigate through all the elements of a multi-dimensional tensor and calculate sums along 
    /// specified dimensions. You don't need to call this method directly - it's used internally by 
    /// the Sum method.</para>
    /// </remarks>
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
            result[resultIndices] = _numOps.Add(result[resultIndices], currentSum);
            return;
        }

        if (axes.Contains(depth))
        {
            for (int i = 0; i < Shape[depth]; i++)
            {
                indices[depth] = i;
                SumRecursive(input, result, axes, indices, depth + 1, _numOps.Add(currentSum, this[indices]));
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
    /// Multiplies all elements in the tensor by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <returns>A new tensor with all elements multiplied by the scalar.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method multiplies every value in your tensor by a single number.</para>
    /// 
    /// <para>A "scalar" is just a fancy word for a single number (as opposed to a vector, matrix, or tensor).</para>
    /// 
    /// <para>For example, if you have a tensor of measurements in inches and want to convert to centimeters,
    /// you could multiply by 2.54 (since 1 inch = 2.54 cm).</para>
    /// 
    /// <para>This is useful for scaling, normalizing, or converting units in your data. It's like
    /// adjusting the volume on a stereo - one control affects all the sound.</para>
    /// </remarks>
    public Tensor<T> Multiply(T scalar)
    {
        return new Tensor<T>(Shape, _data.Multiply(scalar));
    }

    /// <summary>
    /// Multiplies this tensor by a matrix along the last dimension.
    /// </summary>
    /// <param name="matrix">The matrix to multiply with.</param>
    /// <returns>A new tensor containing the result of the multiplication.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the number of rows in the matrix doesn't match the last dimension of the tensor,
    /// or when the tensor's rank is not 2 or 3.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method performs matrix multiplication between this tensor and the provided matrix.
    /// For 2D tensors (matrices), it performs standard matrix multiplication.
    /// For 3D tensors, it performs batched matrix multiplication where each 2D slice in the tensor
    /// is multiplied by the same matrix.
    /// </para>
    /// <para>
    /// The dimensions of the resulting tensor depend on the input tensor's rank:
    /// - For 2D tensors of shape [m, n] multiplied by a matrix of shape [n, p],
    ///   the result is a tensor of shape [m, p].
    /// - For 3D tensors of shape [b, m, n] multiplied by a matrix of shape [n, p],
    ///   the result is a tensor of shape [b, m, p].
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies your tensor with a matrix.
    /// 
    /// Think of it like this:
    /// - If you have a 2D tensor (a table of numbers with rows and columns), this multiplication
    ///   is like standard matrix multiplication from linear algebra.
    /// - If you have a 3D tensor (like multiple tables stacked on top of each other), this method
    ///   multiplies each table by the same matrix.
    /// 
    /// The operation is only valid when the width of your tensor matches the height of the matrix.
    /// For example, if your tensor has shape [12, 4] (12 rows, 4 columns), the matrix must have
    /// 4 rows (its number of columns can be anything).
    /// 
    /// This operation is commonly used in neural networks for applying transformations to features
    /// or for connecting layers together.
    /// </para>
    /// </remarks>
    public Tensor<T> Multiply(Matrix<T> matrix)
    {
        // Handle 2D tensor (matrix multiplication)
        if (this.Rank == 2)
        {
            if (this.Shape[1] != matrix.Rows)
                throw new ArgumentException($"Matrix rows ({matrix.Rows}) must match the last dimension of the tensor ({this.Shape[1]}).");

            var result = new Tensor<T>([this.Shape[0], matrix.Columns]);

            for (int i = 0; i < this.Shape[0]; i++)
            {
                for (int k = 0; k < matrix.Columns; k++)
                {
                    T sum = _numOps.Zero;
                    for (int j = 0; j < this.Shape[1]; j++)
                    {
                        sum = _numOps.Add(sum, _numOps.Multiply(this[i, j], matrix[j, k]));
                    }
                    result[i, k] = sum;
                }
            }

            return result;
        }
        // Handle 3D tensor (batched matrix multiplication)
        else if (this.Rank == 3)
        {
            if (this.Shape[2] != matrix.Rows)
                throw new ArgumentException($"Matrix rows ({matrix.Rows}) must match the last dimension of the tensor ({this.Shape[2]}).");

            var result = new Tensor<T>([this.Shape[0], this.Shape[1], matrix.Columns]);

            for (int i = 0; i < this.Shape[0]; i++)
            {
                for (int j = 0; j < this.Shape[1]; j++)
                {
                    for (int k = 0; k < matrix.Columns; k++)
                    {
                        T sum = _numOps.Zero;
                        for (int l = 0; l < this.Shape[2]; l++)
                        {
                            sum = _numOps.Add(sum, _numOps.Multiply(this[i, j, l], matrix[l, k]));
                        }
                        result[i, j, k] = sum;
                    }
                }
            }

            return result;
        }
        else
        {
            throw new ArgumentException($"Tensor must be 2D or 3D for matrix multiplication, but has rank {this.Rank}");
        }
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
    /// <para><b>For Beginners:</b> This method replaces an entire row in your tensor with new values.</para>
    /// 
    /// <para>A tensor can be thought of as a multi-dimensional array. In a 2D tensor, each row represents 
    /// a horizontal line of data (going from left to right).</para>
    /// 
    /// <para>For example, if your tensor represents a dataset where each row is a data sample,
    /// this method would replace one sample with new data.</para>
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
    /// Copies data from a source tensor into a destination tensor that is being constructed by the Stack operation.
    /// </summary>
    /// <param name="source">The tensor to copy data from.</param>
    /// <param name="destination">The tensor to copy data to.</param>
    /// <param name="destIndices">The current indices in the destination tensor where data should be placed.</param>
    /// <param name="stackAxis">The axis along which tensors are being stacked.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This helper method is used when combining multiple tensors into a single larger tensor.</para>
    /// 
    /// <para>When stacking tensors (like stacking sheets of paper), we need to carefully copy each value from 
    /// the original tensors to the correct position in the new combined tensor. This method handles that
    /// copying process by recursively traversing through all dimensions of the tensors.</para>
    /// 
    /// <para>For example, when stacking 3 images of size [28�28] along a new first dimension, 
    /// the result will be a tensor of shape [3�28�28].</para>
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
    /// <para><b>For Beginners:</b> This method extracts a smaller region from a 4D tensor, similar to cropping an image.</para>
    /// 
    /// <para>4D tensors are commonly used for image data, where:
    /// - The first dimension (batch) represents multiple images in a set
    /// - The second dimension (channel) represents color channels (like Red, Green, Blue)
    /// - The third and fourth dimensions (height, width) represent the image dimensions</para>
    /// 
    /// <para>For example, if you have a collection of color photos and want to extract just the faces from each photo,
    /// you could use this method to "crop" the relevant portion from each image.</para>
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
    /// Extracts a vector from the tensor at the specified index.
    /// </summary>
    /// <param name="index">The index of the vector to extract.</param>
    /// <returns>A vector containing the data at the specified index.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the tensor has fewer than 2 dimensions.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a single row from your tensor as a vector.</para>
    /// 
    /// <para>Think of a tensor as a multi-dimensional table. If it's a 2D tensor (like a spreadsheet),
    /// this method would extract an entire row of data at the position you specify.</para>
    /// 
    /// <para>For example, in a dataset where each row represents a data sample (like information about 
    /// a person), this method would extract all the information for a single sample.</para>
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
    /// Adds another tensor to this tensor with broadcasting support.
    /// </summary>
    /// <param name="other">The tensor to add.</param>
    /// <returns>A new tensor containing the element-wise sum.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors cannot be broadcast together.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adds two tensors together, supporting tensors of different shapes.
    /// Broadcasting automatically expands smaller dimensions to match larger ones when possible.</para>
    /// 
    /// <para>For example, adding a tensor of shape [3,4] and a tensor of shape [4] will work because the
    /// second tensor is automatically expanded to shape [1,4] and then [3,4].</para>
    /// 
    /// <para>Broadcasting follows these rules:
    /// 1. Dimensions starting from the right are compared
    /// 2. Dimensions are compatible when they are equal or one of them is 1
    /// 3. The resulting dimension will be the larger of the two</para>
    /// </remarks>
    public Tensor<T> Add(Tensor<T> other)
    {
        // If shapes are the same, use direct addition
        if (Shape.SequenceEqual(other.Shape))
        {
            var result = new Tensor<T>(Shape);
            for (int i = 0; i < _data.Length; i++)
            {
                result._data[i] = _numOps.Add(_data[i], other._data[i]);
            }

            return result;
        }

        // Otherwise, use broadcasting
        return BroadcastAdd(other);
    }

    /// <summary>
    /// Performs element-wise addition with broadcasting support for tensors of different shapes.
    /// </summary>
    /// <param name="other">The tensor to add.</param>
    /// <returns>A new tensor containing the element-wise sum.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adds each element in this tensor with the corresponding element 
    /// in another tensor. If the tensors have different shapes, broadcasting rules are applied to make them compatible.</para>
    /// 
    /// <para>For example, if you add a tensor of shape [3,4] with a tensor of shape [1,4], the second tensor
    /// will be "expanded" to match the shape of the first one before addition.</para>
    /// </remarks>
    private Tensor<T> BroadcastAdd(Tensor<T> other)
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

            result.SetFlatIndexValue(result.GetFlatIndex(index),
                _numOps.Add(this.GetFlatIndexValue(this.GetFlatIndex(thisIndices)),
                            other.GetFlatIndexValue(other.GetFlatIndex(otherIndices))));
        }

        return result;
    }

    /// <summary>
    /// Subtracts another tensor from this tensor with broadcasting support.
    /// </summary>
    /// <param name="other">The tensor to subtract.</param>
    /// <returns>A new tensor containing the element-wise difference.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors cannot be broadcast together.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method subtracts one tensor from another, supporting tensors of different shapes.
    /// Broadcasting automatically expands smaller dimensions to match larger ones when possible.</para>
    /// 
    /// <para>For example, subtracting a tensor of shape [4] from a tensor of shape [3,4] will work because the
    /// second tensor is automatically expanded to shape [1,4] and then [3,4].</para>
    /// 
    /// <para>Broadcasting follows the same rules as addition, making it easier to perform operations
    /// on tensors with different but compatible shapes.</para>
    /// </remarks>
    public Tensor<T> Subtract(Tensor<T> other)
    {
        // If shapes are the same, use direct subtraction
        if (Shape.SequenceEqual(other.Shape))
        {
            var result = new Tensor<T>(Shape);
            for (int i = 0; i < _data.Length; i++)
            {
                result._data[i] = _numOps.Subtract(_data[i], other._data[i]);
            }
            return result;
        }

        // Otherwise, use broadcasting
        return BroadcastSubtract(other);
    }

    /// <summary>
    /// Performs element-wise subtraction with broadcasting support for tensors of different shapes.
    /// </summary>
    /// <param name="other">The tensor to subtract.</param>
    /// <returns>A new tensor containing the element-wise difference.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method subtracts each element in another tensor from the corresponding element 
    /// in this tensor. If the tensors have different shapes, broadcasting rules are applied to make them compatible.</para>
    /// 
    /// <para>For example, if you subtract a tensor of shape [1,4] from a tensor of shape [3,4], the second tensor
    /// will be "expanded" to match the shape of the first one before subtraction.</para>
    /// </remarks>
    private Tensor<T> BroadcastSubtract(Tensor<T> other)
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

            // Perform subtraction
            result[index] = _numOps.Subtract(this[thisIndices], other[otherIndices]);
        }

        return result;
    }

    /// <summary>
    /// Performs element-wise multiplication with broadcasting support.
    /// </summary>
    /// <param name="other">The tensor to multiply with.</param>
    /// <returns>A new tensor containing the element-wise product.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors cannot be broadcast together.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method multiplies each element in this tensor with the corresponding element in the other tensor.
    /// Broadcasting allows tensors of different shapes to be multiplied together by automatically expanding
    /// smaller dimensions to match larger ones.</para>
    /// 
    /// <para>For example, you can multiply a 3�4 tensor with a 1�4 tensor
    /// (which will be treated as if it were repeated 3 times).</para>
    /// 
    /// <para>This is particularly useful in machine learning when applying the same operation across multiple
    /// data points or features.</para>
    /// </remarks>
    public Tensor<T> PointwiseMultiply(Tensor<T> other)
    {
        if (this.Shape.SequenceEqual(other.Shape))
        {
            // Simple case: tensors have the same shape
            var result = new Tensor<T>(this.Shape);
            for (int i = 0; i < this.Length; i++)
            {
                result._data[i] = _numOps.Multiply(this._data[i], other._data[i]);
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
    /// Divides this tensor by another tensor with broadcasting support.
    /// </summary>
    /// <param name="other">The tensor to divide by.</param>
    /// <returns>A new tensor containing the element-wise quotient.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors cannot be broadcast together.</exception>
    /// <exception cref="DivideByZeroException">Thrown when a division by zero is attempted.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method divides each element in this tensor by the corresponding element 
    /// in another tensor. Broadcasting automatically expands smaller dimensions to match larger ones when possible.</para>
    /// 
    /// <para>For example, dividing a tensor of shape [3,4] by a tensor of shape [1,4] will work because the
    /// second tensor is automatically expanded to match the shape of the first one.</para>
    /// 
    /// <para>The method checks for division by zero and throws an exception if detected. In numerical
    /// applications, you might want to handle this case differently, such as by replacing zeros with a small
    /// number or implementing special handling for infinity values.</para>
    /// </remarks>
    public Tensor<T> Divide(Tensor<T> other)
    {
        // If shapes are the same, use direct division
        if (Shape.SequenceEqual(other.Shape))
        {
            var result = new Tensor<T>(Shape);
            for (int i = 0; i < _data.Length; i++)
            {
                // Check for division by zero if required
                if (_numOps.Equals(other._data[i], _numOps.Zero))
                    throw new DivideByZeroException("Division by zero encountered.");

                result._data[i] = _numOps.Divide(_data[i], other._data[i]);
            }
            return result;
        }

        // Otherwise, use broadcasting
        return BroadcastDivide(other);
    }

    /// <summary>
    /// Performs element-wise division with broadcasting support for tensors of different shapes.
    /// </summary>
    /// <param name="other">The tensor to divide by.</param>
    /// <returns>A new tensor containing the element-wise quotient.</returns>
    /// <exception cref="DivideByZeroException">Thrown when a division by zero is attempted.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method divides each element in this tensor by the corresponding element 
    /// in another tensor. If the tensors have different shapes, broadcasting rules are applied to make them compatible.</para>
    /// 
    /// <para>For example, if you divide a tensor of shape [3,4] by a tensor of shape [1,4], the second tensor
    /// will be "expanded" to match the shape of the first one before division.</para>
    /// 
    /// <para>Division by zero is checked and will throw an exception if encountered.</para>
    /// </remarks>
    private Tensor<T> BroadcastDivide(Tensor<T> other)
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

            // Check for division by zero
            if (_numOps.Equals(other[otherIndices], _numOps.Zero))
                throw new DivideByZeroException("Division by zero encountered.");

            // Perform division
            result[index] = _numOps.Divide(this[thisIndices], other[otherIndices]);
        }

        return result;
    }

    /// <summary>
    /// Performs element-wise multiplication with broadcasting support for tensors of different shapes.
    /// </summary>
    /// <param name="other">The tensor to multiply with.</param>
    /// <returns>A new tensor containing the element-wise product.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method multiplies each element in one tensor with the corresponding element 
    /// in another tensor. If the tensors have different shapes, broadcasting rules are applied to make them compatible.</para>
    /// 
    /// <para>For example, if you multiply a tensor of shape [3,4] with a tensor of shape [1,4], the second tensor
    /// will be "expanded" to match the shape of the first one before multiplication.</para>
    /// 
    /// <para>This is different from regular tensor multiplication which follows matrix multiplication rules.</para>
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
            result[index] = _numOps.Multiply(this[thisIndices], other[otherIndices]);
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
    /// <b>For Beginners:</b> Matrix multiplication is a fundamental operation in linear algebra and machine learning.
    /// 
    /// For two matrices A and B to be multiplied:
    /// - The number of columns in A must equal the number of rows in B
    /// - The result will have dimensions: (rows of A) � (columns of B)
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
    /// Performs matrix multiplication between two 2D tensors.
    /// </summary>
    /// <param name="other">The tensor to multiply with (right operand).</param>
    /// <returns>A new tensor containing the result of the matrix multiplication.</returns>
    /// <remarks>
    /// <para>
    /// This is an alias for MatrixMultiply, providing a shorter, more commonly used name
    /// for matrix multiplication operations. The method performs standard matrix multiplication
    /// where the number of columns in the first matrix must equal the number of rows in the second.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> MatMul (short for "Matrix Multiplication") combines two matrices
    /// in a specific mathematical way. It's different from element-wise multiplication:
    ///
    /// - Element-wise multiplication: multiply corresponding elements (A[i,j] * B[i,j])
    /// - Matrix multiplication: dot products of rows and columns
    ///
    /// For matrix multiplication to work:
    /// - First matrix dimensions: M x N
    /// - Second matrix dimensions: N x P
    /// - Result matrix dimensions: M x P
    ///
    /// The middle dimension (N) must match!
    ///
    /// Example: If A is 2x3 and B is 3x4, the result will be 2x4.
    ///
    /// This operation is fundamental in neural networks, where it's used to propagate
    /// data through layers and compute weighted sums of inputs.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// Thrown when either tensor is not 2D or when the dimensions are incompatible for multiplication.
    /// </exception>
    public Tensor<T> MatMul(Tensor<T> other)
    {
        return MatrixMultiply(other);
    }

    /// <summary>
    /// Generates all possible index combinations for iterating through a tensor.
    /// </summary>
    /// <returns>An enumerable sequence of index arrays, each representing a position in the tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a list of all possible positions (indices) in the tensor.
    /// Think of it as generating all possible coordinates to access each element in the tensor.</para>
    /// 
    /// <para>For example, in a 2�3 tensor, this would generate the coordinates: [0,0], [0,1], [0,2], [1,0], [1,1], [1,2].</para>
    /// 
    /// <para>This is primarily used internally to efficiently loop through all elements in a tensor.</para>
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
    /// Calculates the shape that results from broadcasting two tensors together.
    /// </summary>
    /// <param name="shape1">The shape of the first tensor.</param>
    /// <param name="shape2">The shape of the second tensor.</param>
    /// <returns>The resulting broadcast shape as an array of integers.</returns>
    /// <exception cref="ArgumentException">Thrown when the shapes cannot be broadcast together.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Broadcasting is a way to perform operations between tensors of different shapes.
    /// This method determines what shape will result when two tensors are combined.</para>
    /// 
    /// <para>The broadcasting rules are:</para>
    /// <list type="number">
    ///   <item>Start comparing dimensions from the right (last dimension)</item>
    ///   <item>Two dimensions are compatible when they are equal or one of them is 1</item>
    ///   <item>The output dimension will be the larger of the two input dimensions</item>
    /// </list>
    /// 
    /// <para>For example, broadcasting shapes [3,1,5] and [1,4,5] results in shape [3,4,5].</para>
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
    /// Calculates the arithmetic mean (average) of all values in the tensor.
    /// </summary>
    /// <returns>The mean value of all elements in the tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates the average of all values in your tensor.</para>
    /// 
    /// <para>It works by:
    /// 1. Adding up all the values in the tensor
    /// 2. Dividing the sum by the total number of values</para>
    /// 
    /// <para>The mean is a common statistical measure that represents the "center" or "typical value" of your data.</para>
    /// 
    /// <para>For example, if your tensor contains temperature readings over time, the mean would give you
    /// the average temperature for the entire period.</para>
    /// </remarks>
    public T Mean()
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < _data.Length; i++)
        {
            sum = _numOps.Add(sum, _data[i]);
        }

        return _numOps.Divide(sum, _numOps.FromDouble(_data.Length));
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
    /// <para><b>For Beginners:</b> This method works only on 2D tensors (matrices) and extracts a rectangular region.
    /// Think of it like selecting a range of cells in a spreadsheet. The parameters define the top-left corner
    /// (startRow, startCol) and the bottom-right corner (endRow-1, endCol-1) of the selection.
    /// Note that the end indices are exclusive, meaning they point to the position just after the last element you want.</para>
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
    /// Converts this tensor to a matrix.
    /// </summary>
    /// <returns>A matrix representation of this tensor.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the tensor cannot be reasonably converted to a matrix.</exception>
    /// <remarks>
    /// - For rank 2 tensors: Directly converts to a matrix with the same dimensions
    /// - For rank 1 tensors: Converts to a column matrix (n�1)
    /// - For rank > 2: Reshapes the tensor by flattening all dimensions after the first into a single dimension
    /// </remarks>
    public Matrix<T> ToMatrix()
    {
        // For rank 2 tensors, direct conversion
        if (Rank == 2)
        {
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
        // For rank 1 tensors (vectors), convert to a column matrix
        else if (Rank == 1)
        {
            var matrix = new Matrix<T>(Shape[0], 1);
            for (int i = 0; i < Shape[0]; i++)
            {
                matrix[i, 0] = this[i];
            }
            return matrix;
        }
        // For higher rank tensors, flatten all dimensions after the first
        else if (Rank > 2)
        {
            // Calculate the product of all dimensions after the first
            int secondDimSize = 1;
            for (int i = 1; i < Shape.Length; i++)
            {
                secondDimSize *= Shape[i];
            }

            // Create a matrix with first dimension preserved and all others flattened
            var matrix = new Matrix<T>(Shape[0], secondDimSize);

            // Map the multidimensional tensor to the 2D matrix
            for (int i = 0; i < Shape[0]; i++)
            {
                // Track position in the flattened second dimension
                int flatIndex = 0;

                // Use a recursive helper to fill in the matrix
                int[] indices = new int[Rank];
                indices[0] = i;
                FillMatrixRecursive(matrix, indices, 1, ref flatIndex);
            }

            return matrix;
        }

        throw new InvalidOperationException("Cannot convert tensor to matrix: unsupported rank.");
    }

    /// <summary>
    /// Recursively fills a matrix from a tensor with rank > 2.
    /// </summary>
    private void FillMatrixRecursive(Matrix<T> matrix, int[] indices, int dimension, ref int flatIndex)
    {
        if (dimension == Rank)
        {
            // We have a complete set of indices, copy the value to the matrix
            matrix[indices[0], flatIndex++] = this[indices];
            return;
        }

        // Recursively iterate through each value in the current dimension
        for (int i = 0; i < Shape[dimension]; i++)
        {
            indices[dimension] = i;
            FillMatrixRecursive(matrix, indices, dimension + 1, ref flatIndex);
        }
    }

    /// <summary>
    /// Retrieves the value at a specific position in the flattened tensor.
    /// </summary>
    /// <param name="flatIndex">The index in the flattened (1D) representation of the tensor.</param>
    /// <returns>The value at the specified flat index.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you access a value using a single index number,
    /// even if your tensor has multiple dimensions.</para>
    /// 
    /// <para>Think of it as if all the values in your tensor were laid out in a single line,
    /// and you're picking one value from that line using its position number.</para>
    /// 
    /// <para>For example, in a 2�3 tensor (2 rows, 3 columns), the flat indices would map like this:
    /// [0,0]=0, [0,1]=1, [0,2]=2, [1,0]=3, [1,1]=4, [1,2]=5</para>
    /// 
    /// <para>So if you want the value at row 1, column 0, you could use either the multi-dimensional
    /// access with [1,0] or the flat index access with 3.</para>
    /// </remarks>
    public T GetFlatIndexValue(int flatIndex)
    {
        int[] indices = new int[Rank];
        GetIndicesFromFlatIndex(flatIndex, indices);
        return this[indices];
    }

    /// <summary>
    /// Converts a flat index to multi-dimensional indices based on the tensor's shape.
    /// </summary>
    /// <param name="flatIndex">The flat (linear) index to convert.</param>
    /// <param name="indices">Array to store the resulting multi-dimensional indices.</param>
    /// <remarks>
    /// <para>This is a helper method used internally for tensor operations.</para>
    /// 
    /// <para><b>For Beginners:</b> In a multi-dimensional tensor, we need to convert between a single 
    /// number (flat index) and multiple coordinates (like row, column, etc.). This method takes a 
    /// single number and calculates what position it corresponds to in each dimension of the tensor.</para>
    /// 
    /// <para>For example, in a 3�4 tensor, the flat index 5 would correspond to position [1,1] 
    /// (second row, second column).</para>
    /// </remarks>
    public void GetIndicesFromFlatIndex(int flatIndex, int[] indices)
    {
        for (int i = Rank - 1; i >= 0; i--)
        {
            indices[i] = flatIndex % Shape[i];
            flatIndex /= Shape[i];
        }
    }

    /// <summary>
    /// Gets a representative sample of values from the tensor.
    /// </summary>
    /// <param name="maxSampleSize">The maximum number of values to include in the sample.</param>
    /// <returns>An array containing sampled values from the tensor.</returns>
    /// <remarks>
    /// This method extracts a representative sample of values from the tensor for analysis purposes.
    /// It uses a systematic sampling approach to ensure the sample represents different regions of the tensor.
    /// If the tensor has fewer elements than the requested sample size, all elements are returned.
    /// </remarks>
    public Vector<T> GetSample(int maxSampleSize)
    {
        // Handle edge cases
        if (maxSampleSize <= 0)
        {
            throw new ArgumentException("Sample size must be positive", nameof(maxSampleSize));
        }

        // If tensor is smaller than requested sample, return all elements
        if (Length <= maxSampleSize)
        {
            var allValues = new T[Length];
            for (int i = 0; i < Length; i++)
            {
                allValues[i] = GetFlatIndexValue(i);
            }

            return new Vector<T>(allValues);
        }

        // For larger tensors, use systematic sampling
        var result = new T[maxSampleSize];

        // Calculate sampling interval to cover the entire tensor
        double interval = (double)Length / maxSampleSize;

        // Use systematic sampling with a small random offset for each sample
        // to avoid potential aliasing with regular patterns in the data
        var random = new Random(42); // Fixed seed for reproducibility

        for (int i = 0; i < maxSampleSize; i++)
        {
            // Calculate index with small random jitter (�10% of interval)
            int jitter = (int)(random.NextDouble() * 0.2 * interval - 0.1 * interval);
            int index = Math.Min(Length - 1, Math.Max(0, (int)(i * interval + jitter)));

            result[i] = GetFlatIndexValue(index);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Sets the value at the specified flat index in the tensor.
    /// </summary>
    /// <param name="flatIndex">The flat (linear) index into the tensor's data.</param>
    /// <param name="value">The value to set.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the flat index is out of range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> A tensor can have multiple dimensions (like a cube or hypercube), 
    /// but internally it's stored as a one-dimensional array. The flat index treats the tensor as this 
    /// one-dimensional array, allowing you to access any element with a single number regardless of 
    /// the tensor's actual shape. Think of it as numbering all cells in a spreadsheet from 0 to N-1 
    /// in row-by-row order.</para>
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
    /// Sets the value at a specific position in the flattened tensor.
    /// </summary>
    /// <param name="flatIndex">The index in the flattened (1D) representation of the tensor.</param>
    /// <param name="value">The value to set at the specified position.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you change a value using a single index number,
    /// even if your tensor has multiple dimensions.</para>
    /// 
    /// <para>Think of it as if all the values in your tensor were laid out in a single line,
    /// and you're changing one value in that line using its position number.</para>
    /// 
    /// <para>For example, in a 2�3 tensor (2 rows, 3 columns), the flat indices would map like this:
    /// [0,0]=0, [0,1]=1, [0,2]=2, [1,0]=3, [1,1]=4, [1,2]=5</para>
    /// 
    /// <para>So if you want to change the value at row 1, column 0, you could use either the multi-dimensional
    /// access with [1,0] or the flat index access with 3.</para>
    /// </remarks>
    public void SetFlatIndexValue(int flatIndex, T value)
    {
        int[] indices = new int[Rank];
        GetIndicesFromFlatIndex(flatIndex, indices);
        this[indices] = value;
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
    /// <para><b>For Beginners:</b> This method extracts a single row from your tensor.</para>
    /// 
    /// <para>In a 2D tensor (like a table or spreadsheet), this would extract an entire row of data
    /// (a horizontal line going from left to right).</para>
    /// 
    /// <para>For example, in a dataset where each row represents a sample or observation,
    /// this method would extract all features for a single sample.</para>
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
    /// Creates a tensor with all elements initialized to the specified value.
    /// </summary>
    /// <param name="shape">The shape of the tensor to create.</param>
    /// <param name="value">The value to fill the tensor with.</param>
    /// <returns>A new tensor filled with the specified value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a new tensor where every element has the same value.</para>
    /// 
    /// <para>For example, CreateDefault([2, 3], 1.0) would create a 2�3 tensor filled with the value 1.0, like this:
    /// [[1.0, 1.0, 1.0],
    ///  [1.0, 1.0, 1.0]]</para>
    /// 
    /// <para>This is useful when you need a starting tensor with a specific value, such as zeros or ones.</para>
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
    /// Performs element-wise multiplication of two tensors.
    /// </summary>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise product of the input tensors.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Element-wise multiplication means that each element in the first tensor 
    /// is multiplied by the corresponding element in the second tensor at the same position. 
    /// For example, if you have two 2x2 tensors, the element at position [0,0] in the first tensor 
    /// will be multiplied by the element at position [0,0] in the second tensor, and so on.
    /// This is different from matrix multiplication which involves more complex operations.</para>
    /// </remarks>
    public static Tensor<T> ElementwiseMultiply(Tensor<T> a, Tensor<T> b)
    {
        TensorValidator.ValidateShape(a, b.Shape);

        Tensor<T> result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result._data[i] = _numOps.Multiply(a._data[i], b._data[i]);
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
    /// <para><b>For Beginners:</b> This method multiplies each element in this tensor with the corresponding element in the 
    /// other tensor. This is different from matrix multiplication!</para>
    /// 
    /// <para>For example, if tensor A is [[2, 3], [4, 5]] and tensor B is [[1, 2], [3, 4]], then A.ElementwiseMultiply(B) 
    /// would result in [[2, 6], [12, 20]].</para>
    /// 
    /// <para>Element-wise multiplication is sometimes called the Hadamard product and is often used in neural networks 
    /// for operations like applying masks or gates to feature maps.</para>
    /// 
    /// <para>Both tensors must have identical shapes for this operation.</para>
    /// </remarks>
    public Tensor<T> ElementwiseMultiply(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same dimensions for element-wise multiplication.");

        Vector<T> result = _data.PointwiseMultiply(other._data);
        return new Tensor<T>(Shape, result);
    }

    /// <summary>
    /// Applies a transformation function to each element of the tensor.
    /// </summary>
    /// <param name="transformer">A function that takes an element value and its index and returns a new value.</param>
    /// <returns>A new tensor containing the transformed values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you apply a custom function to every element in the tensor.
    /// The function receives both the element's value and its position (index).</para>
    /// 
    /// <para>For example, you could use this to square every element, add a constant to specific positions,
    /// or apply any other mathematical operation to the tensor's elements.</para>
    /// 
    /// <para>This creates a new tensor and doesn't modify the original tensor.</para>
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
    /// Extracts a slice from the tensor at the specified batch index.
    /// </summary>
    /// <param name="batchIndex">The index of the batch to extract.</param>
    /// <returns>A new tensor containing the extracted slice.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> In machine learning, data is often organized in batches. A batch is simply a group of 
    /// similar items processed together for efficiency.</para>
    /// 
    /// <para>This method extracts a single item (slice) from a batch of data. For example, if you have a tensor with 
    /// shape [32, 784] representing 32 images with 784 features each, GetSlice(5) would return the 6th image (index 5)
    /// as a tensor with shape [784].</para>
    /// 
    /// <para>Think of it like taking one cookie (the slice) from a tray of cookies (the batch).</para>
    /// 
    /// <para>This method assumes the first dimension is the batch dimension.</para>
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
    /// Creates a tensor from a vector.
    /// </summary>
    /// <param name="vector">The source vector.</param>
    /// <returns>A new tensor with shape [vector.Length] containing the vector's data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts a vector (a simple list of values) into a tensor.</para>
    /// 
    /// <para>A vector is a one-dimensional collection of values. This method wraps that collection
    /// in a tensor structure, which allows you to perform more complex operations on the data.</para>
    /// 
    /// <para>The resulting tensor will have a rank of 1 (one dimension) and its length will be
    /// the same as the original vector's length.</para>
    /// 
    /// <para>For example, if you have a vector of 10 temperature readings and want to apply
    /// tensor operations to it, you would first convert it to a tensor using this method.</para>
    /// </remarks>
    public static Tensor<T> FromVector(Vector<T> vector)
    {
        return new Tensor<T>([vector.Length], vector);
    }

    /// <summary>
    /// Creates a new tensor from a vector with an optional specified shape.
    /// </summary>
    /// <param name="vector">The source vector containing the tensor data.</param>
    /// <param name="shape">Optional shape for the resulting tensor. If not provided, creates a 1D tensor.</param>
    /// <returns>A new tensor with the data from the vector and the specified shape.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the vector is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the shape is invalid or incompatible with the vector's length.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a tensor using the data from the provided vector. The elements are copied in row-major order,
    /// which means the rightmost indices vary the fastest. If a shape is provided, the method verifies that the total
    /// number of elements in the tensor (product of shape dimensions) matches the length of the vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a simple list of numbers (vector) into a
    /// multi-dimensional array (tensor). Think of it like transforming a long line of numbers into
    /// a grid, cube, or even higher-dimensional structure.
    /// </para>
    /// </remarks>
    public static Tensor<T> FromVector(Vector<T> vector, int[]? shape = null)
    {
        if (vector == null)
        {
            throw new ArgumentNullException(nameof(vector), "Source vector cannot be null.");
        }
    
        // If no shape is provided, create a 1D tensor with the same length as the vector
        if (shape == null || shape.Length == 0)
        {
            return new Tensor<T>([vector.Length], vector);
        }
    
        // Calculate the total number of elements based on the shape
        int totalElements = 1;
        foreach (int dim in shape)
        {
            if (dim <= 0)
            {
                throw new ArgumentException($"Invalid dimension size {dim}. All dimensions must be positive.", nameof(shape));
            }
        
            // Check for potential integer overflow
            if (int.MaxValue / dim < totalElements)
            {
                throw new ArgumentException("The product of dimensions is too large and would cause an overflow.", nameof(shape));
            }
        
            totalElements *= dim;
        }
    
        // Verify that the vector has the correct number of elements
        if (totalElements != vector.Length)
        {
            throw new ArgumentException(
                $"Vector length ({vector.Length}) does not match the product of the specified dimensions ({totalElements}).",
                nameof(shape));
        }
    
        // Create a new tensor with the specified shape and data
        return new Tensor<T>(shape, vector);
    }

    /// <summary>
    /// Creates a tensor from a matrix.
    /// </summary>
    /// <param name="matrix">The source matrix.</param>
    /// <returns>A new tensor with shape [matrix.Rows, matrix.Columns] containing the matrix's data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts a matrix (a 2D grid of values) into a tensor.</para>
    /// 
    /// <para>A matrix is a two-dimensional grid of values, like a spreadsheet with rows and columns.
    /// This method transforms that grid into a tensor structure, which can handle more dimensions
    /// and provides additional operations.</para>
    /// 
    /// <para>The resulting tensor will have a rank of 2 (two dimensions) with the first dimension
    /// being the number of rows and the second dimension being the number of columns from the original matrix.</para>
    /// 
    /// <para>For example, if you have a 3�4 matrix representing student test scores (3 students, 4 tests),
    /// this method would convert it to a tensor with the same structure but with the ability to perform
    /// more advanced operations on the data.</para>
    /// 
    /// <para>Internally, the matrix is first converted to a single column of values (column vector)
    /// before being reshaped into the tensor, but this is handled automatically.</para>
    /// </remarks>
    public static Tensor<T> FromMatrix(Matrix<T> matrix)
    {
        return new Tensor<T>([matrix.Rows, matrix.Columns], matrix.ToColumnVector());
    }

    /// <summary>
    /// Converts the tensor to a one-dimensional array.
    /// </summary>
    /// <returns>A one-dimensional array containing all the tensor's elements in row-major order.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method flattens your multi-dimensional tensor into a simple one-dimensional array.
    /// Think of it like taking a 3D cube of data and lining up all the values in a single row.</para>
    /// 
    /// <para>The elements are arranged in row-major order, which means:
    /// - For a 2D tensor (matrix), it reads row by row from left to right
    /// - For a 3D tensor, it reads layer by layer, and within each layer, row by row
    /// - The rightmost index varies fastest</para>
    /// 
    /// <para>For example, a 2×3 tensor [[1,2,3], [4,5,6]] would become [1,2,3,4,5,6].</para>
    /// 
    /// <para>This is useful when you need to:
    /// - Export tensor data to systems that expect simple arrays
    /// - Serialize the tensor data for storage or transmission
    /// - Convert to formats expected by other libraries or APIs</para>
    /// </remarks>
    public T[] ToArray()
    {
        T[] array = new T[Length];
        for (int i = 0; i < Length; i++)
        {
            array[i] = _data[i];
        }
        return array;
    }

    /// <summary>
    /// Gets the value at the specified flat index.
    /// </summary>
    /// <param name="flatIndex">The flat (linear) index in the tensor's internal storage.</param>
    /// <returns>The value at the specified flat index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the flat index is out of range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows direct access to tensor elements using a single index,
    /// treating the tensor as if it were a one-dimensional array. The flat index corresponds to the position
    /// in the internal storage where elements are arranged in row-major order.</para>
    /// 
    /// <para>For example, in a 2×3 tensor, flat index 0 is [0,0], index 1 is [0,1], index 2 is [0,2],
    /// index 3 is [1,0], and so on.</para>
    /// 
    /// <para>This method is mainly used internally by other operations for efficient element access.</para>
    /// </remarks>
    public T GetFlatIndexValue(int flatIndex)
    {
        if (flatIndex < 0 || flatIndex >= Length)
            throw new ArgumentOutOfRangeException(nameof(flatIndex), $"Flat index {flatIndex} is out of range for tensor with length {Length}.");
        
        return _data[flatIndex];
    }

    /// <summary>
    /// Sets the value at the specified flat index.
    /// </summary>
    /// <param name="flatIndex">The flat (linear) index in the tensor's internal storage.</param>
    /// <param name="value">The value to set at the specified index.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the flat index is out of range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows direct modification of tensor elements using a single index,
    /// treating the tensor as if it were a one-dimensional array. The flat index corresponds to the position
    /// in the internal storage where elements are arranged in row-major order.</para>
    /// 
    /// <para>For example, in a 2×3 tensor, setting value at flat index 3 modifies the element at position [1,0].</para>
    /// 
    /// <para>This method is mainly used internally by other operations for efficient element modification.</para>
    /// </remarks>
    public void SetFlatIndexValue(int flatIndex, T value)
    {
        if (flatIndex < 0 || flatIndex >= Length)
            throw new ArgumentOutOfRangeException(nameof(flatIndex), $"Flat index {flatIndex} is out of range for tensor with length {Length}.");
        
        _data[flatIndex] = value;
    }

    /// <summary>
    /// Creates a new tensor with a single scalar value.
    /// </summary>
    /// <param name="value">The scalar value to store in the tensor.</param>
    /// <returns>A new tensor containing only the specified scalar value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a tensor with just one number in it.
    /// It's useful when you need to convert a single value into a tensor format.</para>
    /// 
    /// <para>The resulting tensor has a shape of [1], meaning it's a one-dimensional
    /// tensor with a single element. This is the simplest possible tensor.</para>
    /// </remarks>
    public static Tensor<T> FromScalar(T value)
    {
        // Create a new tensor with shape [1] (a single-element tensor)
        var tensor = new Tensor<T>([1])
        {
            // Set the first (and only) element to the provided value
            [0] = value
        };

        return tensor;
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    /// <param name="left">The first tensor.</param>
    /// <param name="right">The second tensor.</param>
    /// <returns>A new tensor containing the sum of the two tensors.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator adds two tensors together by adding their corresponding elements.
    /// Both tensors must have exactly the same shape for this to work.
    /// 
    /// For example, if you have two 2�3 matrices:
    /// ```
    /// A = [[1, 2, 3],     B = [[5, 6, 7],
    ///      [4, 5, 6]]          [8, 9, 10]]
    /// ```
    /// 
    /// Then A + B would result in:
    /// ```
    /// [[1+5, 2+6, 3+7],    [[6, 8, 10],
    ///  [4+8, 5+9, 6+10]] =  [12, 14, 16]]
    /// ```
    /// </para>
    /// </remarks>
    public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right)
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
    /// <para><b>For Beginners:</b> Tensor multiplication follows specific rules from linear algebra.
    /// For 2D tensors (matrices), this performs matrix multiplication where:
    /// - The number of columns in the first tensor must equal the number of rows in the second tensor
    /// - The result will have dimensions [rows of first tensor, columns of second tensor]
    /// 
    /// For example, multiplying a 2�3 tensor by a 3�4 tensor results in a 2�4 tensor.
    /// This is different from element-wise multiplication, which would require both tensors to have the same shape.
    /// </para>
    /// </remarks>
    public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Multiplies this tensor by another tensor.
    /// </summary>
    /// <param name="other">The tensor to multiply by.</param>
    /// <returns>A new tensor containing the result of the multiplication.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tensor multiplication follows specific rules from linear algebra.
    /// For 2D tensors (matrices), this performs matrix multiplication where:
    /// - The number of columns in the first tensor must equal the number of rows in the second tensor
    /// - The result will have dimensions [rows of first tensor, columns of second tensor]
    /// 
    /// For example, multiplying a 2�3 tensor by a 3�4 tensor results in a 2�4 tensor.
    /// This is different from element-wise multiplication, which would require both tensors to have the same shape.
    /// </para>
    /// </remarks>
        public Tensor<T> Multiply(Tensor<T> other)
    {
        // For simplicity, we'll implement matrix multiplication for 2D tensors
        if (Shape.Length != 2 || other.Shape.Length != 2)
        {
            throw new NotSupportedException("Multiplication is currently only supported for 2D tensors (matrices).");
        }

        if (Shape[1] != other.Shape[0])
        {
            throw new ArgumentException("The number of columns in the first tensor must equal the number of rows in the second tensor.");
        }

        int resultRows = Shape[0];
        int resultCols = other.Shape[1];
        int commonDim = Shape[1];

        var result = new Tensor<T>(new[] { resultRows, resultCols });

        for (int i = 0; i < resultRows; i++)
        {
            for (int j = 0; j < resultCols; j++)
            {
                T sum = _numOps.Zero;
                for (int k = 0; k < commonDim; k++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(this[i, k], other[k, j]));
                }

                result[i, j] = sum;
            }
        }

        return result;
    }

    /// <summary>
    /// Transposes the tensor.
    /// </summary>
    /// <returns>A new tensor that is the transpose of this tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Transposing a tensor means swapping its dimensions. 
    /// For a 2D tensor (matrix), it means turning rows into columns and vice versa.
    /// 
    /// For example, if you have a 2�3 matrix:
    /// ```
    /// A = [[1, 2, 3],
    ///      [4, 5, 6]]
    /// ```
    /// 
    /// Then A.Transpose() would result in a 3�2 matrix:
    /// ```
    /// [[1, 4],
    ///  [2, 5],
    ///  [3, 6]]
    /// ```
    /// </para>
    /// </remarks>
    public Tensor<T> Transpose()
    {
        if (Shape.Length != 2)
        {
            throw new NotSupportedException("Transpose is currently only supported for 2D tensors (matrices).");
        }

        var result = new Tensor<T>([Shape[1], Shape[0]]);

        for (int i = 0; i < Shape[0]; i++)
        {
            for (int j = 0; j < Shape[1]; j++)
            {
                result[j, i] = this[i, j];
            }
        }

        return result;
    }

    /// <summary>
    /// Creates a deep copy of this tensor.
    /// </summary>
    /// <returns>A new tensor with the same shape and values as this tensor.</returns>
    public new Tensor<T> Clone()
    {
        return (Tensor<T>)base.Clone();
    }

    /// <summary>
    /// Concatenates multiple tensors along the specified axis.
    /// </summary>
    /// <param name="tensors">The array of tensors to concatenate.</param>
    /// <param name="axis">The axis along which to concatenate the tensors.</param>
    /// <returns>A new tensor with the same rank as the input tensors.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have incompatible shapes or the axis is invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Think of concatenation as joining multiple arrays together. For example, if you have two 
    /// tensors representing images (each with shape [3, 4, 4] for 3 color channels and 4x4 pixels), concatenating them along 
    /// axis 0 would give you a tensor with shape [6, 4, 4] - essentially stacking the images on top of each other.</para>
    /// 
    /// <para>The "axis" parameter determines which dimension to join along. Axis 0 is typically the batch dimension, 
    /// axis 1 might be rows, axis 2 might be columns, and so on.</para>
    /// 
    /// <para>All input tensors must have the same shape except along the concatenation axis.</para>
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
    /// <para><b>For Beginners:</b> This helper method is used when joining tensors together. It takes data from one tensor 
    /// and places it at the correct position in another tensor.</para>
    /// 
    /// <para>The method uses recursion (a function calling itself) to navigate through all dimensions of the tensors
    /// and copy values one by one to the right locations.</para>
    /// 
    /// <para>This is a helper method used by the Concatenate method to combine multiple tensors.</para>
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
    /// Sets a slice of the tensor's data from a vector.
    /// </summary>
    /// <param name="start">The starting index in the flattened data.</param>
    /// <param name="slice">The vector containing the data to set.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method replaces a portion of the tensor's data with values from a vector.
    /// It works directly with the flattened (one-dimensional) representation of the tensor.
    /// 
    /// Think of it like pasting a section of data into the tensor's internal storage.
    /// This is useful when you need to update a continuous segment of the tensor's data
    /// without worrying about its multi-dimensional structure.
    /// </para>
    /// </remarks>
    public void SetSlice(int start, Vector<T> slice)
    {
        for (int i = 0; i < slice.Length; i++)
        {
            _data[start + i] = slice[i];
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
    /// <para><b>For Beginners:</b> This is a more flexible version of the SetSlice method that lets you
    /// replace a slice along any dimension, not just the first one.
    /// 
    /// For example, with a 3D tensor of shape [4, 5, 6]:
    /// - SetSlice(0, 2, newSlice) would replace the slice at index 2 along dimension 0, requiring newSlice to have shape [5, 6]
    /// - SetSlice(1, 3, newSlice) would replace the slice at index 3 along dimension 1, requiring newSlice to have shape [4, 6]
    /// - SetSlice(2, 4, newSlice) would replace the slice at index 4 along dimension 2, requiring newSlice to have shape [4, 5]
    /// 
    /// Think of it like cutting through your data from different angles and replacing that slice with new data.
    /// </para>
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
    /// Extracts a slice of the tensor along the specified axis.
    /// </summary>
    /// <param name="axis">The axis along which to slice.</param>
    /// <param name="start">The starting index of the slice.</param>
    /// <param name="end">The ending index of the slice (exclusive). If null, slices to the end of the axis.</param>
    /// <returns>A new tensor containing the specified slice.</returns>
    /// <exception cref="ArgumentException">Thrown when axis or indices are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you take a "slice" or section of your tensor along any dimension.
    /// For example, if your tensor represents a stack of images (3D tensor), you could extract images 5 through 10
    /// by using axis=0, start=5, end=11. Or you could extract just the middle portion of each image by slicing
    /// along the height or width dimensions.</para>
    /// <para>This method creates a new tensor that is a subset of the original tensor along the specified axis.</para>
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
    /// Computes the sum along the specified axis.
    /// </summary>
    /// <param name="axis">The axis along which to compute the sum.</param>
    /// <returns>A new tensor with the specified axis removed, containing sum values.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the axis is invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adds up all values along a specific dimension of your tensor.
    /// For example, if you have a 2D tensor representing sales data for multiple products across multiple months,
    /// summing along axis 0 would give you the total sales for each product across all months,
    /// while summing along axis 1 would give you the total sales across all products for each month.</para>
    /// <para>The resulting tensor has one fewer dimension than the original tensor.</para>
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
            T sum = _numOps.Zero;
            for (int j = 0; j < axisSize; j++)
            {
                sum = _numOps.Add(sum, _data[i + j]);
            }

            result._data[i / axisSize] = sum;
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
    /// <para><b>For Beginners:</b> This method finds the largest value along a specific dimension of your tensor.
    /// For example, if you have a 2D tensor representing test scores for multiple students across multiple subjects,
    /// finding the max along axis 0 would give you the highest score for each subject across all students,
    /// while finding the max along axis 1 would give you each student's highest score across all subjects.</para>
    /// <para>The resulting tensor has one fewer dimension than the original tensor.</para>
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
                if (_numOps.GreaterThan(_data[i + j], max))
                    max = _data[i + j];
            }

            result._data[i / axisSize] = max;
        }

        return result;
    }

    /// <summary>
    /// Computes the mean values along the specified axis.
    /// </summary>
    /// <param name="axis">The axis along which to compute the mean.</param>
    /// <returns>A new tensor with the specified axis removed, containing mean values.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the axis is invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates the average value along a specific dimension of your tensor.
    /// For example, if you have a 2D tensor representing test scores for multiple students across multiple subjects,
    /// calculating the mean along axis 0 would give you the average score for each subject across all students,
    /// while calculating the mean along axis 1 would give you each student's average score across all subjects.</para>
    /// <para>The resulting tensor has one fewer dimension than the original tensor.</para>
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
            T sum = _numOps.Zero;
            for (int j = 0; j < axisSize; j++)
            {
                sum = _numOps.Add(sum, _data[i + j]);
            }

            result._data[i / axisSize] = _numOps.Divide(sum, _numOps.FromDouble(axisSize));
        }

        return result;
    }

    /// <summary>
    /// Creates a new instance of the tensor with the specified shape.
    /// </summary>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>A new tensor with the specified shape.</returns>
    protected override TensorBase<T> CreateInstance(int[] shape)
    {
        return new Tensor<T>(shape);
    }

    /// <summary>
    /// Creates a new instance of the tensor with the specified data and shape.
    /// </summary>
    /// <param name="data">The data to populate the new tensor with.</param>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>A new tensor with the specified data and shape.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a new tensor with the given data and shape.
    /// It's useful when you want to create a tensor from existing data, such as when reshaping or
    /// performing operations that result in new tensors.</para>
    /// </remarks>
    protected override TensorBase<T> CreateInstance(T[] data, int[] shape)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));
        if (shape == null)
            throw new ArgumentNullException(nameof(shape));

        int totalSize = shape.Aggregate(1, (acc, dim) => acc * dim);
        if (data.Length != totalSize)
            throw new ArgumentException("The number of elements in the data array does not match the specified shape.");

        return new Tensor<T>(shape, new Vector<T>(data));
    }

    /// <summary>
    /// Creates a new instance of the tensor with the specified shape and a different element type.
    /// </summary>
    /// <typeparam name="TResult">The type of elements in the new tensor.</typeparam>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>A new tensor with the specified shape and element type.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is used when you need to create a new tensor with a different
    /// data type than the current tensor. This is common in operations that change the data type,
    /// such as converting a tensor of integers to a tensor of floating-point numbers.</para>
    /// </remarks>
    protected override TensorBase<TResult> CreateInstance<TResult>(params int[] shape)
    {
        if (shape == null)
            throw new ArgumentNullException(nameof(shape));

        return new Tensor<TResult>(shape);
    }
}