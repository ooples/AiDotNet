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
public class Tensor<T> : IEnumerable<T>
{
    private readonly Vector<T> _data;
    private readonly int[] _dimensions;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the dimensions of the tensor as an array of integers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The shape describes how many elements exist in each dimension.
    /// For example, a shape of [2, 3] means a 2×3 matrix (2 rows, 3 columns).
    /// 
    /// Think of Shape as the "size" of your tensor in each direction:
    /// - For a 1D tensor (vector) of length 5, the shape would be [5]
    /// - For a 2D tensor (matrix) with 3 rows and 4 columns, the shape would be [3, 4]
    /// - For a 3D tensor with dimensions 2×3×4, the shape would be [2, 3, 4]
    /// </para>
    /// </remarks>
    public int[] Shape => _dimensions;

    /// <summary>
    /// Gets the number of dimensions (axes) in the tensor.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rank tells you how many dimensions your tensor has.
    /// 
    /// For example:
    /// - A vector (like a simple list of numbers) has rank 1
    /// - A matrix (like a table of numbers with rows and columns) has rank 2
    /// - A 3D array (like a cube of numbers) has rank 3
    /// 
    /// The rank helps you understand the complexity of your data structure.
    /// </para>
    /// </remarks>
    public int Rank => _dimensions.Length;

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Length tells you the total count of all values stored in the tensor.
    /// 
    /// This is calculated by multiplying all the dimension sizes together. For example:
    /// - A vector of length 5 has 5 elements
    /// - A 3×4 matrix has 12 elements (3 × 4 = 12)
    /// - A 2×3×4 tensor has 24 elements (2 × 3 × 4 = 24)
    /// </para>
    /// </remarks>
    public int Length => _data.Length;

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
    /// - new Tensor&lt;float&gt;([2, 3]) creates a 2×3 matrix of zeros
    /// </para>
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
    /// <para><b>For Beginners:</b> This constructor creates a tensor with a specific shape and fills it with
    /// the values you provide in the data parameter.
    /// 
    /// The data is stored in "row-major order," which means we fill the tensor one row at a time.
    /// For a 2×3 matrix, the data would be arranged as:
    /// [row1-col1, row1-col2, row1-col3, row2-col1, row2-col2, row2-col3]
    /// 
    /// The length of your data must match the total number of elements needed for the tensor's shape.
    /// </para>
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
    /// <para><b>For Beginners:</b> This indexer lets you access or change individual elements in the tensor.
    /// You specify the position using indices for each dimension.
    /// 
    /// For example:
    /// - In a vector: myVector[5] accesses the 6th element (indices start at 0)
    /// - In a matrix: myMatrix[1, 2] accesses the element in the 2nd row, 3rd column
    /// - In a 3D tensor: my3DTensor[0, 1, 2] accesses the element at position [0,1,2]
    /// 
    /// You can also use it to set values: myMatrix[1, 2] = 42;
    /// </para>
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
    /// Adds two tensors element-wise.
    /// </summary>
    /// <param name="left">The first tensor.</param>
    /// <param name="right">The second tensor.</param>
    /// <returns>A new tensor containing the sum of the two tensors.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator adds two tensors together by adding their corresponding elements.
    /// Both tensors must have exactly the same shape for this to work.
    /// 
    /// For example, if you have two 2×3 matrices:
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
    /// For example, multiplying a 2×3 tensor by a 3×4 tensor results in a 2×4 tensor.
    /// This is different from element-wise multiplication, which would require both tensors to have the same shape.
    /// </para>
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
    /// Creates a deep copy of this tensor.
    /// </summary>
    /// <returns>A new tensor with the same shape and values as this tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A "deep copy" means creating a completely new tensor with the same values,
    /// but stored in a different location in memory. This is important because:
    /// 
    /// - Changes to the copy won't affect the original tensor
    /// - Changes to the original won't affect the copy
    /// 
    /// This is different from a "reference" or "shallow copy" where two variables would point to the same data.
    /// Use this method when you need to modify a tensor without changing the original.
    /// </para>
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
    /// Creates an empty tensor with no dimensions.
    /// </summary>
    /// <returns>An empty tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> An empty tensor is a tensor with no elements. It's like an empty array or list.
    /// This is different from a tensor filled with zeros, which would have a specific shape and contain zero values.
    /// Empty tensors are useful as placeholders or when you need to build a tensor incrementally.</para>
    /// </remarks>
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
    /// (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32</para>
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
    /// Scales the tensor in-place by multiplying each element by a factor.
    /// </summary>
    /// <param name="factor">The scaling factor.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method does the same thing as the Scale method, but instead of creating
    /// a new tensor, it changes the values directly in the current tensor.</para>
    /// 
    /// <para>For example, if your tensor contains [1,2,3] and you call ScaleInPlace(2), your tensor will
    /// now contain [2,4,6].</para>
    /// </remarks>
    public void ScaleInPlace(T factor)
    {
        for (int i = 0; i < this.Length; i++)
        {
            this[i] = _numOps.Multiply(this[i], factor);
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
    /// - If you have three 2×3 tensors (like three rectangular sheets of paper) and stack them with axis=0,
    ///   you'll get a 3×2×3 tensor (like a stack of three sheets).
    /// - If you stack them with axis=1, you'll get a 2×3×3 tensor (like sheets arranged side by side).
    /// - If you stack them with axis=2, you'll get a 2×3×3 tensor (like sheets arranged in a grid).</para>
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
    /// <para>For example, when stacking 3 images of size [28×28] along a new first dimension, 
    /// the result will be a tensor of shape [3×28×28].</para>
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
    /// Creates a tensor with all elements initialized to the specified value.
    /// </summary>
    /// <param name="shape">The shape of the tensor to create.</param>
    /// <param name="value">The value to fill the tensor with.</param>
    /// <returns>A new tensor filled with the specified value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a new tensor where every element has the same value.</para>
    /// 
    /// <para>For example, CreateDefault([2, 3], 1.0) would create a 2×3 tensor filled with the value 1.0, like this:
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
    /// Subtracts another tensor from this tensor element-wise.
    /// </summary>
    /// <param name="other">The tensor to subtract.</param>
    /// <returns>A new tensor containing the result of the subtraction.</returns>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method subtracts each element in the "other" tensor from the corresponding element 
    /// in this tensor.</para>
    /// 
    /// <para>For example, if tensor A is [[5, 6], [7, 8]] and tensor B is [[1, 2], [3, 4]], then A.Subtract(B) would result 
    /// in [[4, 4], [4, 4]].</para>
    /// 
    /// <para>Both tensors must have identical shapes for this operation to work.</para>
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
    /// Performs tensor multiplication with broadcasting support.
    /// </summary>
    /// <param name="other">The tensor to multiply with.</param>
    /// <returns>A new tensor containing the result of the multiplication.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when tensor shapes are not compatible for multiplication with broadcasting.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tensor multiplication combines two tensors to create a new one. Broadcasting 
    /// automatically handles tensors of different shapes by "expanding" the smaller one when possible.</para>
    /// 
    /// <para>For example, if you have a 3×4 tensor and a 4×2 tensor, multiplication will give you a 3×2 result.
    /// This is similar to matrix multiplication you might have learned in math class.</para>
    /// 
    /// <para>Broadcasting makes it possible to multiply tensors of different dimensions without manually
    /// copying data, which is very useful in machine learning operations.</para>
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
    /// Generates all possible index combinations for iterating through a tensor.
    /// </summary>
    /// <returns>An enumerable sequence of index arrays, each representing a position in the tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a list of all possible positions (indices) in the tensor.
    /// Think of it as generating all possible coordinates to access each element in the tensor.</para>
    /// 
    /// <para>For example, in a 2×3 tensor, this would generate the coordinates: [0,0], [0,1], [0,2], [1,0], [1,1], [1,2].</para>
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
    /// Subtracts a scalar from each element of the tensor.
    /// </summary>
    /// <param name="scalar">The scalar value to subtract.</param>
    /// <returns>A new tensor containing the result of the subtraction.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method subtracts the same value (scalar) from every element in the tensor.</para>
    /// 
    /// <para>For example, if you have a tensor with values [[1,2],[3,4]] and subtract the scalar 1,
    /// the result will be [[0,1],[2,3]].</para>
    /// 
    /// <para>This creates a new tensor and doesn't modify the original tensor.</para>
    /// </remarks>
    public Tensor<T> ElementwiseSubtract(T scalar)
    {
        var result = new Tensor<T>(Shape);
        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = _numOps.Subtract(_data[i], scalar);
        }

        return result;
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
                    result[i, j, k] = _numOps.Add(this[i, j, k], vector[k]);
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
    /// <para>This is a helper method used internally for tensor operations.</para>
    /// 
    /// <para><b>For Beginners:</b> In a multi-dimensional tensor, we need to convert between a single 
    /// number (flat index) and multiple coordinates (like row, column, etc.). This method takes a 
    /// single number and calculates what position it corresponds to in each dimension of the tensor.</para>
    /// 
    /// <para>For example, in a 3×4 tensor, the flat index 5 would correspond to position [1,1] 
    /// (second row, second column).</para>
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
    /// <para><b>For Beginners:</b> This method allows you to convert a 2D tensor to a Matrix object,
    /// which might have specialized methods for matrix operations.</para>
    /// 
    /// <para>A 2D tensor and a matrix are conceptually the same thing - a rectangular grid of numbers.
    /// This method simply changes the representation from one class to another, making it easier to
    /// use matrix-specific operations if needed.</para>
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
        SumRecursive(this, result, axes, indices, 0, _numOps.Zero);

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
    /// Checks if two tensor shapes are compatible for element-wise multiplication with broadcasting.
    /// </summary>
    /// <param name="shape1">First tensor shape.</param>
    /// <param name="shape2">Second tensor shape.</param>
    /// <returns>True if shapes are compatible, false otherwise.</returns>
    /// <remarks>
    /// <para>This method determines if two tensors can be multiplied together using broadcasting rules.</para>
    /// 
    /// <para><b>For Beginners:</b> Broadcasting is a powerful feature that allows operations between tensors 
    /// of different shapes. For multiplication to work with broadcasting, each dimension must either:</para>
    /// <para>1. Be the same size in both tensors, or</para>
    /// <para>2. Be size 1 in one of the tensors, or</para>
    /// <para>3. Not exist in one tensor (which is treated as having size 1)</para>
    /// 
    /// <para>For example, a 3×4 tensor can be multiplied with a 3×1 tensor (the second dimension will be 
    /// "broadcast" from 1 to 4), but not with a 2×4 tensor (the first dimensions 3 and 2 are incompatible).</para>
    /// </remarks>
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
    /// <remarks>
    /// <para>This method determines the final shape when two tensors are combined using broadcasting.</para>
    /// 
    /// <para><b>For Beginners:</b> When performing operations between tensors of different shapes, 
    /// broadcasting rules determine the shape of the result. The output shape follows these rules:</para>
    /// <para>1. The result will have at least as many dimensions as the tensor with the most dimensions</para>
    /// <para>2. For each dimension, the size will be the maximum of the corresponding dimensions in the input tensors</para>
    /// 
    /// <para>For example, if you combine a 3×1 tensor with a 1×4 tensor, the result will be a 3×4 tensor.</para>
    /// </remarks>
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
    /// <para><b>For Beginners:</b> This method replaces an entire column in your tensor with new values.</para>
    /// 
    /// <para>A tensor can be thought of as a multi-dimensional array. In a 2D tensor, each column represents 
    /// a vertical line of data (going from top to bottom).</para>
    /// 
    /// <para>For example, if your tensor represents a dataset where each column is a feature (like height, weight, etc.),
    /// this method would replace one feature with new values across all samples.</para>
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
    /// <para><b>For Beginners:</b> This method extracts a single column from your tensor.</para>
    /// 
    /// <para>Think of a tensor as a multi-dimensional table. If it's a 2D tensor (like a spreadsheet),
    /// this method would extract an entire column of data (a vertical line going from top to bottom).</para>
    /// 
    /// <para>For example, in a dataset where each column represents a feature (like height, weight, etc.),
    /// this method would extract all values for a single feature.</para>
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
    /// <para><b>For Beginners:</b> This method replaces an entire row in your tensor with new values.</para>
    /// 
    /// <para>This method is similar to SetRow, but uses a more general "index" parameter name. It places
    /// the values from your vector into the tensor at the specified row index.</para>
    /// 
    /// <para>For example, if your tensor represents a dataset where each row is a data sample,
    /// this method would replace one sample with new data.</para>
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
    /// <para><b>For Beginners:</b> This method changes how your data is organized without changing the actual values.</para>
    /// 
    /// <para>Think of it like rearranging items in a container - the items stay the same, but their organization changes.
    /// The total number of elements must remain the same.</para>
    /// 
    /// <para>For example, you could reshape a 4×3 tensor (4 rows, 3 columns) into a 2×6 tensor (2 rows, 6 columns).
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
    /// Helper method for element-wise multiplication of tensors.
    /// </summary>
    /// <param name="a">First tensor operand.</param>
    /// <param name="b">Second tensor operand.</param>
    /// <param name="result">Tensor to store the multiplication result.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method multiplies two tensors together element by element.</para>
    /// 
    /// <para>Element-wise multiplication means that each value in tensor A is multiplied by the 
    /// corresponding value in tensor B at the same position, and the result is stored in a new tensor.</para>
    /// 
    /// <para>For example, if we have two 2×2 tensors:
    /// Tensor A: [[1, 2], [3, 4]]
    /// Tensor B: [[5, 6], [7, 8]]
    /// 
    /// The result would be:
    /// [[1×5, 2×6], [3×7, 4×8]] = [[5, 12], [21, 32]]</para>
    /// </remarks>
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
    /// <remarks>
    /// <para><b>For Beginners:</b> This is an internal method that helps multiply tensors with multiple dimensions.</para>
    /// 
    /// <para>Recursion is a programming technique where a function calls itself to solve smaller parts of a problem.
    /// In this case, we use recursion to navigate through all the elements of multi-dimensional tensors.</para>
    /// 
    /// <para>The "depth" parameter keeps track of which dimension we're currently processing, and "indices" 
    /// stores the current position in the tensor we're working with.</para>
    /// </remarks>
    private static void MultiplyTensorsRecursive(Tensor<T> a, Tensor<T> b, Tensor<T> result, int[] indices, int depth)
    {
        if (depth == result.Rank)
        {
            result[indices] = _numOps.Multiply(a[indices], b[indices]);
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
    /// <para>For example, in a 2×3 tensor (2 rows, 3 columns), the flat indices would map like this:
    /// [0,0]=0, [0,1]=1, [0,2]=2, [1,0]=3, [1,1]=4, [1,2]=5</para>
    /// 
    /// <para>So if you want the value at row 1, column 0, you could use either the multi-dimensional
    /// access with [1,0] or the flat index access with 3.</para>
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
    /// <para><b>For Beginners:</b> This method lets you change a value using a single index number,
    /// even if your tensor has multiple dimensions.</para>
    /// 
    /// <para>Think of it as if all the values in your tensor were laid out in a single line,
    /// and you're changing one value in that line using its position number.</para>
    /// 
    /// <para>For example, in a 2×3 tensor (2 rows, 3 columns), the flat indices would map like this:
    /// [0,0]=0, [0,1]=1, [0,2]=2, [1,0]=3, [1,1]=4, [1,2]=5</para>
    /// 
    /// <para>So if you want to change the value at row 1, column 0, you could use either the multi-dimensional
    /// access with [1,0] or the flat index access with 3.</para>
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
    /// <para><b>For Beginners:</b> This method converts a position specified by multiple coordinates
    /// (one for each dimension) into a single index number.</para>
    /// 
    /// <para>Think of it like converting an apartment address (building, floor, apartment number)
    /// into a single unique ID number. This is necessary because internally, the tensor stores
    /// all its data in a single long list, even though conceptually it's multi-dimensional.</para>
    /// 
    /// <para>For example, if you have a 2D tensor (like a grid), you might refer to a position
    /// using row and column coordinates [2,3]. This method converts those coordinates
    /// to a single number that represents the same position in a flattened version of the tensor.</para>
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
    /// <para><b>For Beginners:</b> This method adds two tensors together by adding their corresponding elements.</para>
    /// 
    /// <para>Element-wise addition means that each value in this tensor is added to the value
    /// at the same position in the other tensor. It's like adding two spreadsheets cell by cell.</para>
    /// 
    /// <para>For example, if you have two tensors representing measurements from two experiments,
    /// adding them would give you the combined measurements at each point.</para>
    /// 
    /// <para>Both tensors must have exactly the same shape (dimensions) for this operation to work,
    /// just like you can only add spreadsheets with the same number of rows and columns.</para>
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
    /// Converts a rank-1 tensor to a vector.
    /// </summary>
    /// <returns>A vector containing the same data as the tensor.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the tensor's rank is not 1.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts a one-dimensional tensor into a vector.</para>
    /// 
    /// <para>A vector is a simpler data structure that represents a sequence of values in a single dimension.
    /// You can think of it as a list or a single row/column of data.</para>
    /// 
    /// <para>This conversion is only possible if your tensor has exactly one dimension
    /// (like a single row or column of data). You can't convert a 2D or higher tensor to a vector
    /// because a vector can only represent one-dimensional data.</para>
    /// 
    /// <para>For example, if you have a 1D tensor with 5 elements, you can convert it to
    /// a vector with 5 elements containing the same values. This is useful when you need to
    /// use functions that work with vectors rather than tensors.</para>
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
    /// <para>For example, if you have a 3×4 matrix representing student test scores (3 students, 4 tests),
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
}