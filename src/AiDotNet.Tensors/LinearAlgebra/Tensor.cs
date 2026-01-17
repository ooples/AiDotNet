using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra;

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
    /// Creates a new tensor with the specified dimensions, initialized with default values.
    /// </summary>
    /// <param name="dimensions">An array specifying the size of each dimension.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor creates an empty tensor with the shape you specify.
    /// All elements will be initialized to their default values (usually 0).
    /// 
    /// For example:
    /// - new Tensor&lt;float&gt;([5]) creates a vector with 5 zeros
    /// - new Tensor&lt;float&gt;([2, 3]) creates a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 matrix of zeros
    /// </para>
    /// </remarks>
    public Tensor(int[] dimensions) : base(dimensions)
    {
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
    /// For a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 matrix, the data would be arranged as:
    /// [row1-col1, row1-col2, row1-col3, row2-col1, row2-col2, row2-col3]
    /// 
    /// The length of your data must match the total number of elements needed for the tensor's shape.
    /// </para>
    /// </remarks>
    public Tensor(int[] dimensions, Vector<T> data) : base(data, dimensions)
    {
    }

    /// <summary>
    /// Creates a new tensor with the specified dimensions using a raw array.
    /// </summary>
    /// <param name="data">The data to populate the tensor with.</param>
    /// <param name="dimensions">An array specifying the size of each dimension.</param>
    public Tensor(T[] data, int[] dimensions) : base(data, dimensions)
    {
    }

    /// <summary>
    /// Private constructor for zero-copy tensor creation from a Vector.
    /// </summary>
    private Tensor(Vector<T> data, int[] dimensions) : base(data, dimensions)
    {
    }

    /// <summary>
    /// Creates a new tensor from existing memory without copying data.
    /// </summary>
    /// <param name="memory">The memory to use as the tensor's backing store.</param>
    /// <param name="dimensions">An array specifying the size of each dimension.</param>
    /// <returns>A new tensor using the provided memory.</returns>
    /// <remarks>
    /// <para><b>Performance:</b> This method does NOT copy data. The tensor directly uses
    /// the provided memory. This is useful for high-performance scenarios where
    /// memory pooling or ArrayPool is used.</para>
    /// <para><b>Warning:</b> The caller must ensure the memory remains valid for the
    /// lifetime of the tensor. If using ArrayPool, do NOT return the array to the pool
    /// until the tensor is no longer in use.</para>
    /// </remarks>
    public static Tensor<T> FromMemory(Memory<T> memory, int[] dimensions)
    {
        var vector = Vector<T>.FromMemory(memory);
        return new Tensor<T>(vector, dimensions);
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
            throw new ArgumentException($"Matrix size ({matrix.Rows}ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â{matrix.Columns} = {matrix.Rows * matrix.Columns}) " +
                                        $"does not match the specified tensor dimensions (total elements: {totalSize})");
        }

        if (dimensions.Length == 2 && (dimensions[0] != matrix.Rows || dimensions[1] != matrix.Columns))
        {
            throw new ArgumentException($"For a 2D tensor, matrix dimensions must match exactly. " +
                                        $"Expected: [{dimensions[0]}, {dimensions[1]}], " +
                                        $"Got: [{matrix.Rows}, {matrix.Columns}]");
        }

        // Use vectorized Copy operation to copy entire matrix data at once (5-10x faster than nested loops)
        // Matrix is stored in row-major order, which matches tensor storage
        _numOps.Copy(matrix.AsSpan(), _data.AsWritableSpan());
    }

    /// <summary>
    /// Gets or sets the value at the specified flat (linear) index.
    /// </summary>
    /// <param name="flatIndex">The flat index into the tensor's underlying storage (0 to Length-1).</param>
    /// <returns>The value at the specified flat index.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This indexer allows you to access tensor elements using a single
    /// index that treats the tensor as a 1D array. The flat index corresponds to row-major ordering
    /// where the last dimension varies fastest.
    ///
    /// For example, for a 2x3 tensor:
    /// - Index 0 corresponds to [0,0]
    /// - Index 1 corresponds to [0,1]
    /// - Index 2 corresponds to [0,2]
    /// - Index 3 corresponds to [1,0]
    /// - And so on...
    ///
    /// This is useful when you need to iterate through all elements without worrying about dimensions.
    /// </para>
    /// </remarks>
    public override T this[int flatIndex]
    {
        get => GetFlat(flatIndex);
        set => SetFlat(flatIndex, value);
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
    /// For example, if you have a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2 tensor:
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
    /// Converts the tensor to a different numeric type (precision casting).
    /// </summary>
    /// <typeparam name="TOut">The target numeric type to convert to.</typeparam>
    /// <returns>A new tensor with the same shape but elements converted to the target type.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts all values in the tensor from one numeric type to another.
    /// This is essential for mixed-precision training where we need to convert between:
    /// - float (32-bit) and Half (16-bit) for memory efficiency
    /// - Half (16-bit) and double (64-bit) for numerical stability
    ///
    /// For example:
    /// - Converting from float to Half reduces memory usage by 50%
    /// - Converting from Half to float allows more precise accumulation
    /// - Converting from Half to double provides maximum numerical precision
    ///
    /// In mixed-precision training:
    /// - Forward/backward passes often use FP16 (Half) for speed
    /// - Gradient accumulation uses FP32 (float) for stability
    /// - Master weights are kept in FP32
    /// </para>
    /// <para><b>Technical Details:</b> The conversion uses the INumericOperations interface to handle
    /// type conversions. The specific conversion path depends on the source and target types:
    /// - Half ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ float: Lossless, expands precision
    /// - float ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ Half: May lose precision and overflow
    /// - float ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ double: Lossless, expands precision
    /// - double ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ float: May lose precision
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Convert FP32 tensor to FP16 for forward pass
    /// Tensor&lt;float&gt; weights = new Tensor&lt;float&gt;([100, 50]);
    /// Tensor&lt;Half&gt; weightsHalf = weights.Cast&lt;Half&gt;();
    ///
    /// // Convert FP16 gradients back to FP32 for accumulation
    /// Tensor&lt;Half&gt; gradientsHalf = layer.Backward(outputGradient);
    /// Tensor&lt;float&gt; gradients = gradientsHalf.Cast&lt;float&gt;();
    /// </code>
    /// </example>
    public Tensor<TOut> Cast<TOut>()
    {
        var sourceOps = MathHelper.GetNumericOperations<T>();
        var targetOps = MathHelper.GetNumericOperations<TOut>();

        // Create output tensor with same shape
        var resultData = new Vector<TOut>(this.Length);

        // Convert each element
        for (int i = 0; i < this.Length; i++)
        {
            T sourceValue = _data[i];

            // Use the precision conversion methods in INumericOperations
            // Determine the most efficient conversion path
            if (typeof(T) == typeof(TOut))
            {
                // Same type, just copy (this shouldn't normally happen, but handle it)
                resultData[i] = (TOut)(object)sourceValue!;
            }
            else if (typeof(TOut) == typeof(float))
            {
                // Convert to float
                float floatValue = sourceOps.ToFloat(sourceValue);
                resultData[i] = (TOut)(object)floatValue;
            }
            else if (typeof(TOut) == typeof(Half))
            {
                // Convert to Half
                Half halfValue = sourceOps.ToHalf(sourceValue);
                resultData[i] = (TOut)(object)halfValue;
            }
            else if (typeof(TOut) == typeof(double))
            {
                // Convert to double
                double doubleValue = sourceOps.ToDouble(sourceValue);
                resultData[i] = (TOut)(object)doubleValue;
            }
            // Target type is not float/Half/double, check source type for efficient conversion
            else if (typeof(T) == typeof(float))
            {
                // Source is float, convert to target
                float floatValue = (float)(object)sourceValue!;
                resultData[i] = targetOps.FromFloat(floatValue);
            }
            else if (typeof(T) == typeof(Half))
            {
                // Source is Half, convert to target
                Half halfValue = (Half)(object)sourceValue!;
                resultData[i] = targetOps.FromHalf(halfValue);
            }
            else if (typeof(T) == typeof(double))
            {
                // Source is double, preserve precision by converting directly
                double doubleValue = (double)(object)sourceValue!;
                resultData[i] = targetOps.FromDouble(doubleValue);
            }
            else
            {
                // Fallback: convert through double as intermediate type
                double intermediate = sourceOps.ToDouble(sourceValue);
                resultData[i] = targetOps.FromDouble(intermediate);
            }
        }

        return new Tensor<TOut>(this.Shape, resultData);
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
        CopySubTensorData(this, subTensor, currentIndices, indices.Length, indices.Length);

        return subTensor;
    }

    /// <summary>
    /// Helper method to recursively copy data from a source tensor to a destination sub-tensor.
    /// </summary>
    /// <param name="source">The source tensor to copy from.</param>
    /// <param name="destination">The destination tensor to copy to.</param>
    /// <param name="currentIndices">The current indices being processed.</param>
    /// <param name="fixedDimensions">The number of dimensions that were originally fixed in SubTensor call.</param>
    /// <param name="currentDimension">The current dimension being iterated (starts at fixedDimensions).</param>
    private static void CopySubTensorData(Tensor<T> source, Tensor<T> destination, int[] currentIndices, int fixedDimensions, int currentDimension)
    {
        if (currentDimension == source.Shape.Length)
        {
            // Extract destination indices from the unfixed portion of currentIndices
            int[] destIndices = new int[destination.Shape.Length];
            for (int i = 0; i < destIndices.Length; i++)
            {
                destIndices[i] = currentIndices[fixedDimensions + i];
            }
            destination[destIndices] = source[currentIndices];
            return;
        }

        for (int i = 0; i < source.Shape[currentDimension]; i++)
        {
            currentIndices[currentDimension] = i;
            CopySubTensorData(source, destination, currentIndices, fixedDimensions, currentDimension + 1);
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
        // indices.Length specifies how many dimensions to fix
        // subTensor.Rank is the remaining dimensions
        // Together they must equal the parent tensor's rank
        if (indices.Length + subTensor.Rank != Rank)
            throw new ArgumentException($"Number of indices ({indices.Length}) plus sub-tensor rank ({subTensor.Rank}) must equal tensor rank ({Rank}).");

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
        var random = RandomHelper.CreateSecureRandom();
        var numOps = MathHelper.GetNumericOperations<T>();

        // Use flat indexing for better performance (avoids multi-dimensional index calculation overhead)
        var flattenedSize = dimensions.Aggregate(1, (a, b) => a * b);
        for (int i = 0; i < flattenedSize; i++)
        {
            // Generate a random value between 0 and 1
            tensor._data[i] = numOps.FromDouble(random.NextDouble());
        }

        return tensor;
    }

    /// <summary>
    /// Creates a tensor filled with ones with the specified dimensions.
    /// </summary>
    /// <param name="dimensions">An array specifying the size of each dimension.</param>
    /// <returns>A new tensor filled with ones.</returns>
    /// <exception cref="ArgumentException">Thrown when dimensions are null or empty.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a tensor where every element is 1.
    /// Ones tensors are commonly used in neural networks for:
    /// - Creating mask tensors
    /// - Computing (1 - x) operations in gating mechanisms
    /// - Bias initialization
    /// </para>
    /// </remarks>
    public static Tensor<T> CreateOnes(params int[] dimensions)
    {
        if (dimensions == null || dimensions.Length == 0)
            throw new ArgumentException("Dimensions cannot be null or empty.", nameof(dimensions));

        return CreateDefault(dimensions, _numOps.One);
    }

    /// <summary>
    /// Creates a tensor filled with zeros with the specified dimensions.
    /// </summary>
    /// <param name="dimensions">An array specifying the size of each dimension.</param>
    /// <returns>A new tensor filled with zeros.</returns>
    /// <exception cref="ArgumentException">Thrown when dimensions are null or empty.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a tensor where every element is 0.
    /// Zero tensors are commonly used in neural networks for:
    /// - Initializing accumulators
    /// - Padding operations
    /// - Creating empty output tensors before filling them
    /// </para>
    /// </remarks>
    public static Tensor<T> CreateZeros(params int[] dimensions)
    {
        if (dimensions == null || dimensions.Length == 0)
            throw new ArgumentException("Dimensions cannot be null or empty.", nameof(dimensions));

        return CreateDefault(dimensions, _numOps.Zero);
    }

    /// <summary>
    /// Creates an identity matrix as a 2D tensor.
    /// </summary>
    /// <param name="size">The size of the square identity matrix.</param>
    /// <returns>A new 2D tensor representing an identity matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when size is less than 1.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> An identity matrix is a square matrix with 1s on the diagonal
    /// and 0s everywhere else. When you multiply any matrix by the identity matrix, you get
    /// the original matrix back - it's like multiplying by 1 for numbers.
    /// 
    /// For example, a 3x3 identity matrix looks like:
    /// [[1, 0, 0],
    ///  [0, 1, 0],
    ///  [0, 0, 1]]
    /// 
    /// Identity matrices are used in:
    /// - Neural network weight initialization
    /// - Skip connections and residual networks
    /// - Regularization terms
    /// </para>
    /// </remarks>
    public static Tensor<T> CreateIdentity(int size)
    {
        if (size < 1)
            throw new ArgumentException("Size must be at least 1.", nameof(size));

        var tensor = new Tensor<T>([size, size]);

        // Fill with zeros first (default), then set diagonal to 1
        for (int i = 0; i < size; i++)
        {
            tensor[i, i] = _numOps.One;
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
        // Use vectorized Subtract operation for SIMD acceleration (5-15x faster with AVX2)
        _numOps.Subtract(_data.AsSpan(), other._data.AsSpan(), result._data.AsWritableSpan());

        return result;
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
        // Support both 2D and 3D tensors
        // For 2D: [batch, features] + [features] -> broadcasts vector across batch
        // For 3D: [batch, seq, features] + [features] -> broadcasts vector across batch and seq
        if (this.Rank == 2)
        {
            if (this.Shape[1] != vector.Length)
                throw new ArgumentException($"Vector length ({vector.Length}) must match the last dimension of the tensor ({this.Shape[1]}).");

            var result = new Tensor<T>(this.Shape);
            int rowLength = this.Shape[1];
            // Use vectorized Add for each row (5-15x faster with AVX2)
            var srcSpan = _data.AsSpan();
            var destSpan = result._data.AsWritableSpan();
            for (int i = 0; i < this.Shape[0]; i++)
            {
                int offset = i * rowLength;
                var sourceRow = srcSpan.Slice(offset, rowLength);
                var destRow = destSpan.Slice(offset, rowLength);
                _numOps.Add(sourceRow, vector.AsSpan(), destRow);
            }
            return result;
        }
        else if (this.Rank == 3)
        {
            if (this.Shape[2] != vector.Length)
                throw new ArgumentException($"Vector length ({vector.Length}) must match the last dimension of the tensor ({this.Shape[2]}).");

            var result = new Tensor<T>(this.Shape);
            int lastDimLength = this.Shape[2];
            int sliceSize = this.Shape[1] * this.Shape[2];
            // Use vectorized Add for each row in the last dimension (5-15x faster with AVX2)
            var srcSpan = _data.AsSpan();
            var destSpan = result._data.AsWritableSpan();
            for (int i = 0; i < this.Shape[0]; i++)
            {
                for (int j = 0; j < this.Shape[1]; j++)
                {
                    int offset = i * sliceSize + j * lastDimLength;
                    var sourceSlice = srcSpan.Slice(offset, lastDimLength);
                    var destSlice = destSpan.Slice(offset, lastDimLength);
                    _numOps.Add(sourceSlice, vector.AsSpan(), destSlice);
                }
            }
            return result;
        }
        else
        {
            throw new ArgumentException($"Add(Vector) is only supported for 2D and 3D tensors. Got rank {this.Rank}.");
        }
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

        // TensorValidator.ValidateShape(slice, [..Shape.Skip(1)]);

        int sliceSize = slice.Length;
        int offset = index * sliceSize;

        // Use vectorized Copy operation for SIMD acceleration (5-15x faster with AVX2)
        // Use internal AsWritableSpan to get writable span - do NOT use implicit T[] conversion
        var destSpan = _data.AsWritableSpan().Slice(offset, sliceSize);
        _numOps.Copy(slice._data.AsSpan(), destSpan);
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
    /// (1ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â4) + (2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â5) + (3ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â6) = 4 + 10 + 18 = 32</para>
    /// 
    /// <para>Both tensors must have identical shapes for this operation.</para>
    /// </remarks>
    public T DotProduct(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for dot product.");

        // Use vectorized Dot product for SIMD acceleration (10-15x faster with AVX2)
        return _numOps.Dot(_data.AsSpan(), other._data.AsSpan());
    }

    /// <summary>
    /// Fills the entire tensor with a specified value.
    /// </summary>
    /// <param name="value">The value to fill the tensor with.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method replaces all elements in the tensor with the same value.
    /// It's like painting all cells in a spreadsheet with the same color.</para>
    /// <para><b>Performance:</b> Uses vectorized Fill operation for SIMD acceleration (5-15x faster with AVX2).</para>
    /// </remarks>
    public void Fill(T value)
    {
        _numOps.Fill(_data.AsWritableSpan(), value);
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
        // Use internal Span methods - do NOT use implicit T[] conversion as it creates copies
        var sourceSpan = _data.AsSpan().Slice(offset, sliceSize);
        _numOps.Copy(sourceSpan, sliceData.AsWritableSpan());

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
    /// <para><b>Performance:</b> Uses vectorized MultiplyScalar operation for SIMD acceleration (5-15x faster with AVX2).</para>
    /// </remarks>
    public Tensor<T> Scale(T factor)
    {
        var result = new Tensor<T>(this.Shape);
        _numOps.MultiplyScalar(_data.AsSpan(), factor, result._data.AsWritableSpan());
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
    /// - If you have three 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 tensors (like three rectangular sheets of paper) and stack them with axis=0,
    ///   you'll get a 3ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 tensor (like a stack of three sheets).
    /// - If you stack them with axis=1, you'll get a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 tensor (like sheets arranged side by side).
    /// - If you stack them with axis=2, you'll get a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 tensor (like sheets arranged in a grid).</para>
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
    /// Transposes the tensor by rearranging its dimensions according to the specified permutation.
    /// </summary>
    /// <param name="permutation">An array specifying the new order of dimensions.</param>
    /// <returns>A new tensor with rearranged dimensions.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the permutation array length doesn't match the tensor rank or contains invalid values.
    /// </exception>
    /// <remarks>
    /// <b>For Beginners:</b> Transposing a tensor means rearranging its dimensions.
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
        // Use vectorized Subtract operation for SIMD acceleration (5-15x faster with AVX2)
        _numOps.Subtract(_data.AsSpan(), other._data.AsSpan(), result._data.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Subtracts another tensor from this tensor in-place, modifying this tensor.
    /// </summary>
    /// <param name="other">The tensor to subtract.</param>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    /// <remarks>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated subtraction.</para>
    /// </remarks>
    public void SubtractInPlace(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for subtraction.");

        _numOps.Subtract(_data.AsSpan(), other._data.AsSpan(), _data.AsWritableSpan());
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
            // Sum all elements using vectorized Sum operation for SIMD acceleration (5-15x faster with AVX2)
            T sum = _numOps.Sum(_data.AsSpan());
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
        if (start < 0 || start >= _data.Length)
            throw new ArgumentOutOfRangeException(nameof(start), "Start index must be within bounds of the tensor data.");
        if (length < 0 || start + length > _data.Length)
            throw new ArgumentOutOfRangeException(nameof(length), "Length must not exceed remaining elements from start.");

        var result = new Vector<T>(length);
        var sourceSpan = _data.AsSpan().Slice(start, length);
        var destSpan = result.AsWritableSpan();
        _numOps.Copy(sourceSpan, destSpan);
        return result;
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
        // Use vectorized Max to find the value quickly (5-15x faster with AVX2)
        T maxVal = _numOps.Max(_data.AsSpan());

        // Find the index of the max value (requires linear scan)
        int maxIndex = 0;
        for (int i = 0; i < _data.Length; i++)
        {
            if (_numOps.Equals(_data[i], maxVal))
            {
                maxIndex = i;
                break;
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
    /// <para>For example, you could reshape a 4ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 tensor (4 rows, 3 columns) into a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â6 tensor (2 rows, 6 columns).
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
        // Use vectorized Copy operation for SIMD acceleration (5-15x faster with AVX2)
        _numOps.Copy(_data.AsSpan(), reshaped._data.AsWritableSpan());

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
    /// Multiplies this tensor by a scalar value in-place.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <remarks>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated multiplication.</para>
    /// </remarks>
    public void MultiplyInPlace(T scalar)
    {
        _numOps.MultiplyScalar(_data.AsSpan(), scalar, _data.AsWritableSpan());
    }

    /// <summary>
    /// Divides each element of this tensor by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <returns>A new tensor with each element divided by the scalar.</returns>
    /// <remarks>
    /// <para><b>Performance:</b> Uses SIMD-accelerated operations.</para>
    /// </remarks>
    public Tensor<T> Divide(T scalar)
    {
        var result = new Tensor<T>(Shape);
        _numOps.DivideScalar(_data.AsSpan(), scalar, result._data.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// Divides each element of this tensor by a scalar value in-place.
    /// </summary>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <remarks>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated division.</para>
    /// </remarks>
    public void DivideInPlace(T scalar)
    {
        _numOps.DivideScalar(_data.AsSpan(), scalar, _data.AsWritableSpan());
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
    /// <b>For Beginners:</b> This operation performs matrix multiplication between each 2D slice of the 3D tensor
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
        int lastDim = this.Shape[2];

        // Extract a row vector and matrix column for vectorized dot product
        T[] tensorRow = new T[lastDim];
        T[] matrixCol = new T[lastDim];

        for (int i = 0; i < this.Shape[0]; i++)
        {
            for (int j = 0; j < this.Shape[1]; j++)
            {
                // Extract tensor row [i,j,:] for this position
                int tensorOffset = (i * this.Shape[1] + j) * lastDim;
                for (int l = 0; l < lastDim; l++)
                {
                    tensorRow[l] = _data[tensorOffset + l];
                }

                for (int k = 0; k < matrix.Columns; k++)
                {
                    // Extract matrix column [:, k]
                    for (int l = 0; l < lastDim; l++)
                    {
                        matrixCol[l] = matrix[l, k];
                    }

                    // Use vectorized Dot product for sum of element-wise products (10-15x faster with AVX2)
                    result[i, j, k] = _numOps.Dot(new ReadOnlySpan<T>(tensorRow), new ReadOnlySpan<T>(matrixCol));
                }
            }
        }

        return result;
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
    /// <para>For example, when stacking 3 images of size [28ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â28] along a new first dimension, 
    /// the result will be a tensor of shape [3ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â28 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 28].</para>
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
    /// Performs element-wise multiplication with broadcasting support.
    /// </summary>
    /// <param name="other">The tensor to multiply with.</param>
    /// <returns>A new tensor containing the element-wise product.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method multiplies each element in this tensor with the corresponding element in the other tensor.
    /// 
    /// Broadcasting allows tensors of different shapes to be multiplied together by automatically expanding
    /// smaller dimensions to match larger ones. For example, you can multiply a 3ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â4 tensor with a 1ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â4 tensor
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
            // Use vectorized Multiply operation for SIMD acceleration (5-15x faster with AVX2)
            _numOps.Multiply(_data.AsSpan(), other._data.AsSpan(), result._data.AsWritableSpan());
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
    /// Performs matrix multiplication between two tensors with support for N-dimensional batched operations.
    /// </summary>
    /// <param name="other">The second tensor to multiply with.</param>
    /// <returns>A new tensor containing the result of the matrix multiplication.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when either tensor has fewer than 2 dimensions or when the inner dimensions don't match.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method supports both 2D matrix multiplication and N-dimensional batched matrix multiplication
    /// following NumPy-style broadcasting semantics. For tensors with shapes [..., M, K] and [..., K, N],
    /// the result has shape [..., M, N].
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Matrix multiplication is a fundamental operation in linear algebra and machine learning.
    /// </para>
    /// <para>
    /// For two matrices A and B to be multiplied:
    /// </para>
    /// <para>
    /// - The number of columns in A must equal the number of rows in B
    /// - The result will have dimensions: (rows of A) x (columns of B)
    /// </para>
    /// <para>
    /// For batched operations, shape [2, 3, 4] @ [2, 4, 5] results in [2, 3, 5].
    /// </para>
    /// <para>
    /// This is different from element-wise multiplication where corresponding elements are simply multiplied together.
    /// </para>
    /// </remarks>
    public Tensor<T> MatrixMultiply(Tensor<T> other)
    {
        if (this.Rank < 2 || other.Rank < 2)
        {
            throw new ArgumentException("MatMul requires tensors with at least 2 dimensions.");
        }

        // Get matrix dimensions (last 2 dims)
        int M = this.Shape[^2];
        int K1 = this.Shape[^1];
        int K2 = other.Shape[^2];
        int N = other.Shape[^1];

        if (K1 != K2)
        {
            throw new ArgumentException($"Incompatible matrix dimensions for multiplication: {K1} vs {K2}.");
        }

        // Handle simple 2D case
        if (this.Rank == 2 && other.Rank == 2)
        {
            return this.Multiply(other);
        }

        // Handle batched matrix multiplication
        return BatchedMatrixMultiply(other);
    }

    /// <summary>
    /// Performs batched matrix multiplication for N-dimensional tensors.
    /// </summary>
    /// <param name="other">The other tensor to multiply with.</param>
    /// <returns>The result of batched matrix multiplication.</returns>
    private Tensor<T> BatchedMatrixMultiply(Tensor<T> other)
    {
        int M = this.Shape[^2];
        int K = this.Shape[^1];
        int N = other.Shape[^1];

        // Calculate batch dimensions (all but last 2)
        var thisBatchShape = this.Shape.Take(this.Rank - 2).ToArray();
        var otherBatchShape = other.Shape.Take(other.Rank - 2).ToArray();

        // Calculate broadcasted batch shape
        var maxBatchRank = Math.Max(thisBatchShape.Length, otherBatchShape.Length);
        var batchShape = new int[maxBatchRank];

        // Pad shorter batch shape with 1s from the left
        var paddedThis = new int[maxBatchRank];
        var paddedOther = new int[maxBatchRank];
        for (int i = 0; i < maxBatchRank; i++)
        {
            paddedThis[i] = i < maxBatchRank - thisBatchShape.Length ? 1 : thisBatchShape[i - (maxBatchRank - thisBatchShape.Length)];
            paddedOther[i] = i < maxBatchRank - otherBatchShape.Length ? 1 : otherBatchShape[i - (maxBatchRank - otherBatchShape.Length)];

            // Broadcasting: dimension must be equal or one of them must be 1
            if (paddedThis[i] != paddedOther[i] && paddedThis[i] != 1 && paddedOther[i] != 1)
            {
                throw new ArgumentException($"Cannot broadcast batch dimensions: {string.Join(",", thisBatchShape)} vs {string.Join(",", otherBatchShape)}");
            }
            batchShape[i] = Math.Max(paddedThis[i], paddedOther[i]);
        }

        // Calculate total batch size
        var totalBatchSize = batchShape.Length > 0 ? batchShape.Aggregate(1, (a, b) => a * b) : 1;

        // Result shape
        var resultShape = batchShape.Concat(new[] { M, N }).ToArray();
        var result = new Tensor<T>(resultShape);

        // Flatten batch dimensions for iteration
        var thisMatrixStride = M * K;
        var otherMatrixStride = K * N;
        var resultMatrixStride = M * N;

        // Calculate strides for each tensor
        var thisStrides = CalculateBatchStrides(paddedThis);
        var otherStrides = CalculateBatchStrides(paddedOther);

        var thisData = this._data;
        var otherData = other._data;
        var resultData = result._data;

        for (int batchIdx = 0; batchIdx < totalBatchSize; batchIdx++)
        {
            // Calculate batch indices
            var batchIndices = new int[maxBatchRank];
            int remaining = batchIdx;
            for (int d = maxBatchRank - 1; d >= 0; d--)
            {
                batchIndices[d] = remaining % batchShape[d];
                remaining /= batchShape[d];
            }

            // Calculate source indices with broadcasting
            int thisOffset = 0;
            int otherOffset = 0;
            for (int d = 0; d < maxBatchRank; d++)
            {
                int thisIdx = paddedThis[d] == 1 ? 0 : batchIndices[d];
                int otherIdx = paddedOther[d] == 1 ? 0 : batchIndices[d];
                thisOffset += thisIdx * thisStrides[d] * thisMatrixStride;
                otherOffset += otherIdx * otherStrides[d] * otherMatrixStride;
            }

            int resultOffset = batchIdx * resultMatrixStride;

            // Perform matrix multiplication for this batch
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    T sum = _numOps.Zero;
                    for (int k = 0; k < K; k++)
                    {
                        var a = thisData[thisOffset + i * K + k];
                        var b = otherData[otherOffset + k * N + j];
                        sum = _numOps.Add(sum, _numOps.Multiply(a, b));
                    }
                    resultData[resultOffset + i * N + j] = sum;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates the strides for batch dimensions.
    /// </summary>
    private static int[] CalculateBatchStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        if (shape.Length == 0) return strides;

        strides[shape.Length - 1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    /// <summary>
    /// Generates all possible index combinations for iterating through a tensor.
    /// </summary>
    /// <returns>An enumerable sequence of index arrays, each representing a position in the tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a list of all possible positions (indices) in the tensor.
    /// Think of it as generating all possible coordinates to access each element in the tensor.</para>
    /// 
    /// <para>For example, in a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 tensor, this would generate the coordinates: [0,0], [0,1], [0,2], [1,0], [1,1], [1,2].</para>
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
        // Use vectorized Sum for SIMD acceleration (8-12x speedup with AVX2)
        T sum = _numOps.Sum(_data.AsSpan());

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

        // Use vectorized Copy operation per row when slicing full width, otherwise element-by-element
        int sourceCols = this.Shape[1];
        if (startCol == 0 && endCol == sourceCols)
        {
            // Full width slice - use vectorized Copy per row (5-10x faster)
            var srcSpan = _data.AsSpan();
            var destSpan = result._data.AsWritableSpan();
            for (int i = 0; i < newRows; i++)
            {
                int sourceOffset = (startRow + i) * sourceCols;
                int destOffset = i * newCols;
                _numOps.Copy(srcSpan.Slice(sourceOffset, newCols), destSpan.Slice(destOffset, newCols));
            }
        }
        else
        {
            // Partial width slice - must copy element by element
            for (int i = 0; i < newRows; i++)
            {
                for (int j = 0; j < newCols; j++)
                {
                    result[i, j] = this[startRow + i, startCol + j];
                }
            }
        }

        return result;
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
    /// <para>For example, in a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 tensor (2 rows, 3 columns), the flat indices would map like this:
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
    /// <para>For example, in a 3ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â4 tensor, the flat index 5 would correspond to position [1,1] 
    /// (second row, second column).</para>
    /// </remarks>
    private void GetIndicesFromFlatIndex(int flatIndex, int[] indices)
    {
        for (int i = Rank - 1; i >= 0; i--)
        {
            indices[i] = flatIndex % Shape[i];
            flatIndex /= Shape[i];
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
    /// <para>For example, in a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 tensor (2 rows, 3 columns), the flat indices would map like this:
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
    /// <para>For example, CreateDefault([2, 3], 1.0) would create a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 tensor filled with the value 1.0, like this:
    /// [[1.0, 1.0, 1.0],
    ///  [1.0, 1.0, 1.0]]</para>
    /// 
    /// <para>This is useful when you need a starting tensor with a specific value, such as zeros or ones.</para>
    /// </remarks>
    public static Tensor<T> CreateDefault(int[] shape, T value)
    {
        var tensor = new Tensor<T>(shape);
        // Use vectorized Fill operation for SIMD acceleration (5-15x faster with AVX2)
        _numOps.Fill(tensor._data.AsWritableSpan(), value);

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
        // TensorValidator.ValidateShape(a, b.Shape);

        Tensor<T> result = new Tensor<T>(a.Shape);
        // Use vectorized Multiply operation for SIMD acceleration (5-15x faster with AVX2)
        _numOps.Multiply(a._data.AsSpan(), b._data.AsSpan(), result._data.AsWritableSpan());

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

        // Use the Vector's ElementwiseMultiply method to perform the operation
        var resultData = _data.ElementwiseMultiply(other._data);
        return new Tensor<T>(Shape, resultData);
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
    /// Gets a slice of the tensor along a specified dimension.
    /// </summary>
    /// <param name="index">The index along the dimension to slice.</param>
    /// <param name="dimension">The dimension to slice along (0-indexed).</param>
    /// <returns>A new tensor with the specified dimension removed.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index or dimension is out of bounds.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a "slice" of data from a multi-dimensional tensor
    /// along any dimension you specify.</para>
    ///
    /// <para>For example, if you have a tensor with shape [batchSize, timeSteps, features] and want to
    /// get all the data for a specific time step (keeping the batch dimension), you would use
    /// GetSliceAlongDimension(timeStepIndex, 1).</para>
    ///
    /// <para>This is particularly useful for:
    /// <list type="bullet">
    /// <item><description>LSTM/RNN processing: iterating through time steps while keeping batch dimension</description></item>
    /// <item><description>Multi-head attention: extracting specific heads or positions</description></item>
    /// <item><description>Any operation that needs to extract along a non-batch dimension</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public Tensor<T> GetSliceAlongDimension(int index, int dimension)
    {
        if (dimension < 0 || dimension >= Shape.Length)
            throw new ArgumentOutOfRangeException(nameof(dimension), $"Dimension {dimension} is out of range for tensor with {Shape.Length} dimensions.");
        if (index < 0 || index >= Shape[dimension])
            throw new ArgumentOutOfRangeException(nameof(index), $"Index {index} is out of range for dimension {dimension} with size {Shape[dimension]}.");

        // Create new shape without the sliced dimension
        int[] newShape = new int[Shape.Length - 1];
        for (int d = 0, nd = 0; d < Shape.Length; d++)
        {
            if (d != dimension)
                newShape[nd++] = Shape[d];
        }

        var result = new Tensor<T>(newShape);

        // Calculate strides for source tensor
        int[] strides = new int[Shape.Length];
        strides[Shape.Length - 1] = 1;
        for (int d = Shape.Length - 2; d >= 0; d--)
            strides[d] = strides[d + 1] * Shape[d + 1];

        // Calculate strides for destination tensor
        int[] resultStrides = new int[newShape.Length];
        if (newShape.Length > 0)
        {
            resultStrides[newShape.Length - 1] = 1;
            for (int d = newShape.Length - 2; d >= 0; d--)
                resultStrides[d] = resultStrides[d + 1] * newShape[d + 1];
        }

        // Copy elements into temporary array, then copy to result
        int resultLength = newShape.Length > 0 ? newShape.Aggregate(1, (a, b) => a * b) : 1;
        T[] destArray = new T[resultLength];
        int resultIdx = 0;
        CopySliceRecursive(_data.ToArray(), destArray, Shape, newShape, strides, dimension, index, 0, 0, ref resultIdx);
        result.CopyFromArray(destArray);

        return result;
    }

    private void CopySliceRecursive(T[] source, T[] dest, int[] shape, int[] newShape,
        int[] strides, int sliceDim, int sliceIdx, int currentDim, int sourceOffset, ref int destIdx)
    {
        if (currentDim == shape.Length)
        {
            dest[destIdx++] = source[sourceOffset];
            return;
        }

        if (currentDim == sliceDim)
        {
            // Skip to the specific index in the slice dimension
            int newSourceOffset = sourceOffset + sliceIdx * strides[currentDim];
            CopySliceRecursive(source, dest, shape, newShape, strides, sliceDim, sliceIdx, currentDim + 1, newSourceOffset, ref destIdx);
        }
        else
        {
            // Iterate through all indices in non-slice dimensions
            for (int i = 0; i < shape[currentDim]; i++)
            {
                int newSourceOffset = sourceOffset + i * strides[currentDim];
                CopySliceRecursive(source, dest, shape, newShape, strides, sliceDim, sliceIdx, currentDim + 1, newSourceOffset, ref destIdx);
            }
        }
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
    /// <para>For example, if you have a 3ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â4 matrix representing student test scores (3 students, 4 tests),
    /// this method would convert it to a tensor with the same structure but with the ability to perform
    /// more advanced operations on the data.</para>
    /// 
    /// <para>Internally, the matrix is converted to a row-major flattened vector before being stored
    /// in the tensor's internal data array. This matches Tensor's row-major storage order.</para>
    /// </remarks>
    public static Tensor<T> FromMatrix(Matrix<T> matrix)
    {
        // Use ToRowVector() for row-major order, consistent with Tensor's internal storage
        // and ToMatrix() method which also uses row-major element-wise copy.
        // ToColumnVector() would produce column-major data causing transposition.
        return FromRowMatrix(matrix);
    }

    /// <summary>
    /// Creates a tensor from a matrix using row-major order (standard C# memory layout).
    /// </summary>
    /// <param name="matrix">The matrix to convert.</param>
    /// <returns>A new tensor with the same values as the matrix in row-major order.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Row-major order means the matrix is stored row by row.
    /// For a 2x3 matrix [[1,2,3],[4,5,6]], the internal storage is [1,2,3,4,5,6].</para>
    /// <para>This is the standard layout for C# arrays and is consistent with Tensor's
    /// internal storage and the ToMatrix() method.</para>
    /// </remarks>
    public static Tensor<T> FromRowMatrix(Matrix<T> matrix)
    {
        return new Tensor<T>([matrix.Rows, matrix.Columns], matrix.ToRowVector());
    }

    /// <summary>
    /// Creates a tensor from a matrix using column-major order (Fortran/MATLAB layout).
    /// </summary>
    /// <param name="matrix">The matrix to convert.</param>
    /// <returns>A new tensor with the same values as the matrix in column-major order.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Column-major order means the matrix is stored column by column.
    /// For a 2x3 matrix [[1,2,3],[4,5,6]], the internal storage is [1,4,2,5,3,6].</para>
    /// <para><b>Warning:</b> This layout is different from Tensor's native row-major order.
    /// Using this method will result in a tensor where element access via indices will
    /// return values as if the matrix was transposed. Only use this if you specifically
    /// need column-major compatibility (e.g., interop with Fortran or MATLAB libraries).</para>
    /// </remarks>
    public static Tensor<T> FromColumnMatrix(Matrix<T> matrix)
    {
        return new Tensor<T>([matrix.Rows, matrix.Columns], matrix.ToColumnVector());
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
        var tensor = new Tensor<T>([1]);

        // Set the first (and only) element to the provided value
        tensor[0] = value;

        return tensor;
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
    /// Adds two tensors element-wise.
    /// </summary>
    /// <param name="left">The first tensor.</param>
    /// <param name="right">The second tensor.</param>
    /// <returns>A new tensor containing the sum of the two tensors.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This operator adds two tensors together by adding their corresponding elements.
    /// Both tensors must have exactly the same shape for this to work.
    /// 
    /// For example, if you have two 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 matrices:
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
    /// For example, multiplying a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 tensor by a 3ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â4 tensor results in a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â4 tensor.
    /// This is different from element-wise multiplication, which would require both tensors to have the same shape.
    /// </para>
    /// </remarks>
    public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Adds another tensor to this tensor element-wise.
    /// </summary>
    /// <param name="other">The tensor to add.</param>
    /// <returns>A new tensor containing the sum of this tensor and the other tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adds two tensors together by adding their corresponding elements.
    /// Both tensors must have exactly the same shape for this to work.
    /// 
    /// For example, if you have two 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 matrices:
    /// ```
    /// A = [[1, 2, 3],     B = [[5, 6, 7],
    ///      [4, 5, 6]]          [8, 9, 10]]
    /// ```
    /// 
    /// Then A.Add(B) would result in:
    /// ```
    /// [[1+5, 2+6, 3+7],    [[6, 8, 10],
    ///  [4+8, 5+9, 6+10]] =  [12, 14, 16]]
    /// ```
    /// </para>
    /// </remarks>
    public Tensor<T> Add(Tensor<T> other)
    {
        // TensorValidator.ValidateShape(this, other.Shape);

        var result = new Tensor<T>(Shape);
        // Use vectorized Add operation for SIMD acceleration (5-15x faster with AVX2)
        _numOps.Add(_data.AsSpan(), other._data.AsSpan(), result._data.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// Adds another tensor to this tensor in-place, modifying this tensor.
    /// </summary>
    /// <param name="other">The tensor to add.</param>
    /// <exception cref="ArgumentException">Thrown when tensors have different shapes.</exception>
    /// <remarks>
    /// <para><b>Performance:</b> Zero-allocation SIMD-accelerated addition.</para>
    /// </remarks>
    public void AddInPlace(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for addition.");

        _numOps.Add(_data.AsSpan(), other._data.AsSpan(), _data.AsWritableSpan());
    }

    /// <summary>
    /// Adds two tensors with broadcasting support, following NumPy/PyTorch broadcasting rules.
    /// </summary>
    /// <param name="other">The tensor to add. Can have different shape if broadcastable.</param>
    /// <returns>A new tensor containing the element-wise sum with broadcasting.</returns>
    /// <exception cref="ArgumentException">Thrown when shapes are not broadcastable.</exception>
    /// <remarks>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be added together by automatically expanding
    /// dimensions of size 1 to match the other tensor. This follows NumPy/PyTorch broadcasting semantics.
    /// </para>
    /// <para><b>For Beginners:</b> Broadcasting lets you add tensors of different shapes.
    ///
    /// For example:
    /// - [4, 3, 2] + [2] broadcasts the [2] across all positions
    /// - [4, 3, 2] + [1, 3, 1] broadcasts along dimensions 0 and 2
    /// - [batch, channels, H, W] + [1, channels, 1, 1] adds per-channel bias
    ///
    /// The rule is: dimensions are compatible if they're equal or one of them is 1.
    /// </para>
    /// </remarks>
    public Tensor<T> BroadcastAdd(Tensor<T> other)
    {
        // Check if shapes are already identical - use fast path
        if (Shape.SequenceEqual(other.Shape))
        {
            return Add(other);
        }

        // Get broadcast shape
        int[] broadcastShape = GetBroadcastShape(this.Shape, other.Shape);
        var result = new Tensor<T>(broadcastShape);

        // Pad shapes to same rank for easier indexing
        int maxRank = broadcastShape.Length;
        int[] thisShape = new int[maxRank];
        int[] otherShape = new int[maxRank];

        // Right-align shapes (prepend 1s)
        int thisOffset = maxRank - this.Rank;
        int otherOffset = maxRank - other.Rank;

        for (int i = 0; i < maxRank; i++)
        {
            thisShape[i] = i < thisOffset ? 1 : this.Shape[i - thisOffset];
            otherShape[i] = i < otherOffset ? 1 : other.Shape[i - otherOffset];
        }

        // Iterate over the result tensor
        int[] thisIndices = new int[this.Rank];
        int[] otherIndices = new int[other.Rank];

        foreach (var index in result.GetIndices())
        {
            // Map result index to this tensor's index (accounting for broadcasting)
            for (int i = 0; i < this.Rank; i++)
            {
                int broadcastIdx = i + thisOffset;
                thisIndices[i] = thisShape[broadcastIdx] == 1 ? 0 : index[broadcastIdx];
            }

            // Map result index to other tensor's index (accounting for broadcasting)
            for (int i = 0; i < other.Rank; i++)
            {
                int broadcastIdx = i + otherOffset;
                otherIndices[i] = otherShape[broadcastIdx] == 1 ? 0 : index[broadcastIdx];
            }

            // Perform addition
            result[index] = _numOps.Add(this[thisIndices], other[otherIndices]);
        }

        return result;
    }

    /// <summary>
    /// Subtracts another tensor from this tensor with NumPy-style broadcasting.
    /// </summary>
    /// <param name="other">The tensor to subtract.</param>
    /// <returns>A new tensor containing the element-wise difference with broadcasting.</returns>
    /// <remarks>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be subtracted by automatically
    /// expanding the smaller tensor. For example, [B,H,W,C] - [B,1,1,C] broadcasts the [B,1,1,C]
    /// tensor across the spatial dimensions.
    /// </para>
    /// <para>
    /// The rule is: dimensions are compatible if they're equal or one of them is 1.
    /// </para>
    /// </remarks>
    public Tensor<T> BroadcastSubtract(Tensor<T> other)
    {
        // Check if shapes are already identical - use fast path
        if (Shape.SequenceEqual(other.Shape))
        {
            return Subtract(other);
        }

        // Get broadcast shape
        int[] broadcastShape = GetBroadcastShape(this.Shape, other.Shape);
        var result = new Tensor<T>(broadcastShape);

        // Pad shapes to same rank for easier indexing
        int maxRank = broadcastShape.Length;
        int[] thisShape = new int[maxRank];
        int[] otherShape = new int[maxRank];

        // Right-align shapes (prepend 1s)
        int thisOffset = maxRank - this.Rank;
        int otherOffset = maxRank - other.Rank;

        for (int i = 0; i < maxRank; i++)
        {
            thisShape[i] = i < thisOffset ? 1 : this.Shape[i - thisOffset];
            otherShape[i] = i < otherOffset ? 1 : other.Shape[i - otherOffset];
        }

        // Iterate over the result tensor
        int[] thisIndices = new int[this.Rank];
        int[] otherIndices = new int[other.Rank];

        foreach (var index in result.GetIndices())
        {
            // Map result index to this tensor's index (accounting for broadcasting)
            for (int i = 0; i < this.Rank; i++)
            {
                int broadcastIdx = i + thisOffset;
                thisIndices[i] = thisShape[broadcastIdx] == 1 ? 0 : index[broadcastIdx];
            }

            // Map result index to other tensor's index (accounting for broadcasting)
            for (int i = 0; i < other.Rank; i++)
            {
                int broadcastIdx = i + otherOffset;
                otherIndices[i] = otherShape[broadcastIdx] == 1 ? 0 : index[broadcastIdx];
            }

            // Perform subtraction
            result[index] = _numOps.Subtract(this[thisIndices], other[otherIndices]);
        }

        return result;
    }

    /// <summary>
    /// Multiplies this tensor by another tensor with NumPy-style broadcasting.
    /// </summary>
    /// <param name="other">The tensor to multiply by.</param>
    /// <returns>A new tensor containing the element-wise product with broadcasting.</returns>
    /// <remarks>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be multiplied together by automatically
    /// expanding the smaller tensor. For example, [B,H,W,C] * [B,1,1,C] broadcasts the [B,1,1,C]
    /// tensor across the spatial dimensions.
    /// </para>
    /// </remarks>
    public Tensor<T> BroadcastMultiply(Tensor<T> other)
    {
        // Check if shapes are already identical - use fast path (element-wise multiply)
        if (Shape.SequenceEqual(other.Shape))
        {
            // Element-wise multiplication, not matrix multiplication
            var fastResult = new Tensor<T>(Shape);
            var srcSpan = _data.AsSpan();
            var otherSpan = other._data.AsSpan();
            var destSpan = fastResult._data.AsWritableSpan();
            for (int i = 0; i < Length; i++)
            {
                destSpan[i] = _numOps.Multiply(srcSpan[i], otherSpan[i]);
            }
            return fastResult;
        }

        // Get broadcast shape
        int[] broadcastShape = GetBroadcastShape(this.Shape, other.Shape);
        var result = new Tensor<T>(broadcastShape);

        // Pad shapes to same rank for easier indexing
        int maxRank = broadcastShape.Length;
        int[] thisShape = new int[maxRank];
        int[] otherShape = new int[maxRank];

        // Right-align shapes (prepend 1s)
        int thisOffset = maxRank - this.Rank;
        int otherOffset = maxRank - other.Rank;

        for (int i = 0; i < maxRank; i++)
        {
            thisShape[i] = i < thisOffset ? 1 : this.Shape[i - thisOffset];
            otherShape[i] = i < otherOffset ? 1 : other.Shape[i - otherOffset];
        }

        // Iterate over the result tensor
        int[] thisIndices = new int[this.Rank];
        int[] otherIndices = new int[other.Rank];

        foreach (var index in result.GetIndices())
        {
            // Map result index to this tensor's index (accounting for broadcasting)
            for (int i = 0; i < this.Rank; i++)
            {
                int broadcastIdx = i + thisOffset;
                thisIndices[i] = thisShape[broadcastIdx] == 1 ? 0 : index[broadcastIdx];
            }

            // Map result index to other tensor's index (accounting for broadcasting)
            for (int i = 0; i < other.Rank; i++)
            {
                int broadcastIdx = i + otherOffset;
                otherIndices[i] = otherShape[broadcastIdx] == 1 ? 0 : index[broadcastIdx];
            }

            // Perform multiplication
            result[index] = _numOps.Multiply(this[thisIndices], other[otherIndices]);
        }

        return result;
    }

    /// <summary>
    /// Divides this tensor by another tensor with NumPy-style broadcasting.
    /// </summary>
    /// <param name="other">The tensor to divide by.</param>
    /// <returns>A new tensor containing the element-wise quotient with broadcasting.</returns>
    /// <remarks>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be divided by automatically
    /// expanding the smaller tensor. For example, [B,H,W,C] / [B,1,1,C] broadcasts the [B,1,1,C]
    /// tensor across the spatial dimensions.
    /// </para>
    /// <para>
    /// The rule is: dimensions are compatible if they're equal or one of them is 1.
    /// </para>
    /// </remarks>
    public Tensor<T> BroadcastDivide(Tensor<T> other)
    {
        // Check if shapes are already identical - use fast path (element-wise divide)
        if (Shape.SequenceEqual(other.Shape))
        {
            var fastResult = new Tensor<T>(Shape);
            var srcSpan = _data.AsSpan();
            var otherSpan = other._data.AsSpan();
            var destSpan = fastResult._data.AsWritableSpan();
            for (int i = 0; i < Length; i++)
            {
                destSpan[i] = _numOps.Divide(srcSpan[i], otherSpan[i]);
            }
            return fastResult;
        }

        // Get broadcast shape
        int[] broadcastShape = GetBroadcastShape(this.Shape, other.Shape);
        var result = new Tensor<T>(broadcastShape);

        // Pad shapes to same rank for easier indexing
        int maxRank = broadcastShape.Length;
        int[] thisShape = new int[maxRank];
        int[] otherShape = new int[maxRank];

        // Right-align shapes (prepend 1s)
        int thisOffset = maxRank - this.Rank;
        int otherOffset = maxRank - other.Rank;

        for (int i = 0; i < maxRank; i++)
        {
            thisShape[i] = i < thisOffset ? 1 : this.Shape[i - thisOffset];
            otherShape[i] = i < otherOffset ? 1 : other.Shape[i - otherOffset];
        }

        // Iterate over the result tensor
        int[] thisIndices = new int[this.Rank];
        int[] otherIndices = new int[other.Rank];

        foreach (var index in result.GetIndices())
        {
            // Map result index to this tensor's index (accounting for broadcasting)
            for (int i = 0; i < this.Rank; i++)
            {
                int broadcastIdx = i + thisOffset;
                thisIndices[i] = thisShape[broadcastIdx] == 1 ? 0 : index[broadcastIdx];
            }

            // Map result index to other tensor's index (accounting for broadcasting)
            for (int i = 0; i < other.Rank; i++)
            {
                int broadcastIdx = i + otherOffset;
                otherIndices[i] = otherShape[broadcastIdx] == 1 ? 0 : index[broadcastIdx];
            }

            // Perform division
            result[index] = _numOps.Divide(this[thisIndices], other[otherIndices]);
        }

        return result;
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
    /// For example, multiplying a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â3 tensor by a 3ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â4 tensor results in a 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â4 tensor.
    /// This is different from element-wise multiplication, which would require both tensors to have the same shape.
    /// </para>
    /// </remarks>
    public Tensor<T> Multiply(Tensor<T> other)
    {
        // Support 2D matrix multiplication and 3D batch matrix multiplication
        if (Shape.Length == 2 && other.Shape.Length == 2)
        {
            // Standard 2D matrix multiplication
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
        else if (Shape.Length == 3 && other.Shape.Length == 3)
        {
            // 3D batch matrix multiplication: (batch, m, k) @ (batch, k, n) -> (batch, m, n)
            if (Shape[0] != other.Shape[0])
            {
                throw new ArgumentException("Batch dimensions must match for batch matrix multiplication.");
            }

            if (Shape[2] != other.Shape[1])
            {
                throw new ArgumentException("The number of columns in the first tensor must equal the number of rows in the second tensor for each batch.");
            }

            int batchSize = Shape[0];
            int resultRows = Shape[1];
            int resultCols = other.Shape[2];
            int commonDim = Shape[2];

            var result = new Tensor<T>(new[] { batchSize, resultRows, resultCols });

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < resultRows; i++)
                {
                    for (int j = 0; j < resultCols; j++)
                    {
                        T sum = _numOps.Zero;
                        for (int k = 0; k < commonDim; k++)
                        {
                            sum = _numOps.Add(sum, _numOps.Multiply(this[b, i, k], other[b, k, j]));
                        }
                        result[b, i, j] = sum;
                    }
                }
            }

            return result;
        }
        else if (Shape.Length == 3 && other.Shape.Length == 2)
        {
            // 3D @ 2D with broadcasting: (batch, m, k) @ (k, n) -> (batch, m, n)
            // The 2D matrix is broadcast across the batch dimension
            if (Shape[2] != other.Shape[0])
            {
                throw new ArgumentException($"Matrix dimensions don't match for multiplication: ({Shape[1]}, {Shape[2]}) @ ({other.Shape[0]}, {other.Shape[1]})");
            }

            int batchSize = Shape[0];
            int resultRows = Shape[1];
            int resultCols = other.Shape[1];
            int commonDim = Shape[2];

            var result = new Tensor<T>(new[] { batchSize, resultRows, resultCols });

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < resultRows; i++)
                {
                    for (int j = 0; j < resultCols; j++)
                    {
                        T sum = _numOps.Zero;
                        for (int k = 0; k < commonDim; k++)
                        {
                            sum = _numOps.Add(sum, _numOps.Multiply(this[b, i, k], other[k, j]));
                        }
                        result[b, i, j] = sum;
                    }
                }
            }

            return result;
        }
        else if (Shape.Length == 4 && other.Shape.Length == 4)
        {
            // 4D batch-head matrix multiplication for multi-head attention:
            // (batch, heads, m, k) @ (batch, heads, k, n) -> (batch, heads, m, n)
            if (Shape[0] != other.Shape[0])
            {
                throw new ArgumentException($"Batch dimensions must match: {Shape[0]} vs {other.Shape[0]}");
            }

            if (Shape[1] != other.Shape[1])
            {
                throw new ArgumentException($"Head dimensions must match: {Shape[1]} vs {other.Shape[1]}");
            }

            if (Shape[3] != other.Shape[2])
            {
                throw new ArgumentException($"Matrix dimensions don't match for multiplication: inner dims {Shape[3]} vs {other.Shape[2]}");
            }

            int batchSize = Shape[0];
            int numHeads = Shape[1];
            int resultRows = Shape[2];
            int resultCols = other.Shape[3];
            int commonDim = Shape[3];

            var result = new Tensor<T>(new[] { batchSize, numHeads, resultRows, resultCols });

            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int i = 0; i < resultRows; i++)
                    {
                        for (int j = 0; j < resultCols; j++)
                        {
                            T sum = _numOps.Zero;
                            for (int k = 0; k < commonDim; k++)
                            {
                                sum = _numOps.Add(sum, _numOps.Multiply(this[b, h, i, k], other[b, h, k, j]));
                            }
                            result[b, h, i, j] = sum;
                        }
                    }
                }
            }

            return result;
        }
        else if (Shape.Length == 4 && other.Shape.Length == 2)
        {
            // 4D @ 2D with broadcasting: (batch, heads, m, k) @ (k, n) -> (batch, heads, m, n)
            // The 2D matrix is broadcast across batch and head dimensions
            if (Shape[3] != other.Shape[0])
            {
                throw new ArgumentException($"Matrix dimensions don't match for multiplication: inner dim {Shape[3]} vs {other.Shape[0]}");
            }

            int batchSize = Shape[0];
            int numHeads = Shape[1];
            int resultRows = Shape[2];
            int resultCols = other.Shape[1];
            int commonDim = Shape[3];

            var result = new Tensor<T>(new[] { batchSize, numHeads, resultRows, resultCols });

            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int i = 0; i < resultRows; i++)
                    {
                        for (int j = 0; j < resultCols; j++)
                        {
                            T sum = _numOps.Zero;
                            for (int k = 0; k < commonDim; k++)
                            {
                                sum = _numOps.Add(sum, _numOps.Multiply(this[b, h, i, k], other[k, j]));
                            }
                            result[b, h, i, j] = sum;
                        }
                    }
                }
            }

            return result;
        }
        else
        {
            throw new NotSupportedException($"Multiplication is not supported for tensors with shapes {string.Join("x", Shape)} and {string.Join("x", other.Shape)}. Supported: 2D×2D, 3D×3D, 3D×2D, 4D×4D, 4D×2D.");
        }
    }

    /// <summary>
    /// Transposes the tensor.
    /// </summary>
    /// <returns>A new tensor that is the transpose of this tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Transposing a tensor means swapping its dimensions.
    ///
    /// For different tensor ranks:
    /// - 1D tensors: Returns a copy (transpose has no effect on vectors)
    /// - 2D tensors: Swaps rows and columns (standard matrix transpose)
    /// - N-D tensors: Reverses all dimensions (e.g., shape [2,3,4] becomes [4,3,2])
    ///
    /// For example, if you have a 2x3 matrix:
    /// ```
    /// A = [[1, 2, 3],
    ///      [4, 5, 6]]
    /// ```
    /// 
    /// Then A.Transpose() would result in a 3x2 matrix:
    /// ```
    /// [[1, 4],
    ///  [2, 5],
    ///  [3, 6]]
    /// ```
    /// </para>
    /// </remarks>
    public Tensor<T> Transpose()
    {
        if (Shape.Length == 1)
        {
            // 1D tensor: return a copy (transpose has no effect)
            return Clone();
        }
        else if (Shape.Length == 2)
        {
            // 2D tensor: swap rows and columns
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
        else
        {
            // N-dimensional tensor: reverse all dimensions (default behavior)
            // For example, [2,3,4] becomes [4,3,2]
            var permutation = Enumerable.Range(0, Rank).Reverse().ToArray();
            return Transpose(permutation);
        }
    }

    /// <summary>
    /// Swaps the last two dimensions of the tensor.
    /// </summary>
    /// <returns>A new tensor with the last two dimensions swapped.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is commonly used in batch matrix operations where
    /// you want to transpose the matrix part of a tensor while keeping batch dimensions intact.
    ///
    /// For example, for a tensor with shape [batch, rows, cols], this will produce
    /// a tensor with shape [batch, cols, rows].</para>
    /// </remarks>
    public Tensor<T> TransposeLast2D()
    {
        if (Rank < 2)
        {
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to transpose last 2D.");
        }

        // Create permutation that swaps only the last two dimensions
        var permutation = Enumerable.Range(0, Rank).ToArray();
        permutation[Rank - 2] = Rank - 1;
        permutation[Rank - 1] = Rank - 2;

        return Transpose(permutation);
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

        // TensorValidator.ValidateShape(slice, expectedSliceShape);

        // Calculate strides for source tensor
        int[] strides = new int[Rank];
        strides[Rank - 1] = 1;
        for (int d = Rank - 2; d >= 0; d--)
            strides[d] = strides[d + 1] * Shape[d + 1];

        // Use recursive helper to copy elements
        // Get data as array, modify it, then copy back
        T[] destArray = _data.ToArray();
        int sliceIdx = 0;
        SetSliceRecursive(destArray, slice._data.ToArray(), Shape, expectedSliceShape, strides, dimension, index, 0, 0, ref sliceIdx);
        CopyFromArray(destArray);
    }

    private void SetSliceRecursive(T[] dest, T[] source, int[] destShape, int[] sourceShape,
        int[] destStrides, int sliceDim, int sliceIdx, int currentDim, int destOffset, ref int sourceIdx)
    {
        if (currentDim == destShape.Length)
        {
            dest[destOffset] = source[sourceIdx++];
            return;
        }

        if (currentDim == sliceDim)
        {
            // Skip to the specific index in the slice dimension
            int newDestOffset = destOffset + sliceIdx * destStrides[currentDim];
            SetSliceRecursive(dest, source, destShape, sourceShape, destStrides, sliceDim, sliceIdx, currentDim + 1, newDestOffset, ref sourceIdx);
        }
        else
        {
            // Iterate through all indices in non-slice dimensions
            for (int i = 0; i < destShape[currentDim]; i++)
            {
                int newDestOffset = destOffset + i * destStrides[currentDim];
                SetSliceRecursive(dest, source, destShape, sourceShape, destStrides, sliceDim, sliceIdx, currentDim + 1, newDestOffset, ref sourceIdx);
            }
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

        // Use vectorized Sum for each slice (5-15x faster with AVX2)
        for (int i = 0; i < _data.Length; i += axisSize)
        {
            var slice = new ReadOnlySpan<T>(_data, i, axisSize);
            result._data[i / axisSize] = _numOps.Sum(slice);
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

        // Use vectorized Max for each slice (5-15x faster with AVX2)
        for (int i = 0; i < _data.Length; i += axisSize)
        {
            var slice = new ReadOnlySpan<T>(_data, i, axisSize);
            result._data[i / axisSize] = _numOps.Max(slice);
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
        T divisor = _numOps.FromDouble(axisSize);

        // Use vectorized Sum for each slice (5-15x faster with AVX2)
        for (int i = 0; i < _data.Length; i += axisSize)
        {
            var slice = new ReadOnlySpan<T>(_data, i, axisSize);
            T sum = _numOps.Sum(slice);
            result._data[i / axisSize] = _numOps.Divide(sum, divisor);
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
