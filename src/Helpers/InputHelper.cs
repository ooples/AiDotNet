namespace AiDotNet.Helpers;

/// <summary>
/// Provides helper methods for input-related operations.
/// </summary>
public static class InputHelper<T, TInput>
{
    /// <summary>
    /// Gets the batch size from the input data.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <returns>The batch size of the input data.</returns>
    public static int GetBatchSize(TInput input)
    {
        return input switch
        {
            Matrix<T> matrix => matrix.Rows,
            Tensor<T> tensor => tensor.Shape[0],
            _ => throw new ArgumentException("Unsupported input type")
        };
    }

    /// <summary>
    /// Gets the size of the input data.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <returns>The size of the input data.</returns>
    public static int GetInputSize(TInput input)
    {
        return input switch
        {
            Matrix<T> matrix => matrix.Columns,
            Tensor<T> tensor => tensor.Shape.Length >= 2
                ? (tensor.Shape.Length == 2
                    ? tensor.Shape[1]
                    : tensor.Shape.Skip(1).Aggregate((a, b) => a * b))
                : tensor.Shape[0],
            _ => throw new ArgumentException("Unsupported input type")
        };
    }

    /// <summary>
    /// Gets an element at the specified position from the input data structure.
    /// </summary>
    /// <param name="input">The input data structure.</param>
    /// <param name="row">The row index.</param>
    /// <param name="column">The column index.</param>
    /// <returns>The element at the specified position.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the input is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the input type is not supported.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the indices are out of range.</exception>
    /// <remarks>
    /// <para>
    /// This method extracts a specific element from the input data structure based on row and column indices.
    /// It supports Matrix&lt;T&gt;, Vector&lt;T&gt;, and Tensor&lt;T&gt; input types.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps you access specific values from your data.
    /// Think of it like looking up a value in a spreadsheet by providing the row and column numbers.
    /// </para>
    /// </remarks>
    public static T GetElement(TInput input, int row, int column)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input data cannot be null.");
        }

        // Handle Matrix<T>
        if (input is Matrix<T> matrix)
        {
            if (row < 0 || row >= matrix.Rows || column < 0 || column >= matrix.Columns)
            {
                throw new ArgumentOutOfRangeException(
                    $"Indices ({row}, {column}) are out of range for matrix with dimensions {matrix.Rows}x{matrix.Columns}.");
            }
            return matrix[row, column];
        }

        // Handle Vector<T>
        else if (input is Vector<T> vector)
        {
            // For vectors, we'll use the row index if column is 0, or column index if row is 0
            if (column == 0 && row >= 0 && row < vector.Length)
            {
                return vector[row];
            }
            else if (row == 0 && column >= 0 && column < vector.Length)
            {
                return vector[column];
            }

            throw new ArgumentOutOfRangeException(
                $"Invalid indices ({row}, {column}) for vector with length {vector.Length}. " +
                $"For vectors, either row or column must be 0, and the other index must be within bounds.");
        }

        // Handle Tensor<T>
        else if (input is Tensor<T> tensor)
        {
            if (tensor.Shape.Length < 2)
            {
                // Handle 1D tensor
                if (tensor.Shape.Length == 1)
                {
                    int index = column == 0 ? row : column;
                    if (index < 0 || index >= tensor.Shape[0])
                    {
                        throw new ArgumentOutOfRangeException(
                            $"Index {index} is out of range for 1D tensor with length {tensor.Shape[0]}.");
                    }
                    return tensor[index];
                }

                throw new ArgumentException(
                    $"Tensor has {tensor.Shape.Length} dimensions. At least 1 dimension is required to access elements using indices.");
            }

            // Handle 2D or higher tensor
            if (row < 0 || row >= tensor.Shape[0] || column < 0 ||
                (tensor.Shape.Length > 1 && column >= tensor.Shape[1]))
            {
                throw new ArgumentOutOfRangeException(
                    $"Indices ({row}, {column}) are out of range for tensor with shape {string.Join("x", tensor.Shape)}.");
            }

            return tensor[row, column];
        }

        throw new ArgumentException(
            $"Unsupported input type: {input.GetType().Name}. " +
            $"The GetElement method only supports Matrix<T>, Vector<T>, and Tensor<T>.");
    }

    /// <summary>
    /// Extracts a batch of data from the input based on the specified indices.
    /// </summary>
    /// <typeparam name="TInput">The type of input data structure.</typeparam>
    /// <typeparam name="T">The numeric data type of the elements.</typeparam>
    /// <param name="input">The input data structure.</param>
    /// <param name="indices">The indices of the elements or rows to include in the batch.</param>
    /// <returns>A new data structure containing only the selected elements.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the input or indices are null.</exception>
    /// <exception cref="ArgumentException">Thrown when the input type is not supported or indices are invalid.</exception>
    public static TInput GetBatch(TInput input, int[] indices)
    {
        // Validate input parameters
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input data cannot be null.");
        }

        if (indices == null)
        {
            throw new ArgumentNullException(nameof(indices), "Indices array cannot be null.");
        }

        if (indices.Length == 0)
        {
            throw new ArgumentException("Indices array cannot be empty.", nameof(indices));
        }

        try
        {
            // Handle Matrix<T>
            if (input is Matrix<T> matrix)
            {
                return (TInput)GetMatrixBatch(matrix, indices);
            }
            // Handle Vector<T>
            else if (input is Vector<T> vector)
            {
                return (TInput)GetVectorBatch(vector, indices);
            }
            // Handle Tensor<T>
            else if (input is Tensor<T> tensor)
            {
                return (TInput)GetTensorBatch(tensor, indices);
            }

            throw new ArgumentException(
                $"Unsupported input type: {input.GetType().Name}. " +
                $"The GetBatch method only supports Matrix<T>, Vector<T>, and Tensor<T>.");
        }
        catch (InvalidCastException ex)
        {
            throw new InvalidOperationException(
                $"Failed to cast result to type {typeof(TInput).Name}. Ensure the input type is compatible with batch operation.", ex);
        }
    }

    /// <summary>
    /// Extracts a batch from a matrix based on the specified row indices.
    /// </summary>
    private static object GetMatrixBatch(Matrix<T> matrix, int[] indices)
    {
        // Pre-validate indices in a single pass for better error messages
        int maxIndex = matrix.Rows - 1;
        int? outOfRangeIndex = null;

        for (int i = 0; i < indices.Length && outOfRangeIndex == null; i++)
        {
            if (indices[i] < 0 || indices[i] > maxIndex)
            {
                outOfRangeIndex = indices[i];
            }
        }

        if (outOfRangeIndex.HasValue)
        {
            throw new ArgumentOutOfRangeException(nameof(indices),
                $"Index {outOfRangeIndex} is out of range for matrix with {matrix.Rows} rows.");
        }

        // Create result and populate in a single pass
        var result = new Matrix<T>(indices.Length, matrix.Columns);
        for (int i = 0; i < indices.Length; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = matrix[indices[i], j];
            }
        }

        return result;
    }

    /// <summary>
    /// Extracts a batch from a vector based on the specified indices.
    /// </summary>
    private static object GetVectorBatch(Vector<T> vector, int[] indices)
    {
        // Pre-validate indices in a single pass
        int maxIndex = vector.Length - 1;
        int? outOfRangeIndex = null;

        for (int i = 0; i < indices.Length && outOfRangeIndex == null; i++)
        {
            if (indices[i] < 0 || indices[i] > maxIndex)
            {
                outOfRangeIndex = indices[i];
            }
        }

        if (outOfRangeIndex.HasValue)
        {
            throw new ArgumentOutOfRangeException(nameof(indices),
                $"Index {outOfRangeIndex} is out of range for vector with length {vector.Length}.");
        }

        // Create result and populate in a single pass
        var result = new Vector<T>(indices.Length);
        for (int i = 0; i < indices.Length; i++)
        {
            result[i] = vector[indices[i]];
        }

        return result;
    }

    /// <summary>
    /// Extracts a batch from a tensor based on the specified indices along the first dimension.
    /// </summary>
    private static object GetTensorBatch(Tensor<T> tensor, int[] indices)
    {
        // Check tensor rank
        if (tensor.Shape.Length < 1)
        {
            throw new ArgumentException("Cannot get batch from a scalar tensor.", nameof(tensor));
        }

        // Pre-validate indices in a single pass
        int maxIndex = tensor.Shape[0] - 1;
        int? outOfRangeIndex = null;

        for (int i = 0; i < indices.Length && outOfRangeIndex == null; i++)
        {
            if (indices[i] < 0 || indices[i] > maxIndex)
            {
                outOfRangeIndex = indices[i];
            }
        }

        if (outOfRangeIndex.HasValue)
        {
            throw new ArgumentOutOfRangeException(nameof(indices),
                $"Index {outOfRangeIndex} is out of range for tensor with first dimension size {tensor.Shape[0]}.");
        }

        // Create new tensor shape with updated first dimension
        int[] newShape = new int[tensor.Shape.Length];
        newShape[0] = indices.Length;

        // Calculate slice size once
        int sliceSize = 1;
        for (int i = 1; i < tensor.Shape.Length; i++)
        {
            newShape[i] = tensor.Shape[i];
            sliceSize *= tensor.Shape[i];
        }

        // Create a new tensor with the selected slices
        var result = new Tensor<T>(newShape);

        // Use optimized approach based on tensor rank
        if (tensor.Shape.Length == 1)
        {
            // Most efficient approach for 1D tensors - direct indexing
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = tensor[indices[i]];
            }
        }
        else if (tensor.Shape.Length == 2)
        {
            // Efficient approach for 2D tensors - row-wise copying
            for (int i = 0; i < indices.Length; i++)
            {
                int sourceIndex = indices[i];
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    result[i, j] = tensor[sourceIndex, j];
                }
            }
        }
        else
        {
            // Optimized block transfer for higher-dimensional tensors
            // Get internal data arrays using the most efficient API
            // This assumes direct access to the underlying data
            for (int i = 0; i < indices.Length; i++)
            {
                // Use tensor.GetSlice with proper offset calculation
                var sourceSlice = tensor.GetSlice(indices[i]);
                result.SetSlice(i, sourceSlice);
            }
        }

        return result;
    }

    /// <summary>
    /// Creates a batch containing a single item.
    /// </summary>
    /// <typeparam name="TInput">The type of the input and output data structure.</typeparam>
    /// <typeparam name="T">The element type of the data structure.</typeparam>
    /// <param name="item">The single item to include in the batch.</param>
    /// <returns>A batch containing only the provided item.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the item is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a type conversion fails.</exception>
    /// <exception cref="NotSupportedException">Thrown when the item type is not supported.</exception>
    public static TInput CreateSingleItemBatch(TInput item)
    {
        if (item == null)
        {
            throw new ArgumentNullException(nameof(item), "Item cannot be null.");
        }

        try
        {
            // Handle Vector<T> input
            if (item is Vector<T> vector)
            {
                return CreateBatchFromVector(vector);
            }

            // Handle Matrix<T> input
            if (item is Matrix<T> matrix)
            {
                return CreateBatchFromMatrix(matrix);
            }

            // Handle Tensor<T> input
            if (item is Tensor<T> tensor)
            {
                return CreateBatchFromTensor(tensor);
            }

            // Handle scalar value
            if (item is T scalarValue)
            {
                return CreateBatchFromScalar(scalarValue);
            }

            // If we've reached here, we're dealing with an unrecognized type
            // Return as-is, but log a warning for diagnostic purposes
            System.Diagnostics.Debug.WriteLine($"Warning: Unrecognized type {item.GetType().Name} in CreateSingleItemBatch. Returning original item.");
            return item;
        }
        catch (InvalidCastException ex)
        {
            throw new InvalidOperationException(
                $"Failed to convert {item.GetType().Name} to a batch format. Ensure the input type is compatible with batch operations.", ex);
        }
    }

    /// <summary>
    /// Creates a batch from a single vector.
    /// </summary>
    private static TInput CreateBatchFromVector(Vector<T> vector)
    {
        if (vector.Length == 0)
        {
            throw new ArgumentException("Cannot create a batch from an empty vector.", nameof(vector));
        }

        try
        {
            // Create a matrix with a single row containing the vector data
            var batchMatrix = new Matrix<T>(1, vector.Length);

            // Optimize for contiguous memory copying when possible
            if (vector is IArrayAccessible arrayVector && batchMatrix is IArraySettable arrayMatrix)
            {
                // Use direct array access for better performance
                var sourceArray = arrayVector.GetArray();
                arrayMatrix.SetArray(0, 0, sourceArray, 0, vector.Length);
            }
            else
            {
                // Fall back to element-by-element copying
                for (int i = 0; i < vector.Length; i++)
                {
                    batchMatrix[0, i] = vector[i];
                }
            }

            return (TInput)(object)batchMatrix;
        }
        catch (Exception ex) when (ex is not ArgumentException and not InvalidCastException)
        {
            throw new InvalidOperationException("Failed to create batch from vector.", ex);
        }
    }

    /// <summary>
    /// Creates a batch from a single matrix, ensuring it's in the correct batch format.
    /// </summary>
    private static TInput CreateBatchFromMatrix(Matrix<T> matrix)
    {
        if (matrix.Rows == 0 || matrix.Columns == 0)
        {
            throw new ArgumentException("Cannot create a batch from an empty matrix.", nameof(matrix));
        }

        try
        {
            if (matrix.Rows == 1)
            {
                // Already a single-row matrix, return as is
                return (TInput)(object)matrix;
            }

            if (matrix.Columns == 1)
            {
                var batchMatrix = new Matrix<T>(1, matrix.Rows);

                for (int i = 0; i < matrix.Rows; i++)
                {
                    batchMatrix[0, i] = matrix[i, 0];
                }

                return (TInput)(object)batchMatrix;
            }

            var singleRow = new Matrix<T>(1, matrix.Columns);

            for (int i = 0; i < matrix.Columns; i++)
            {
                singleRow[0, i] = matrix[0, i];
            }

            return (TInput)(object)singleRow;
        }
        catch (Exception ex) when (ex is not ArgumentException and not InvalidCastException)
        {
            throw new InvalidOperationException("Failed to create batch from matrix.", ex);
        }
    }

    /// <summary>
    /// Creates a batch from a single tensor, ensuring the first dimension is 1.
    /// </summary>
    private static TInput CreateBatchFromTensor(Tensor<T> tensor)
    {
        if (tensor.Length == 0)
        {
            throw new ArgumentException("Cannot create a batch from an empty tensor.", nameof(tensor));
        }

        try
        {
            // If first dimension is already 1, return as is
            if (tensor.Shape.Length > 0 && tensor.Shape[0] == 1)
            {
                return (TInput)(object)tensor;
            }

            // Handle different tensor ranks
            if (tensor.Shape.Length == 0)
            {
                // Scalar tensor - create 1D tensor with single element
                var batchTensor = new Tensor<T>(new[] { 1 });
                batchTensor[0] = tensor.GetFlatIndexValue(0);
                return (TInput)(object)batchTensor;
            }
            else if (tensor.Shape.Length == 1)
            {
                // 1D tensor with shape [n] - create 2D tensor with shape [1, n]
                int[] newShape = new[] { 1, tensor.Shape[0] };
                var batchTensor = new Tensor<T>(newShape);
                for (int i = 0; i < tensor.Shape[0]; i++)
                {
                    batchTensor[0, i] = tensor[i];
                }
                return (TInput)(object)batchTensor;
            }
            else
            {
                // 2D or higher tensor - prepend batch dimension of 1
                int[] newShape = new int[tensor.Shape.Length + 1];
                newShape[0] = 1;
                for (int i = 0; i < tensor.Shape.Length; i++)
                {
                    newShape[i + 1] = tensor.Shape[i];
                }

                var batchTensor = new Tensor<T>(newShape);

                // Copy all data - the tensor becomes the single item in the batch
                // For a tensor of shape [h, w, c], the batch tensor is [1, h, w, c]
                // We need to copy all elements preserving their relative positions
                CopyTensorData(tensor, batchTensor, 0);

                return (TInput)(object)batchTensor;
            }
        }
        catch (Exception ex) when (ex is not ArgumentException and not InvalidCastException)
        {
            throw new InvalidOperationException("Failed to create batch from tensor.", ex);
        }
    }

    /// <summary>
    /// Copies tensor data into a batch tensor at the specified batch index.
    /// </summary>
    private static void CopyTensorData(Tensor<T> source, Tensor<T> destination, int batchIndex)
    {
        // Calculate the number of elements in the source tensor
        int sourceLength = source.Length;

        // Calculate the starting flat index in the destination
        int destStride = sourceLength;
        int destStart = batchIndex * destStride;

        // Copy element by element using flat indexing
        for (int i = 0; i < sourceLength; i++)
        {
            destination.SetFlatIndexValue(destStart + i, source.GetFlatIndexValue(i));
        }
    }

    public interface IArrayAccessible
    {
        T[] GetArray();
        int GetOffset();
    }

    public interface IArraySettable
    {
        void SetArray(int row, int col, T[] array, int offset, int length);
    }

    /// <summary>
    /// Creates a batch from a single scalar value.
    /// </summary>
    private static TInput CreateBatchFromScalar(T scalarValue)
    {
        try
        {
            var vector = new Vector<T>(1);
            vector[0] = scalarValue;

            return (TInput)(object)vector;
        }
        catch (Exception ex) when (ex is not ArgumentException and not InvalidCastException)
        {
            throw new InvalidOperationException("Failed to create batch from scalar value.", ex);
        }
    }

    /// <summary>
    /// Retrieves a single item from a batch of input data.
    /// </summary>
    /// <param name="input">The batch of input data.</param>
    /// <param name="index">The index of the item to retrieve.</param>
    /// <returns>A single item from the batch at the specified index.</returns>
    /// <exception cref="ArgumentNullException">Thrown when input is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is negative or exceeds input dimensions.</exception>
    /// <exception cref="NotSupportedException">Thrown when the input type is not supported.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a single example from your data.
    /// Think of it like selecting one row from a spreadsheet of many rows.
    /// </para>
    /// </remarks>
    public static TInput GetItem(TInput input, int index)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input), "Input cannot be null.");

        if (index < 0)
            throw new ArgumentOutOfRangeException(nameof(index), "Index cannot be negative.");

        // Handle Vector<T> input
        if (input is Vector<T> vector)
        {
            if (index >= vector.Length)
                throw new ArgumentOutOfRangeException(nameof(index), "Index exceeds vector length.");

            // Create a new vector with a single element
            var singletonVector = new Vector<T>(1);
            singletonVector[0] = vector[index];

            return (TInput)(object)singletonVector;
        }

        // Handle Matrix<T> input
        if (input is Matrix<T> matrix)
        {
            if (index >= matrix.Rows)
                throw new ArgumentOutOfRangeException(nameof(index), "Index exceeds matrix row count.");

            // Return a single row as a Vector<T>
            var rowVector = new Vector<T>(matrix.Columns);
            for (int i = 0; i < matrix.Columns; i++)
            {
                rowVector[i] = matrix[index, i];
            }

            return (TInput)(object)rowVector;
        }

        // Handle Tensor<T> input
        if (input is Tensor<T> tensor)
        {
            if (index >= tensor.Shape[0])
                throw new ArgumentOutOfRangeException(nameof(index), "Index exceeds tensor's first dimension.");

            // Extract a slice from the tensor
            return (TInput)(object)tensor.Slice(index);
        }

        // If input is not one of our supported types
        throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported. Expected Vector<T>, Matrix<T>, or Tensor<T>.");
    }

    /// <summary>
    /// Retrieves a specific feature value from an input item.
    /// </summary>
    /// <param name="input">The input item.</param>
    /// <param name="featureIndex">The index of the feature to retrieve.</param>
    /// <returns>The value of the specified feature.</returns>
    /// <exception cref="ArgumentNullException">Thrown when input is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when featureIndex is negative or exceeds feature count.</exception>
    /// <exception cref="NotSupportedException">Thrown when the input type is not supported.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts one specific value from your data.
    /// If your data represents features of something (like height, weight, color of an object),
    /// this method lets you get just one of those features.
    /// </para>
    /// </remarks>
    public static T GetFeatureValue(TInput input, int featureIndex)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input), "Input cannot be null.");

        if (featureIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(featureIndex), "Feature index cannot be negative.");

        // Handle scalar value case (when TInput is the same as T)
        if (input is T scalarValue && featureIndex == 0)
        {
            return scalarValue;
        }

        // Handle Vector<T> input
        if (input is Vector<T> vector)
        {
            if (featureIndex >= vector.Length)
                throw new ArgumentOutOfRangeException(nameof(featureIndex), "Feature index exceeds vector length.");

            return vector[featureIndex];
        }

        // Handle Matrix<T> input
        if (input is Matrix<T> matrix)
        {
            if (matrix.Rows == 0)
                throw new ArgumentException("Matrix has no rows.", nameof(input));

            if (featureIndex >= matrix.Columns)
                throw new ArgumentOutOfRangeException(nameof(featureIndex), "Feature index exceeds matrix column count.");

            // Assuming this is a single row matrix or we're getting a feature from the first row
            return matrix[0, featureIndex];
        }

        // Handle Tensor<T> input
        if (input is Tensor<T> tensor)
        {
            // For a rank-1 tensor, just index directly
            if (tensor.Rank == 1)
            {
                if (featureIndex >= tensor.Shape[0])
                    throw new ArgumentOutOfRangeException(nameof(featureIndex), "Feature index exceeds tensor dimension.");

                return tensor[featureIndex];
            }

            // For higher rank tensors, assume we want the first element along higher dimensions
            if (featureIndex >= tensor.Shape[tensor.Rank - 1])
                throw new ArgumentOutOfRangeException(nameof(featureIndex), "Feature index exceeds tensor's last dimension.");

            // Create index array with all zeros except the last dimension
            int[] indices = new int[tensor.Rank];
            indices[tensor.Rank - 1] = featureIndex;

            return tensor[indices];
        }

        // If input is not one of our supported types
        throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported. Expected T, Vector<T>, Matrix<T>, or Tensor<T>.");
    }
}
