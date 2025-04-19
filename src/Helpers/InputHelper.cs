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
    /// <param name="input">The input data structure.</param>
    /// <param name="indices">The indices of the elements or rows to include in the batch.</param>
    /// <returns>A new data structure containing only the selected elements.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the input or indices are null.</exception>
    /// <exception cref="ArgumentException">Thrown when the input type is not supported or indices are invalid.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a subset of the input data by selecting elements or rows based on the provided indices.
    /// For Matrix&lt;T&gt;, it selects rows at the specified indices.
    /// For Vector&lt;T&gt;, it selects elements at the specified indices.
    /// For Tensor&lt;T&gt;, it selects slices along the first dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like creating a smaller version of your data
    /// by picking only specific rows or elements. It's similar to selecting certain rows in a spreadsheet.
    /// </para>
    /// </remarks>
    public static TInput GetBatch(TInput input, int[] indices)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input data cannot be null.");
        }
    
        if (indices == null || indices.Length == 0)
        {
            throw new ArgumentNullException(nameof(indices), "Indices array cannot be null or empty.");
        }
    
        // Handle Matrix<T>
        if (input is Matrix<T> matrix)
        {
            // Validate indices
            foreach (var idx in indices)
            {
                if (idx < 0 || idx >= matrix.Rows)
                {
                    throw new ArgumentOutOfRangeException(nameof(indices), 
                        $"Index {idx} is out of range for matrix with {matrix.Rows} rows.");
                }
            }
        
            // Create a new matrix with the selected rows
            var result = new Matrix<T>(indices.Length, matrix.Columns);
        
            for (int i = 0; i < indices.Length; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    result[i, j] = matrix[indices[i], j];
                }
            }
        
            return (TInput)(object)result;
        }
    
        // Handle Vector<T>
        else if (input is Vector<T> vector)
        {
            // Validate indices
            foreach (var idx in indices)
            {
                if (idx < 0 || idx >= vector.Length)
                {
                    throw new ArgumentOutOfRangeException(nameof(indices), 
                        $"Index {idx} is out of range for vector with length {vector.Length}.");
                }
            }
        
            // Create a new vector with the selected elements
            var result = new Vector<T>(indices.Length);
        
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = vector[indices[i]];
            }
        
            return (TInput)(object)result;
        }
    
        // Handle Tensor<T>
        else if (input is Tensor<T> tensor)
        {
            // For tensors, we'll select along the first dimension
            if (tensor.Shape.Length < 1)
            {
                throw new ArgumentException("Cannot get batch from a scalar tensor.", nameof(input));
            }
        
            // Validate indices
            foreach (var idx in indices)
            {
                if (idx < 0 || idx >= tensor.Shape[0])
                {
                    throw new ArgumentOutOfRangeException(nameof(indices), 
                        $"Index {idx} is out of range for tensor with first dimension size {tensor.Shape[0]}.");
                }
            }
        
            // Create a new tensor shape with updated first dimension
            int[] newShape = new int[tensor.Shape.Length];
            newShape[0] = indices.Length;
            for (int i = 1; i < tensor.Shape.Length; i++)
            {
                newShape[i] = tensor.Shape[i];
            }
        
            // Create a new tensor with the selected slices
            var result = new Tensor<T>(newShape);
        
            // Copy the selected slices
            if (tensor.Shape.Length == 1)
            {
                // Handle 1D tensor (vector-like)
                for (int i = 0; i < indices.Length; i++)
                {
                    result[i] = tensor[indices[i]];
                }
            }
            else if (tensor.Shape.Length == 2)
            {
                // Handle 2D tensor (matrix-like)
                for (int i = 0; i < indices.Length; i++)
                {
                    for (int j = 0; j < tensor.Shape[1]; j++)
                    {
                        result[i, j] = tensor[indices[i], j];
                    }
                }
            }
            else
            {
                // Handle higher-dimensional tensors
                // This is a simplified approach - for complex tensors, you may need more specialized handling
                var resultVector = result.ToVector();
                var inputVector = tensor.ToVector();
            
                int sliceSize = 1;
                for (int i = 1; i < tensor.Shape.Length; i++)
                {
                    sliceSize *= tensor.Shape[i];
                }
            
                for (int i = 0; i < indices.Length; i++)
                {
                    int sourceOffset = indices[i] * sliceSize;
                    int destOffset = i * sliceSize;
                
                    for (int j = 0; j < sliceSize; j++)
                    {
                        resultVector[destOffset + j] = inputVector[sourceOffset + j];
                    }
                }
            
                result = Tensor<T>.FromVector(resultVector, newShape);
            }
        
            return (TInput)(object)result;
        }
    
        throw new ArgumentException(
            $"Unsupported input type: {input.GetType().Name}. " +
            $"The GetBatch method only supports Matrix<T>, Vector<T>, and Tensor<T>.");
    }

    /// <summary>
    /// Creates a batch containing a single item.
    /// </summary>
    /// <param name="item">The single item to include in the batch.</param>
    /// <returns>A batch containing only the provided item.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the item is null.</exception>
    /// <exception cref="NotSupportedException">Thrown when the item type is not supported.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a batch (a data structure that can hold multiple items) containing only the provided item.
    /// It's useful when you have a single data point but need to use methods that expect batched input.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes a single example and packages it as if it were a batch.
    /// It's like putting a single item into a container designed for multiple items.
    /// </para>
    /// </remarks>
    public static TInput CreateSingleItemBatch(TInput item)
    {
        if (item == null)
            throw new ArgumentNullException(nameof(item), "Item cannot be null.");

        // Handle Vector<T> input
        if (item is Vector<T> vector)
        {
            // Create a matrix with a single row containing the vector data
            var newMatrix = new Matrix<T>(1, vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                newMatrix[0, i] = vector[i];
            }

            return (TInput)(object)newMatrix;
        }

        // Handle Matrix<T> input - if it's already a matrix, ensure it has only one row
        if (item is Matrix<T> matrix)
        {
            if (matrix.Rows == 1)
            {
                // Already a single-row matrix, return as is
                return item;
            }
            else if (matrix.Columns == 1)
            {
                // It's a single-column matrix, transpose it to make it a single-row matrix
                var transposed = new Matrix<T>(1, matrix.Rows);
                for (int i = 0; i < matrix.Rows; i++)
                {
                    transposed[0, i] = matrix[i, 0];
                }
                return (TInput)(object)transposed;
            }
            else
            {
                // Multi-dimensional matrix - create a new matrix with just the first row
                var singleRow = new Matrix<T>(1, matrix.Columns);
                for (int i = 0; i < matrix.Columns; i++)
                {
                    singleRow[0, i] = matrix[0, i];
                }
                return (TInput)(object)singleRow;
            }
        }

        // Handle Tensor<T> input
        if (item is Tensor<T> tensor)
        {
            // Create a new tensor with first dimension of size 1
            int[] newShape = new int[tensor.Shape.Length];
            newShape[0] = 1;
            for (int i = 1; i < tensor.Shape.Length; i++)
            {
                newShape[i] = tensor.Shape[i];
            }

            var batchTensor = new Tensor<T>(newShape);

            // Copy the data from the original tensor to the first slice of the batch tensor
            if (tensor.Shape.Length == 1)
            {
                // Handle 1D tensor
                for (int i = 0; i < tensor.Shape[0]; i++)
                {
                    batchTensor[0, i] = tensor[i];
                }
            }
            else
            {
                // For higher dimensions, we need to copy the entire tensor to the first slice
                // This is a simplified approach - for complex tensors, more specialized handling may be needed
                var batchVector = batchTensor.ToVector();
                var inputVector = tensor.ToVector();

                int sliceSize = inputVector.Length;
                for (int i = 0; i < sliceSize; i++)
                {
                    batchVector[i] = inputVector[i];
                }

                batchTensor = Tensor<T>.FromVector(batchVector, newShape);
            }

            return (TInput)(object)batchTensor;
        }

        // For scalar values (when TInput is T), wrap in a Vector
        if (item is T scalarValue)
        {
            var newVector = new Vector<T>(1);
            newVector[0] = scalarValue;

            return (TInput)(object)newVector;
        }

        // If the item is already in a batch format or we don't know how to convert it,
        // return it as is and let the caller handle any potential issues
        return item;
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