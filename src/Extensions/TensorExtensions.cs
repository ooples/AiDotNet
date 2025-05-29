namespace AiDotNet.Extensions;

public static class TensorExtensions
{
    /// <summary>
    /// Converts a tensor to a matrix for various calculations.
    /// </summary>
    /// <param name="tensor">The tensor to convert.</param>
    /// <returns>A matrix representation of the tensor.</returns>
    /// <remarks>
    /// Helper method for converting tensors to matrices for mathematical operations.
    /// </remarks>
    public static Matrix<T> ConvertToMatrix<T>(this Tensor<T> tensor)
    {
        // For 2D tensor, convert directly to matrix
        if (tensor.Rank == 2)
        {
            Matrix<T> matrix = new Matrix<T>(tensor.Shape[0], tensor.Shape[1]);
            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    matrix[i, j] = tensor[i, j];
                }
            }
            return matrix;
        }
        // For vector, return as column matrix
        else if (tensor.Rank == 1)
        {
            Matrix<T> matrix = new Matrix<T>(tensor.Shape[0], 1);
            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                matrix[i, 0] = tensor[i];
            }
            return matrix;
        }
        // Handle other cases
        else
        {
            throw new ArgumentException("Tensor must be 1D or 2D for conversion to matrix.");
        }
    }

    /// <summary>
    /// Converts a flattened vector back into a tensor with the specified shape.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor elements.</typeparam>
    /// <param name="tensor">The tensor to populate with values.</param>
    /// <param name="flattenedValues">The flattened vector containing values to reshape into the tensor.</param>
    /// <returns>The tensor populated with the values from the flattened vector.</returns>
    /// <remarks>
    /// <para>
    /// This method takes a flattened vector and reshapes it back into a tensor with the shape of the current tensor.
    /// It's the inverse operation of flattening a tensor into a vector.
    /// </para>
    /// <para><b>For Beginners:</b> This is like taking a long line of numbers and arranging them back into a
    /// multi-dimensional structure (like rows and columns). It's similar to taking a string of text and
    /// formatting it back into paragraphs.
    /// </para>
    /// </remarks>
    public static Tensor<T> Unflatten<T>(this Tensor<T> tensor, Vector<T> flattenedValues)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
            
        if (flattenedValues == null)
            throw new ArgumentNullException(nameof(flattenedValues));
            
        // Calculate the total size of the tensor
        int totalSize = tensor.Shape.Aggregate(1, (acc, dim) => acc * dim);
            
        if (flattenedValues.Length != totalSize)
            throw new ArgumentException($"The size of the flattened vector ({flattenedValues.Length}) does not match the tensor shape total size ({totalSize})");
            
        // Create a new tensor with the same shape
        var result = new Tensor<T>(tensor.Shape);
            
        // Copy the values from the flattened vector to the tensor
        int index = 0;
        result.ForEachPosition((position, _) =>
        {
            result[position] = flattenedValues[index++];
            return true;
        });
            
        return result;
    }

    /// <summary>
    /// Iterates through all positions in the tensor and applies a function to each position.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor elements.</typeparam>
    /// <param name="tensor">The tensor to iterate through.</param>
    /// <param name="action">The action to apply at each position. Takes the position array and current value as parameters.
    /// Return true to continue iteration, false to stop.</param>
    /// <remarks>
    /// <para>
    /// This method provides a way to iterate through all positions in a tensor and perform an operation at each position.
    /// The action receives the current position (as an array of indices) and the current value at that position.
    /// </para>
    /// <para><b>For Beginners:</b> This is like visiting every cell in a spreadsheet one by one and performing
    /// some action at each cell. You can use this to transform, inspect, or extract data from the tensor.
    /// </para>
    /// </remarks>
    public static void ForEachPosition<T>(this Tensor<T> tensor, Func<int[], T, bool> action)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        
        if (action == null)
            throw new ArgumentNullException(nameof(action));
        
        // Create an array to hold the current position
        int[] position = new int[tensor.Shape.Length];
    
        // Start recursive iteration
        ForEachPositionRecursive(tensor, position, 0, action);
    }

    /// <summary>
    /// Helper method for recursive iteration through tensor positions.
    /// </summary>
    private static bool ForEachPositionRecursive<T>(Tensor<T> tensor, int[] position, int dimension, Func<int[], T, bool> action)
    {
        if (dimension == tensor.Shape.Length)
        {
            // We've reached a complete position, apply the action
            return action(position, tensor[position]);
        }
        else
        {
            // Iterate through the current dimension
            for (int i = 0; i < tensor.Shape[dimension]; i++)
            {
                position[dimension] = i;
                if (!ForEachPositionRecursive(tensor, position, dimension + 1, action))
                {
                    return false; // Stop iteration if the action returns false
                }
            }

            return true;
        }
    }

    public static bool TensorEquals<T>(this Tensor<T> a, Tensor<T> b)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (!a.Shape.SequenceEqual(b.Shape))
            return false;

        for (int i = 0; i < a.Length; i++)
        {
            if (!numOps.Equals(a[i], b[i]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Concatenates two tensors along the last dimension.
    /// </summary>
    /// <param name="tensorA">The first tensor.</param>
    /// <param name="tensorB">The second tensor.</param>
    /// <returns>The concatenated tensor.</returns>
    public static Tensor<T> ConcatenateTensors<T>(this Tensor<T> tensorA, Tensor<T> tensorB)
    {
        // Get shapes and verify they can be concatenated
        int[] shapeA = tensorA.Shape;
        int[] shapeB = tensorB.Shape;
    
        if (shapeA.Length != shapeB.Length)
        {
            throw new ArgumentException("Tensors must have the same number of dimensions to concatenate");
        }
    
        for (int i = 0; i < shapeA.Length - 1; i++)
        {
            if (shapeA[i] != shapeB[i])
            {
                throw new ArgumentException("All dimensions except the last must match for concatenation");
            }
        }
    
        // Calculate shape of result tensor
        int[] resultShape = new int[shapeA.Length];
        Array.Copy(shapeA, resultShape, shapeA.Length);
        resultShape[resultShape.Length - 1] = shapeA[shapeA.Length - 1] + shapeB[shapeB.Length - 1];
    
        // Create result tensor
        var result = new Tensor<T>(resultShape);
    
        // Handle different dimensionality cases
        if (shapeA.Length == 2)
        {
            // 2D tensors (batch, features)
            for (int b = 0; b < shapeA[0]; b++)
            {
                // Copy values from first tensor
                for (int f = 0; f < shapeA[1]; f++)
                {
                    result[b, f] = tensorA[b, f];
                }
            
                // Copy values from second tensor
                for (int f = 0; f < shapeB[1]; f++)
                {
                    result[b, shapeA[1] + f] = tensorB[b, f];
                }
            }
        }
        else if (shapeA.Length == 3)
        {
            // 3D tensors (batch, sequence, features)
            for (int b = 0; b < shapeA[0]; b++)
            {
                for (int s = 0; s < shapeA[1]; s++)
                {
                    // Copy values from first tensor
                    for (int f = 0; f < shapeA[2]; f++)
                    {
                        result[b, s, f] = tensorA[b, s, f];
                    }
                
                    // Copy values from second tensor
                    for (int f = 0; f < shapeB[2]; f++)
                    {
                        result[b, s, shapeA[2] + f] = tensorB[b, s, f];
                    }
                }
            }
        }
        else
        {
            throw new NotSupportedException("Concatenation currently supports only 2D and 3D tensors");
        }
    
        return result;
    }

    /// <summary>
    /// Performs a batched matrix multiplication on a batch of matrices.
    /// </summary>
    /// <typeparam name="T">The numeric data type used in the tensors.</typeparam>
    /// <param name="tensor">The input tensor representing a batch of matrices.</param>
    /// <param name="matrices">A batch of matrices to multiply with.</param>
    /// <returns>A new tensor containing the result of the batched matrix multiplication.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor or matrices is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when tensor and matrices have incompatible shapes for multiplication.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method performs batched matrix multiplication between the input tensor and a batch of matrices.
    /// For a tensor of shape [batch_size, m, n] and matrices of shape [batch_size, n, p],
    /// the resulting tensor will have shape [batch_size, m, p].
    /// </para>
    /// <para>
    /// Each matrix in the batch is multiplied with the corresponding slice in the input tensor.
    /// This is more efficient than performing individual multiplications in a loop.
    /// </para>
    /// <para>
    /// For large tensors, this method uses parallel processing to improve performance.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies multiple matrices at once (in a batch).
    /// 
    /// It's like having multiple related calculations that you want to do in parallel:
    /// - If you have 10 matrices in your tensor and 10 corresponding matrices in the 'matrices' parameter
    /// - This method will multiply each pair together and give you 10 result matrices
    /// 
    /// This is commonly used in neural networks when processing multiple samples at once (batch processing).
    /// </para>
    /// </remarks>
    public static Tensor<T> BatchMatrixMultiply<T>(this Tensor<T> tensor, Tensor<T> matrices)
    {
        // Validate inputs
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor), "Input tensor cannot be null.");

        if (matrices == null)
            throw new ArgumentNullException(nameof(matrices), "Matrices tensor cannot be null.");

        var numOps = MathHelper.GetNumericOperations<T>();

        // Validate tensor dimensions
        if (tensor.Rank != 3)
            throw new ArgumentException($"Input tensor must be 3D for batch matrix multiplication, but has rank {tensor.Rank}", nameof(tensor));

        if (matrices.Rank != 3)
            throw new ArgumentException($"Matrices tensor must be 3D for batch matrix multiplication, but has rank {matrices.Rank}", nameof(matrices));

        // Check batch size compatibility
        if (tensor.Shape[0] != matrices.Shape[0])
            throw new ArgumentException($"Batch size mismatch: tensor has batch size {tensor.Shape[0]}, matrices has batch size {matrices.Shape[0]}");

        // Check matrix multiplication compatibility
        if (tensor.Shape[2] != matrices.Shape[1])
            throw new ArgumentException($"Incompatible matrix dimensions: tensor has shape [..., {tensor.Shape[2]}], matrices has shape [..., {matrices.Shape[1]}, ...]");

        int batchSize = tensor.Shape[0];
        int m = tensor.Shape[1];
        int n = tensor.Shape[2];
        int p = matrices.Shape[2];

        // Create result tensor
        var result = new Tensor<T>([batchSize, m, p]);

        // For small tensors, use a simple approach
        if (batchSize * m * p < 10000)
        {
            // Perform batched multiplication sequentially
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        T sum = numOps.Zero;
                        for (int k = 0; k < n; k++)
                        {
                            sum = numOps.Add(sum, numOps.Multiply(tensor[b, i, k], matrices[b, k, j]));
                        }
                        result[b, i, j] = sum;
                    }
                }
            }
        }
        else
        {
            // For larger tensors, use parallel processing
            Parallel.For(0, batchSize, b =>
            {
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        T sum = numOps.Zero;
                        for (int k = 0; k < n; k++)
                        {
                            sum = numOps.Add(sum, numOps.Multiply(tensor[b, i, k], matrices[b, k, j]));
                        }
                        result[b, i, j] = sum;
                    }
                }
            });
        }

        return result;
    }

    /// <summary>
    /// Transposes a tensor by swapping specified dimensions.
    /// </summary>
    /// <typeparam name="T">The numeric data type used in the tensor.</typeparam>
    /// <param name="tensor">The input tensor to transpose.</param>
    /// <param name="dim1">First dimension to swap (zero-based).</param>
    /// <param name="dim2">Second dimension to swap (zero-based).</param>
    /// <returns>A new tensor with the specified dimensions transposed.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when dimension indices are invalid for the tensor's rank.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method creates a new tensor with two dimensions swapped. The original tensor is not modified.
    /// For example, transposing dimensions 1 and 2 of a tensor with shape [2, 3, 4] would result in a
    /// tensor with shape [2, 4, 3].
    /// </para>
    /// <para>
    /// For the common case of a 2D tensor (matrix), calling Transpose(0, 1) performs a standard matrix transpose.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you swap any two dimensions of your tensor.
    /// 
    /// For example:
    /// - In a 2D tensor (like a table with rows and columns), swapping dimensions 0 and 1 
    ///   turns rows into columns and columns into rows.
    /// - In a 3D tensor (like a stack of tables), you could swap dimension 0 and 2 to 
    ///   reorganize which dimensions represent depth, rows, and columns.
    /// 
    /// Transposing is useful for reorganizing data to match the expected input format of 
    /// various algorithms or to perform certain mathematical operations.
    /// </para>
    /// </remarks>
    public static Tensor<T> Transpose<T>(this Tensor<T> tensor, int dim1, int dim2)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor), "Input tensor cannot be null.");

        // Validate dimension indices
        if (dim1 < 0 || dim1 >= tensor.Rank)
            throw new ArgumentException($"First dimension index ({dim1}) is out of range for tensor with rank {tensor.Rank}.", nameof(dim1));

        if (dim2 < 0 || dim2 >= tensor.Rank)
            throw new ArgumentException($"Second dimension index ({dim2}) is out of range for tensor with rank {tensor.Rank}.", nameof(dim2));

        if (dim1 == dim2)
            return tensor.Clone(); // No need to transpose if dimensions are the same

        // Create new shape with dimensions swapped
        int[] newShape = (int[])tensor.Shape.Clone();
        (newShape[dim1], newShape[dim2]) = (newShape[dim2], newShape[dim1]);

        // Create result tensor
        var result = new Tensor<T>(newShape);

        // Helper method to convert a flat index to multidimensional indices
        int[] GetIndices(int flatIndex, int[] shape)
        {
            int[] indices = new int[shape.Length];
            int remaining = flatIndex;

            for (int i = shape.Length - 1; i >= 0; i--)
            {
                indices[i] = remaining % shape[i];
                remaining /= shape[i];
            }

            return indices;
        }

        // Helper method to calculate flat index from multidimensional indices
        int GetFlatIndex(int[] indices, int[] shape)
        {
            int flatIndex = 0;
            int multiplier = 1;

            for (int i = shape.Length - 1; i >= 0; i--)
            {
                flatIndex += indices[i] * multiplier;
                multiplier *= shape[i];
            }

            return flatIndex;
        }

        // Process all elements
        for (int i = 0; i < tensor.Length; i++)
        {
            // Get multidimensional indices for the current element
            int[] indices = GetIndices(i, tensor.Shape);

            // Swap the target dimensions
            (indices[dim1], indices[dim2]) = (indices[dim2], indices[dim1]);

            // Calculate the corresponding index in the result tensor
            int resultIndex = GetFlatIndex(indices, newShape);

            // Copy the value
            result.SetFlatIndexValue(resultIndex, tensor.GetFlatIndexValue(i));
        }

        return result;
    }

    /// <summary>
    /// Transposes the last two dimensions of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric data type used in the tensor.</typeparam>
    /// <param name="tensor">The input tensor to transpose.</param>
    /// <returns>A new tensor with the last two dimensions transposed.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the tensor's rank is less than 2.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method transposes the last two dimensions of the input tensor. For a 2D tensor (matrix),
    /// this is equivalent to a standard matrix transpose. For tensors with higher ranks, only the
    /// last two dimensions are affected.
    /// </para>
    /// <para>
    /// For example:
    /// - A 2D tensor of shape [m, n] becomes a tensor of shape [n, m]
    /// - A 3D tensor of shape [b, m, n] becomes a tensor of shape [b, n, m]
    /// - A 4D tensor of shape [a, b, m, n] becomes a tensor of shape [a, b, n, m]
    /// </para>
    /// <para><b>For Beginners:</b> This method flips the rows and columns of a matrix.
    /// 
    /// In a standard matrix, transposing means:
    /// - The rows become columns
    /// - The columns become rows
    /// 
    /// For example, if you have a matrix with 3 rows and 4 columns, after transposing
    /// it will have 4 rows and 3 columns.
    /// 
    /// If you're working with higher-dimensional tensors (3D, 4D, etc.), this method
    /// only flips the last two dimensions while keeping the earlier dimensions unchanged.
    /// </para>
    /// </remarks>
    public static Tensor<T> TransposeLastDimensions<T>(this Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor), "Input tensor cannot be null.");

        // Validate dimensions
        if (tensor.Rank < 2)
            throw new ArgumentException($"Tensor must have at least 2 dimensions for transposition, but has rank {tensor.Rank}", nameof(tensor));

        // Use the general transpose method with the last two dimensions
        return tensor.Transpose(tensor.Rank - 2, tensor.Rank - 1);
    }

    /// <summary>
    /// Multiplies a 2D tensor (matrix) by a vector.
    /// </summary>
    /// <typeparam name="T">The numeric data type used in the tensor and vector.</typeparam>
    /// <param name="tensor">The 2D tensor to multiply.</param>
    /// <param name="vector">The vector to multiply with.</param>
    /// <returns>A new vector containing the result of the multiplication.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor or vector is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the tensor is not 2D or the dimensions are incompatible for multiplication.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method performs matrix-vector multiplication between a 2D tensor and a vector.
    /// For a tensor of shape [m, n] and a vector of length n, the result is a vector of length m.
    /// </para>
    /// <para>
    /// The operation performs the following calculation for each element i of the result:
    /// result[i] = sum(tensor[i,j] * vector[j]) for all j from 0 to n-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a matrix (2D tensor) by a vector.
    /// 
    /// In this multiplication:
    /// - The number of columns in your matrix must match the length of your vector
    /// - The result will be a new vector with the same number of elements as rows in your matrix
    /// 
    /// For example, if you have a matrix with 3 rows and 4 columns, and a vector with 4 elements,
    /// the result will be a vector with 3 elements.
    /// 
    /// This operation is commonly used in neural networks, linear transformations, and when applying
    /// weights to features.
    /// </para>
    /// </remarks>
    public static Vector<T> MultiplyVector<T>(this Tensor<T> tensor, Vector<T> vector)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor), "Input tensor cannot be null.");

        if (vector == null)
            throw new ArgumentNullException(nameof(vector), "Input vector cannot be null.");

        var numOps = MathHelper.GetNumericOperations<T>();

        // Validate tensor dimensions
        if (tensor.Rank != 2)
            throw new ArgumentException($"Tensor must be 2D for matrix-vector multiplication, but has rank {tensor.Rank}", nameof(tensor));

        // Check multiplication compatibility
        if (tensor.Shape[1] != vector.Length)
            throw new ArgumentException($"Incompatible dimensions: tensor has {tensor.Shape[1]} columns, but vector has {vector.Length} elements");

        int m = tensor.Shape[0];
        int n = tensor.Shape[1];

        // Create result vector
        var result = new Vector<T>(m);

        // Perform matrix-vector multiplication
        for (int i = 0; i < m; i++)
        {
            T sum = numOps.Zero;
            for (int j = 0; j < n; j++)
            {
                sum = numOps.Add(sum, numOps.Multiply(tensor[i, j], vector[j]));
            }
            result[i] = sum;
        }

        return result;
    }

    /// <summary>
    /// Performs element-wise multiplication between two tensors (Hadamard product).
    /// </summary>
    /// <typeparam name="T">The numeric data type used in the tensors.</typeparam>
    /// <param name="tensor">The first tensor.</param>
    /// <param name="other">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise product of the input tensors.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either tensor is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the tensors have different shapes.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method performs element-wise multiplication (Hadamard product) between two tensors of the same shape.
    /// Each element in the result is the product of the corresponding elements in the input tensors.
    /// </para>
    /// <para>
    /// For large tensors, this method uses parallel processing to improve performance.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two tensors element by element.
    /// 
    /// Unlike matrix multiplication where rows and columns interact in a specific way, element-wise
    /// multiplication simply multiplies corresponding elements:
    /// - result[i,j,k] = tensor1[i,j,k] * tensor2[i,j,k]
    /// 
    /// Both tensors must have exactly the same shape. This operation is commonly used in neural networks
    /// for masking, attention mechanisms, and applying element-wise transformations.
    /// </para>
    /// </remarks>
    public static Tensor<T> ElementwiseMultiply<T>(this Tensor<T> tensor, Tensor<T> other)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor), "First tensor cannot be null.");

        if (other == null)
            throw new ArgumentNullException(nameof(other), "Second tensor cannot be null.");

        var numOps = MathHelper.GetNumericOperations<T>();

        // Validate tensor dimensions
        if (tensor.Rank != other.Rank)
            throw new ArgumentException($"Tensors must have the same rank, but got {tensor.Rank} and {other.Rank}");

        // Check shape compatibility
        for (int i = 0; i < tensor.Rank; i++)
        {
            if (tensor.Shape[i] != other.Shape[i])
                throw new ArgumentException($"Tensors must have the same shape, but got {string.Join("×", tensor.Shape)} and {string.Join("×", other.Shape)}");
        }

        // Create result tensor
        var result = new Tensor<T>(tensor.Shape);

        // For small tensors, use a simple approach
        if (tensor.Length < 100000)
        {
            // Perform element-wise multiplication sequentially
            for (int i = 0; i < tensor.Length; i++)
            {
                result.SetFlatIndexValue(i, numOps.Multiply(tensor.GetFlatIndexValue(i), other.GetFlatIndexValue(i)));
            }
        }
        else
        {
            // For larger tensors, use parallel processing
            Parallel.For(0, tensor.Length, i =>
            {
                result.SetFlatIndexValue(i, numOps.Multiply(tensor.GetFlatIndexValue(i), other.GetFlatIndexValue(i)));
            });
        }

        return result;
    }

    /// <summary>
    /// Creates a deep copy of the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric data type used in the tensor.</typeparam>
    /// <param name="tensor">The tensor to clone.</param>
    /// <returns>A new tensor containing the same values as the original.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the input tensor, meaning that modifications to the cloned tensor
    /// will not affect the original tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact copy of your tensor.
    /// 
    /// Think of it like photocopying a document - the copy has all the same information as the original,
    /// but if you make changes to one, it doesn't affect the other.
    /// 
    /// This is useful when you need to modify a tensor but want to keep the original unchanged.
    /// </para>
    /// </remarks>
    public static Tensor<T> Clone<T>(this Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor), "Input tensor cannot be null.");

        // Create new tensor with the same shape
        var clone = new Tensor<T>(tensor.Shape);

        // Copy all values
        for (int i = 0; i < tensor.Length; i++)
        {
            clone.SetFlatIndexValue(i, tensor.GetFlatIndexValue(i));
        }

        return clone;
    }

    /// <summary>
    /// Reshapes a tensor into a new shape with the same total number of elements.
    /// </summary>
    /// <typeparam name="T">The numeric data type used in the tensor.</typeparam>
    /// <param name="tensor">The tensor to reshape.</param>
    /// <param name="newShape">The new shape for the tensor.</param>
    /// <returns>A new tensor with the specified shape, containing the same data as the original tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor or newShape is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the new shape has a different total number of elements than the original tensor.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method creates a new tensor with a different shape but the same data as the original tensor.
    /// The total number of elements (product of all dimensions) must be the same in both shapes.
    /// </para>
    /// <para>
    /// The reshaping is done in row-major order, meaning that elements are rearranged as if the tensor
    /// is flattened and then reshaped.
    /// </para>
    /// <para><b>For Beginners:</b> This method changes the arrangement of your tensor's dimensions
    /// while keeping all the same values.
    /// 
    /// It's like rearranging eggs in a carton:
    /// - If you have 12 eggs in a 3×4 carton, you could rearrange them into a 2×6 carton or a 1×12 carton
    /// - The important thing is the total number of eggs (elements) stays the same
    /// 
    /// This is useful when you need to change the structure of your data to match the expected input format
    /// of a specific algorithm or operation.
    /// </para>
    /// </remarks>
    public static Tensor<T> Reshape<T>(this Tensor<T> tensor, params int[] newShape)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor), "Input tensor cannot be null.");

        if (newShape == null)
            throw new ArgumentNullException(nameof(newShape), "New shape cannot be null.");

        if (newShape.Length == 0)
            throw new ArgumentException("New shape cannot be empty.", nameof(newShape));

        // Calculate total elements in new shape
        int newTotalElements = 1;
        for (int i = 0; i < newShape.Length; i++)
        {
            if (newShape[i] <= 0)
                throw new ArgumentException($"All dimensions in the new shape must be positive, but got {newShape[i]} at index {i}.", nameof(newShape));

            newTotalElements *= newShape[i];
        }

        // Verify total elements match
        if (newTotalElements != tensor.Length)
            throw new ArgumentException($"New shape has {newTotalElements} elements, but tensor has {tensor.Length} elements. Reshaping requires the same number of elements.");

        // Create new tensor with the desired shape
        var result = new Tensor<T>(newShape);

        // Copy all values (the internal storage order remains the same)
        for (int i = 0; i < tensor.Length; i++)
        {
            result.SetFlatIndexValue(i, tensor.GetFlatIndexValue(i));
        }

        return result;
    }

    /// <summary>
    /// Returns a slice of the tensor along the specified dimension.
    /// </summary>
    /// <typeparam name="T">The numeric data type used in the tensor.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="dimension">The dimension to slice along.</param>
    /// <param name="index">The index in the specified dimension to extract.</param>
    /// <returns>A tensor with one fewer dimension, representing the slice at the specified index.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the dimension or index is invalid.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method extracts a slice of the tensor along a specified dimension at a given index.
    /// The resulting tensor has one fewer dimension than the input tensor.
    /// </para>
    /// <para>
    /// For example, if you have a 3D tensor of shape [2, 3, 4]:
    /// - Slicing along dimension 0 at index 1 would give a 2D tensor of shape [3, 4]
    /// - Slicing along dimension 1 at index 0 would give a 2D tensor of shape [2, 4]
    /// - Slicing along dimension 2 at index 3 would give a 2D tensor of shape [2, 3]
    /// </para>
    /// <para><b>For Beginners:</b> This method extracts a specific "slice" from your tensor.
    /// 
    /// Think of it like cutting a slice from a 3D cake:
    /// - If you cut horizontally, you get a 2D rectangle
    /// - If you cut vertically, you get a different 2D rectangle
    /// 
    /// The dimension parameter tells which direction to cut, and the index tells how far along that direction to make the cut.
    /// 
    /// This is useful for:
    /// - Extracting specific samples from a batch of data
    /// - Breaking down a complex tensor into simpler components
    /// - Focusing on a particular subset of your data
    /// </para>
    /// </remarks>
    public static Tensor<T> Slice<T>(this Tensor<T> tensor, int dimension, int index)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor), "Input tensor cannot be null.");

        // Validate dimension
        if (dimension < 0 || dimension >= tensor.Rank)
            throw new ArgumentException($"Dimension {dimension} is invalid for tensor with rank {tensor.Rank}.", nameof(dimension));

        // Validate index
        if (index < 0 || index >= tensor.Shape[dimension])
            throw new ArgumentException($"Index {index} is out of range for dimension {dimension} with size {tensor.Shape[dimension]}.", nameof(index));

        // Create new shape with the specified dimension removed
        int[] newShape = new int[tensor.Rank - 1];
        for (int i = 0, j = 0; i < tensor.Rank; i++)
        {
            if (i != dimension)
            {
                newShape[j++] = tensor.Shape[i];
            }
        }

        // Create result tensor
        var result = new Tensor<T>(newShape);

        // Helper method to get a slice of indices for specified dimension and index
        int[] GetSliceIndices(int[] fullIndices, int sliceDimension, int sliceIndex)
        {
            // Insert the slice index at the specified dimension
            int[] indices = new int[fullIndices.Length + 1];
            int j = 0;

            for (int i = 0; i < indices.Length; i++)
            {
                if (i == sliceDimension)
                {
                    indices[i] = sliceIndex;
                }
                else
                {
                    indices[i] = fullIndices[j++];
                }
            }

            return indices;
        }

        // Copy all values from the slice
        for (int i = 0; i < result.Length; i++)
        {
            // Get indices in the result tensor
            int[] resultIndices = result.GetIndices(i);

            // Convert to indices in the original tensor
            int[] tensorIndices = GetSliceIndices(resultIndices, dimension, index);

            // Copy the value
            result.SetFlatIndexValue(i, tensor[tensorIndices]);
        }

        return result;
    }

    /// <summary>
    /// Gets the multi-dimensional indices corresponding to a flat index.
    /// </summary>
    /// <typeparam name="T">The numeric data type used in the tensor.</typeparam>
    /// <param name="tensor">The tensor to query.</param>
    /// <param name="flatIndex">The flat (linear) index.</param>
    /// <returns>An array of indices corresponding to the flat index.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when the flat index is negative or exceeds the tensor's total length.
    /// </exception>
    public static int[] GetIndices<T>(this Tensor<T> tensor, int flatIndex)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor), "Input tensor cannot be null.");

        if (flatIndex < 0 || flatIndex >= tensor.Length)
            throw new ArgumentOutOfRangeException(nameof(flatIndex), $"Flat index {flatIndex} is out of range for tensor with length {tensor.Length}.");

        // Convert flat index to multi-dimensional indices
        int[] indices = new int[tensor.Rank];
        int remaining = flatIndex;

        // Calculate indices in row-major order (last dimension varies fastest)
        for (int i = tensor.Rank - 1; i >= 0; i--)
        {
            indices[i] = remaining % tensor.Shape[i];
            remaining /= tensor.Shape[i];
        }

        return indices;
    }

    /// <summary>
    /// Finds the indices of the maximum values along a specified axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor elements.</typeparam>
    /// <param name="tensor">The tensor to search.</param>
    /// <param name="axis">The axis along which to find the maximum values.</param>
    /// <returns>A vector containing the indices of the maximum values.</returns>
    public static Vector<int> ArgMax<T>(this Tensor<T> tensor, int axis)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        
        if (axis == 1 && tensor.Rank == 2)
        {
            // For a 2D tensor, find argmax along axis 1 (columns)
            var result = new Vector<int>(tensor.Shape[0]);
            
            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                int maxIndex = 0;
                T maxValue = tensor[i, 0];
                
                for (int j = 1; j < tensor.Shape[1]; j++)
                {
                    if (numOps.GreaterThan(tensor[i, j], maxValue))
                    {
                        maxValue = tensor[i, j];
                        maxIndex = j;
                    }
                }
                
                result[i] = maxIndex;
            }
            
            return result;
        }
        else if (axis == 0 && tensor.Rank == 1)
        {
            // For a 1D tensor, find the single argmax
            int maxIndex = 0;
            T maxValue = tensor[0];
            
            for (int i = 1; i < tensor.Length; i++)
            {
                if (numOps.GreaterThan(tensor[i], maxValue))
                {
                    maxValue = tensor[i];
                    maxIndex = i;
                }
            }
            
            return new Vector<int>(new[] { maxIndex });
        }
        else if (axis == 0 && tensor.Rank == 2)
        {
            // For a 2D tensor, find argmax along axis 0 (rows)
            var result = new Vector<int>(tensor.Shape[1]);
            
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                int maxIndex = 0;
                T maxValue = tensor[0, j];
                
                for (int i = 1; i < tensor.Shape[0]; i++)
                {
                    if (numOps.GreaterThan(tensor[i, j], maxValue))
                    {
                        maxValue = tensor[i, j];
                        maxIndex = i;
                    }
                }
                
                result[j] = maxIndex;
            }
            
            return result;
        }
        else if (axis >= 0 && axis < tensor.Rank)
        {
            // General case for any tensor rank and axis
            // Calculate the shape of the result
            var resultShape = new List<int>();
            for (int i = 0; i < tensor.Rank; i++)
            {
                if (i != axis)
                    resultShape.Add(tensor.Shape[i]);
            }
            
            // If result would be scalar, return single element vector
            if (resultShape.Count == 0)
            {
                int maxIndex = 0;
                T maxValue = tensor[new int[tensor.Rank]];
                
                var indices = new int[tensor.Rank];
                tensor.ForEachPosition((position, value) =>
                {
                    if (numOps.GreaterThan(value, maxValue))
                    {
                        maxValue = value;
                        maxIndex = position[axis];
                    }
                    return true;
                });
                
                return new Vector<int>(new[] { maxIndex });
            }
            
            // Calculate total size of result
            int resultSize = resultShape.Aggregate(1, (a, b) => a * b);
            var result = new Vector<int>(resultSize);
            
            // Iterate through all positions in the result
            int resultIndex = 0;
            var currentPosition = new int[resultShape.Count];
            
            do
            {
                // Map result position to tensor position
                var tensorPosition = new int[tensor.Rank];
                int dimIndex = 0;
                for (int i = 0; i < tensor.Rank; i++)
                {
                    if (i != axis)
                    {
                        tensorPosition[i] = currentPosition[dimIndex++];
                    }
                }
                
                // Find max along the axis
                int maxIndex = 0;
                tensorPosition[axis] = 0;
                T maxValue = tensor[tensorPosition];
                
                for (int i = 1; i < tensor.Shape[axis]; i++)
                {
                    tensorPosition[axis] = i;
                    T value = tensor[tensorPosition];
                    if (numOps.GreaterThan(value, maxValue))
                    {
                        maxValue = value;
                        maxIndex = i;
                    }
                }
                
                result[resultIndex++] = maxIndex;
                
                // Increment position
                for (int i = currentPosition.Length - 1; i >= 0; i--)
                {
                    currentPosition[i]++;
                    if (currentPosition[i] < resultShape[i])
                        break;
                    currentPosition[i] = 0;
                }
            } while (resultIndex < resultSize);
            
            return result;
        }
        else
        {
            throw new ArgumentException($"Invalid axis {axis} for tensor with rank {tensor.Rank}");
        }
    }
    
    /// <summary>
    /// Gathers values from a tensor using indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor elements.</typeparam>
    /// <param name="tensorObj">The tensor object to gather from.</param>
    /// <param name="indices">The indices to use for gathering.</param>
    /// <returns>A vector containing the gathered values.</returns>
    public static Vector<T> GatherValues<T>(this object tensorObj, Vector<int> indices)
    {
        if (tensorObj is Tensor<T> tensor)
        {
            var result = new Vector<T>(indices.Length);
            
            if (tensor.Rank == 2)
            {
                // For 2D tensor, gather from each row using the corresponding index
                for (int i = 0; i < indices.Length; i++)
                {
                    result[i] = tensor[i, indices[i]];
                }
            }
            else if (tensor.Rank == 1)
            {
                // For 1D tensor, gather using indices directly
                for (int i = 0; i < indices.Length; i++)
                {
                    result[i] = tensor[indices[i]];
                }
            }
            else if (tensor.Rank == 3)
            {
                // For 3D tensor, assume we're gathering from the last dimension
                // This is common in neural networks (batch_size, sequence_length, features)
                int batchSize = tensor.Shape[0];
                int seqLength = tensor.Shape[1];
                
                if (indices.Length == batchSize * seqLength)
                {
                    // Gather from each position in the batch and sequence
                    for (int b = 0; b < batchSize; b++)
                    {
                        for (int s = 0; s < seqLength; s++)
                        {
                            int idx = b * seqLength + s;
                            result[idx] = tensor[b, s, indices[idx]];
                        }
                    }
                }
                else if (indices.Length == batchSize)
                {
                    // Gather one value per batch element (common in RL for action selection)
                    for (int i = 0; i < batchSize; i++)
                    {
                        // Assuming we want the first sequence position
                        result[i] = tensor[i, 0, indices[i]];
                    }
                }
                else
                {
                    throw new ArgumentException($"Indices length {indices.Length} doesn't match expected dimensions for 3D tensor gathering");
                }
            }
            else
            {
                // General case for higher rank tensors
                // Assume we're gathering from the last dimension
                int lastDim = tensor.Rank - 1;
                int totalElements = 1;
                for (int i = 0; i < lastDim; i++)
                {
                    totalElements *= tensor.Shape[i];
                }
                
                if (indices.Length != totalElements)
                {
                    throw new ArgumentException($"Indices length {indices.Length} doesn't match expected size {totalElements} for gathering from tensor with shape [{string.Join(", ", tensor.Shape)}]");
                }
                
                // Iterate through all positions except the last dimension
                var position = new int[tensor.Rank];
                for (int i = 0; i < indices.Length; i++)
                {
                    // Calculate position from flat index
                    int temp = i;
                    for (int dim = lastDim - 1; dim >= 0; dim--)
                    {
                        position[dim] = temp % tensor.Shape[dim];
                        temp /= tensor.Shape[dim];
                    }
                    
                    // Set the last dimension using the gather index
                    position[lastDim] = indices[i];
                    
                    // Get the value
                    result[i] = tensor[position];
                }
            }
            
            return result;
        }
        else
        {
            throw new ArgumentException("Input must be a Tensor<T>");
        }
    }
}