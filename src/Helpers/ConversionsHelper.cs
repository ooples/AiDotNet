namespace AiDotNet.Helpers;

/// <summary>
/// Provides utility methods for converting between different data structures used in machine learning models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This helper class contains methods to convert between different mathematical 
/// objects used in AI (Tensor, Matrix, Vector). Think of it like a universal adapter that lets 
/// different parts of your AI system work together even when they expect different data formats.</para>
/// 
/// <para>For example, if one algorithm outputs a Matrix but another needs a Tensor, these methods
/// help you convert between them without writing complex conversion code yourself.</para>
/// </remarks>
public static class ConversionsHelper
{
    /// <summary>
    /// Converts an input of generic type to a Matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <typeparam name="TInput">The type of the input data (must be either Matrix&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <param name="input">The input data to convert.</param>
    /// <returns>A Matrix&lt;T&gt; representation of the input data.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the input cannot be converted to a Matrix&lt;T&gt;.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes data in one format and converts it to a matrix format.
    /// A matrix is like a table of numbers arranged in rows and columns.</para>
    /// 
    /// <para>If your data is already a matrix, it simply returns it. If it's a tensor (which can have more
    /// dimensions), it flattens or reshapes it to fit into a 2D matrix structure.</para>
    /// </remarks>
    public static Matrix<T> ConvertToMatrix<T, TInput>(TInput input)
    {
        if (input is Matrix<T> matrix)
        {
            return matrix;
        }
        else if (input is Tensor<T> tensor)
        {
            // Use the built-in ToMatrix method if it's a 2D tensor
            if (tensor.Rank == 2)
            {
                return tensor.ToMatrix();
            }
            else
            {
                // For higher-dimensional tensors, reshape to 2D first
                var reshapedTensor = tensor.Reshape(tensor.Length / tensor.Shape[tensor.Rank - 1], tensor.Shape[tensor.Rank - 1]);
                return reshapedTensor.ToMatrix();
            }
        }
        
        throw new InvalidOperationException($"Cannot convert {typeof(TInput).Name} to Matrix<{typeof(T).Name}>. Expected Matrix<T> or Tensor<T>.");
    }

    /// <summary>
    /// Converts an output of generic type to a Vector.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <typeparam name="TOutput">The type of the output data (must be either Vector&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <param name="output">The output data to convert.</param>
    /// <returns>A Vector&lt;T&gt; representation of the output data.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the output cannot be converted to a Vector&lt;T&gt;.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes data in various formats and converts it to a vector format.
    /// A vector is essentially a list of numbers arranged in a specific order.</para>
    /// 
    /// <para>If your data is already a vector, it simply returns it. If it's a tensor (which can have multiple
    /// dimensions), it flattens it into a one-dimensional vector.</para>
    /// </remarks>
    public static Vector<T> ConvertToVector<T, TOutput>(TOutput output)
    {
        if (output is Vector<T> vector)
        {
            return vector;
        }
        else if (output is Tensor<T> tensor)
        {
            // Check tensor shape and handle based on dimensions
            if (tensor.Shape.Length == 2 && tensor.Shape[1] == 1)
            {
                // It's already a column vector format ([n, 1]), just extract the column
                var result = new Vector<T>(tensor.Shape[0]);
                for (int i = 0; i < tensor.Shape[0]; i++)
                {
                    result[i] = tensor[i, 0];
                }
                return result;
            }
            else if (tensor.Shape.Length == 1)
            {
                // It's a 1D tensor, convert directly
                var result = new Vector<T>(tensor.Shape[0]);
                for (int i = 0; i < tensor.Shape[0]; i++)
                {
                    result[i] = tensor[i];
                }
                return result;
            }
            else if (tensor.Shape.Length == 2)
            {
                // For multi-class classification with one-hot encoding
                // Extract class indices (or create a summary value)
                int numSamples = tensor.Shape[0];
                var result = new Vector<T>(numSamples);
                var numOps = MathHelper.GetNumericOperations<T>();

                // For each sample, find the index of the max value (for classification)
                // or average the values (for regression with multiple outputs)
                for (int i = 0; i < numSamples; i++)
                {
                    if (tensor.Shape[1] > 1)
                    {
                        // For classification: find the most likely class
                        int maxIndex = 0;
                        var maxValue = tensor[i, 0];

                        for (int j = 1; j < tensor.Shape[1]; j++)
                        {
                            var currentValue = tensor[i, j];
                            if (numOps.GreaterThan(currentValue, maxValue))
                            {
                                maxValue = currentValue;
                                maxIndex = j;
                            }
                        }

                        // Store the class index as a numeric value
                        result[i] = numOps.FromDouble(maxIndex);
                    }
                    else
                    {
                        // Single column tensor (should be handled by first case, but just in case)
                        result[i] = tensor[i, 0];
                    }
                }

                return result;
            }
            else
            {
                // For higher-dimensional tensors, create sample-preserving projection
                // Collapse all dimensions except the first
                int numSamples = tensor.Shape[0];
                var result = new Vector<T>(numSamples);
                var numOps = MathHelper.GetNumericOperations<T>();

                // Calculate elements per sample
                int elementsPerSample = 1;
                for (int i = 1; i < tensor.Shape.Length; i++)
                {
                    elementsPerSample *= tensor.Shape[i];
                }

                // For each sample, compute a representative value (e.g., average)
                for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
                {
                    var sum = numOps.Zero;
                    int count = 0;

                    // Calculate flat index range for this sample
                    int startIdx = sampleIdx * elementsPerSample;
                    int endIdx = startIdx + elementsPerSample;

                    // Sum all elements for this sample
                    for (int flatIdx = startIdx; flatIdx < endIdx; flatIdx++)
                    {
                        // Convert flat index to multi-dimensional indices
                        int[] indices = new int[tensor.Shape.Length];
                        tensor.GetIndicesFromFlatIndex(flatIdx, indices);

                        // Add to sum
                        sum = numOps.Add(sum, tensor[indices]);
                        count++;
                    }

                    // Store average as the representative value
                    result[sampleIdx] = numOps.Divide(sum, numOps.FromDouble(count));
                }

                return result;
            }
        }

        throw new InvalidOperationException($"Cannot convert {typeof(TOutput).Name} to Vector<{typeof(T).Name}>. Expected Vector<T> or Tensor<T>.");
    }

    /// <summary>
    /// Converts an output value to a scalar value of type T.
    /// </summary>
    /// <typeparam name="T">The numeric type to convert to (e.g., float, double).</typeparam>
    /// <typeparam name="TOutput">The type of the output value.</typeparam>
    /// <param name="output">The output value to convert.</param>
    /// <returns>A scalar value of type T extracted from the output.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the output cannot be converted to a scalar value of type T.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts a single number from more complex data structures.
    /// It's like finding the first item in a list or the first cell in a table.</para>
    /// 
    /// <para>If your data is already a single value, it simply returns it. If it's a vector, it returns
    /// the first element. If it's a tensor or matrix, it returns the first element.</para>
    /// </remarks>
    public static T ConvertToScalar<T, TOutput>(TOutput output)
    {
        if (output is T scalar)
        {
            return scalar;
        }
        else if (output is Vector<T> vector)
        {
            if (vector.Length == 0)
            {
                throw new InvalidOperationException("Cannot extract scalar from empty vector.");
            }
            return vector[0];
        }
        else if (output is Matrix<T> matrix)
        {
            if (matrix.Rows == 0 || matrix.Columns == 0)
            {
                throw new InvalidOperationException("Cannot extract scalar from empty matrix.");
            }
            return matrix[0, 0];
        }
        else if (output is Tensor<T> tensor)
        {
            if (tensor.Length == 0)
            {
                throw new InvalidOperationException("Cannot extract scalar from empty tensor.");
            }
            return tensor[0];
        }

        throw new InvalidOperationException($"Cannot convert {typeof(TOutput).Name} to scalar of type {typeof(T).Name}. Expected T, Vector<T>, Matrix<T>, or Tensor<T>.");
    }

    /// <summary>
    /// Converts a generic object to a Vector.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <param name="obj">The object to convert, which must be either Vector&lt;T&gt; or Tensor&lt;T&gt;.</param>
    /// <returns>A Vector&lt;T&gt; representation of the object, or null if the input is null.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the object cannot be converted to a Vector&lt;T&gt;.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is useful when you have an object but don't know its exact type.</para>
    /// 
    /// <para>It checks if the object is a vector or tensor, and converts it to a vector if possible.
    /// This is helpful when working with dynamic data where the type might not be known at compile time.</para>
    /// </remarks>
    public static Vector<T>? ConvertObjectToVector<T>(object? obj)
    {
        if (obj == null)
        {
            return null;
        }
        
        if (obj is Vector<T> vector)
        {
            return vector;
        }
        else if (obj is Tensor<T> tensor)
        {
            // Use the built-in Flatten method
            return tensor.ToVector();
        }
        
        throw new InvalidOperationException($"Cannot convert {obj.GetType().Name} to Vector<{typeof(T).Name}>. Expected Vector<T> or Tensor<T>.");
    }

    /// <summary>
    /// Converts a fit function that works with generic types to one that works with Matrix and Vector.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <typeparam name="TInput">The type of the input data (must be either Matrix&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <typeparam name="TOutput">The type of the output data (must be either Vector&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <param name="fitFunction">The original fit function that works with TInput and TOutput.</param>
    /// <returns>A converted function that works with Matrix&lt;T&gt; and Vector&lt;T&gt;.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the conversion between types cannot be performed.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adapts a machine learning function so it can work with different data types.</para>
    /// 
    /// <para>Think of it like an adapter that lets you plug a device from one country into a power socket in another country.
    /// It converts the input data to the format expected by the function, runs the function, and then converts
    /// the output back to the format you need.</para>
    /// </remarks>
    public static Func<Matrix<T>, Vector<T>> ConvertFitFunction<T, TInput, TOutput>(Func<TInput, TOutput> fitFunction)
    {
        return (Matrix<T> matrix) =>
        {
            // Convert Matrix<T> to TInput
            TInput convertedInput;
            if (typeof(TInput) == typeof(Matrix<T>))
            {
                convertedInput = (TInput)(object)matrix;
            }
            else if (typeof(TInput) == typeof(Tensor<T>))
            {
                // Use Tensor.FromMatrix instead of manual conversion
                convertedInput = (TInput)(object)Tensor<T>.FromMatrix(matrix);
            }
            else
            {
                throw new InvalidOperationException($"Cannot convert Matrix<{typeof(T).Name}> to {typeof(TInput).Name}. Expected Matrix<T> or Tensor<T>.");
            }

            // Apply the original fit function
            TOutput result = fitFunction(convertedInput);

            // Convert TOutput to Vector<T>
            return ConvertToVector<T, TOutput>(result);
        };
    }

    /// <summary>
    /// Converts a tensor to a matrix with the specified dimensions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <param name="tensor">The tensor to convert.</param>
    /// <param name="rows">The number of rows in the resulting matrix.</param>
    /// <param name="columns">The number of columns in the resulting matrix.</param>
    /// <returns>A Matrix&lt;T&gt; representation of the tensor with the specified dimensions.</returns>
    /// <exception cref="ArgumentException">Thrown when the tensor size doesn't match the specified dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reshapes a tensor into a matrix with specific dimensions.</para>
    /// 
    /// <para>It's like taking a ball of clay and reshaping it into a rectangular form with a specific
    /// number of rows and columns. The total amount of clay (data) remains the same, but its
    /// arrangement changes.</para>
    /// </remarks>
    public static Matrix<T> TensorToMatrix<T>(Tensor<T> tensor, int rows, int columns)
    {
        if (tensor.Length != rows * columns)
        {
            throw new ArgumentException($"Tensor size {tensor.Length} doesn't match the specified matrix dimensions {rows}x{columns}");
        }

        // Reshape the tensor to the desired dimensions and use built-in ToMatrix
        return tensor.Reshape(rows, columns).ToMatrix();
    }

    /// <summary>
    /// Converts a matrix to a tensor with the specified shape.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <param name="matrix">The matrix to convert.</param>
    /// <param name="shape">The shape of the resulting tensor.</param>
    /// <returns>A Tensor&lt;T&gt; representation of the matrix with the specified shape.</returns>
    /// <exception cref="ArgumentException">Thrown when the matrix size doesn't match the product of the shape dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method transforms a 2D matrix into a tensor with a specific shape.</para>
    /// 
    /// <para>It's like taking a flat sheet of paper (the matrix) and folding it into a 3D structure (the tensor).
    /// The amount of data stays the same, but it's organized in a different dimensional structure.</para>
    /// </remarks>
    public static Tensor<T> MatrixToTensor<T>(Matrix<T> matrix, int[] shape)
    {
        int totalSize = 1;
        foreach (int dim in shape)
        {
            totalSize *= dim;
        }

        if (matrix.Rows * matrix.Columns != totalSize)
        {
            throw new ArgumentException($"Matrix size {matrix.Rows * matrix.Columns} doesn't match the specified tensor shape with size {totalSize}");
        }

        // Use Tensor.FromMatrix and then reshape
        Tensor<T> tensor = Tensor<T>.FromMatrix(matrix);
        return tensor.Reshape(shape);
    }

    /// <summary>
    /// Converts a vector to a tensor with the specified shape.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <param name="vector">The vector to convert.</param>
    /// <param name="shape">The shape of the resulting tensor.</param>
    /// <returns>A Tensor&lt;T&gt; representation of the vector with the specified shape.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector length doesn't match the product of the shape dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method transforms a one-dimensional vector into a multi-dimensional tensor.</para>
    /// 
    /// <para>Imagine taking a long string of beads (the vector) and arranging them into a specific
    /// three-dimensional shape. The number of beads stays the same, but they're now organized in a
    /// structured multi-dimensional form.</para>
    /// </remarks>
    public static Tensor<T> VectorToTensor<T>(Vector<T> vector, int[] shape)
    {
        int totalSize = 1;
        foreach (int dim in shape)
        {
            totalSize *= dim;
        }

        if (vector.Length != totalSize)
        {
            throw new ArgumentException($"Vector length {vector.Length} doesn't match the specified tensor shape with size {totalSize}");
        }

        // Use Tensor.FromVector and then reshape
        Tensor<T> tensor = Tensor<T>.FromVector(vector);
        return tensor.Reshape(shape);
    }
}