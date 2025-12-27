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
            // Handle empty or scalar tensors
            if (tensor.Rank == 0 || tensor.Length == 0)
            {
                return Matrix<T>.Empty();
            }

            // Use the built-in ToMatrix method if it's a 2D tensor
            if (tensor.Rank == 2)
            {
                return tensor.ToMatrix();
            }
            else if (tensor.Rank == 1)
            {
                // For 1D tensors, create a row matrix (1 x Length)
                var reshapedTensor = tensor.Reshape(1, tensor.Length);
                return reshapedTensor.ToMatrix();
            }
            else
            {
                // For higher-dimensional tensors (3D+), reshape to 2D first
                // Flatten all dimensions except the last into rows
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
    /// <typeparam name="TOutput">The type of the output data (can be Vector&lt;T&gt;, Tensor&lt;T&gt;, T[], or T scalar).</typeparam>
    /// <param name="output">The output data to convert.</param>
    /// <returns>A Vector&lt;T&gt; representation of the output data.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the output cannot be converted to a Vector&lt;T&gt;.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes data in various formats and converts it to a vector format.
    /// A vector is essentially a list of numbers arranged in a specific order.</para>
    ///
    /// <para>If your data is already a vector, it simply returns it. If it's a tensor (which can have multiple
    /// dimensions), it flattens it into a one-dimensional vector. If it's an array, it wraps it in a Vector.
    /// For scalar values (single numbers), it creates a 2-element vector for binary classification.</para>
    /// </remarks>
    public static Vector<T> ConvertToVector<T, TOutput>(TOutput output)
    {
        if (output is Vector<T> vector)
        {
            return vector;
        }
        else if (output is Tensor<T> tensor)
        {
            // Handle empty tensors
            if (tensor.Rank == 0 || tensor.Length == 0)
            {
                return Vector<T>.Empty();
            }

            // Use the built-in Flatten method to convert tensor to vector
            return tensor.ToVector();
        }
        else if (output is T[] array)
        {
            // Wrap array in a Vector
            return new Vector<T>(array);
        }
        else if (output is T scalar)
        {
            // For scalar output (binary classification), create 2-element vector
            var numOps = MathHelper.GetNumericOperations<T>();
            var one = numOps.FromDouble(1.0);
            var zero = numOps.FromDouble(0.0);

            // Validate scalar is a valid probability in [0, 1]
            if (numOps.LessThan(scalar, zero) || numOps.GreaterThan(scalar, one))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(output),
                    scalar,
                    $"Scalar output must be a probability in [0, 1] for binary classification conversion. " +
                    $"Consider using Vector<T> or T[] for non-probability outputs.");
            }

            // Return [P(class 0), P(class 1)] for binary classification
            var complement = numOps.Subtract(one, scalar);
            return new Vector<T>(new[] { complement, scalar });
        }

        throw new InvalidOperationException($"Cannot convert {typeof(TOutput).Name} to Vector<{typeof(T).Name}>. Expected Vector<T>, Tensor<T>, T[], or T scalar.");
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
            // Create zero indices for all dimensions to get the first element
            var indices = new int[tensor.Rank];
            return tensor[indices];
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
            // Handle empty tensors
            if (tensor.Rank == 0 || tensor.Length == 0)
            {
                return Vector<T>.Empty();
            }

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
                var tensor = Tensor<T>.FromRowMatrix(matrix);
                convertedInput = (TInput)(object)tensor;
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
        Tensor<T> tensor = Tensor<T>.FromRowMatrix(matrix);
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

    /// <summary>
    /// Converts a Matrix or Vector to a Tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <param name="input">The input data to convert (Matrix&lt;T&gt; or Vector&lt;T&gt;).</param>
    /// <returns>A Tensor&lt;T&gt; representation of the input data.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the input cannot be converted to a Tensor&lt;T&gt;.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes data in matrix or vector format and converts it to a tensor format.
    /// A tensor is a multi-dimensional array that can represent matrices (2D), vectors (1D), or higher dimensions.</para>
    ///
    /// <para>If your data is a matrix, it creates a 2D tensor. If it's a vector, it creates a 1D tensor.
    /// If it's already a tensor, it simply returns it.</para>
    /// </remarks>
    public static Tensor<T> ConvertToTensor<T>(object input)
    {
        if (input is Tensor<T> tensor)
        {
            return tensor;
        }
        else if (input is Matrix<T> matrix)
        {
            return Tensor<T>.FromRowMatrix(matrix);
        }
        else if (input is Vector<T> vector)
        {
            return Tensor<T>.FromVector(vector);
        }

        throw new InvalidOperationException($"Cannot convert {input.GetType().Name} to Tensor<{typeof(T).Name}>. Expected Matrix<T>, Vector<T>, or Tensor<T>.");
    }

    /// <summary>
    /// Converts a Vector to the generic TInput type (Vector, Matrix, or Tensor) using a reference input for shape information.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    /// <typeparam name="TInput">The target type (Vector&lt;T&gt;, Matrix&lt;T&gt;, or Tensor&lt;T&gt;).</typeparam>
    /// <param name="vector">The vector to convert.</param>
    /// <param name="referenceInput">A reference input providing shape information for Matrix or Tensor conversion.</param>
    /// <returns>The vector converted to TInput type with the same shape as referenceInput.</returns>
    /// <exception cref="InvalidOperationException">Thrown when conversion is not supported.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts a flat vector back to its original format (vector, matrix, or tensor)
    /// by using a reference sample to determine the correct shape.</para>
    ///
    /// <para>Think of it like taking apart LEGO blocks into a line, then using the original picture to rebuild them
    /// into the correct structure.</para>
    /// </remarks>
    public static TInput ConvertVectorToInput<T, TInput>(Vector<T> vector, TInput referenceInput)
    {
        if (typeof(TInput) == typeof(Vector<T>))
        {
            return (TInput)(object)vector;
        }
        else if (typeof(TInput) == typeof(Matrix<T>) && referenceInput is Matrix<T> refMatrix)
        {
            // Use reference matrix shape and built-in FromVector method
            int rows = refMatrix.Rows;
            int cols = refMatrix.Columns;

            if (vector.Length != rows * cols)
            {
                throw new InvalidOperationException(
                    $"Vector length {vector.Length} doesn't match reference matrix size {rows}x{cols} = {rows * cols}");
            }

            // Matrix.FromVector creates a column matrix - need to reshape to match reference
            return (TInput)(object)TensorToMatrix(Tensor<T>.FromVector(vector, new[] { rows, cols }), rows, cols);
        }
        else if (typeof(TInput) == typeof(Tensor<T>) && referenceInput is Tensor<T> refTensor)
        {
            // Use reference tensor shape and built-in FromVector method with shape parameter
            if (vector.Length != refTensor.Length)
            {
                throw new InvalidOperationException(
                    $"Vector length {vector.Length} doesn't match reference tensor size {refTensor.Length}");
            }

            return (TInput)(object)Tensor<T>.FromVector(vector, refTensor.Shape);
        }
        else
        {
            throw new InvalidOperationException(
                $"Cannot convert Vector<T> to {typeof(TInput).Name}. " +
                "Supported types: Vector<T>, Matrix<T>, Tensor<T>.");
        }
    }

    /// <summary>
    /// Converts a Vector to the generic TInput type (Vector, Tensor, Matrix, T[], or scalar T) without requiring a reference input.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    /// <typeparam name="TInput">The target type (Vector&lt;T&gt;, Tensor&lt;T&gt;, Matrix&lt;T&gt;, T[], or scalar T).</typeparam>
    /// <param name="vector">The vector to convert.</param>
    /// <returns>The vector converted to TInput type with inferred shape.</returns>
    /// <exception cref="InvalidOperationException">Thrown when conversion is not supported.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts a flat vector to the target type without needing
    /// a reference sample. It infers reasonable default shapes based on the target type.</para>
    ///
    /// <para><b>Production Use Case:</b> This is designed for pipeline parallelism and distributed training
    /// where intermediate stages receive activation vectors from previous stages and need to convert them
    /// to the expected input type without having access to the original input's shape.</para>
    ///
    /// <para><b>Shape Inference Rules:</b></para>
    /// <list type="bullet">
    /// <item><description>For Vector&lt;T&gt;: Returns the vector as-is</description></item>
    /// <item><description>For Tensor&lt;T&gt;: Creates a batch-size-1 tensor with shape [1, vector.Length]</description></item>
    /// <item><description>For Matrix&lt;T&gt;: Creates a row matrix with shape [1, vector.Length]</description></item>
    /// <item><description>For T[]: Converts vector to array</description></item>
    /// <item><description>For scalar T: Returns the first element (for single-output models)</description></item>
    /// </list>
    ///
    /// <para><b>Example Usage:</b></para>
    /// <code>
    /// // Pipeline stage receives 128-element activation vector from previous stage
    /// Vector&lt;double&gt; receivedActivations = GetActivationsFromPreviousStage();
    ///
    /// // Convert to Tensor&lt;double&gt; with shape [1, 128] for this stage's model
    /// Tensor&lt;double&gt; stageInput = ConversionsHelper.ConvertVectorToInputWithoutReference&lt;double, Tensor&lt;double&gt;&gt;(receivedActivations);
    /// </code>
    /// </remarks>
    public static TInput ConvertVectorToInputWithoutReference<T, TInput>(Vector<T> vector)
    {
        if (typeof(TInput) == typeof(Vector<T>))
        {
            // Vector to Vector: Direct conversion
            return (TInput)(object)vector;
        }
        else if (typeof(TInput) == typeof(Tensor<T>))
        {
            // Vector to Tensor: Create batch-size-1 tensor with shape [1, vector.Length]
            // This is the standard shape for passing activations in neural networks and pipelines
            return (TInput)(object)Tensor<T>.FromVector(vector, new[] { 1, vector.Length });
        }
        else if (typeof(TInput) == typeof(T[]))
        {
            // Vector to T[]: Convert vector to array
            return (TInput)(object)vector.ToArray();
        }
        else if (typeof(TInput) == typeof(T))
        {
            // Vector to scalar T: Return first element
            // This is used for single-output models (e.g., binary classification, regression)
            if (vector.Length == 0)
            {
                throw new InvalidOperationException("Cannot convert empty vector to scalar.");
            }
            // T is constrained to numeric types (double, float, etc.) which are non-nullable value types
            // The boxing/unboxing here is safe because numeric types cannot be null
            object boxedValue = vector[0] ?? throw new InvalidOperationException("Unexpected null value in vector.");
            return (TInput)boxedValue;
        }
        else if (typeof(TInput) == typeof(Matrix<T>))
        {
            // Vector to Matrix: Create a row matrix (1 x n) as the default shape
            // This is consistent with batch-first convention used in neural networks
            return (TInput)(object)TensorToMatrix(Tensor<T>.FromVector(vector, new[] { 1, vector.Length }), 1, vector.Length);
        }
        else
        {
            throw new InvalidOperationException(
                $"Cannot convert Vector<T> to {typeof(TInput).Name} without reference input. " +
                $"Supported types for shape-free conversion: Vector<T>, Tensor<T>, T[], scalar T, Matrix<T>.");
        }
    }

    /// <summary>
    /// Gets the sample count from supported input/output data types.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <typeparam name="TData">The data type to inspect.</typeparam>
    /// <param name="data">The data instance to inspect.</param>
    /// <returns>The number of samples in the data.</returns>
    public static int GetSampleCount<T, TData>(TData data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (data is Matrix<T> matrix)
        {
            return matrix.Rows;
        }

        if (data is Vector<T> vector)
        {
            return vector.Length;
        }

        if (data is Tensor<T> tensor)
        {
            if (tensor.Rank == 0)
            {
                return tensor.Length > 0 ? 1 : 0;
            }

            return tensor.Shape[0];
        }

        if (data is T)
        {
            return 1;
        }

        throw new InvalidOperationException(
            $"Cannot determine sample count for type {typeof(TData).Name}. " +
            "Expected Matrix<T>, Vector<T>, Tensor<T>, or scalar T.");
    }
}
