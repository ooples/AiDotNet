namespace AiDotNet.Helpers;

/// <summary>
/// Provides helper methods for outlier removal algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This helper class provides common functionality used by different
/// outlier removal algorithms. It handles converting between different data types and ensures
/// that the algorithms can work with various input and output formats.</para>
/// </remarks>
public static class OutlierRemovalHelper<T, TInput, TOutput>
{
    /// <summary>
    /// Converts generic input and output types to concrete Matrix and Vector types.
    /// </summary>
    /// <param name="inputs">The generic input data.</param>
    /// <param name="outputs">The generic output data.</param>
    /// <returns>A tuple containing the concrete Matrix and Vector representations.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts your data into a standard format that
    /// all the outlier detection algorithms can work with. It's like translating different
    /// languages into a common language that everyone understands.</para>
    /// </remarks>
    public static (Matrix<T> InputMatrix, Vector<T> OutputVector) ConvertToMatrixVector(TInput inputs, TOutput outputs)
    {
        if (inputs is Matrix<T> inputMatrix && outputs is Vector<T> outputVector)
        {
            return (inputMatrix, outputVector);
        }
        else if (inputs is Tensor<T> inputTensor && outputs is Tensor<T> outputTensor)
        {
            // Convert Tensor to Matrix and Vector
            if (inputTensor.Shape.Length != 2)
            {
                throw new InvalidOperationException("Input tensor must be 2-dimensional (matrix-like)");
            }

            if (outputTensor.Shape.Length != 1)
            {
                throw new InvalidOperationException("Output tensor must be 1-dimensional (vector-like)");
            }

            // Convert tensor to matrix - create a new matrix with the same data
            var inputRows = inputTensor.Shape[0];
            var inputCols = inputTensor.Shape[1];
            var inputMatrix2 = new Matrix<T>(inputRows, inputCols);

            for (int i = 0; i < inputRows; i++)
            {
                for (int j = 0; j < inputCols; j++)
                {
                    inputMatrix2[i, j] = inputTensor[i, j];
                }
            }

            return (inputMatrix2, outputTensor.ToVector());
        }

        throw new InvalidOperationException(
            $"Unsupported combination of input type {typeof(TInput).Name} and output type {typeof(TOutput).Name}. " +
            "Currently supported combinations are: " +
            $"(Matrix<{typeof(T).Name}>, Vector<{typeof(T).Name}>) for linear models and " +
            $"(Tensor<{typeof(T).Name}>, Tensor<{typeof(T).Name}>) for tensor models.");
    }

    /// <summary>
    /// Converts Matrix and Vector data back to the original generic types.
    /// </summary>
    /// <param name="cleanedInputMatrix">The cleaned matrix of inputs.</param>
    /// <param name="cleanedOutputVector">The cleaned vector of outputs.</param>
    /// <param name="inputType">The original type of the input data.</param>
    /// <param name="outputType">The original type of the output data.</param>
    /// <returns>A tuple containing the converted data in their original types.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> After processing the data, this method converts it back to
    /// its original format so it can be used with the rest of your code. It's like translating
    /// from the common language back to each person's native language.</para>
    /// </remarks>
    public static (TInput CleanedInputs, TOutput CleanedOutputs) ConvertToOriginalTypes(
        Matrix<T> cleanedInputMatrix,
        Vector<T> cleanedOutputVector,
        Type inputType,
        Type outputType)
    {
        if (inputType == typeof(Matrix<T>) && outputType == typeof(Vector<T>))
        {
            return ((TInput)(object)cleanedInputMatrix, (TOutput)(object)cleanedOutputVector);
        }
        else if (inputType == typeof(Tensor<T>) && outputType == typeof(Tensor<T>))
        {
            // Convert Matrix to Tensor
            var inputTensor = Tensor<T>.FromRowMatrix(cleanedInputMatrix);

            // Convert Vector to Tensor
            var outputTensor = Tensor<T>.FromVector(cleanedOutputVector);

            return ((TInput)(object)inputTensor, (TOutput)(object)outputTensor);
        }

        throw new InvalidOperationException(
            $"Unsupported combination of input type {inputType.Name} and output type {outputType.Name}. " +
            "Currently supported combinations are: " +
            $"(Matrix<{typeof(T).Name}>, Vector<{typeof(T).Name}>) for linear models and " +
            $"(Tensor<{typeof(T).Name}>, Tensor<{typeof(T).Name}>) for tensor models.");
    }
}
