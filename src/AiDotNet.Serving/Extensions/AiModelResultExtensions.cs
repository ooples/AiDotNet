using AiDotNet.Models.Results;
using AiDotNet.Serving.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Extensions;

/// <summary>
/// Extension methods for converting AiModelResult to IServableModel.
/// </summary>
/// <remarks>
/// <para>
/// These extensions enable seamless integration between trained models and the serving infrastructure.
/// Use these methods to convert a AiModelResult into a format suitable for the REST API.
/// </para>
/// <para><b>For Beginners:</b> After training a model, you need to convert it to a serving format.
///
/// Example:
/// <code>
/// // Train a model using AiModelBuilder
/// var trainedModel = builder.Build(trainingData, validationData);
///
/// // Convert to servable model for the REST API
/// var servableModel = trainedModel.ToServableModel("my-model", 10, 1);
///
/// // Register with the model repository
/// modelRepository.LoadModel("my-model", servableModel);
/// </code>
/// </para>
/// </remarks>
public static class AiModelResultExtensions
{
    /// <summary>
    /// Converts a AiModelResult with Vector input/output to an IServableModel.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <param name="result">The prediction model result to convert.</param>
    /// <param name="modelName">The name to assign to the servable model.</param>
    /// <param name="inputDimension">The expected number of input features.</param>
    /// <param name="outputDimension">The number of output dimensions.</param>
    /// <param name="enableBatching">Whether the model supports serving-side batching.</param>
    /// <param name="enableSpeculativeDecoding">Whether the model supports speculative decoding.</param>
    /// <returns>An IServableModel that wraps the prediction model result.</returns>
    /// <exception cref="ArgumentNullException">Thrown when result or modelName is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the model cannot be converted.</exception>
    /// <remarks>
    /// <para>
    /// This method is designed for models with Vector&lt;T&gt; input and output types.
    /// The resulting servable model can be registered with the ModelRepository for REST API serving.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when your model takes a vector of features and outputs a vector.
    ///
    /// Example:
    /// <code>
    /// var servable = trainedModel.ToServableModel("classifier", 784, 10);
    /// modelRepository.LoadModel("classifier", servable);
    /// </code>
    /// </para>
    /// </remarks>
    public static IServableModel<T> ToServableModel<T>(
        this AiModelResult<T, Vector<T>, Vector<T>> result,
        string modelName,
        int inputDimension,
        int outputDimension,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));
        }

        // Create prediction function that uses the AiModelResult
        Vector<T> predictFunc(Vector<T> input)
        {
            return result.Predict(input);
        }

        // Create batch prediction function
        Matrix<T> predictBatchFunc(Matrix<T> inputs)
        {
            // Predict each row and combine into output matrix
            var resultMatrix = new Matrix<T>(inputs.Rows, outputDimension);

            for (int i = 0; i < inputs.Rows; i++)
            {
                var inputVector = inputs.GetRow(i);
                var outputVector = result.Predict(inputVector);

                for (int j = 0; j < outputDimension && j < outputVector.Length; j++)
                {
                    resultMatrix[i, j] = outputVector[j];
                }
            }

            return resultMatrix;
        }

        return new ServableModelWrapper<T>(
            modelName,
            inputDimension,
            outputDimension,
            predictFunc,
            predictBatchFunc,
            enableBatching,
            enableSpeculativeDecoding);
    }

    /// <summary>
    /// Converts a AiModelResult with Matrix input and Vector output to an IServableModel.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <param name="result">The prediction model result to convert.</param>
    /// <param name="modelName">The name to assign to the servable model.</param>
    /// <param name="inputDimension">The expected number of input features.</param>
    /// <param name="outputDimension">The number of output dimensions.</param>
    /// <param name="enableBatching">Whether the model supports serving-side batching.</param>
    /// <param name="enableSpeculativeDecoding">Whether the model supports speculative decoding.</param>
    /// <returns>An IServableModel that wraps the prediction model result.</returns>
    /// <exception cref="ArgumentNullException">Thrown when result or modelName is null.</exception>
    /// <remarks>
    /// <para>
    /// This method is designed for models with Matrix&lt;T&gt; input and Vector&lt;T&gt; output types.
    /// This is common for regression and classification models that take feature matrices.
    /// </para>
    /// <para><b>For Beginners:</b> Use this for models that typically take batches of samples.
    ///
    /// Example:
    /// <code>
    /// var servable = regressionModel.ToServableModel("regressor", 50, 1);
    /// modelRepository.LoadModel("regressor", servable);
    /// </code>
    /// </para>
    /// </remarks>
    public static IServableModel<T> ToServableModel<T>(
        this AiModelResult<T, Matrix<T>, Vector<T>> result,
        string modelName,
        int inputDimension,
        int outputDimension,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));
        }

        // Create prediction function that converts Vector to single-row Matrix
        Vector<T> predictFunc(Vector<T> input)
        {
            // Convert single vector to a 1-row matrix
            var inputMatrix = new Matrix<T>(1, input.Length);
            for (int i = 0; i < input.Length; i++)
            {
                inputMatrix[0, i] = input[i];
            }

            var output = result.Predict(inputMatrix);
            return output;
        }

        // Create batch prediction function
        Matrix<T> predictBatchFunc(Matrix<T> inputs)
        {
            // For Matrix->Vector models, we predict row by row
            var resultMatrix = new Matrix<T>(inputs.Rows, outputDimension);

            for (int i = 0; i < inputs.Rows; i++)
            {
                // Create a single-row matrix for this input
                var inputMatrix = new Matrix<T>(1, inputs.Columns);
                for (int j = 0; j < inputs.Columns; j++)
                {
                    inputMatrix[0, j] = inputs[i, j];
                }

                var outputVector = result.Predict(inputMatrix);

                for (int j = 0; j < outputDimension && j < outputVector.Length; j++)
                {
                    resultMatrix[i, j] = outputVector[j];
                }
            }

            return resultMatrix;
        }

        return new ServableModelWrapper<T>(
            modelName,
            inputDimension,
            outputDimension,
            predictFunc,
            predictBatchFunc,
            enableBatching,
            enableSpeculativeDecoding);
    }

    /// <summary>
    /// Converts a AiModelResult to an IServableModel using custom conversion functions.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <typeparam name="TInput">The input type of the prediction model.</typeparam>
    /// <typeparam name="TOutput">The output type of the prediction model.</typeparam>
    /// <param name="result">The prediction model result to convert.</param>
    /// <param name="modelName">The name to assign to the servable model.</param>
    /// <param name="inputDimension">The expected number of input features.</param>
    /// <param name="outputDimension">The number of output dimensions.</param>
    /// <param name="inputConverter">Function to convert Vector&lt;T&gt; to TInput.</param>
    /// <param name="outputConverter">Function to convert TOutput to Vector&lt;T&gt;.</param>
    /// <param name="enableBatching">Whether the model supports serving-side batching.</param>
    /// <param name="enableSpeculativeDecoding">Whether the model supports speculative decoding.</param>
    /// <returns>An IServableModel that wraps the prediction model result.</returns>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    /// <remarks>
    /// <para>
    /// This method provides maximum flexibility for converting any AiModelResult type.
    /// You provide custom conversion functions to handle the type transformations.
    /// </para>
    /// <para><b>For Beginners:</b> Use this for models with custom input/output types.
    ///
    /// Example:
    /// <code>
    /// var servable = customModel.ToServableModel(
    ///     "custom-model",
    ///     10,
    ///     5,
    ///     vector => new CustomInput(vector),
    ///     output => output.ToVector());
    /// </code>
    /// </para>
    /// </remarks>
    public static IServableModel<T> ToServableModel<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        string modelName,
        int inputDimension,
        int outputDimension,
        Func<Vector<T>, TInput> inputConverter,
        Func<TOutput, Vector<T>> outputConverter,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));
        }

        if (inputConverter == null)
        {
            throw new ArgumentNullException(nameof(inputConverter));
        }

        if (outputConverter == null)
        {
            throw new ArgumentNullException(nameof(outputConverter));
        }

        // Create prediction function using the converters
        Vector<T> predictFunc(Vector<T> input)
        {
            var convertedInput = inputConverter(input);
            var output = result.Predict(convertedInput);
            return outputConverter(output);
        }

        // Create batch prediction function
        Matrix<T> predictBatchFunc(Matrix<T> inputs)
        {
            var resultMatrix = new Matrix<T>(inputs.Rows, outputDimension);

            for (int i = 0; i < inputs.Rows; i++)
            {
                var inputVector = inputs.GetRow(i);
                var outputVector = predictFunc(inputVector);

                for (int j = 0; j < outputDimension && j < outputVector.Length; j++)
                {
                    resultMatrix[i, j] = outputVector[j];
                }
            }

            return resultMatrix;
        }

        return new ServableModelWrapper<T>(
            modelName,
            inputDimension,
            outputDimension,
            predictFunc,
            predictBatchFunc,
            enableBatching,
            enableSpeculativeDecoding);
    }
}
