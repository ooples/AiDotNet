using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection;

/// <summary>
/// Adapts any <see cref="IAnomalyDetector{T}"/> to the legacy <see cref="IOutlierRemoval{T, TInput, TOutput}"/> interface.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This adapter class allows you to use any of the new anomaly detection
/// algorithms (like Isolation Forest, Local Outlier Factor, etc.) with the existing data
/// preprocessing pipeline that expects the older <see cref="IOutlierRemoval{T, TInput, TOutput}"/> interface.
/// </para>
/// <para>
/// <b>Usage Example:</b>
/// <code>
/// // Create an anomaly detector
/// var detector = new IsolationForest&lt;double&gt;();
///
/// // Wrap it for use with DefaultDataPreprocessor
/// var outlierRemoval = new OutlierRemovalAdapter&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(detector);
///
/// // Use with AiModelBuilder
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureOutlierRemoval(outlierRemoval);
/// </code>
/// </para>
/// </remarks>
public class OutlierRemovalAdapter<T, TInput, TOutput> : IOutlierRemoval<T, TInput, TOutput>
{
    private readonly IAnomalyDetector<T> _detector;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the underlying anomaly detector.
    /// </summary>
    public IAnomalyDetector<T> Detector => _detector;

    /// <summary>
    /// Creates a new adapter that wraps an anomaly detector for use with the legacy outlier removal interface.
    /// </summary>
    /// <param name="detector">The anomaly detector to use for identifying outliers.</param>
    /// <exception cref="ArgumentNullException">Thrown when detector is null.</exception>
    public OutlierRemovalAdapter(IAnomalyDetector<T> detector)
    {
        _detector = detector ?? throw new ArgumentNullException(nameof(detector));
    }

    /// <summary>
    /// Removes outliers from the input data using the configured anomaly detector.
    /// </summary>
    /// <param name="inputs">
    /// The input feature matrix where each row represents a data point and each column represents a feature.
    /// </param>
    /// <param name="outputs">
    /// The target values vector where each element corresponds to a row in the input matrix.
    /// </param>
    /// <returns>
    /// A tuple containing:
    /// - CleanedInputs: A matrix of input features with outliers removed
    /// - CleanedOutputs: A vector of output values corresponding to the cleaned inputs
    /// </returns>
    public (TInput CleanedInputs, TOutput CleanedOutputs) RemoveOutliers(TInput inputs, TOutput outputs)
    {
        // Convert to concrete types for anomaly detection
        var (inputMatrix, outputVector) = ConvertToMatrixVector(inputs, outputs);

        // Fit the detector if not already fitted
        if (!_detector.IsFitted)
        {
            _detector.Fit(inputMatrix);
        }

        // Get predictions: 1 = inlier, -1 = outlier
        var predictions = _detector.Predict(inputMatrix);

        // Filter out outliers
        var cleanedInputs = new List<Vector<T>>();
        var cleanedOutputs = new List<T>();
        T one = NumOps.FromDouble(1);

        for (int i = 0; i < predictions.Length; i++)
        {
            if (NumOps.Equals(predictions[i], one))
            {
                cleanedInputs.Add(new Vector<T>(inputMatrix.GetRow(i)));
                cleanedOutputs.Add(outputVector[i]);
            }
        }

        // Convert back to matrix/vector
        var cleanedInputMatrix = cleanedInputs.Count > 0
            ? new Matrix<T>(cleanedInputs)
            : new Matrix<T>(0, inputMatrix.Columns);
        var cleanedOutputVector = new Vector<T>(cleanedOutputs);

        // Convert back to original types
        return ConvertToOriginalTypes(cleanedInputMatrix, cleanedOutputVector, typeof(TInput), typeof(TOutput));
    }

    private static (Matrix<T> inputs, Vector<T> outputs) ConvertToMatrixVector(TInput inputs, TOutput outputs)
    {
        Matrix<T> inputMatrix;
        Vector<T> outputVector;

        if (inputs is Matrix<T> matrix)
        {
            inputMatrix = matrix;
        }
        else if (inputs is Tensor<T> tensor && tensor.Rank == 2)
        {
            inputMatrix = new Matrix<T>(tensor.Shape[0], tensor.Shape[1]);
            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    inputMatrix[i, j] = tensor[i, j];
                }
            }
        }
        else
        {
            throw new InvalidOperationException(
                $"Unsupported input type {typeof(TInput).Name}. Expected Matrix<{typeof(T).Name}> or 2D Tensor<{typeof(T).Name}>.");
        }

        if (outputs is Vector<T> vector)
        {
            outputVector = vector;
        }
        else if (outputs is Tensor<T> tensorOut && tensorOut.Rank == 1)
        {
            outputVector = new Vector<T>(tensorOut.Shape[0]);
            for (int i = 0; i < tensorOut.Shape[0]; i++)
            {
                outputVector[i] = tensorOut[i];
            }
        }
        else
        {
            throw new InvalidOperationException(
                $"Unsupported output type {typeof(TOutput).Name}. Expected Vector<{typeof(T).Name}> or 1D Tensor<{typeof(T).Name}>.");
        }

        return (inputMatrix, outputVector);
    }

    private static (TInput, TOutput) ConvertToOriginalTypes(Matrix<T> inputMatrix, Vector<T> outputVector,
        Type inputType, Type outputType)
    {
        object cleanedInputs;
        object cleanedOutputs;

        if (inputType == typeof(Matrix<T>))
        {
            cleanedInputs = inputMatrix;
        }
        else if (inputType.IsGenericType && inputType.GetGenericTypeDefinition() == typeof(Tensor<>))
        {
            var tensor = new Tensor<T>(new[] { inputMatrix.Rows, inputMatrix.Columns });
            for (int i = 0; i < inputMatrix.Rows; i++)
            {
                for (int j = 0; j < inputMatrix.Columns; j++)
                {
                    tensor[i, j] = inputMatrix[i, j];
                }
            }
            cleanedInputs = tensor;
        }
        else
        {
            throw new InvalidOperationException($"Cannot convert to input type {inputType.Name}.");
        }

        if (outputType == typeof(Vector<T>))
        {
            cleanedOutputs = outputVector;
        }
        else if (outputType.IsGenericType && outputType.GetGenericTypeDefinition() == typeof(Tensor<>))
        {
            var tensor = new Tensor<T>(new[] { outputVector.Length });
            for (int i = 0; i < outputVector.Length; i++)
            {
                tensor[i] = outputVector[i];
            }
            cleanedOutputs = tensor;
        }
        else
        {
            throw new InvalidOperationException($"Cannot convert to output type {outputType.Name}.");
        }

        return ((TInput)cleanedInputs, (TOutput)cleanedOutputs);
    }
}
