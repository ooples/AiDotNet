using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Models;

/// <summary>
/// Generic wrapper that adapts various AiDotNet models to the IServableModel interface.
/// This allows any model with a Predict method to be served via the REST API.
/// </summary>
/// <typeparam name="T">The numeric type used by the model</typeparam>
public class ServableModelWrapper<T> : IServableModel<T>, IServableModelInferenceOptions
{
    private readonly Func<Vector<T>, Vector<T>> _predictFunc;
    private readonly Func<Matrix<T>, Matrix<T>>? _predictBatchFunc;
    private readonly string _modelName;
    private readonly int _inputDimension;
    private readonly int _outputDimension;
    private readonly bool _enableBatching;
    private readonly bool _enableSpeculativeDecoding;

    /// <summary>
    /// Initializes a new instance of the ServableModelWrapper with custom prediction functions.
    /// </summary>
    /// <param name="modelName">The name of the model</param>
    /// <param name="inputDimension">The expected number of input features</param>
    /// <param name="outputDimension">The number of output dimensions</param>
    /// <param name="predictFunc">Function to perform single prediction</param>
    /// <param name="predictBatchFunc">Optional function to perform batch prediction. If not provided, batch prediction will use multiple single predictions.</param>
    /// <param name="enableBatching">Whether this model supports serving-side batching.</param>
    /// <param name="enableSpeculativeDecoding">Whether this model supports speculative decoding in serving/session workflows.</param>
    public ServableModelWrapper(
        string modelName,
        int inputDimension,
        int outputDimension,
        Func<Vector<T>, Vector<T>> predictFunc,
        Func<Matrix<T>, Matrix<T>>? predictBatchFunc = null,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        _modelName = modelName ?? throw new ArgumentNullException(nameof(modelName));
        _inputDimension = inputDimension;
        _outputDimension = outputDimension;
        _predictFunc = predictFunc ?? throw new ArgumentNullException(nameof(predictFunc));
        _predictBatchFunc = predictBatchFunc;
        _enableBatching = enableBatching;
        _enableSpeculativeDecoding = enableSpeculativeDecoding;
    }

    /// <summary>
    /// Initializes a new instance of the ServableModelWrapper from an IRegression model.
    /// </summary>
    /// <param name="modelName">The name of the model</param>
    /// <param name="regressionModel">The regression model to wrap</param>
    /// <param name="inputDimension">The expected number of input features</param>
    public ServableModelWrapper(
        string modelName,
        IRegression<T> regressionModel,
        int inputDimension)
    {
        _modelName = modelName ?? throw new ArgumentNullException(nameof(modelName));
        _inputDimension = inputDimension;
        _outputDimension = 1; // Regression models typically output a single value
        _enableBatching = true;
        _enableSpeculativeDecoding = false;

        if (regressionModel == null)
        {
            throw new ArgumentNullException(nameof(regressionModel));
        }

        _predictFunc = input =>
        {
            // Regression predict typically takes Matrix and returns Vector
            var inputMatrix = new Matrix<T>(1, input.Length);
            for (int i = 0; i < input.Length; i++)
            {
                inputMatrix[0, i] = input[i];
            }

            var predictions = regressionModel.Predict(inputMatrix);
            return new Vector<T>(new[] { predictions[0] });
        };

        _predictBatchFunc = inputs =>
        {
            var predictions = regressionModel.Predict(inputs);
            var result = new Matrix<T>(inputs.Rows, 1);
            for (int i = 0; i < predictions.Length; i++)
            {
                result[i, 0] = predictions[i];
            }
            return result;
        };
    }

    /// <inheritdoc/>
    public string ModelName => _modelName;

    /// <inheritdoc/>
    public int InputDimension => _inputDimension;

    /// <inheritdoc/>
    public int OutputDimension => _outputDimension;

    /// <inheritdoc/>
    public Vector<T> Predict(Vector<T> input)
    {
        if (input.Length != _inputDimension)
        {
            throw new ArgumentException(
                $"Input dimension mismatch. Expected {_inputDimension}, got {input.Length}",
                nameof(input));
        }

        return _predictFunc(input);
    }

    /// <inheritdoc/>
    public Matrix<T> PredictBatch(Matrix<T> inputs)
    {
        if (inputs.Columns != _inputDimension)
        {
            throw new ArgumentException(
                $"Input dimension mismatch. Expected {_inputDimension}, got {inputs.Columns}",
                nameof(inputs));
        }

        // If a batch prediction function was provided, use it
        if (_predictBatchFunc != null)
        {
            return _predictBatchFunc(inputs);
        }

        // Otherwise, fall back to multiple single predictions
        var result = new Matrix<T>(inputs.Rows, _outputDimension);
        for (int i = 0; i < inputs.Rows; i++)
        {
            var inputVector = inputs.GetRow(i);
            var outputVector = Predict(inputVector);
            for (int j = 0; j < _outputDimension; j++)
            {
                result[i, j] = outputVector[j];
            }
        }

        return result;
    }

    bool IServableModelInferenceOptions.EnableBatching => _enableBatching;
    bool IServableModelInferenceOptions.EnableSpeculativeDecoding => _enableSpeculativeDecoding;
}
