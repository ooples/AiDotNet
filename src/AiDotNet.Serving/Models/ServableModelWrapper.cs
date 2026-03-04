using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

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
    private readonly int[] _inputShape;
    private readonly int[] _outputShape;
    private readonly DynamicShapeInfo _dynamicShapeInfo;
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
    /// <param name="enableBatching">Whether serving-side batching is enabled for this model.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled for this model.</param>
    /// <param name="inputShape">Optional full input shape array. If null, derived from inputDimension.</param>
    /// <param name="outputShape">Optional full output shape array. If null, derived from outputDimension.</param>
    /// <param name="dynamicShapeInfo">Optional dynamic shape information. If null, all dimensions are fixed.</param>
    public ServableModelWrapper(
        string modelName,
        int inputDimension,
        int outputDimension,
        Func<Vector<T>, Vector<T>> predictFunc,
        Func<Matrix<T>, Matrix<T>>? predictBatchFunc = null,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false,
        int[]? inputShape = null,
        int[]? outputShape = null,
        DynamicShapeInfo? dynamicShapeInfo = null)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        _modelName = modelName;
        _inputDimension = inputDimension;
        _outputDimension = outputDimension;
        _inputShape = inputShape ?? new[] { inputDimension };
        _outputShape = outputShape ?? new[] { outputDimension };
        _dynamicShapeInfo = dynamicShapeInfo ?? DynamicShapeInfo.None;
        Guard.NotNull(predictFunc);
        _predictFunc = predictFunc;
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
    /// <param name="enableBatching">Whether serving-side batching is enabled for this model.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled for this model.</param>
    public ServableModelWrapper(
        string modelName,
        IRegression<T> regressionModel,
        int inputDimension,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        _modelName = modelName;
        _inputDimension = inputDimension;
        _outputDimension = 1; // Regression models typically output a single value
        _inputShape = new[] { inputDimension };
        _outputShape = new[] { 1 };
        _dynamicShapeInfo = DynamicShapeInfo.None;
        _enableBatching = enableBatching;
        _enableSpeculativeDecoding = enableSpeculativeDecoding;

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

    /// <summary>
    /// Initializes a new instance of the ServableModelWrapper from a Matrix-to-Vector model
    /// (regression, classification, clustering, survival, causal, online learning, time series).
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="model">The model that accepts Matrix input and returns Vector output.</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled.</param>
    public ServableModelWrapper(
        string modelName,
        IFullModel<T, Matrix<T>, Vector<T>> model,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        Guard.NotNull(model);

        _modelName = modelName;
        _enableBatching = enableBatching;
        _enableSpeculativeDecoding = enableSpeculativeDecoding;

        // Extract shape from IModelShape if available
        if (model is IModelShape shapeModel)
        {
            var inShape = shapeModel.GetInputShape();
            var outShape = shapeModel.GetOutputShape();
            _inputShape = inShape;
            _outputShape = outShape;
            _inputDimension = inShape.Length > 0 ? inShape[inShape.Length - 1] : 0;
            _outputDimension = outShape.Length > 0 ? outShape[outShape.Length - 1] : 0;
            _dynamicShapeInfo = shapeModel.GetDynamicShapeInfo();
        }
        else
        {
            _inputDimension = 0;
            _outputDimension = 0;
            _inputShape = Array.Empty<int>();
            _outputShape = Array.Empty<int>();
            _dynamicShapeInfo = DynamicShapeInfo.None;
        }

        _predictFunc = input =>
        {
            // Wrap single Vector into a 1-row Matrix for Matrix→Vector models
            var inputMatrix = new Matrix<T>(1, input.Length);
            for (int i = 0; i < input.Length; i++)
            {
                inputMatrix[0, i] = input[i];
            }

            return model.Predict(inputMatrix);
        };

        _predictBatchFunc = inputs =>
        {
            var predictions = model.Predict(inputs);
            var result = new Matrix<T>(inputs.Rows, predictions.Length / Math.Max(inputs.Rows, 1));
            // If predictions is a single Vector from batch, wrap it appropriately
            if (inputs.Rows == 1)
            {
                for (int j = 0; j < predictions.Length; j++)
                {
                    result[0, j] = predictions[j];
                }
            }
            else
            {
                // For batch processing, call predict for each row
                for (int i = 0; i < inputs.Rows; i++)
                {
                    var row = inputs.GetRow(i);
                    var rowMatrix = new Matrix<T>(1, row.Length);
                    for (int j = 0; j < row.Length; j++)
                    {
                        rowMatrix[0, j] = row[j];
                    }

                    var pred = model.Predict(rowMatrix);
                    for (int j = 0; j < pred.Length; j++)
                    {
                        result[i, j] = pred[j];
                    }
                }
            }
            return result;
        };
    }

    /// <summary>
    /// Initializes a new instance of the ServableModelWrapper from a Tensor-to-Tensor model
    /// (neural networks, diffusion models).
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="model">The model that accepts Tensor input and returns Tensor output.</param>
    /// <param name="inputShape">The shape to reshape flat input vectors into tensors.</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled.</param>
    public ServableModelWrapper(
        string modelName,
        IFullModel<T, Tensor<T>, Tensor<T>> model,
        int[] inputShape,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        Guard.NotNull(model);

        _modelName = modelName;
        _enableBatching = enableBatching;
        _enableSpeculativeDecoding = enableSpeculativeDecoding;

        // Use provided inputShape and extract output shape from IModelShape if available
        _inputShape = inputShape ?? Array.Empty<int>();
        _inputDimension = 1;
        foreach (int dim in _inputShape)
        {
            if (dim > 0)
            {
                _inputDimension *= dim;
            }
        }

        if (model is IModelShape shapeModel)
        {
            _outputShape = shapeModel.GetOutputShape();
            _dynamicShapeInfo = shapeModel.GetDynamicShapeInfo();
        }
        else
        {
            _outputShape = Array.Empty<int>();
            _dynamicShapeInfo = DynamicShapeInfo.None;
        }

        _outputDimension = 1;
        foreach (int dim in _outputShape)
        {
            if (dim > 0)
            {
                _outputDimension *= dim;
            }
        }

        _predictFunc = input =>
        {
            // Reshape flat Vector into Tensor using inputShape
            var tensor = new Tensor<T>(_inputShape, input);
            var result = model.Predict(tensor);

            // Flatten Tensor output back to Vector
            var outputVector = new Vector<T>(result.Length);
            for (int i = 0; i < result.Length; i++)
            {
                outputVector[i] = result[i];
            }
            return outputVector;
        };

        _predictBatchFunc = null; // Falls back to row-by-row single predictions
    }

    /// <summary>
    /// Initializes a new instance of the ServableModelWrapper from a Vector-to-Vector model
    /// (reinforcement learning agents).
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="model">The model that accepts Vector input and returns Vector output.</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled.</param>
    public ServableModelWrapper(
        string modelName,
        IFullModel<T, Vector<T>, Vector<T>> model,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        Guard.NotNull(model);

        _modelName = modelName;
        _enableBatching = enableBatching;
        _enableSpeculativeDecoding = enableSpeculativeDecoding;

        if (model is IModelShape shapeModel)
        {
            var inShape = shapeModel.GetInputShape();
            var outShape = shapeModel.GetOutputShape();
            _inputShape = inShape;
            _outputShape = outShape;
            _inputDimension = inShape.Length > 0 ? inShape[inShape.Length - 1] : 0;
            _outputDimension = outShape.Length > 0 ? outShape[outShape.Length - 1] : 0;
            _dynamicShapeInfo = shapeModel.GetDynamicShapeInfo();
        }
        else
        {
            _inputDimension = 0;
            _outputDimension = 0;
            _inputShape = Array.Empty<int>();
            _outputShape = Array.Empty<int>();
            _dynamicShapeInfo = DynamicShapeInfo.None;
        }

        // Direct pass-through: Vector→Vector models need no adaptation
        _predictFunc = input => model.Predict(input);
        _predictBatchFunc = null; // Falls back to row-by-row single predictions
    }

    /// <summary>
    /// Creates a ServableModelWrapper by automatically detecting the model's input/output type pattern
    /// and selecting the appropriate adapter.
    /// </summary>
    /// <param name="modelName">The name for the servable model.</param>
    /// <param name="model">The model instance (must implement IModelSerializer and one of the IFullModel variants).</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled.</param>
    /// <returns>A ServableModelWrapper configured for the detected model type.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model does not implement a supported IFullModel variant.</exception>
    public static ServableModelWrapper<T> FromModel(
        string modelName,
        IModelSerializer model,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false)
    {
        Guard.NotNullOrWhiteSpace(modelName);
        Guard.NotNull(model);

        // Check Vector→Vector first (RL agents) — most specific
        if (model is IFullModel<T, Vector<T>, Vector<T>> vectorModel)
        {
            return new ServableModelWrapper<T>(modelName, vectorModel, enableBatching, enableSpeculativeDecoding);
        }

        // Check Matrix→Vector (regression, classification, clustering, etc.)
        if (model is IFullModel<T, Matrix<T>, Vector<T>> matrixModel)
        {
            return new ServableModelWrapper<T>(modelName, matrixModel, enableBatching, enableSpeculativeDecoding);
        }

        // Check Tensor→Tensor (neural networks, diffusion)
        if (model is IFullModel<T, Tensor<T>, Tensor<T>> tensorModel)
        {
            // Get input shape from IModelShape
            int[] inputShape;
            if (model is IModelShape shapeModel)
            {
                inputShape = shapeModel.GetInputShape();
            }
            else
            {
                throw new InvalidOperationException(
                    $"Model '{modelName}' (type: {model.GetType().Name}) is a Tensor model but does not implement IModelShape. " +
                    "Tensor models must implement IModelShape to provide the input shape for serving.");
            }

            return new ServableModelWrapper<T>(modelName, tensorModel, inputShape, enableBatching, enableSpeculativeDecoding);
        }

        throw new InvalidOperationException(
            $"Model '{modelName}' (type: {model.GetType().Name}) does not implement a supported IFullModel variant. " +
            "Supported patterns: IFullModel<T, Vector<T>, Vector<T>>, IFullModel<T, Matrix<T>, Vector<T>>, " +
            "IFullModel<T, Tensor<T>, Tensor<T>>. Use the constructor overload with custom predict functions instead.");
    }

    /// <summary>
    /// Loads an AIMF model file and creates a ServableModelWrapper in one step.
    /// Combines ModelLoader.Load with FromModel for the common serving use case.
    /// </summary>
    /// <param name="filePath">The path to the AIMF model file.</param>
    /// <param name="modelName">The name for the servable model.</param>
    /// <param name="enableBatching">Whether serving-side batching is enabled.</param>
    /// <param name="enableSpeculativeDecoding">Whether speculative decoding is enabled.</param>
    /// <param name="licenseKey">Optional license key for encrypted AIMF models.</param>
    /// <param name="decryptionToken">Optional server-side decryption token.</param>
    /// <returns>A ServableModelWrapper ready for serving.</returns>
    public static ServableModelWrapper<T> LoadServable(
        string filePath,
        string modelName,
        bool enableBatching = true,
        bool enableSpeculativeDecoding = false,
        string? licenseKey = null,
        byte[]? decryptionToken = null)
    {
        var model = ModelLoader.Load<T>(filePath, licenseKey, decryptionToken);
        return FromModel(modelName, model, enableBatching, enableSpeculativeDecoding);
    }

    /// <inheritdoc/>
    public string ModelName => _modelName;

    /// <inheritdoc/>
    public int InputDimension => _inputDimension;

    /// <inheritdoc/>
    public int OutputDimension => _outputDimension;

    /// <inheritdoc/>
    public int[] InputShape => (int[])_inputShape.Clone();

    /// <inheritdoc/>
    public int[] OutputShape => (int[])_outputShape.Clone();

    /// <inheritdoc/>
    public DynamicShapeInfo DynamicShapeInfo => _dynamicShapeInfo;

    bool IServableModelInferenceOptions.EnableBatching => _enableBatching;

    bool IServableModelInferenceOptions.EnableSpeculativeDecoding => _enableSpeculativeDecoding;

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
}
