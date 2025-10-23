using System.Text.Json;

namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the complete result of a prediction model building process, including the trained model,
/// optimization results, and normalization information.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This class wraps a trained prediction model along with all the information
/// needed to use it effectively. It includes:
/// - The trained model itself
/// - Information about how the model was optimized
/// - Details about how the data was normalized
///
/// This wrapper allows the model to be used for predictions, saved, loaded, and provides access
/// to all the metadata about how it was created and how well it performs.
/// </remarks>
public class PredictionModelResult<T, TInput, TOutput> : IPredictiveModel<T, TInput, TOutput>, IFullModel<T, TInput, TOutput>
{
    private IFullModel<T, TInput, TOutput>? _innerModel;
    private OptimizationResult<T, TInput, TOutput>? _optimizationResult;
    private NormalizationInfo<T, TInput, TOutput>? _normalizationInfo;

    /// <summary>
    /// Gets the inner model used for predictions.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? Model => _innerModel;

    /// <summary>
    /// Gets the optimization result containing detailed information about the training process.
    /// </summary>
    public OptimizationResult<T, TInput, TOutput>? OptimizationResult => _optimizationResult;

    /// <summary>
    /// Gets the normalization information used to preprocess the data.
    /// </summary>
    public NormalizationInfo<T, TInput, TOutput>? NormalizationInfo => _normalizationInfo;

    /// <summary>
    /// Initializes a new instance of the PredictionModelResult class.
    /// </summary>
    /// <param name="model">The trained model.</param>
    /// <param name="optimizationResult">The optimization results.</param>
    /// <param name="normalizationInfo">The normalization information.</param>
    public PredictionModelResult(
        IFullModel<T, TInput, TOutput>? model,
        OptimizationResult<T, TInput, TOutput> optimizationResult,
        NormalizationInfo<T, TInput, TOutput> normalizationInfo)
    {
        _innerModel = model;
        _optimizationResult = optimizationResult ?? throw new ArgumentNullException(nameof(optimizationResult));
        _normalizationInfo = normalizationInfo ?? throw new ArgumentNullException(nameof(normalizationInfo));
    }

    /// <summary>
    /// Initializes a new instance of the PredictionModelResult class for deserialization.
    /// </summary>
    public PredictionModelResult()
    {
        _innerModel = null;
        _optimizationResult = null;
        _normalizationInfo = null;
    }

    #region IPredictiveModel Implementation

    /// <summary>
    /// Makes predictions using the trained model on new input data.
    /// </summary>
    /// <param name="input">The input data to make predictions for.</param>
    /// <returns>The predicted output values.</returns>
    public TOutput Predict(TInput input)
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot make predictions.");
        }

        return _innerModel.Predict(input);
    }

    /// <summary>
    /// Retrieves metadata and performance information about the trained model.
    /// </summary>
    /// <returns>Metadata about the model's performance and configuration.</returns>
    public ModelMetaData<T> GetModelMetadata()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot retrieve metadata.");
        }

        return _innerModel.GetModelMetaData();
    }

    #endregion

    #region IModel Implementation

    /// <summary>
    /// Trains the model using input features and their corresponding target values.
    /// </summary>
    /// <param name="input">The input training data.</param>
    /// <param name="expectedOutput">The expected output values.</param>
    public void Train(TInput input, TOutput expectedOutput)
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot train.");
        }

        _innerModel.Train(input, expectedOutput);
    }

    /// <summary>
    /// Retrieves metadata and performance metrics about the trained model.
    /// </summary>
    /// <returns>An object containing metadata and performance metrics about the trained model.</returns>
    public ModelMetaData<T> GetModelMetaData()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot retrieve metadata.");
        }

        return _innerModel.GetModelMetaData();
    }

    #endregion

    #region IModelSerializer Implementation

    /// <summary>
    /// Converts the current state of the model into a binary format.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    public byte[] Serialize()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot serialize.");
        }

        var data = new
        {
            Model = _innerModel.Serialize(),
            OptimizationResult = _optimizationResult,
            NormalizationInfo = _normalizationInfo
        };

        return JsonSerializer.SerializeToUtf8Bytes(data);
    }

    /// <summary>
    /// Loads a previously serialized model from binary data.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model data.</param>
    public void Deserialize(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var jsonData = JsonSerializer.Deserialize<Dictionary<string, object>>(data);
        if (jsonData == null)
        {
            throw new InvalidOperationException("Failed to deserialize model data.");
        }

        // Note: This is a simplified implementation. A full implementation would need to:
        // 1. Deserialize the inner model type information
        // 2. Reconstruct the inner model using the appropriate type
        // 3. Properly deserialize OptimizationResult and NormalizationInfo
        throw new NotImplementedException("Full deserialization logic needs to be implemented based on specific model types.");
    }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model should be saved.</param>
    public void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var data = Serialize();
        File.WriteAllBytes(filePath, data);
    }

    /// <summary>
    /// Loads the model from a file.
    /// </summary>
    /// <param name="filePath">The path to the file containing the saved model.</param>
    public void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Model file not found at path: {filePath}", filePath);
        }

        var data = File.ReadAllBytes(filePath);
        Deserialize(data);
    }

    #endregion

    #region IParameterizable Implementation

    /// <summary>
    /// Gets the parameters that can be optimized.
    /// </summary>
    /// <returns>A vector containing the model parameters.</returns>
    public Vector<T> GetParameters()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot get parameters.");
        }

        return _innerModel.GetParameters();
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">The parameter vector to set.</param>
    public void SetParameters(Vector<T> parameters)
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot set parameters.");
        }

        _innerModel.SetParameters(parameters);
    }

    /// <summary>
    /// Gets the number of parameters in the model.
    /// </summary>
    public int ParameterCount
    {
        get
        {
            if (_innerModel == null)
            {
                return 0;
            }

            return _innerModel.ParameterCount;
        }
    }

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to use for the new instance.</param>
    /// <returns>A new model instance with the specified parameters.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model or required metadata is not initialized.</exception>
    public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot create instance with parameters.");
        }

        if (_optimizationResult == null)
        {
            throw new InvalidOperationException("OptimizationResult is missing. Cannot create instance with parameters.");
        }

        if (_normalizationInfo == null)
        {
            throw new InvalidOperationException("NormalizationInfo is missing. Cannot create instance with parameters.");
        }

        var newInnerModel = _innerModel.WithParameters(parameters);
        return new PredictionModelResult<T, TInput, TOutput>(
            newInnerModel,
            _optimizationResult,
            _normalizationInfo);
    }

    #endregion

    #region IFeatureAware Implementation

    /// <summary>
    /// Gets the indices of features that are actively used by this model.
    /// </summary>
    /// <returns>An enumerable of feature indices.</returns>
    public IEnumerable<int> GetActiveFeatureIndices()
    {
        if (_innerModel == null)
        {
            return Enumerable.Empty<int>();
        }

        return _innerModel.GetActiveFeatureIndices();
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    /// <param name="featureIndices">The feature indices to set as active.</param>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot set active feature indices.");
        }

        _innerModel.SetActiveFeatureIndices(featureIndices);
    }

    /// <summary>
    /// Checks if a specific feature is used by this model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is used; otherwise, false.</returns>
    public bool IsFeatureUsed(int featureIndex)
    {
        if (_innerModel == null)
        {
            return false;
        }

        return _innerModel.IsFeatureUsed(featureIndex);
    }

    #endregion

    #region IFeatureImportance Implementation

    /// <summary>
    /// Gets the feature importance scores.
    /// </summary>
    /// <returns>A dictionary mapping feature names to importance scores.</returns>
    public Dictionary<string, T> GetFeatureImportance()
    {
        if (_innerModel == null)
        {
            return new Dictionary<string, T>();
        }

        return _innerModel.GetFeatureImportance();
    }

    #endregion

    #region ICloneable Implementation

    /// <summary>
    /// Creates a deep copy of this object.
    /// </summary>
    /// <returns>A deep copy of the prediction model result.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model or required metadata is not initialized.</exception>
    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot create deep copy.");
        }

        if (_optimizationResult == null)
        {
            throw new InvalidOperationException("OptimizationResult is missing. Cannot create deep copy.");
        }

        if (_normalizationInfo == null)
        {
            throw new InvalidOperationException("NormalizationInfo is missing. Cannot create deep copy.");
        }

        var copiedInnerModel = _innerModel.DeepCopy();
        return new PredictionModelResult<T, TInput, TOutput>(
            copiedInnerModel,
            _optimizationResult,
            _normalizationInfo);
    }

    /// <summary>
    /// Creates a shallow copy of this object.
    /// </summary>
    /// <returns>A shallow copy of the prediction model result.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model or required metadata is not initialized.</exception>
    public IFullModel<T, TInput, TOutput> Clone()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot create clone.");
        }

        if (_optimizationResult == null)
        {
            throw new InvalidOperationException("OptimizationResult is missing. Cannot create clone.");
        }

        if (_normalizationInfo == null)
        {
            throw new InvalidOperationException("NormalizationInfo is missing. Cannot create clone.");
        }

        var clonedInnerModel = _innerModel.Clone();
        return new PredictionModelResult<T, TInput, TOutput>(
            clonedInnerModel,
            _optimizationResult,
            _normalizationInfo);
    }

    #endregion
}
