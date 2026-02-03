using System.Text;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using Newtonsoft.Json;

namespace AiDotNet.OnlineLearning;

/// <summary>
/// Abstract base class for online (incremental) learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This base class provides common functionality for online learning models including
/// sample counting, learning rate scheduling, and incremental updates.
/// </para>
/// <para>
/// <b>For Beginners:</b> This class contains shared code that all online learning models need,
/// so each specific model doesn't have to reimplement it.
///
/// Key shared functionality:
/// - Tracking how many samples have been seen
/// - Managing the learning rate (step size for updates)
/// - Converting between single-sample and batch updates
/// - Standard IFullModel interface implementation
/// </para>
/// </remarks>
public abstract class OnlineLearningModelBase<T> : IOnlineLearningModel<T>
{
    /// <summary>
    /// Numeric operations helper for generic math.
    /// </summary>
    protected readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The default loss function for gradient computation.
    /// </summary>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Number of samples the model has been trained on.
    /// </summary>
    protected long SampleCount;

    /// <summary>
    /// Initial learning rate.
    /// </summary>
    protected readonly double InitialLearningRate;

    /// <summary>
    /// Learning rate decay schedule type.
    /// </summary>
    protected readonly LearningRateSchedule LearningRateScheduleType;

    /// <summary>
    /// Indicates whether the model has been initialized.
    /// </summary>
    protected bool IsInitialized;

    /// <summary>
    /// Gets the number of features the model was trained on.
    /// </summary>
    protected int NumFeatures { get; set; }

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Gets whether the model is trained (has seen at least one sample).
    /// </summary>
    public bool IsTrained => IsInitialized && SampleCount > 0;

    /// <summary>
    /// Gets the default loss function.
    /// </summary>
    public ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    public virtual int ParameterCount => NumFeatures;

    /// <summary>
    /// Gets whether JIT compilation is supported.
    /// </summary>
    public virtual bool SupportsJitCompilation => false;

    /// <summary>
    /// Initializes a new instance of the OnlineLearningModelBase class.
    /// </summary>
    /// <param name="initialLearningRate">Initial learning rate. Default is 0.01.</param>
    /// <param name="learningRateSchedule">Learning rate schedule. Default is InverseScaling.</param>
    protected OnlineLearningModelBase(
        double initialLearningRate = 0.01,
        LearningRateSchedule learningRateSchedule = LearningRateSchedule.InverseScaling)
    {
        InitialLearningRate = initialLearningRate;
        LearningRateScheduleType = learningRateSchedule;
        _defaultLossFunction = new MeanSquaredErrorLoss<T>();
    }

    #region IOnlineLearningModel Implementation

    /// <summary>
    /// Updates the model with a single training example.
    /// </summary>
    public abstract void PartialFit(Vector<T> x, T y);

    /// <summary>
    /// Updates the model with a mini-batch of training examples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Default implementation calls PartialFit for each sample.
    /// Subclasses can override for more efficient batch processing.
    /// </para>
    /// </remarks>
    public virtual void PartialFit(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException(
                $"Number of samples in X ({x.Rows}) must match length of y ({y.Length}).");
        }

        for (int i = 0; i < x.Rows; i++)
        {
            var xi = new Vector<T>(x.Columns);
            for (int j = 0; j < x.Columns; j++)
            {
                xi[j] = x[i, j];
            }
            PartialFit(xi, y[i]);
        }
    }

    /// <summary>
    /// Predicts the target value for a single sample.
    /// </summary>
    public abstract T PredictSingle(Vector<T> x);

    /// <summary>
    /// Gets the number of samples the model has seen.
    /// </summary>
    public long GetSampleCount() => SampleCount;

    /// <summary>
    /// Resets the model to its initial state.
    /// </summary>
    public virtual void Reset()
    {
        SampleCount = 0;
        IsInitialized = false;
        NumFeatures = 0;
    }

    /// <summary>
    /// Gets the current learning rate based on the schedule.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Learning rate typically decreases over time:
    /// - Early training: Large steps to quickly reach good region
    /// - Later training: Small steps to fine-tune without overshooting
    ///
    /// Common schedules:
    /// - Constant: Never changes (simplest)
    /// - Inverse Scaling: η = η₀ / (1 + α × t)
    /// - Exponential: η = η₀ × decay^t
    /// </para>
    /// </remarks>
    public virtual T GetLearningRate()
    {
        double effectiveRate = LearningRateScheduleType switch
        {
            LearningRateSchedule.Constant => InitialLearningRate,
            LearningRateSchedule.InverseScaling => InitialLearningRate / (1.0 + 0.0001 * SampleCount),
            LearningRateSchedule.Exponential => InitialLearningRate * Math.Pow(0.9999, SampleCount),
            LearningRateSchedule.StepDecay => InitialLearningRate * Math.Pow(0.5, SampleCount / 10000),
            _ => InitialLearningRate
        };

        return NumOps.FromDouble(Math.Max(effectiveRate, 1e-10));
    }

    #endregion

    #region IFullModel Implementation

    /// <summary>
    /// Gets the model type.
    /// </summary>
    public virtual ModelType GetModelType() => ModelType.None;

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            FeatureCount = NumFeatures,
            Complexity = NumFeatures,
            Description = $"{GetModelType()} online learning model",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "SampleCount", SampleCount },
                { "LearningRate", NumOps.ToDouble(GetLearningRate()) },
                { "IsInitialized", IsInitialized }
            }
        };
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    public virtual byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumFeatures", NumFeatures },
            { "SampleCount", SampleCount },
            { "IsInitialized", IsInitialized }
        };

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));

        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    public virtual void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata?.ModelData is null)
        {
            throw new InvalidOperationException("Deserialization failed: Invalid model data.");
        }

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(modelDataString);

        if (modelDataObj is null)
        {
            throw new InvalidOperationException("Deserialization failed: Invalid model data.");
        }

        NumFeatures = modelDataObj["NumFeatures"]?.ToObject<int>() ?? 0;
        SampleCount = modelDataObj["SampleCount"]?.ToObject<long>() ?? 0;
        IsInitialized = modelDataObj["IsInitialized"]?.ToObject<bool>() ?? false;
    }

    /// <summary>
    /// Standard model training - equivalent to PartialFit.
    /// </summary>
    public virtual void Train(Matrix<T> x, Vector<T> y)
    {
        PartialFit(x, y);
    }

    /// <summary>
    /// Standard prediction - returns predictions for all samples.
    /// </summary>
    public virtual Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            var xi = new Vector<T>(input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                xi[j] = input[i, j];
            }
            predictions[i] = PredictSingle(xi);
        }
        return predictions;
    }

    /// <summary>
    /// Gets all model parameters as a single vector.
    /// </summary>
    public abstract Vector<T> GetParameters();

    /// <summary>
    /// Creates a new instance of the model with specified parameters.
    /// </summary>
    public abstract IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters);

    /// <summary>
    /// Sets the parameters for this model.
    /// </summary>
    public abstract void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Creates a new instance of the same type.
    /// </summary>
    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance();

    /// <summary>
    /// Gets the indices of features that are actively used in the model.
    /// </summary>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        for (int i = 0; i < NumFeatures; i++)
        {
            yield return i;
        }
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Default: no-op
    }

    /// <summary>
    /// Determines whether a specific feature is used in the model.
    /// </summary>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= NumFeatures)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex));
        }
        return true;
    }

    /// <summary>
    /// Gets the feature importance scores.
    /// </summary>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();
        for (int i = 0; i < NumFeatures; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = NumOps.One;
        }
        return result;
    }

    /// <summary>
    /// Creates a deep copy of the model.
    /// </summary>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        byte[] serialized = Serialize();
        var copy = CreateNewInstance();
        copy.Deserialize(serialized);
        return copy;
    }

    /// <summary>
    /// Creates a clone of the model.
    /// </summary>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        return DeepCopy();
    }

    /// <summary>
    /// Computes gradients for the given input and target.
    /// </summary>
    public virtual Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return new Vector<T>(ParameterCount);
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Default: no-op (online models typically handle updates in PartialFit)
    }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    public virtual void SaveModel(string filePath)
    {
        byte[] serializedData = Serialize();
        File.WriteAllBytes(filePath, serializedData);
    }

    /// <summary>
    /// Loads the model from a file.
    /// </summary>
    public virtual void LoadModel(string filePath)
    {
        byte[] serializedData = File.ReadAllBytes(filePath);
        Deserialize(serializedData);
    }

    /// <summary>
    /// Saves the model's state to a stream.
    /// </summary>
    public virtual void SaveState(Stream stream)
    {
        byte[] serializedData = Serialize();
        stream.Write(serializedData, 0, serializedData.Length);
    }

    /// <summary>
    /// Loads the model's state from a stream.
    /// </summary>
    public virtual void LoadState(Stream stream)
    {
        using var memoryStream = new MemoryStream();
        stream.CopyTo(memoryStream);
        byte[] serializedData = memoryStream.ToArray();
        Deserialize(serializedData);
    }

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation is not supported for this online learning model.");
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Initializes the model for a given number of features.
    /// </summary>
    protected virtual void Initialize(int numFeatures)
    {
        NumFeatures = numFeatures;
        IsInitialized = true;
    }

    /// <summary>
    /// Ensures the model is initialized, initializing if needed.
    /// </summary>
    protected void EnsureInitialized(Vector<T> x)
    {
        if (!IsInitialized)
        {
            Initialize(x.Length);
        }
        else if (x.Length != NumFeatures)
        {
            throw new ArgumentException(
                $"Feature count mismatch: expected {NumFeatures}, got {x.Length}.");
        }
    }

    #endregion
}

/// <summary>
/// Learning rate schedule types for online learning.
/// </summary>
public enum LearningRateSchedule
{
    /// <summary>
    /// Constant learning rate throughout training.
    /// </summary>
    Constant,

    /// <summary>
    /// Learning rate decreases as 1/t (inverse time scaling).
    /// </summary>
    InverseScaling,

    /// <summary>
    /// Learning rate decreases exponentially.
    /// </summary>
    Exponential,

    /// <summary>
    /// Learning rate drops by half every N samples.
    /// </summary>
    StepDecay
}
