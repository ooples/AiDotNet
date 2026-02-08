using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Gpu;
using Newtonsoft.Json;

namespace AiDotNet.Classification.MultiLabel;

/// <summary>
/// Base class for multi-label classification models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides common functionality for multi-label
/// classifiers. Multi-label classification assigns multiple labels to each sample, unlike
/// traditional classification which assigns exactly one label.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class MultiLabelClassifierBase<T> : IMultiLabelClassifier<T>, IConfigurableModel<T>
{
    /// <summary>
    /// Gets the numeric operations provider for type T.
    /// </summary>
    protected INumericOperations<T> NumOps { get; }

    /// <summary>
    /// Gets the classifier options.
    /// </summary>
    protected ClassifierOptions<T> Options { get; }

    /// <inheritdoc/>
    public virtual ModelOptions GetOptions() => Options;

    /// <summary>
    /// Gets the regularization method used to prevent overfitting.
    /// </summary>
    protected IRegularization<T, Matrix<T>, Vector<T>> Regularization { get; }

    /// <summary>
    /// Gets or sets the number of possible labels.
    /// </summary>
    public int NumLabels { get; set; }

    /// <summary>
    /// Gets or sets the number of features.
    /// </summary>
    public int NumFeatures { get; set; }

    /// <summary>
    /// Gets or sets the number of classes (typically 2 for binary classification per label).
    /// </summary>
    public int NumClasses { get; set; }

    /// <summary>
    /// Gets or sets the classification task type.
    /// </summary>
    public ClassificationTaskType TaskType { get; set; }

    /// <summary>
    /// Gets or sets the label names if available.
    /// </summary>
    public string[]? LabelNames { get; set; }

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// The default loss function for this classifier.
    /// </summary>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Initializes a new instance of the MultiLabelClassifierBase class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Regularization method to prevent overfitting.</param>
    protected MultiLabelClassifierBase(
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new ClassifierOptions<T>();
        Regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
        TaskType = Options.TaskType;
        _defaultLossFunction = new BinaryCrossEntropyLoss<T>();
    }

    /// <summary>
    /// Trains the multi-label classifier.
    /// </summary>
    /// <param name="features">Feature matrix [n_samples, n_features].</param>
    /// <param name="labels">Binary label matrix [n_samples, n_labels].</param>
    public void Train(Matrix<T> features, Matrix<T> labels)
    {
        NumFeatures = features.Columns;
        NumLabels = labels.Columns;
        NumClasses = 2; // Binary classification per label
        TrainMultiLabelCore(features, labels);
    }

    /// <summary>
    /// Core training implementation to be overridden by derived classes.
    /// </summary>
    /// <param name="features">Feature matrix.</param>
    /// <param name="labels">Label matrix.</param>
    protected abstract void TrainMultiLabelCore(Matrix<T> features, Matrix<T> labels);

    /// <summary>
    /// Predicts binary label indicators for input samples.
    /// </summary>
    /// <param name="features">Feature matrix.</param>
    /// <returns>Binary label matrix.</returns>
    public virtual Matrix<T> Predict(Matrix<T> features)
    {
        if (NumLabels == 0)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var probabilities = PredictProbabilities(features);
        var predictions = new Matrix<T>(features.Rows, NumLabels);

        for (int i = 0; i < features.Rows; i++)
        {
            for (int l = 0; l < NumLabels; l++)
            {
                predictions[i, l] = NumOps.ToDouble(probabilities[i, l]) > 0.5 ? NumOps.One : NumOps.Zero;
            }
        }

        return predictions;
    }

    /// <summary>
    /// Predicts label probabilities for input samples.
    /// </summary>
    /// <param name="features">Feature matrix.</param>
    /// <returns>Probability matrix.</returns>
    public Matrix<T> PredictProbabilities(Matrix<T> features)
    {
        return PredictMultiLabelProbabilities(features);
    }

    /// <summary>
    /// Core probability prediction implementation to be overridden by derived classes.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Probability matrix.</returns>
    public abstract Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input);

    /// <inheritdoc />
    public abstract Vector<T> GetParameters();

    /// <inheritdoc />
    public abstract void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Gets the model type for this classifier.
    /// </summary>
    protected abstract ModelType GetModelType();

    /// <summary>
    /// Creates a new instance of this model type.
    /// </summary>
    protected abstract IFullModel<T, Matrix<T>, Matrix<T>> CreateNewInstance();

    /// <inheritdoc />
    public virtual IFullModel<T, Matrix<T>, Matrix<T>> WithParameters(Vector<T> parameters)
    {
        var clone = CreateNewInstance();
        clone.SetParameters(parameters);
        return clone;
    }

    /// <inheritdoc />
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var parameters = GetParameters();
        for (int i = 0; i < parameters.Length && i < gradients.Length; i++)
        {
            parameters[i] = NumOps.Subtract(parameters[i],
                NumOps.Multiply(learningRate, gradients[i]));
        }
        SetParameters(parameters);
    }

    /// <inheritdoc />
    public virtual int ParameterCount => GetParameters().Length;

    /// <inheritdoc />
    public virtual ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <inheritdoc />
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            FeatureCount = NumFeatures,
            Complexity = NumFeatures * NumLabels,
            Description = $"{GetModelType()} multi-label classifier with {NumFeatures} features and {NumLabels} labels",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumLabels", NumLabels },
                { "NumClasses", NumClasses },
                { "TaskType", TaskType.ToString() },
                { "LabelNames", LabelNames ?? Array.Empty<string>() }
            }
        };
    }

    /// <inheritdoc />
    public virtual byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumLabels", NumLabels },
            { "NumFeatures", NumFeatures },
            { "NumClasses", NumClasses },
            { "TaskType", (int)TaskType },
            { "LabelNames", LabelNames ?? Array.Empty<string>() },
            { "Parameters", GetParameters().ToArray().Select(NumOps.ToDouble).ToArray() },
            { "RegularizationOptions", Regularization.GetOptions() }
        };

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));

        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <inheritdoc />
    public virtual void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata is null || modelMetadata.ModelData is null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(modelDataString);

        if (modelDataObj is null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        NumLabels = modelDataObj["NumLabels"]?.ToObject<int>() ?? 0;
        NumFeatures = modelDataObj["NumFeatures"]?.ToObject<int>() ?? 0;
        NumClasses = modelDataObj["NumClasses"]?.ToObject<int>() ?? 2;
        TaskType = (ClassificationTaskType)(modelDataObj["TaskType"]?.ToObject<int>() ?? 0);
        LabelNames = modelDataObj["LabelNames"]?.ToObject<string[]>();

        var parametersAsDoubles = modelDataObj["Parameters"]?.ToObject<double[]>() ?? Array.Empty<double>();
        if (parametersAsDoubles.Length > 0)
        {
            var parameters = new Vector<T>(parametersAsDoubles.Length);
            for (int i = 0; i < parametersAsDoubles.Length; i++)
            {
                parameters[i] = NumOps.FromDouble(parametersAsDoubles[i]);
            }
            SetParameters(parameters);
        }
    }

    /// <inheritdoc />
    public virtual void SaveModel(string path)
    {
        byte[] serializedData = Serialize();
        System.IO.File.WriteAllBytes(path, serializedData);
    }

    /// <inheritdoc />
    public virtual void LoadModel(string path)
    {
        byte[] serializedData = System.IO.File.ReadAllBytes(path);
        Deserialize(serializedData);
    }

    /// <inheritdoc />
    public virtual void SaveState(System.IO.Stream stream)
    {
        byte[] serializedData = Serialize();
        stream.Write(serializedData, 0, serializedData.Length);
    }

    /// <inheritdoc />
    public virtual void LoadState(System.IO.Stream stream)
    {
        using var memoryStream = new System.IO.MemoryStream();
        stream.CopyTo(memoryStream);
        byte[] serializedData = memoryStream.ToArray();
        Deserialize(serializedData);
    }

    /// <inheritdoc />
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        for (int i = 0; i < NumFeatures; i++)
        {
            yield return i;
        }
    }

    /// <inheritdoc />
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Default implementation: no-op
    }

    /// <inheritdoc />
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= NumFeatures)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex),
                $"Feature index must be between 0 and {NumFeatures - 1}");
        }
        return true;
    }

    /// <inheritdoc />
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();
        for (int i = 0; i < NumFeatures; i++)
        {
            string featureName = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[featureName] = NumOps.One;
        }
        return result;
    }

    /// <inheritdoc />
    public virtual IFullModel<T, Matrix<T>, Matrix<T>> DeepCopy()
    {
        byte[] serialized = Serialize();
        var copy = CreateNewInstance();
        copy.Deserialize(serialized);
        return copy;
    }

    /// <inheritdoc />
    public virtual IFullModel<T, Matrix<T>, Matrix<T>> Clone()
    {
        return DeepCopy();
    }

    /// <inheritdoc />
    public virtual Vector<T> ComputeGradients(Matrix<T> input, Matrix<T> target, ILossFunction<T>? lossFunction = null)
    {
        var loss = lossFunction ?? _defaultLossFunction;
        var predictions = PredictProbabilities(input);
        var parameters = GetParameters();
        var gradients = new Vector<T>(parameters.Length);

        double totalGradient = 0;
        for (int i = 0; i < input.Rows; i++)
        {
            for (int l = 0; l < NumLabels; l++)
            {
                double pred = NumOps.ToDouble(predictions[i, l]);
                double actual = NumOps.ToDouble(target[i, l]);
                double deriv = (pred - actual) / (pred * (1 - pred) + 1e-15);
                totalGradient += deriv;
            }
        }

        double avgGradient = parameters.Length > 0 ? totalGradient / (input.Rows * NumLabels * parameters.Length) : 0;
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = NumOps.FromDouble(avgGradient);
        }

        return gradients;
    }

    #region IJitCompilable Implementation

    /// <inheritdoc />
    public virtual bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation is not supported for this multi-label classifier.");
    }

    #endregion

    /// <summary>
    /// Binary cross-entropy loss for multi-label classification.
    /// </summary>
    private class BinaryCrossEntropyLoss<TLoss> : ILossFunction<TLoss>
    {
        private static INumericOperations<TLoss> Ops => MathHelper.GetNumericOperations<TLoss>();

        public TLoss CalculateLoss(Vector<TLoss> predicted, Vector<TLoss> actual)
        {
            double loss = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                double p = Math.Max(1e-15, Math.Min(1 - 1e-15, Ops.ToDouble(predicted[i])));
                double y = Ops.ToDouble(actual[i]);
                loss -= y * Math.Log(p) + (1 - y) * Math.Log(1 - p);
            }
            return Ops.FromDouble(loss / Math.Max(1, predicted.Length));
        }

        public Vector<TLoss> CalculateDerivative(Vector<TLoss> predicted, Vector<TLoss> actual)
        {
            var derivative = new Vector<TLoss>(predicted.Length);
            for (int i = 0; i < predicted.Length; i++)
            {
                double p = Math.Max(1e-15, Math.Min(1 - 1e-15, Ops.ToDouble(predicted[i])));
                double y = Ops.ToDouble(actual[i]);
                derivative[i] = Ops.FromDouble((p - y) / (p * (1 - p) + 1e-15));
            }
            return derivative;
        }

        public (TLoss Loss, IGpuTensor<TLoss> Gradient) CalculateLossAndGradientGpu(IGpuTensor<TLoss> predicted, IGpuTensor<TLoss> actual)
        {
            var predictedCpu = predicted.ToTensor();
            var actualCpu = actual.ToTensor();
            var predictedVector = new Vector<TLoss>(predictedCpu.Data.ToArray());
            var actualVector = new Vector<TLoss>(actualCpu.Data.ToArray());

            var loss = CalculateLoss(predictedVector, actualVector);
            var gradientVector = CalculateDerivative(predictedVector, actualVector);
            var gradientTensor = new Tensor<TLoss>(predictedCpu.Shape, gradientVector);

            var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
            var backend = engine?.GetBackend() ?? throw new InvalidOperationException("GPU backend not available");
            var gradientGpu = new GpuTensor<TLoss>(backend, gradientTensor, GpuTensorRole.Gradient);

            return (loss, gradientGpu);
        }

        public string Name => "BinaryCrossEntropy";
    }
}
