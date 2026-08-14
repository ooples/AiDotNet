using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using Newtonsoft.Json;
using AiDotNet.Helpers;

using AiDotNet.Models.Parameters;
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
public abstract class MultiLabelClassifierBase<T> : IMultiLabelClassifier<T>, IConfigurableModel<T>, IModelShape,
    IParameterManifestProvider
{
    // --- declared state (ModelStateRegistry) ---
    // Identical in every model base because these bases are siblings over the same interfaces rather
    // than one hierarchy; the logic itself lives once in ModelStateRegistry/ModelStateEnvelope.

    /// <summary>State that is not a parameter vector, declared once and persisted by this base.</summary>
    private readonly AiDotNet.Models.ModelStateRegistry<T> _declaredState = new();
    private bool _declaredStateRegistered;

    /// <summary>
    /// Declare state here that the parameter vector does not carry -- a retained training set,
    /// fitted knots, kernel centres, an ensemble's children. Both halves of the payload are driven
    /// by the declaration, so they cannot drift.
    /// </summary>
    /// <param name="state">The registry to declare into.</param>
    protected virtual void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
    }
    /// <summary>Generated state declarations for fields declared across this model's hierarchy.</summary>
    /// <param name="state">The registry to declare into.</param>
    /// <remarks>
    /// Emitted by ModelStateGenerator into the partial model, so a model author declares nothing. The
    /// hand-written <c>RegisterState</c> beside it exists only for state the classifier genuinely
    /// cannot place; anything it CAN place belongs here, where it cannot be forgotten.
    /// </remarks>
    protected virtual void RegisterGeneratedState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
    }

    /// <summary>The declared state, registered once and lazily so it runs after the constructor.</summary>
    protected AiDotNet.Models.ModelStateRegistry<T> DeclaredState
    {
        get
        {
            if (!_declaredStateRegistered)
            {
                _declaredStateRegistered = true;
                RegisterGeneratedState(_declaredState);
                RegisterState(_declaredState);
            }
            return _declaredState;
        }
    }
    /// <summary>
    /// Gets the hardware-accelerated computation engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

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
        ThrowIfDisposed();
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
        ThrowIfDisposed();
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
                predictions[i, l] = NumOps.GreaterThan(probabilities[i, l], NumOps.FromDouble(0.5)) ? NumOps.One : NumOps.Zero;
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

    /// <summary>
    /// The components the parameters of this model live in. Empty until the model registers
    /// some, in which case the surfaces below fall back to what they always did.
    /// </summary>
    private readonly ParameterComponentRegistry<T> _parameterRegistry = new();
    private bool _componentsRegistered;

    /// <summary>
    /// Declares a component whose parameters belong to the surface of this model.
    /// Registration
    /// order is serialization order, so keep it stable.
    /// </summary>
    protected void RegisterParameterComponent(
        IParameterSource<T>? component,
        [System.Runtime.CompilerServices.CallerArgumentExpression(nameof(component))] string? componentExpression = null,
        [System.Runtime.CompilerServices.CallerMemberName] string? memberName = null)
        => _parameterRegistry.RegisterLegacy(GetType().FullName ?? GetType().Name,
            memberName, componentExpression, component);

    protected void RegisterParameterComponent(string stableId, IParameterSource<T>? component,
        ParameterSlotRole role = ParameterSlotRole.Trainable)
        => _parameterRegistry.Register(stableId, component, role);

    /// <summary>
    /// Declare the trainable components of this model here with
    /// <see cref="RegisterParameterComponent"/>. Called once, lazily, so it runs after the
    /// constructor has built them.
    /// </summary>
    protected virtual void RegisterComponents()
    {
    }

    protected virtual void RegisterGeneratedParameterComponents(ParameterComponentRegistry<T> registry)
    {
    }

    /// <summary>
    /// Runs after <see cref="SetParameters"/> has distributed values into the components.
    /// </summary>
    protected virtual void OnParametersRestored()
    {
    }

    private ParameterComponentRegistry<T> Registry
    {
        get
        {
            if (!_componentsRegistered)
            {
                RegisterGeneratedParameterComponents(_parameterRegistry);
                RegisterComponents();
                _componentsRegistered = true;
            }
            return _parameterRegistry;
        }
    }

    public ParameterLayoutSnapshot ParameterLayout => Registry.ParameterLayout;

    /// <inheritdoc/>
    /// <remarks>
    /// Virtual rather than abstract: a model that registers its components inherits all
    /// three surfaces and writes no parameter plumbing. It was abstract, which FORCED every
    /// descendant to hand-write the triple -- the same defect ModelBase and LayerBase had.
    /// </remarks>
    public virtual Vector<T> GetParameters()
        => Registry.HasComponents ? Registry.GetParameters() : new Vector<T>(0);

    /// <inheritdoc/>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (!Registry.HasComponents) return;
        Registry.SetParameters(parameters);
        OnParametersRestored();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Folds the same enumeration the vector does once components are registered. The
    /// previous expression is kept for models not yet converted -- and it is exactly why the
    /// two could disagree: it described the MODEL, not the vector. Measured on
    /// CausalForest: 5 against a 6-element vector after any restore.
    /// </remarks>
    public virtual long ParameterCount
        => Registry.HasComponents ? Registry.ParameterCount : GetParameters().Length;
    /// <summary>
    /// Gets the model type for this classifier.
    /// </summary>

    /// <summary>
    /// Creates a new instance of this model type.
    /// </summary>
    /// <remarks>
    /// <para>
    /// No longer abstract. Every concrete model used to be forced to write this, and 1147 of them
    /// did -- each one a hand-copied list of constructor arguments that a new option could fall out
    /// of without anything failing. The clone plan records that constructor at compile time instead,
    /// so the base can rebuild the type and a model only overrides this when the generator says it
    /// cannot: a constructor parameter with nothing holding its value, which the build reports by
    /// name rather than leaving to be discovered by a clone that comes back subtly different.
    /// </para>
    /// </remarks>
    protected virtual IFullModel<T, Matrix<T>, Matrix<T>> CreateNewInstance()
        => (IFullModel<T, Matrix<T>, Matrix<T>>)AiDotNet.Models.CloneEngine.CopyConfiguration(this);

    /// <inheritdoc />
    public virtual IFullModel<T, Matrix<T>, Matrix<T>> WithParameters(Vector<T> parameters)
    {
        var clone = CreateNewInstance();
        ((IParameterizable<T, Matrix<T>, Matrix<T>>)clone).SetParameters(parameters);
        return clone;
    }

    /// <inheritdoc />
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var parameters = GetParameters();
        var updated = (Vector<T>)Engine.Subtract(parameters, Engine.Multiply(gradients, learningRate));
        SetParameters(updated);
    }

    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;

    /// <inheritdoc />
    public virtual ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <inheritdoc />
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = NumFeatures,
            Complexity = NumFeatures * NumLabels,
            Description = $"{GetType().Name} multi-label classifier with {NumFeatures} features and {NumLabels} labels",
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
        ThrowIfDisposed();
        ModelPersistenceGuard.EnforceBeforeSerialize();
        return AiDotNet.Models.ModelStateEnvelope.Append(DeclaredState, SerializeInternalUnchecked());
    }

    /// <summary>
    /// Internal, non-virtual, no-guard serialization used by trusted framework
    /// call sites such as <see cref="DeepCopy"/>. Subclasses cannot override
    /// this method, so a subclass override of <see cref="Serialize"/> cannot
    /// intercept the clone path.
    /// </summary>
    private byte[] SerializeInternalUnchecked()
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
        // Strips and applies any declared-state trailer, so the body below reads the payload
        // exactly as it did before this existed.
        modelData = AiDotNet.Models.ModelStateEnvelope.Extract(DeclaredState, modelData);
        ThrowIfDisposed();
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        DeserializeInternalUnchecked(modelData);
    }

    /// <summary>
    /// Internal, non-virtual, no-guard deserialization used by trusted framework
    /// call sites such as <see cref="DeepCopy"/>. Subclasses cannot override
    /// this method, so a subclass override of <see cref="Deserialize"/> cannot
    /// intercept the clone path.
    /// </summary>
    private void DeserializeInternalUnchecked(byte[] modelData)
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
    public virtual int[] GetInputShape()
    {
        return new[] { NumFeatures };
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        return new[] { NumLabels };
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    public virtual void SaveModel(string path)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(path));
        }

        var fullPath = System.IO.Path.GetFullPath(path);
        var directory = System.IO.Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory) && !System.IO.Directory.Exists(directory))
        {
            System.IO.Directory.CreateDirectory(directory);
        }

        byte[] serializedData = Serialize();
        byte[] envelopedData = ModelFileHeader.WrapWithHeader(
            serializedData, this, GetInputShape(), GetOutputShape(), SerializationFormat.Json,
            GetDynamicShapeInfo());
        System.IO.File.WriteAllBytes(fullPath, envelopedData);
    }

    /// <inheritdoc />
    public virtual void LoadModel(string path)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(path));
        }

        var fullPath = System.IO.Path.GetFullPath(path);
        if (!System.IO.File.Exists(fullPath))
        {
            throw new System.IO.FileNotFoundException($"Model file not found: {fullPath}", fullPath);
        }

        byte[] serializedData = System.IO.File.ReadAllBytes(fullPath);

        // Extract payload from AIMF envelope if present; use raw bytes for legacy files
        if (ModelFileHeader.HasHeader(serializedData))
        {
            serializedData = ModelFileHeader.ExtractPayload(serializedData);
        }

        Deserialize(serializedData);
    }

    /// <inheritdoc />
    public virtual void SaveState(System.IO.Stream stream)
    {
        ThrowIfDisposed();
        byte[] serializedData = Serialize();
        stream.Write(serializedData, 0, serializedData.Length);
    }

    /// <inheritdoc />
    public virtual void LoadState(System.IO.Stream stream)
    {
        ThrowIfDisposed();
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
        // Default: no feature selection support. Subclasses that support
        // feature selection should override this method.
        throw new NotSupportedException(
            $"{GetType().Name} does not support feature selection. " +
            "Override SetActiveFeatureIndices to implement this capability.");
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
        // In-memory clone, not a user save/load — wrap in InternalOperation
        // so the persistence guard does not treat this as a billable op, AND
        // route through the private non-virtual SerializeInternalUnchecked /
        // DeserializeInternalUnchecked helpers so a subclass override of the
        // public virtual Serialize / Deserialize methods cannot intercept the
        // clone path (closes the subclass-override bypass surface).
        using (ModelPersistenceGuard.InternalOperation())
        {
            byte[] serialized = SerializeInternalUnchecked();
            var copy = CreateNewInstance();
            if (copy is MultiLabelClassifierBase<T> copyBase)
            {
                copyBase.DeserializeInternalUnchecked(serialized);
            }
            else
            {
                copy.Deserialize(serialized);
            }
            return copy;
        }
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

    // --- IDisposable (issue #1136 plan part 3) ---

    private bool _disposed;

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
        System.GC.SuppressFinalize(this);
    }

    /// <summary>Releases resources held by this multi-label classifier. Override + call base for layer/tensor cleanup.</summary>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
    }

    /// <summary>
    /// Throws <see cref="ObjectDisposedException"/> if <see cref="Dispose"/> has already
    /// been called. Subclasses must call this from any public entry point that touches
    /// model state.
    /// </summary>
    protected void ThrowIfDisposed()
    {
        if (_disposed) throw new System.ObjectDisposedException(GetType().FullName);
    }
}
