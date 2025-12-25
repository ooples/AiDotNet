global using AiDotNet.Factories;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using Newtonsoft.Json;

namespace AiDotNet.Classification;

/// <summary>
/// Provides a base implementation for classification algorithms that predict categorical outcomes.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class implements common functionality for classification models, including
/// prediction, serialization/deserialization, and parameter management. Specific classification
/// algorithms should inherit from this class and implement the Train and Predict methods.
/// </para>
/// <para>
/// The class supports various options like class weighting to handle imbalanced datasets
/// and different classification task types (binary, multi-class, multi-label, ordinal).
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Classification is about predicting which category something belongs to.
/// This base class provides the foundation for different classification techniques, handling
/// common operations like making predictions and saving/loading models. Think of it as
/// a template that specific classification algorithms can customize while reusing the shared
/// functionality.
/// </para>
/// </remarks>
public abstract class ClassifierBase<T> : IClassifier<T>
{
    /// <summary>
    /// Gets the numeric operations for the specified type T.
    /// </summary>
    /// <value>
    /// An object that provides mathematical operations for type T.
    /// </value>
    protected INumericOperations<T> NumOps { get; private set; }

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the execution engine (CPU or GPU) for performing
    /// vectorized operations. The engine is determined by the global AiDotNetEngine configuration
    /// and allows automatic fallback from GPU to CPU when GPU is not available.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This gives access to either CPU or GPU processing for faster computations.
    /// The system automatically chooses the best available option and falls back to CPU
    /// if GPU acceleration is not available.
    /// </para>
    /// </remarks>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the classifier options.
    /// </summary>
    /// <value>
    /// Configuration options for the classifier model.
    /// </value>
    protected ClassifierOptions<T> Options { get; private set; }

    /// <summary>
    /// Gets the regularization method used to prevent overfitting.
    /// </summary>
    /// <value>
    /// An object that implements regularization for the classifier model.
    /// </value>
    protected IRegularization<T, Matrix<T>, Vector<T>> Regularization { get; private set; }

    /// <summary>
    /// Gets the default loss function for this classifier.
    /// </summary>
    /// <value>
    /// The loss function used for gradient computation.
    /// </value>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Gets or sets the number of classes in the classification problem.
    /// </summary>
    /// <value>
    /// The number of distinct classes learned during training.
    /// </value>
    public int NumClasses { get; protected set; }

    /// <summary>
    /// Gets or sets the type of classification task.
    /// </summary>
    /// <value>
    /// The classification task type (Binary, MultiClass, MultiLabel, or Ordinal).
    /// </value>
    public ClassificationTaskType TaskType { get; protected set; }

    /// <summary>
    /// Gets or sets the class labels learned during training.
    /// </summary>
    /// <value>
    /// A vector containing the unique class labels, or null if not yet trained.
    /// </value>
    public Vector<T>? ClassLabels { get; protected set; }

    /// <summary>
    /// Gets or sets the number of features expected by this classifier.
    /// </summary>
    /// <value>
    /// The number of input features the model was trained on.
    /// </value>
    public int NumFeatures { get; protected set; }

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    /// <value>
    /// An array of feature names. If not set, feature indices will be used as names.
    /// </value>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Gets the expected number of parameters for this model.
    /// </summary>
    /// <value>
    /// The total number of parameters in the model, used for serialization and gradient computation.
    /// </value>
    protected virtual int ExpectedParameterCount => NumFeatures * NumClasses;

    /// <summary>
    /// Initializes a new instance of the ClassifierBase class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the classifier model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <param name="lossFunction">Loss function for gradient computation. If null, defaults to Cross Entropy.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This constructor sets up the classification model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data. The loss function determines how
    /// prediction errors are measured during training.
    /// </para>
    /// </remarks>
    protected ClassifierBase(ClassifierOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null, ILossFunction<T>? lossFunction = null)
    {
        Regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new ClassifierOptions<T>();
        TaskType = Options.TaskType;
        _defaultLossFunction = lossFunction ?? new CrossEntropyLoss<T>();
    }

    /// <summary>
    /// Trains the classifier on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target class labels vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to train the classifier.
    /// The target vector should contain class indices (0, 1, 2, ...) for each sample.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Training is the process where the model learns from your data. Different classification algorithms
    /// implement this method differently, but they all aim to learn how to correctly predict the class
    /// labels based on the input features.
    /// </para>
    /// </remarks>
    public abstract void Train(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Predicts class labels for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted class indices for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates predictions for each sample in the input matrix.
    /// The returned vector contains class indices (0, 1, 2, ...).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// After training, this method is used to make predictions on new data. It returns
    /// the predicted class for each input sample as a numeric index.
    /// Use ClassLabels to map these indices back to the original label values if needed.
    /// </para>
    /// </remarks>
    public abstract Vector<T> Predict(Matrix<T> input);

    /// <summary>
    /// Infers the classification task type from the training labels.
    /// </summary>
    /// <param name="y">The target labels vector.</param>
    /// <returns>The inferred classification task type.</returns>
    /// <remarks>
    /// <para>
    /// This method examines the unique values in the target vector to determine
    /// whether this is a binary or multi-class classification problem.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// The model automatically figures out what kind of classification problem you have:
    /// - If there are exactly 2 unique values → Binary classification
    /// - If there are more than 2 unique values → Multi-class classification
    /// </para>
    /// </remarks>
    protected virtual ClassificationTaskType InferTaskType(Vector<T> y)
    {
        var uniqueClasses = new HashSet<double>();
        for (int i = 0; i < y.Length; i++)
        {
            uniqueClasses.Add(NumOps.ToDouble(y[i]));
        }

        return uniqueClasses.Count == 2 ? ClassificationTaskType.Binary : ClassificationTaskType.MultiClass;
    }

    /// <summary>
    /// Extracts unique class labels from the training data.
    /// </summary>
    /// <param name="y">The target labels vector.</param>
    /// <returns>A sorted vector of unique class labels.</returns>
    protected virtual Vector<T> ExtractClassLabels(Vector<T> y)
    {
        var uniqueClasses = new HashSet<double>();
        for (int i = 0; i < y.Length; i++)
        {
            uniqueClasses.Add(NumOps.ToDouble(y[i]));
        }

        var sortedClasses = uniqueClasses.OrderBy(x => x).ToArray();
        var labels = new Vector<T>(sortedClasses.Length);
        for (int i = 0; i < sortedClasses.Length; i++)
        {
            labels[i] = NumOps.FromDouble(sortedClasses[i]);
        }

        return labels;
    }

    /// <summary>
    /// Computes class weights for handling imbalanced datasets.
    /// </summary>
    /// <param name="y">The target labels vector.</param>
    /// <returns>An array of weights for each class.</returns>
    protected virtual double[] ComputeClassWeights(Vector<T> y)
    {
        if (Options.ClassWeights != null && Options.ClassWeights.Length == NumClasses)
        {
            return Options.ClassWeights;
        }

        if (!Options.UseClassWeights)
        {
            var uniformWeights = new double[NumClasses];
            for (int i = 0; i < NumClasses; i++)
            {
                uniformWeights[i] = 1.0;
            }
            return uniformWeights;
        }

        // Count samples per class
        var classCounts = new int[NumClasses];
        for (int i = 0; i < y.Length; i++)
        {
            int classIdx = (int)NumOps.ToDouble(y[i]);
            if (classIdx >= 0 && classIdx < NumClasses)
            {
                classCounts[classIdx]++;
            }
        }

        // Compute weights: n_samples / (n_classes * n_samples_per_class)
        var weights = new double[NumClasses];
        double nSamples = y.Length;
        for (int i = 0; i < NumClasses; i++)
        {
            if (classCounts[i] > 0)
            {
                weights[i] = nSamples / (NumClasses * classCounts[i]);
            }
            else
            {
                weights[i] = 1.0;
            }
        }

        return weights;
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, feature count, complexity,
    /// description, and additional information specific to classification.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Model metadata provides information about the model itself, rather than the predictions it makes.
    /// This includes details about the model's structure (like how many features it uses) and characteristics
    /// (like how many classes it can predict). This information can be useful for understanding and
    /// comparing different models.
    /// </para>
    /// </remarks>
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            FeatureCount = NumFeatures,
            Complexity = NumFeatures * NumClasses,
            Description = $"{GetModelType()} classifier with {NumFeatures} features and {NumClasses} classes",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumClasses", NumClasses },
                { "TaskType", TaskType.ToString() },
                { "ClassLabels", ClassLabels?.ToArray() ?? Array.Empty<T>() }
            }
        };
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to specify the model type.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method simply returns an identifier that indicates what type of classifier this is
    /// (e.g., Naive Bayes, Random Forest). It's used internally by the library to keep track
    /// of different types of models.
    /// </para>
    /// </remarks>
    protected abstract ModelType GetModelType();

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters to a JSON format and then converts it to a byte array.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it.
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumClasses", NumClasses },
            { "NumFeatures", NumFeatures },
            { "TaskType", (int)TaskType },
            { "ClassLabels", ClassLabels?.ToArray() ?? Array.Empty<T>() },
            { "RegularizationOptions", Regularization.GetOptions() }
        };

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));

        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it.
    /// </para>
    /// </remarks>
    public virtual void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata == null || modelMetadata.ModelData == null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(modelDataString);

        if (modelDataObj == null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        var numClassesToken = modelDataObj["NumClasses"];
        var numFeaturesToken = modelDataObj["NumFeatures"];
        var taskTypeToken = modelDataObj["TaskType"];
        var classLabelsToken = modelDataObj["ClassLabels"];

        if (numClassesToken == null || numFeaturesToken == null || taskTypeToken == null)
        {
            throw new InvalidOperationException("Deserialization failed: Missing required classification parameters.");
        }

        NumClasses = numClassesToken.ToObject<int>();
        NumFeatures = numFeaturesToken.ToObject<int>();
        TaskType = (ClassificationTaskType)taskTypeToken.ToObject<int>();

        if (classLabelsToken != null)
        {
            var classLabelsAsDoubles = classLabelsToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (classLabelsAsDoubles.Length > 0)
            {
                ClassLabels = new Vector<T>(classLabelsAsDoubles.Length);
                for (int i = 0; i < classLabelsAsDoubles.Length; i++)
                {
                    ClassLabels[i] = NumOps.FromDouble(classLabelsAsDoubles[i]);
                }
            }
        }

        var regularizationOptionsToken = modelDataObj["RegularizationOptions"];
        if (regularizationOptionsToken != null)
        {
            var regularizationOptionsJson = JsonConvert.SerializeObject(regularizationOptionsToken);
            var regularizationOptions = JsonConvert.DeserializeObject<RegularizationOptions>(regularizationOptionsJson);
            if (regularizationOptions != null)
            {
                Regularization = RegularizationFactory.CreateRegularization<T, Matrix<T>, Vector<T>>(regularizationOptions);
            }
        }
    }

    /// <summary>
    /// Gets all model parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all model parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns a vector containing all model parameters for use with
    /// optimization algorithms or model comparison.
    /// </para>
    /// <para><b>For Beginners:</b> This method packages all the model's parameters into a single collection.
    /// This is useful for optimization algorithms that need to work with all parameters at once.
    /// </para>
    /// </remarks>
    public abstract Vector<T> GetParameters();

    /// <summary>
    /// Creates a new instance of the model with specified parameters.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters.</param>
    /// <returns>A new model instance with the specified parameters.</returns>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has an incorrect length.</exception>
    public abstract IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters);

    /// <summary>
    /// Sets the parameters for this model.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has an incorrect length.</exception>
    public abstract void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Gets the indices of features that are actively used in the model.
    /// </summary>
    /// <returns>An enumerable collection of indices for features that contribute to predictions.</returns>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        // Default implementation: all features are active
        for (int i = 0; i < NumFeatures; i++)
        {
            yield return i;
        }
    }

    /// <summary>
    /// Determines whether a specific feature is used in the model.
    /// </summary>
    /// <param name="featureIndex">The zero-based index of the feature to check.</param>
    /// <returns>True if the feature contributes to predictions; otherwise, false.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the feature index is outside the valid range.</exception>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= NumFeatures)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex),
                $"Feature index must be between 0 and {NumFeatures - 1}");
        }

        // Default implementation: all features are used
        return true;
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to activate.</param>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Default implementation: no-op (subclasses that support feature selection should override)
    }

    /// <summary>
    /// Gets the feature importance scores as a dictionary.
    /// </summary>
    /// <returns>A dictionary mapping feature names to their importance scores.</returns>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();

        for (int i = 0; i < NumFeatures; i++)
        {
            string featureName = FeatureNames != null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[featureName] = NumOps.One; // Default: equal importance
        }

        return result;
    }

    /// <summary>
    /// Creates a deep copy of the classifier model.
    /// </summary>
    /// <returns>A new instance of the model with the same parameters and options.</returns>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        byte[] serialized = Serialize();
        var copy = CreateNewInstance();
        copy.Deserialize(serialized);
        return copy;
    }

    /// <summary>
    /// Creates a new instance of the same type as this classifier.
    /// </summary>
    /// <returns>A new instance of the same classifier type.</returns>
    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance();

    /// <summary>
    /// Creates a clone of the classifier model.
    /// </summary>
    /// <returns>A new instance of the model with the same parameters and options.</returns>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        return DeepCopy();
    }

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    public virtual int ParameterCount => ExpectedParameterCount;

    /// <inheritdoc/>
    public virtual ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <inheritdoc/>
    public abstract Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null);

    /// <inheritdoc/>
    public abstract void ApplyGradients(Vector<T> gradients, T learningRate);

    /// <summary>
    /// Saves the classifier model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model should be saved.</param>
    public virtual void SaveModel(string filePath)
    {
        byte[] serializedData = Serialize();
        File.WriteAllBytes(filePath, serializedData);
    }

    /// <summary>
    /// Loads a classifier model from a file.
    /// </summary>
    /// <param name="filePath">The path to the file containing the saved model.</param>
    public virtual void LoadModel(string filePath)
    {
        byte[] serializedData = File.ReadAllBytes(filePath);
        Deserialize(serializedData);
    }

    /// <summary>
    /// Saves the model's current state to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the model state to.</param>
    public virtual void SaveState(Stream stream)
    {
        byte[] serializedData = Serialize();
        stream.Write(serializedData, 0, serializedData.Length);
    }

    /// <summary>
    /// Loads the model's state from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the model state from.</param>
    public virtual void LoadState(Stream stream)
    {
        using var memoryStream = new MemoryStream();
        stream.CopyTo(memoryStream);
        byte[] serializedData = memoryStream.ToArray();
        Deserialize(serializedData);
    }

    #region IJitCompilable Implementation

    /// <inheritdoc/>
    public virtual bool SupportsJitCompilation => false;

    /// <inheritdoc/>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation is not supported for this classifier. Override this method in derived classes to enable JIT support.");
    }

    #endregion
}
