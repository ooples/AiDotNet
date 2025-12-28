using AiDotNet.Classification;
using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Neighbors;

/// <summary>
/// K-Nearest Neighbors classifier that predicts based on the majority class of nearest neighbors.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning algorithm.
/// It stores all training data and classifies new samples by finding the k closest
/// training samples and voting on their class labels.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// KNN is one of the simplest machine learning algorithms. Think of it as "you are the
/// company you keep." To classify something new:
///
/// 1. Find the k most similar training examples
/// 2. Look at their classes
/// 3. Return the most common class
///
/// Example: Classifying a fruit by its weight and color:
/// - New fruit: 150g, red
/// - 3 nearest neighbors: Apple (160g, red), Apple (145g, red), Cherry (10g, red)
/// - Wait! Cherry is much smaller, so it's not really "near"
/// - Actual 3 nearest: Apple, Apple, Orange -> Predicted: Apple
///
/// This is why feature scaling is important for KNN!
/// </para>
/// </remarks>
public class KNeighborsClassifier<T> : ProbabilisticClassifierBase<T>
{
    /// <summary>
    /// Gets the KNN specific options.
    /// </summary>
    protected new KNeighborsOptions<T> Options => (KNeighborsOptions<T>)base.Options;

    /// <summary>
    /// Stored training features.
    /// </summary>
    private Matrix<T>? _xTrain;

    /// <summary>
    /// Stored training labels.
    /// </summary>
    private Vector<T>? _yTrain;

    /// <summary>
    /// Initializes a new instance of the KNeighborsClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the KNN classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public KNeighborsClassifier(KNeighborsOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new KNeighborsOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.KNeighborsClassifier;

    /// <summary>
    /// Trains the KNN classifier by storing the training data.
    /// </summary>
    /// <remarks>
    /// KNN is a lazy learner - it doesn't actually build a model during training.
    /// It simply stores the training data for use during prediction.
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        if (Options.NNeighbors > x.Rows)
        {
            throw new ArgumentException($"n_neighbors ({Options.NNeighbors}) cannot be greater than number of training samples ({x.Rows}).");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Store training data
        _xTrain = new Matrix<T>(x.Rows, x.Columns);
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                _xTrain[i, j] = x[i, j];
            }
        }

        _yTrain = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            _yTrain[i] = y[i];
        }
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_xTrain == null || _yTrain == null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        var probabilities = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            var sample = new Vector<T>(input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                sample[j] = input[i, j];
            }

            var probs = PredictSampleProbabilities(sample);
            for (int c = 0; c < NumClasses; c++)
            {
                probabilities[i, c] = probs[c];
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Predicts class probabilities for a single sample.
    /// </summary>
    private Vector<T> PredictSampleProbabilities(Vector<T> sample)
    {
        // Compute distances to all training samples
        var distances = new List<(T distance, int index)>();
        for (int i = 0; i < _xTrain!.Rows; i++)
        {
            var trainSample = new Vector<T>(_xTrain.Columns);
            for (int j = 0; j < _xTrain.Columns; j++)
            {
                trainSample[j] = _xTrain[i, j];
            }

            T distance = ComputeDistance(sample, trainSample);
            distances.Add((distance, i));
        }

        // Sort by distance and get k nearest
        var sortedDistances = distances
            .OrderBy(d => NumOps.ToDouble(d.distance))
            .Take(Options.NNeighbors)
            .ToList();

        // Compute weighted votes
        var classVotes = new T[NumClasses];
        for (int c = 0; c < NumClasses; c++)
        {
            classVotes[c] = NumOps.Zero;
        }

        foreach (var (distance, index) in sortedDistances)
        {
            int classIdx = GetClassIndex(_yTrain![index]);
            T weight = GetWeight(distance);
            classVotes[classIdx] = NumOps.Add(classVotes[classIdx], weight);
        }

        // Normalize to probabilities
        T totalWeight = NumOps.Zero;
        for (int c = 0; c < NumClasses; c++)
        {
            totalWeight = NumOps.Add(totalWeight, classVotes[c]);
        }

        var probabilities = new Vector<T>(NumClasses);
        for (int c = 0; c < NumClasses; c++)
        {
            if (NumOps.Compare(totalWeight, NumOps.Zero) > 0)
            {
                probabilities[c] = NumOps.Divide(classVotes[c], totalWeight);
            }
            else
            {
                probabilities[c] = NumOps.FromDouble(1.0 / NumClasses);
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Computes the distance between two samples based on the configured metric.
    /// </summary>
    private T ComputeDistance(Vector<T> a, Vector<T> b)
    {
        return Options.Metric switch
        {
            DistanceMetric.Euclidean => ComputeEuclideanDistance(a, b),
            DistanceMetric.Manhattan => ComputeManhattanDistance(a, b),
            DistanceMetric.Minkowski => ComputeMinkowskiDistance(a, b, Options.P),
            DistanceMetric.Chebyshev => ComputeChebyshevDistance(a, b),
            DistanceMetric.Cosine => ComputeCosineDistance(a, b),
            _ => ComputeEuclideanDistance(a, b)
        };
    }

    /// <summary>
    /// Computes Euclidean (L2) distance.
    /// </summary>
    private T ComputeEuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sumSquared = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(diff, diff));
        }
        return NumOps.Sqrt(sumSquared);
    }

    /// <summary>
    /// Computes Manhattan (L1) distance.
    /// </summary>
    private T ComputeManhattanDistance(Vector<T> a, Vector<T> b)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Abs(diff));
        }
        return sum;
    }

    /// <summary>
    /// Computes Minkowski distance with parameter p.
    /// </summary>
    private T ComputeMinkowskiDistance(Vector<T> a, Vector<T> b, double p)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Abs(NumOps.Subtract(a[i], b[i]));
            sum = NumOps.Add(sum, NumOps.Power(diff, NumOps.FromDouble(p)));
        }
        return NumOps.Power(sum, NumOps.FromDouble(1.0 / p));
    }

    /// <summary>
    /// Computes Chebyshev (L-infinity) distance.
    /// </summary>
    private T ComputeChebyshevDistance(Vector<T> a, Vector<T> b)
    {
        T maxDiff = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Abs(NumOps.Subtract(a[i], b[i]));
            if (NumOps.Compare(diff, maxDiff) > 0)
            {
                maxDiff = diff;
            }
        }
        return maxDiff;
    }

    /// <summary>
    /// Computes cosine distance (1 - cosine similarity).
    /// </summary>
    private T ComputeCosineDistance(Vector<T> a, Vector<T> b)
    {
        T dotProduct = NumOps.Zero;
        T normA = NumOps.Zero;
        T normB = NumOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(a[i], b[i]));
            normA = NumOps.Add(normA, NumOps.Multiply(a[i], a[i]));
            normB = NumOps.Add(normB, NumOps.Multiply(b[i], b[i]));
        }

        normA = NumOps.Sqrt(normA);
        normB = NumOps.Sqrt(normB);

        T epsilon = NumOps.FromDouble(1e-10);
        if (NumOps.Compare(normA, epsilon) < 0 || NumOps.Compare(normB, epsilon) < 0)
        {
            return NumOps.One; // Maximum distance if one vector is zero
        }

        T cosineSimilarity = NumOps.Divide(dotProduct, NumOps.Multiply(normA, normB));
        return NumOps.Subtract(NumOps.One, cosineSimilarity);
    }

    /// <summary>
    /// Gets the weight for a neighbor based on the weighting scheme.
    /// </summary>
    private T GetWeight(T distance)
    {
        return Options.Weights switch
        {
            WeightingScheme.Uniform => NumOps.One,
            WeightingScheme.Distance => GetDistanceWeight(distance),
            _ => NumOps.One
        };
    }

    /// <summary>
    /// Computes weight based on distance (1/distance).
    /// </summary>
    private T GetDistanceWeight(T distance)
    {
        T epsilon = NumOps.FromDouble(1e-10);
        if (NumOps.Compare(distance, epsilon) < 0)
        {
            // Very close to training point, return large weight
            return NumOps.FromDouble(1e10);
        }
        return NumOps.Divide(NumOps.One, distance);
    }

    /// <summary>
    /// Gets the class index for a label.
    /// </summary>
    private int GetClassIndex(T label)
    {
        if (ClassLabels == null)
        {
            throw new InvalidOperationException("Model must be trained before getting class index.");
        }

        double labelValue = NumOps.ToDouble(label);
        for (int i = 0; i < ClassLabels.Length; i++)
        {
            if (Math.Abs(NumOps.ToDouble(ClassLabels[i]) - labelValue) < 1e-10)
            {
                return i;
            }
        }

        throw new ArgumentException($"Label {label} not found in class labels.");
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new KNeighborsClassifier<T>(new KNeighborsOptions<T>
        {
            NNeighbors = Options.NNeighbors,
            Metric = Options.Metric,
            Weights = Options.Weights,
            P = Options.P,
            Algorithm = Options.Algorithm,
            LeafSize = Options.LeafSize
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new KNeighborsClassifier<T>(new KNeighborsOptions<T>
        {
            NNeighbors = Options.NNeighbors,
            Metric = Options.Metric,
            Weights = Options.Weights,
            P = Options.P,
            Algorithm = Options.Algorithm,
            LeafSize = Options.LeafSize
        });

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;

        if (ClassLabels != null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (_xTrain != null)
        {
            clone._xTrain = new Matrix<T>(_xTrain.Rows, _xTrain.Columns);
            for (int i = 0; i < _xTrain.Rows; i++)
            {
                for (int j = 0; j < _xTrain.Columns; j++)
                {
                    clone._xTrain[i, j] = _xTrain[i, j];
                }
            }
        }

        if (_yTrain != null)
        {
            clone._yTrain = new Vector<T>(_yTrain.Length);
            for (int i = 0; i < _yTrain.Length; i++)
            {
                clone._yTrain[i] = _yTrain[i];
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // KNN is a lazy learner - it doesn't have traditional model parameters
        // Return an empty vector for compatibility
        return new Vector<T>(0);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = (KNeighborsClassifier<T>)Clone();
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // KNN is a lazy learner - it doesn't have traditional model parameters
        // This is a no-op for compatibility
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // KNN doesn't use gradient-based optimization
        // Return zero gradients for compatibility
        return new Vector<T>(NumFeatures);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // KNN doesn't use gradient-based optimization
        // This is a no-op for compatibility
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NNeighbors"] = Options.NNeighbors;
        metadata.AdditionalInfo["Metric"] = Options.Metric.ToString();
        metadata.AdditionalInfo["Weights"] = Options.Weights.ToString();
        metadata.AdditionalInfo["TrainingSamples"] = _xTrain?.Rows ?? 0;
        return metadata;
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumClasses", NumClasses },
            { "NumFeatures", NumFeatures },
            { "TaskType", (int)TaskType },
            { "ClassLabels", ClassLabels?.ToArray() ?? Array.Empty<T>() },
            { "RegularizationOptions", Regularization.GetOptions() },
            // KNN-specific options
            { "NNeighbors", Options.NNeighbors },
            { "Metric", (int)Options.Metric },
            { "Weights", (int)Options.Weights },
            { "P", Options.P },
            { "Algorithm", (int)Options.Algorithm },
            { "LeafSize", Options.LeafSize }
        };

        // Serialize _xTrain matrix
        if (_xTrain is not null)
        {
            var xTrainArray = new double[_xTrain.Rows * _xTrain.Columns];
            int idx = 0;
            for (int i = 0; i < _xTrain.Rows; i++)
            {
                for (int j = 0; j < _xTrain.Columns; j++)
                {
                    xTrainArray[idx++] = NumOps.ToDouble(_xTrain[i, j]);
                }
            }
            modelData["XTrain"] = xTrainArray;
            modelData["XTrainRows"] = _xTrain.Rows;
            modelData["XTrainCols"] = _xTrain.Columns;
        }

        // Serialize _yTrain vector
        if (_yTrain is not null)
        {
            var yTrainArray = new double[_yTrain.Length];
            for (int i = 0; i < _yTrain.Length; i++)
            {
                yTrainArray[i] = NumOps.ToDouble(_yTrain[i]);
            }
            modelData["YTrain"] = yTrainArray;
        }

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = System.Text.Encoding.UTF8.GetBytes(
            Newtonsoft.Json.JsonConvert.SerializeObject(modelData));

        return System.Text.Encoding.UTF8.GetBytes(
            Newtonsoft.Json.JsonConvert.SerializeObject(modelMetadata));
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        var jsonString = System.Text.Encoding.UTF8.GetString(modelData);
        var modelMetadata = Newtonsoft.Json.JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata == null || modelMetadata.ModelData == null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        var modelDataString = System.Text.Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = Newtonsoft.Json.JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(modelDataString);

        if (modelDataObj == null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        // Deserialize base properties
        NumClasses = modelDataObj["NumClasses"]?.ToObject<int>() ?? 0;
        NumFeatures = modelDataObj["NumFeatures"]?.ToObject<int>() ?? 0;
        TaskType = (ClassificationTaskType)(modelDataObj["TaskType"]?.ToObject<int>() ?? 0);

        // Deserialize KNN-specific options
        Options.NNeighbors = modelDataObj["NNeighbors"]?.ToObject<int>() ?? 5;
        Options.Metric = (DistanceMetric)(modelDataObj["Metric"]?.ToObject<int>() ?? 0);
        Options.Weights = (WeightingScheme)(modelDataObj["Weights"]?.ToObject<int>() ?? 0);
        Options.P = modelDataObj["P"]?.ToObject<double>() ?? 2.0;
        Options.Algorithm = (KNNAlgorithm)(modelDataObj["Algorithm"]?.ToObject<int>() ?? 0);
        Options.LeafSize = modelDataObj["LeafSize"]?.ToObject<int>() ?? 30;

        var classLabelsToken = modelDataObj["ClassLabels"];
        if (classLabelsToken is not null)
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

        // Deserialize _xTrain matrix
        var xTrainToken = modelDataObj["XTrain"];
        if (xTrainToken is not null)
        {
            var xTrainArray = xTrainToken.ToObject<double[]>() ?? Array.Empty<double>();
            int rows = modelDataObj["XTrainRows"]?.ToObject<int>() ?? 0;
            int cols = modelDataObj["XTrainCols"]?.ToObject<int>() ?? 0;

            if (rows > 0 && cols > 0)
            {
                _xTrain = new Matrix<T>(rows, cols);
                int idx = 0;
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        _xTrain[i, j] = NumOps.FromDouble(xTrainArray[idx++]);
                    }
                }
            }
        }

        // Deserialize _yTrain vector
        var yTrainToken = modelDataObj["YTrain"];
        if (yTrainToken is not null)
        {
            var yTrainArray = yTrainToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (yTrainArray.Length > 0)
            {
                _yTrain = new Vector<T>(yTrainArray.Length);
                for (int i = 0; i < yTrainArray.Length; i++)
                {
                    _yTrain[i] = NumOps.FromDouble(yTrainArray[i]);
                }
            }
        }
    }
}
