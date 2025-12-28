using AiDotNet.Models.Options;

namespace AiDotNet.Classification.NaiveBayes;

/// <summary>
/// Gaussian Naive Bayes classifier for continuous features.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Gaussian Naive Bayes assumes that the continuous features follow a Gaussian (normal)
/// distribution within each class. It estimates the mean and variance of each feature
/// for each class during training, then uses these to compute the probability density
/// during prediction.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// This classifier works well with continuous data (like measurements: height, weight, temperature).
/// It assumes each feature follows a bell-shaped curve (normal distribution) for each class.
///
/// During training, it learns:
/// - The average value of each feature for each class
/// - How spread out (variance) each feature is for each class
///
/// During prediction, it calculates how likely a new data point is under each class's
/// distribution and picks the most likely class.
///
/// Example use cases:
/// - Classifying iris flowers based on petal/sepal measurements
/// - Medical diagnosis based on patient vitals
/// - Weather prediction based on sensor readings
/// </para>
/// </remarks>
public class GaussianNaiveBayes<T> : NaiveBayesBase<T>
{
    /// <summary>
    /// Mean values for each feature in each class.
    /// Shape: [NumClasses, NumFeatures]
    /// </summary>
    private Matrix<T>? _means;

    /// <summary>
    /// Variance values for each feature in each class.
    /// Shape: [NumClasses, NumFeatures]
    /// </summary>
    private Matrix<T>? _variances;

    /// <summary>
    /// Precomputed log(2 * pi) for efficiency in log-likelihood calculation.
    /// </summary>
    private readonly T _log2Pi;

    /// <summary>
    /// Initializes a new instance of the GaussianNaiveBayes class.
    /// </summary>
    /// <param name="options">Configuration options for the Naive Bayes classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public GaussianNaiveBayes(NaiveBayesOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _log2Pi = NumOps.FromDouble(Math.Log(2.0 * Math.PI));
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    /// <returns>ModelType.GaussianNaiveBayes</returns>
    protected override ModelType GetModelType() => ModelType.GaussianNaiveBayes;

    /// <summary>
    /// Computes the mean and variance of each feature for each class.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target class labels vector.</param>
    protected override void ComputeClassParameters(Matrix<T> x, Vector<T> y)
    {
        _means = new Matrix<T>(NumClasses, NumFeatures);
        _variances = new Matrix<T>(NumClasses, NumFeatures);

        // For each class, compute the mean and variance of each feature
        for (int c = 0; c < NumClasses; c++)
        {
            // Collect all samples belonging to this class
            var classIndices = new List<int>();
            for (int i = 0; i < y.Length; i++)
            {
                if (GetClassIndex(y[i]) == c)
                {
                    classIndices.Add(i);
                }
            }

            int classCount = classIndices.Count;

            // Check for empty class to prevent division by zero
            if (classCount == 0)
            {
                // Use zero mean and minimum variance for classes with no samples
                T minVar = NumOps.FromDouble(Options.MinVariance);
                for (int f = 0; f < NumFeatures; f++)
                {
                    _means[c, f] = NumOps.Zero;
                    _variances[c, f] = minVar;
                }
                continue;
            }

            // Compute mean for each feature
            for (int f = 0; f < NumFeatures; f++)
            {
                T sum = NumOps.Zero;
                foreach (int idx in classIndices)
                {
                    sum = NumOps.Add(sum, x[idx, f]);
                }
                _means[c, f] = NumOps.Divide(sum, NumOps.FromDouble(classCount));
            }

            // Compute variance for each feature (using n, not n-1, for sklearn compatibility)
            for (int f = 0; f < NumFeatures; f++)
            {
                T sumSquaredDiff = NumOps.Zero;
                T mean = _means[c, f];

                foreach (int idx in classIndices)
                {
                    T diff = NumOps.Subtract(x[idx, f], mean);
                    sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
                }

                // Use n instead of n-1 for consistency with sklearn
                T variance = NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(classCount));

                // Apply minimum variance to prevent division by zero
                T minVar = NumOps.FromDouble(Options.MinVariance);
                if (NumOps.Compare(variance, minVar) < 0)
                {
                    variance = minVar;
                }

                _variances[c, f] = variance;
            }
        }
    }

    /// <summary>
    /// Computes the log-likelihood of a sample given a class using Gaussian distribution.
    /// </summary>
    /// <param name="sample">The feature vector for a single sample.</param>
    /// <param name="classIndex">The class index.</param>
    /// <returns>The log-likelihood log P(sample|class).</returns>
    /// <remarks>
    /// <para>
    /// The Gaussian log-likelihood for a feature x given class c is:
    /// log P(x|c) = -0.5 * [log(2*pi) + log(variance) + (x - mean)^2 / variance]
    /// </para>
    /// <para>
    /// The total log-likelihood is the sum over all features (assuming independence).
    /// </para>
    /// </remarks>
    protected override T ComputeLogLikelihood(Vector<T> sample, int classIndex)
    {
        if (_means == null || _variances == null)
        {
            throw new InvalidOperationException("Model must be trained before computing log-likelihood.");
        }

        T logLikelihood = NumOps.Zero;
        T half = NumOps.FromDouble(0.5);

        for (int f = 0; f < NumFeatures; f++)
        {
            T mean = _means[classIndex, f];
            T variance = _variances[classIndex, f];

            // Compute (x - mean)^2 / variance
            T diff = NumOps.Subtract(sample[f], mean);
            T squaredDiff = NumOps.Multiply(diff, diff);
            T normalizedSquaredDiff = NumOps.Divide(squaredDiff, variance);

            // Compute log(variance)
            T logVariance = NumOps.Log(variance);

            // log P(x|c) for this feature: -0.5 * (log(2*pi) + log(variance) + (x-mean)^2/variance)
            T featureLogLikelihood = NumOps.Multiply(
                NumOps.Negate(half),
                NumOps.Add(NumOps.Add(_log2Pi, logVariance), normalizedSquaredDiff)
            );

            logLikelihood = NumOps.Add(logLikelihood, featureLogLikelihood);
        }

        return logLikelihood;
    }

    /// <summary>
    /// Creates a new instance of this model type.
    /// </summary>
    /// <returns>A new GaussianNaiveBayes instance.</returns>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new GaussianNaiveBayes<T>(new NaiveBayesOptions<T>
        {
            Alpha = Options.Alpha,
            FitPriors = Options.FitPriors,
            ClassPriors = Options.ClassPriors,
            MinVariance = Options.MinVariance
        });
    }

    /// <summary>
    /// Creates a deep clone of this model.
    /// </summary>
    /// <returns>A cloned GaussianNaiveBayes instance.</returns>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new GaussianNaiveBayes<T>(new NaiveBayesOptions<T>
        {
            Alpha = Options.Alpha,
            FitPriors = Options.FitPriors,
            ClassPriors = Options.ClassPriors?.ToArray(),
            MinVariance = Options.MinVariance
        });

        // Copy trained state
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

        if (LogPriors != null)
        {
            clone.LogPriors = new Vector<T>(LogPriors.Length);
            for (int i = 0; i < LogPriors.Length; i++)
            {
                clone.LogPriors[i] = LogPriors[i];
            }
        }

        if (ClassCounts != null)
        {
            clone.ClassCounts = ClassCounts.ToArray();
        }

        if (_means != null)
        {
            clone._means = new Matrix<T>(_means.Rows, _means.Columns);
            for (int i = 0; i < _means.Rows; i++)
            {
                for (int j = 0; j < _means.Columns; j++)
                {
                    clone._means[i, j] = _means[i, j];
                }
            }
        }

        if (_variances != null)
        {
            clone._variances = new Matrix<T>(_variances.Rows, _variances.Columns);
            for (int i = 0; i < _variances.Rows; i++)
            {
                for (int j = 0; j < _variances.Columns; j++)
                {
                    clone._variances[i, j] = _variances[i, j];
                }
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["MinVariance"] = Options.MinVariance;
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
            { "ClassCounts", ClassCounts ?? Array.Empty<int>() }
        };

        // Serialize LogPriors
        if (LogPriors is not null)
        {
            var logPriorsArray = new double[LogPriors.Length];
            for (int i = 0; i < LogPriors.Length; i++)
            {
                logPriorsArray[i] = NumOps.ToDouble(LogPriors[i]);
            }
            modelData["LogPriors"] = logPriorsArray;
        }

        // Serialize _means matrix
        if (_means is not null)
        {
            var meansArray = new double[_means.Rows * _means.Columns];
            int idx = 0;
            for (int i = 0; i < _means.Rows; i++)
            {
                for (int j = 0; j < _means.Columns; j++)
                {
                    meansArray[idx++] = NumOps.ToDouble(_means[i, j]);
                }
            }
            modelData["Means"] = meansArray;
            modelData["MeansRows"] = _means.Rows;
            modelData["MeansCols"] = _means.Columns;
        }

        // Serialize _variances matrix
        if (_variances is not null)
        {
            var variancesArray = new double[_variances.Rows * _variances.Columns];
            int idx = 0;
            for (int i = 0; i < _variances.Rows; i++)
            {
                for (int j = 0; j < _variances.Columns; j++)
                {
                    variancesArray[idx++] = NumOps.ToDouble(_variances[i, j]);
                }
            }
            modelData["Variances"] = variancesArray;
            modelData["VariancesRows"] = _variances.Rows;
            modelData["VariancesCols"] = _variances.Columns;
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

        // Deserialize ClassCounts
        var classCountsToken = modelDataObj["ClassCounts"];
        if (classCountsToken is not null)
        {
            ClassCounts = classCountsToken.ToObject<int[]>();
        }

        // Deserialize LogPriors
        var logPriorsToken = modelDataObj["LogPriors"];
        if (logPriorsToken is not null)
        {
            var logPriorsArray = logPriorsToken.ToObject<double[]>() ?? Array.Empty<double>();
            LogPriors = new Vector<T>(logPriorsArray.Length);
            for (int i = 0; i < logPriorsArray.Length; i++)
            {
                LogPriors[i] = NumOps.FromDouble(logPriorsArray[i]);
            }
        }

        // Deserialize _means matrix
        var meansToken = modelDataObj["Means"];
        if (meansToken is not null)
        {
            var meansArray = meansToken.ToObject<double[]>() ?? Array.Empty<double>();
            int rows = modelDataObj["MeansRows"]?.ToObject<int>() ?? 0;
            int cols = modelDataObj["MeansCols"]?.ToObject<int>() ?? 0;

            if (rows > 0 && cols > 0)
            {
                _means = new Matrix<T>(rows, cols);
                int idx = 0;
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        _means[i, j] = NumOps.FromDouble(meansArray[idx++]);
                    }
                }
            }
        }

        // Deserialize _variances matrix
        var variancesToken = modelDataObj["Variances"];
        if (variancesToken is not null)
        {
            var variancesArray = variancesToken.ToObject<double[]>() ?? Array.Empty<double>();
            int rows = modelDataObj["VariancesRows"]?.ToObject<int>() ?? 0;
            int cols = modelDataObj["VariancesCols"]?.ToObject<int>() ?? 0;

            if (rows > 0 && cols > 0)
            {
                _variances = new Matrix<T>(rows, cols);
                int idx = 0;
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        _variances[i, j] = NumOps.FromDouble(variancesArray[idx++]);
                    }
                }
            }
        }
    }
}
