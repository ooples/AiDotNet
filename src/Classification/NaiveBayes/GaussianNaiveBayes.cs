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
}
