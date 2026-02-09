using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.Online;

/// <summary>
/// Implements Online (Incremental) Naive Bayes classifier.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Online Naive Bayes is an incremental version of the classic
/// Naive Bayes classifier that can learn from streaming data. It maintains running statistics
/// (mean, variance, counts) that are updated with each new sample.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>For each sample, update per-class feature statistics (mean, variance, count)</item>
/// <item>To predict, calculate P(class|features) using Bayes' rule</item>
/// <item>Assumes features are conditionally independent given the class</item>
/// <item>Uses Gaussian distribution assumption for continuous features</item>
/// </list>
/// </para>
///
/// <para><b>Key formulas:</b>
/// <list type="bullet">
/// <item>P(class|x) ∝ P(class) × ∏ P(xi|class)</item>
/// <item>P(xi|class) = N(xi; μ, σ²) for Gaussian assumption</item>
/// <item>Online mean update: μ_new = μ_old + (x - μ_old) / n</item>
/// <item>Online variance update: Welford's algorithm</item>
/// </list>
/// </para>
///
/// <para><b>Advantages:</b>
/// <list type="bullet">
/// <item>Very fast updates (constant time per sample)</item>
/// <item>Low memory usage (only store statistics, not data)</item>
/// <item>Works well with high-dimensional data</item>
/// <item>Naturally handles class imbalance through prior updates</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class OnlineNaiveBayesClassifier<T> : ClassifierBase<T>, IOnlineClassifier<T>
{
    private readonly OnlineNaiveBayesOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly Random _random;
    private readonly List<T> _knownClasses;
    private readonly Dictionary<int, ClassStatistics> _classStats;

    /// <summary>
    /// Gets the total number of samples the model has seen.
    /// </summary>
    public long SamplesSeen { get; private set; }

    /// <summary>
    /// Gets whether the model has seen at least one sample.
    /// </summary>
    public bool IsWarm => SamplesSeen > 0;

    /// <summary>
    /// Statistics for a single class.
    /// </summary>
    private class ClassStatistics
    {
        public long Count { get; set; }
        public double[]? Means { get; set; }
        public double[]? M2 { get; set; } // For Welford's algorithm
        public double[]? Variances => ComputeVariances();

        private double[]? ComputeVariances()
        {
            if (M2 is null || Count < 2) return null;

            var variances = new double[M2.Length];
            for (int i = 0; i < M2.Length; i++)
            {
                variances[i] = M2[i] / (Count - 1);
            }
            return variances;
        }
    }

    /// <summary>
    /// Creates a new Online Naive Bayes classifier.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public OnlineNaiveBayesClassifier(OnlineNaiveBayesOptions<T>? options = null)
        : base(options)
    {
        _options = options ?? new OnlineNaiveBayesOptions<T>();
        _random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();
        _knownClasses = new List<T>();
        _classStats = new Dictionary<int, ClassStatistics>();
    }

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.OnlineLearning;

    /// <summary>
    /// Updates the model with a single training sample.
    /// </summary>
    public void PartialFit(Vector<T> features, T label)
    {
        if (NumFeatures == 0)
        {
            NumFeatures = features.Length;
        }
        else if (features.Length != NumFeatures)
        {
            throw new ArgumentException(
                $"Feature vector length {features.Length} does not match expected {NumFeatures}.",
                nameof(features));
        }

        int classIdx = GetOrCreateClassIndex(label);

        if (!_classStats.ContainsKey(classIdx))
        {
            _classStats[classIdx] = new ClassStatistics
            {
                Means = new double[features.Length],
                M2 = new double[features.Length]
            };
        }

        var stats = _classStats[classIdx];
        stats.Count++;

        // Welford's online algorithm for mean and variance
        for (int f = 0; f < features.Length; f++)
        {
            double value = NumOps.ToDouble(features[f]);
            double delta = value - stats.Means![f];
            stats.Means[f] += delta / stats.Count;
            double delta2 = value - stats.Means[f];
            stats.M2![f] += delta * delta2;
        }

        SamplesSeen++;
    }

    /// <summary>
    /// Updates the model with a batch of training samples.
    /// </summary>
    public void PartialFit(Matrix<T> features, Vector<T> labels)
    {
        for (int i = 0; i < features.Rows; i++)
        {
            var sample = new Vector<T>(features.Columns);
            for (int j = 0; j < features.Columns; j++)
            {
                sample[j] = features[i, j];
            }
            PartialFit(sample, labels[i]);
        }
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        PartialFit(x, y);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            var features = new Vector<T>(input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                features[j] = input[i, j];
            }
            predictions[i] = PredictSingle(features);
        }

        return predictions;
    }

    private T PredictSingle(Vector<T> features)
    {
        if (_knownClasses.Count == 0)
        {
            return default!;
        }

        double bestLogProb = double.NegativeInfinity;
        int bestClass = 0;

        for (int c = 0; c < _knownClasses.Count; c++)
        {
            double logProb = ComputeLogPosterior(features, c);
            if (logProb > bestLogProb)
            {
                bestLogProb = logProb;
                bestClass = c;
            }
        }

        return _knownClasses[bestClass];
    }

    private double ComputeLogPosterior(Vector<T> features, int classIdx)
    {
        if (!_classStats.ContainsKey(classIdx))
        {
            return double.NegativeInfinity;
        }

        var stats = _classStats[classIdx];

        // Log prior: log(P(class))
        double logPrior = Math.Log((double)stats.Count / SamplesSeen);

        // Log likelihood: sum of log(P(xi|class))
        double logLikelihood = 0;
        var variances = stats.Variances;

        for (int f = 0; f < features.Length; f++)
        {
            double x = NumOps.ToDouble(features[f]);
            double mean = stats.Means![f];
            double variance = variances?[f] ?? 1.0;

            // Add small constant to prevent division by zero
            variance = Math.Max(variance, 1e-9);

            if (_options.UseGaussian)
            {
                // Gaussian log probability
                double logP = -0.5 * Math.Log(2 * Math.PI * variance) -
                             0.5 * (x - mean) * (x - mean) / variance;
                logLikelihood += logP;
            }
            else
            {
                // Simple smoothed probability based on distance from mean
                double distance = Math.Abs(x - mean) / Math.Sqrt(variance);
                double logP = -distance - Math.Log(_options.Alpha + distance);
                logLikelihood += logP;
            }
        }

        return logPrior + logLikelihood;
    }

    /// <summary>
    /// Gets the probability distribution over classes for a sample.
    /// </summary>
    public Vector<T> PredictProbabilities(Vector<T> features)
    {
        if (_knownClasses.Count == 0)
        {
            return new Vector<T>(0);
        }

        var logProbs = new double[_knownClasses.Count];
        double maxLogProb = double.NegativeInfinity;

        for (int c = 0; c < _knownClasses.Count; c++)
        {
            logProbs[c] = ComputeLogPosterior(features, c);
            maxLogProb = Math.Max(maxLogProb, logProbs[c]);
        }

        // Convert to probabilities using log-sum-exp trick
        double sumExp = 0;
        for (int c = 0; c < logProbs.Length; c++)
        {
            sumExp += Math.Exp(logProbs[c] - maxLogProb);
        }

        var probs = new Vector<T>(_knownClasses.Count);
        for (int c = 0; c < logProbs.Length; c++)
        {
            double prob = Math.Exp(logProbs[c] - maxLogProb) / sumExp;
            probs[c] = NumOps.FromDouble(prob);
        }

        return probs;
    }

    private int GetOrCreateClassIndex(T label)
    {
        for (int i = 0; i < _knownClasses.Count; i++)
        {
            if (NumOps.Compare(_knownClasses[i], label) == 0)
            {
                return i;
            }
        }

        _knownClasses.Add(label);
        NumClasses = _knownClasses.Count;
        ClassLabels = new Vector<T>(_knownClasses.ToArray());
        return _knownClasses.Count - 1;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        // Pack means and variances into parameter vector
        if (_classStats.Count == 0 || NumFeatures == 0)
        {
            return new Vector<T>(0);
        }

        int paramsPerClass = NumFeatures * 2; // mean + variance
        var parameters = new Vector<T>(_knownClasses.Count * paramsPerClass);

        int idx = 0;
        for (int c = 0; c < _knownClasses.Count; c++)
        {
            if (_classStats.ContainsKey(c))
            {
                var stats = _classStats[c];
                var variances = stats.Variances ?? new double[NumFeatures];

                for (int f = 0; f < NumFeatures; f++)
                {
                    parameters[idx++] = NumOps.FromDouble(stats.Means?[f] ?? 0);
                }
                for (int f = 0; f < NumFeatures; f++)
                {
                    parameters[idx++] = NumOps.FromDouble(variances[f]);
                }
            }
            else
            {
                idx += paramsPerClass;
            }
        }

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (NumFeatures == 0 || _knownClasses.Count == 0)
        {
            return;
        }

        int paramsPerClass = NumFeatures * 2;
        int idx = 0;

        for (int c = 0; c < _knownClasses.Count && idx + paramsPerClass <= parameters.Length; c++)
        {
            if (!_classStats.ContainsKey(c))
            {
                _classStats[c] = new ClassStatistics
                {
                    Means = new double[NumFeatures],
                    M2 = new double[NumFeatures],
                    Count = 2 // Minimum count so Variances is defined (variance = M2 / (Count - 1))
                };
            }

            var stats = _classStats[c];

            for (int f = 0; f < NumFeatures; f++)
            {
                stats.Means![f] = NumOps.ToDouble(parameters[idx++]);
            }

            // Set M2 based on variance (reverse Welford)
            for (int f = 0; f < NumFeatures; f++)
            {
                double variance = NumOps.ToDouble(parameters[idx++]);
                stats.M2![f] = variance * (stats.Count - 1);
            }
        }

        // Recompute SamplesSeen from class counts
        long totalCount = 0;
        foreach (var s in _classStats.Values)
        {
            totalCount += s.Count;
        }
        SamplesSeen = totalCount;
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var clone = new OnlineNaiveBayesClassifier<T>(_options);
        clone._knownClasses.AddRange(_knownClasses);
        clone.NumClasses = NumClasses;
        clone.NumFeatures = NumFeatures;
        clone.ClassLabels = ClassLabels is not null ? new Vector<T>(ClassLabels.ToArray()) : null;
        clone.SetParameters(parameters);
        return clone;
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new OnlineNaiveBayesClassifier<T>(_options);
    }

    /// <inheritdoc />
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Naive Bayes doesn't use gradients
        return new Vector<T>(GetParameters().Length);
    }

    /// <inheritdoc />
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Naive Bayes doesn't use gradients
    }
}
