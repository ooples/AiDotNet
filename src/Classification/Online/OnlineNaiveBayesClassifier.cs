using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using AiDotNet.Models.Parameters;
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
/// <example>
/// <code>
/// // Create online Naive Bayes for streaming data classification
/// var options = new OnlineNaiveBayesOptions&lt;double&gt;();
///
/// // Prepare training data
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 1.1 }, { 1.2, 0.9 }, { 0.8, 1.0 }, { 5.0, 5.1 }, { 5.2, 4.9 }, { 4.8, 5.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train by updating running statistics (mean, variance) incrementally
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new OnlineNaiveBayesClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict using Bayes' rule with Gaussian likelihood
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.1, 1.0 } });
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("A Comparison of Event Models for Naive Bayes Text Classification", "https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf")]
public partial class OnlineNaiveBayesClassifier<T> : ClassifierBase<T>, IOnlineClassifier<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The per-class means and variances, packed exactly as before. PackParameters and UnpackParameters ARE the previous
    /// GetParameters and SetParameters, renamed rather than rewritten, so the layout is
    /// byte-for-byte unchanged. What changes is that the COUNT now folds this same
    /// component instead of ClassifierBase's NumFeatures * NumClasses formula, which
    /// described the classifier's shape rather than what it serializes.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VariableLengthParameterSource<T>(
            () => PackParameters().Length,
            PackParameters,
            UnpackParameters));
    }
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
        // Plain arrays so a saved model keeps master's JSON number arrays
        public T[]? Means { get; set; }
        public T[]? M2 { get; set; } // For Welford's algorithm
    }

    /// <summary>
    /// Creates a new Online Naive Bayes classifier.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public OnlineNaiveBayesClassifier(OnlineNaiveBayesOptions<T>? options = null)
        : base(options)
    {
        _options = options ?? new OnlineNaiveBayesOptions<T>();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        _knownClasses = new List<T>();
        _classStats = new Dictionary<int, ClassStatistics>();
    }

    /// <inheritdoc />

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
                Means = new T[features.Length],
                M2 = new T[features.Length]
            };
        }

        var stats = _classStats[classIdx];
        stats.Count++;

        // Welford's online algorithm for mean and variance:
        // delta = x - mean; mean += delta / n; M2 += delta * (x - mean)
        var engine = Engine;
        var means = Vector<T>.Wrap(stats.Means ?? throw new InvalidOperationException("Means has not been initialized."));
        var m2 = Vector<T>.Wrap(stats.M2 ?? throw new InvalidOperationException("M2 has not been initialized."));
        var delta = engine.Subtract(features, means);
        var updatedMeans = engine.Add(means, engine.Divide(delta, NumOps.FromDouble(stats.Count)));
        var delta2 = engine.Subtract(features, updatedMeans);
        stats.Means = updatedMeans.ToArray();
        stats.M2 = engine.Add(m2, engine.Multiply(delta, delta2)).ToArray();

        SamplesSeen++;
    }

    /// <summary>
    /// Updates the model with a batch of training samples.
    /// </summary>
    public void PartialFit(Matrix<T> features, Vector<T> labels)
    {
        var labelValues = labels.ToArray();
        for (int i = 0; i < features.Rows; i++)
        {
            PartialFit(features.GetRow(i), labelValues[i]);
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
        var predictions = new T[input.Rows];

        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return Vector<T>.Wrap(predictions);
    }

    private T PredictSingle(Vector<T> features)
    {
        if (_knownClasses.Count == 0)
        {
            throw new InvalidOperationException("No known classes available for prediction. The classifier must be trained first.");
        }

        // Class with the highest log posterior (the first one wins ties)
        var logPosteriors = ComputeLogPosteriors(features);
        int bestClass = Engine.TensorArgMax(logPosteriors.AsTensor().Reshape(1, logPosteriors.Length), 1).ToArray()[0];
        return _knownClasses[bestClass];
    }

    private T ComputeLogPosterior(Vector<T> features, int classIdx)
    {
        if (!_classStats.ContainsKey(classIdx))
        {
            return NumOps.FromDouble(double.NegativeInfinity);
        }

        var engine = Engine;
        var stats = _classStats[classIdx];
        var means = Vector<T>.Wrap(stats.Means ?? throw new InvalidOperationException("Means has not been initialized."));

        // Log prior: log(P(class))
        T logPrior = NumOps.Log(NumOps.Divide(NumOps.FromDouble(stats.Count), NumOps.FromDouble(SamplesSeen)));

        // Variances (1 until two samples were seen), plus a small floor to prevent division by zero
        var variances = GetVariances(stats) ?? FilledVector(features.Length, NumOps.One);
        var variance = engine.TensorClampMin(variances.AsTensor(), NumOps.FromDouble(1e-9));
        var diff = engine.TensorSubtract(features.AsTensor(), means.AsTensor());

        // Log likelihood: sum of log(P(xi|class))
        Tensor<T> logP;
        if (_options.UseGaussian)
        {
            // Gaussian log probability: -0.5 * log(2*pi*variance) - 0.5 * (x - mean)^2 / variance
            T negHalf = NumOps.FromDouble(-0.5);
            var logNorm = engine.TensorMultiplyScalar(
                engine.TensorLog(engine.TensorMultiplyScalar(variance, NumOps.FromDouble(2 * Math.PI))), negHalf);
            var quadratic = engine.TensorMultiplyScalar(
                engine.TensorDivide(engine.TensorMultiply(diff, diff), variance), negHalf);
            logP = engine.TensorAdd(logNorm, quadratic);
        }
        else
        {
            // Simple smoothed probability based on distance from mean: -d - log(alpha + d)
            var distance = engine.TensorDivide(engine.TensorAbs(diff), engine.TensorSqrt(variance));
            var penalty = engine.TensorLog(engine.TensorAddScalar(distance, NumOps.FromDouble(_options.Alpha)));
            logP = engine.TensorNegate(engine.TensorAdd(distance, penalty));
        }

        T logLikelihood = features.Length == 0 ? NumOps.Zero : engine.TensorSum(logP);
        return NumOps.Add(logPrior, logLikelihood);
    }

    /// <summary>
    /// Log posterior of every known class for one sample.
    /// </summary>
    private Vector<T> ComputeLogPosteriors(Vector<T> features)
    {
        var values = new T[_knownClasses.Count];
        for (int c = 0; c < values.Length; c++)
        {
            values[c] = ComputeLogPosterior(features, c);
        }

        return Vector<T>.Wrap(values);
    }

    /// <summary>
    /// M2 / (n - 1), or null before two samples of the class were seen.
    /// </summary>
    private Vector<T>? GetVariances(ClassStatistics stats)
    {
        if (stats.M2 is null || stats.Count < 2) return null;
        return Engine.Divide(Vector<T>.Wrap(stats.M2), NumOps.FromDouble(stats.Count - 1));
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

        // Convert to probabilities using the log-sum-exp (softmax) trick
        return Engine.Softmax(ComputeLogPosteriors(features));
    }

    private int GetOrCreateClassIndex(T label)
    {
        // Known classes are stored in first-seen order; equality test against all of them at once
        if (_knownClasses.Count > 0)
        {
            var matches = Engine.TensorEquals(Vector<T>.Wrap(_knownClasses.ToArray()).AsTensor(), label);
            if (NumOps.Compare(Engine.TensorSum(matches), NumOps.Zero) != 0)
            {
                return Engine.TensorArgMax(matches.Reshape(1, _knownClasses.Count), 1).ToArray()[0];
            }
        }

        _knownClasses.Add(label);
        NumClasses = _knownClasses.Count;
        ClassLabels = Vector<T>.Wrap(_knownClasses.ToArray());
        return _knownClasses.Count - 1;
    }

    /// <inheritdoc />
    private Vector<T> PackParameters()
    {
        // Pack means and variances into parameter vector
        if (_classStats.Count == 0 || NumFeatures == 0)
        {
            return new Vector<T>(0);
        }

        // [means | variances] per known class (zeros for classes without statistics)
        var blocks = new Tensor<T>[_knownClasses.Count * 2];
        for (int c = 0; c < _knownClasses.Count; c++)
        {
            var means = new Vector<T>(NumFeatures);
            var variances = new Vector<T>(NumFeatures);
            if (_classStats.TryGetValue(c, out var stats))
            {
                means = stats.Means is null ? means : FitToLength(Vector<T>.Wrap(stats.Means), NumFeatures);
                var classVariances = GetVariances(stats);
                variances = classVariances is null ? variances : FitToLength(classVariances, NumFeatures);
            }

            blocks[2 * c] = means.AsTensor();
            blocks[2 * c + 1] = variances.AsTensor();
        }

        return Engine.TensorConcatenate(blocks, 0).ToVector();
    }

    /// <inheritdoc />
    private void UnpackParameters(Vector<T> parameters)
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
                    Means = new T[NumFeatures],
                    M2 = new T[NumFeatures],
                    Count = 2 // Minimum count so Variances is defined (variance = M2 / (Count - 1))
                };
            }

            var stats = _classStats[c];

            stats.Means = parameters.GetSubVector(idx, NumFeatures).ToArray();
            idx += NumFeatures;

            // Set M2 based on variance (reverse Welford)
            stats.M2 = Engine.Multiply(parameters.GetSubVector(idx, NumFeatures), NumOps.FromDouble(stats.Count - 1)).ToArray();
            idx += NumFeatures;
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
}
