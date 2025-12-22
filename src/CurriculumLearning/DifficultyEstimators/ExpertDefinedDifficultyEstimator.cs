using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.CurriculumLearning.Interfaces;

namespace AiDotNet.CurriculumLearning.DifficultyEstimators;

/// <summary>
/// Difficulty estimator using pre-defined or expert-provided difficulty scores.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This estimator uses difficulty scores provided by domain
/// experts or predefined heuristics. Unlike model-based estimators, it doesn't require
/// a trained model to estimate difficulty.</para>
///
/// <para><b>Use Cases:</b></para>
/// <list type="bullet">
/// <item><description>Educational datasets with known difficulty levels</description></item>
/// <item><description>Domain-specific complexity measures (e.g., sentence length for NLP)</description></item>
/// <item><description>Manual annotation of sample difficulty</description></item>
/// <item><description>Reproducible experiments with fixed difficulty ordering</description></item>
/// </list>
///
/// <para><b>Difficulty Sources:</b></para>
/// <list type="bullet">
/// <item><description><b>Direct:</b> Explicit difficulty scores per sample</description></item>
/// <item><description><b>Function:</b> A function computing difficulty from sample features</description></item>
/// <item><description><b>Dataset metadata:</b> Difficulty stored with sample data</description></item>
/// </list>
/// </remarks>
public class ExpertDefinedDifficultyEstimator<T, TInput, TOutput> : DifficultyEstimatorBase<T, TInput, TOutput>
{
    private readonly Func<TInput, TOutput, T>? _difficultyFunction;
    private readonly Vector<T>? _precomputedDifficulties;
    private readonly bool _normalize;

    /// <summary>
    /// Gets the name of this estimator.
    /// </summary>
    public override string Name => "ExpertDefined";

    /// <summary>
    /// Gets whether this estimator requires the model.
    /// </summary>
    public override bool RequiresModel => false;

    /// <summary>
    /// Initializes a new instance with precomputed difficulty scores.
    /// </summary>
    /// <param name="difficulties">Vector of difficulty scores (one per sample).</param>
    /// <param name="normalize">Whether to normalize difficulties to [0, 1].</param>
    public ExpertDefinedDifficultyEstimator(
        Vector<T> difficulties,
        bool normalize = true)
    {
        if (difficulties is null) throw new ArgumentNullException(nameof(difficulties));
        _precomputedDifficulties = difficulties;
        _difficultyFunction = null;
        _normalize = normalize;

        if (normalize)
        {
            CachedScores = NormalizeDifficulties(difficulties);
        }
        else
        {
            CachedScores = difficulties;
        }
    }

    /// <summary>
    /// Initializes a new instance with a difficulty function.
    /// </summary>
    /// <param name="difficultyFunction">Function computing difficulty from input and output.</param>
    /// <param name="normalize">Whether to normalize difficulties to [0, 1].</param>
    public ExpertDefinedDifficultyEstimator(
        Func<TInput, TOutput, T> difficultyFunction,
        bool normalize = true)
    {
        if (difficultyFunction is null) throw new ArgumentNullException(nameof(difficultyFunction));
        _difficultyFunction = difficultyFunction;
        _precomputedDifficulties = null;
        _normalize = normalize;
    }

    /// <summary>
    /// Estimates the difficulty of a single sample.
    /// </summary>
    public override T EstimateDifficulty(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        if (_difficultyFunction != null)
        {
            return _difficultyFunction(input, expectedOutput);
        }

        throw new InvalidOperationException(
            "Cannot estimate individual sample difficulty when using precomputed difficulties. " +
            "Use EstimateDifficulties() instead or provide a difficulty function.");
    }

    /// <summary>
    /// Estimates difficulty scores for all samples in a dataset.
    /// </summary>
    public override Vector<T> EstimateDifficulties(
        IDataset<T, TInput, TOutput> dataset,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        if (dataset is null) throw new ArgumentNullException(nameof(dataset));

        if (_precomputedDifficulties != null)
        {
            if (_precomputedDifficulties.Length != dataset.Count)
            {
                throw new ArgumentException(
                    $"Precomputed difficulties count ({_precomputedDifficulties.Length}) " +
                    $"doesn't match dataset size ({dataset.Count}).",
                    nameof(dataset));
            }

            return CachedScores ?? _precomputedDifficulties;
        }

        if (_difficultyFunction != null)
        {
            var difficulties = new T[dataset.Count];

            for (int i = 0; i < dataset.Count; i++)
            {
                var sample = dataset.GetSample(i);
                difficulties[i] = _difficultyFunction(sample.Input, sample.Output);
            }

            var result = new Vector<T>(difficulties);

            if (_normalize)
            {
                result = NormalizeDifficulties(result);
            }

            if (CacheScores)
            {
                CachedScores = result;
            }

            return result;
        }

        throw new InvalidOperationException(
            "ExpertDefinedDifficultyEstimator requires either precomputed difficulties or a difficulty function.");
    }

    /// <summary>
    /// Updates the difficulty estimator (no-op for expert-defined).
    /// </summary>
    public override void Update(int epoch, IFullModel<T, TInput, TOutput> model)
    {
        // Expert-defined difficulties don't change based on model training
    }

    /// <summary>
    /// Creates an estimator with difficulty based on sample index.
    /// </summary>
    /// <param name="totalSamples">Total number of samples.</param>
    /// <param name="ascending">If true, lower indices are easier. If false, higher indices are easier.</param>
    /// <returns>A new expert-defined difficulty estimator.</returns>
    public static ExpertDefinedDifficultyEstimator<T, TInput, TOutput> FromIndex(
        int totalSamples,
        bool ascending = true)
    {
        var difficulties = new T[totalSamples];
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < totalSamples; i++)
        {
            var normalizedIndex = (double)i / (totalSamples - 1);
            difficulties[i] = numOps.FromDouble(ascending ? normalizedIndex : 1.0 - normalizedIndex);
        }

        return new ExpertDefinedDifficultyEstimator<T, TInput, TOutput>(
            new Vector<T>(difficulties),
            normalize: false);
    }

    /// <summary>
    /// Creates an estimator with random difficulty scores.
    /// </summary>
    /// <param name="totalSamples">Total number of samples.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>A new expert-defined difficulty estimator with random difficulties.</returns>
    public static ExpertDefinedDifficultyEstimator<T, TInput, TOutput> Random(
        int totalSamples,
        int? seed = null)
    {
        var random = seed.HasValue ? new Random(seed.Value) : new Random();
        var difficulties = new T[totalSamples];
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < totalSamples; i++)
        {
            difficulties[i] = numOps.FromDouble(random.NextDouble());
        }

        return new ExpertDefinedDifficultyEstimator<T, TInput, TOutput>(
            new Vector<T>(difficulties),
            normalize: false);
    }
}

/// <summary>
/// Difficulty estimator based on sample complexity metrics.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This estimator calculates difficulty from measurable
/// properties of the input data, such as magnitude, dimensionality, or variance.</para>
/// </remarks>
public class ComplexityBasedDifficultyEstimator<T, TInput, TOutput> : DifficultyEstimatorBase<T, TInput, TOutput>
{
    private readonly ComplexityMetric _metric;
    private readonly bool _normalize;

    /// <summary>
    /// Gets the name of this estimator.
    /// </summary>
    public override string Name => $"ComplexityBased_{_metric}";

    /// <summary>
    /// Gets whether this estimator requires the model.
    /// </summary>
    public override bool RequiresModel => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="ComplexityBasedDifficultyEstimator{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="metric">The complexity metric to use.</param>
    /// <param name="normalize">Whether to normalize difficulties.</param>
    public ComplexityBasedDifficultyEstimator(
        ComplexityMetric metric = ComplexityMetric.Magnitude,
        bool normalize = true)
    {
        _metric = metric;
        _normalize = normalize;
    }

    /// <summary>
    /// Estimates difficulty based on input complexity.
    /// </summary>
    public override T EstimateDifficulty(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        return _metric switch
        {
            ComplexityMetric.Magnitude => CalculateMagnitudeComplexity(input),
            ComplexityMetric.Variance => CalculateVarianceComplexity(input),
            ComplexityMetric.Sparsity => CalculateSparsityComplexity(input),
            ComplexityMetric.Entropy => CalculateEntropyComplexity(input),
            _ => CalculateMagnitudeComplexity(input)
        };
    }

    /// <summary>
    /// Estimates difficulty scores for all samples.
    /// </summary>
    public override Vector<T> EstimateDifficulties(
        IDataset<T, TInput, TOutput> dataset,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        var difficulties = base.EstimateDifficulties(dataset, model);

        if (_normalize)
        {
            difficulties = NormalizeDifficulties(difficulties);
        }

        return difficulties;
    }

    /// <summary>
    /// Calculates complexity based on input magnitude.
    /// </summary>
    private T CalculateMagnitudeComplexity(TInput input)
    {
        if (input is Vector<T> vector)
        {
            // L2 norm
            var sumSquares = NumOps.Zero;
            foreach (var value in vector)
            {
                sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(value, value));
            }
            return NumOps.Sqrt(sumSquares);
        }

        if (input is T[] array)
        {
            var sumSquares = NumOps.Zero;
            foreach (var value in array)
            {
                sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(value, value));
            }
            return NumOps.Sqrt(sumSquares);
        }

        if (input is T scalar)
        {
            return NumOps.Abs(scalar);
        }

        return NumOps.One;
    }

    /// <summary>
    /// Calculates complexity based on input variance.
    /// </summary>
    private T CalculateVarianceComplexity(TInput input)
    {
        Vector<T>? vector = null;

        if (input is Vector<T> v)
        {
            vector = v;
        }
        else if (input is T[] array)
        {
            vector = new Vector<T>(array);
        }

        if (vector == null || vector.Length == 0)
        {
            return NumOps.Zero;
        }

        var mean = ComputeMean(vector);
        var variance = NumOps.Zero;

        foreach (var value in vector)
        {
            var diff = NumOps.Subtract(value, mean);
            variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(variance, NumOps.FromDouble(vector.Length));
    }

    /// <summary>
    /// Calculates complexity based on input sparsity (inverse).
    /// </summary>
    private T CalculateSparsityComplexity(TInput input)
    {
        Vector<T>? vector = null;

        if (input is Vector<T> v)
        {
            vector = v;
        }
        else if (input is T[] array)
        {
            vector = new Vector<T>(array);
        }

        if (vector == null || vector.Length == 0)
        {
            return NumOps.Zero;
        }

        // Count non-zero elements
        var nonZeroCount = 0;
        var epsilon = NumOps.FromDouble(1e-10);

        foreach (var value in vector)
        {
            if (NumOps.Compare(NumOps.Abs(value), epsilon) > 0)
            {
                nonZeroCount++;
            }
        }

        // Dense vectors are more complex
        return NumOps.FromDouble((double)nonZeroCount / vector.Length);
    }

    /// <summary>
    /// Calculates complexity based on input entropy.
    /// </summary>
    private T CalculateEntropyComplexity(TInput input)
    {
        Vector<T>? vector = null;

        if (input is Vector<T> v)
        {
            vector = v;
        }
        else if (input is T[] array)
        {
            vector = new Vector<T>(array);
        }

        if (vector == null || vector.Length == 0)
        {
            return NumOps.Zero;
        }

        // Normalize to get distribution
        var sum = NumOps.Zero;
        foreach (var value in vector)
        {
            sum = NumOps.Add(sum, NumOps.Abs(value));
        }

        if (NumOps.Compare(sum, NumOps.Zero) == 0)
        {
            return NumOps.Zero;
        }

        // Calculate entropy
        var entropy = NumOps.Zero;
        var epsilon = NumOps.FromDouble(1e-10);

        foreach (var value in vector)
        {
            var p = NumOps.Divide(NumOps.Abs(value), sum);
            if (NumOps.Compare(p, epsilon) > 0)
            {
                entropy = NumOps.Subtract(entropy, NumOps.Multiply(p, NumOps.Log(p)));
            }
        }

        return entropy;
    }
}

/// <summary>
/// Complexity metric types.
/// </summary>
public enum ComplexityMetric
{
    /// <summary>
    /// Uses L2 norm (magnitude) of the input.
    /// </summary>
    Magnitude,

    /// <summary>
    /// Uses variance of input values.
    /// </summary>
    Variance,

    /// <summary>
    /// Uses density (inverse sparsity) of input.
    /// </summary>
    Sparsity,

    /// <summary>
    /// Uses entropy of input distribution.
    /// </summary>
    Entropy
}
