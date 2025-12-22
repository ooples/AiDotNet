using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.CurriculumLearning.Interfaces;

namespace AiDotNet.CurriculumLearning.DifficultyEstimators;

/// <summary>
/// Difficulty estimator that combines multiple estimators for robust difficulty estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This estimator combines multiple difficulty estimators
/// into a single, more robust estimate. Different estimators may capture different
/// aspects of difficulty, and combining them can provide a more comprehensive view.</para>
///
/// <para><b>Combination Strategies:</b></para>
/// <list type="bullet">
/// <item><description><b>Weighted Average:</b> Sum of (weight × difficulty) for each estimator</description></item>
/// <item><description><b>Maximum:</b> Highest difficulty from any estimator</description></item>
/// <item><description><b>Minimum:</b> Lowest difficulty from any estimator</description></item>
/// <item><description><b>Median:</b> Middle value of all difficulties</description></item>
/// <item><description><b>Rank Average:</b> Average of per-estimator difficulty ranks</description></item>
/// </list>
///
/// <para><b>Benefits:</b></para>
/// <list type="bullet">
/// <item><description>More robust to individual estimator failures</description></item>
/// <item><description>Captures multiple aspects of sample difficulty</description></item>
/// <item><description>Can combine model-based and expert-defined estimates</description></item>
/// </list>
/// </remarks>
public class EnsembleDifficultyEstimator<T, TInput, TOutput>
    : DifficultyEstimatorBase<T, TInput, TOutput>, IEnsembleDifficultyEstimator<T, TInput, TOutput>
{
    private readonly List<IDifficultyEstimator<T, TInput, TOutput>> _estimators;
    private readonly List<T> _weights;
    private readonly EnsembleCombinationMethod _combinationMethod;
    private readonly bool _normalize;

    /// <summary>
    /// Gets the name of this estimator.
    /// </summary>
    public override string Name => $"Ensemble_{_combinationMethod}";

    /// <summary>
    /// Gets whether this estimator requires the model.
    /// </summary>
    public override bool RequiresModel => _estimators.Any(e => e.RequiresModel);

    /// <summary>
    /// Gets the individual estimators in this ensemble.
    /// </summary>
    public IReadOnlyList<IDifficultyEstimator<T, TInput, TOutput>> Estimators => _estimators.AsReadOnly();

    /// <summary>
    /// Gets or sets the weights for each estimator.
    /// </summary>
    public Vector<T> Weights
    {
        get => new Vector<T>(_weights);
        set
        {
            if (value.Length != _estimators.Count)
            {
                throw new ArgumentException(
                    $"Weight count ({value.Length}) must match estimator count ({_estimators.Count}).");
            }
            _weights.Clear();
            _weights.AddRange(value);
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="EnsembleDifficultyEstimator{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="combinationMethod">How to combine estimator outputs.</param>
    /// <param name="normalize">Whether to normalize the final difficulties.</param>
    public EnsembleDifficultyEstimator(
        EnsembleCombinationMethod combinationMethod = EnsembleCombinationMethod.WeightedAverage,
        bool normalize = true)
    {
        _estimators = new List<IDifficultyEstimator<T, TInput, TOutput>>();
        _weights = new List<T>();
        _combinationMethod = combinationMethod;
        _normalize = normalize;
    }

    /// <summary>
    /// Initializes a new instance with predefined estimators.
    /// </summary>
    /// <param name="estimators">List of estimators to combine.</param>
    /// <param name="weights">Weights for each estimator. If null, uniform weights are used.</param>
    /// <param name="combinationMethod">How to combine estimator outputs.</param>
    /// <param name="normalize">Whether to normalize the final difficulties.</param>
    public EnsembleDifficultyEstimator(
        IEnumerable<IDifficultyEstimator<T, TInput, TOutput>> estimators,
        IEnumerable<T>? weights = null,
        EnsembleCombinationMethod combinationMethod = EnsembleCombinationMethod.WeightedAverage,
        bool normalize = true)
        : this(combinationMethod, normalize)
    {
        _estimators.AddRange(estimators);

        if (weights != null)
        {
            _weights.AddRange(weights);
        }
        else
        {
            // Uniform weights
            var uniformWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(_estimators.Count));
            for (int i = 0; i < _estimators.Count; i++)
            {
                _weights.Add(uniformWeight);
            }
        }

        if (_weights.Count != _estimators.Count)
        {
            throw new ArgumentException("Weight count must match estimator count.");
        }
    }

    /// <summary>
    /// Adds an estimator to the ensemble.
    /// </summary>
    public void AddEstimator(IDifficultyEstimator<T, TInput, TOutput> estimator, T weight)
    {
        if (estimator is null) throw new ArgumentNullException(nameof(estimator));
        _estimators.Add(estimator);
        _weights.Add(weight);
    }

    /// <summary>
    /// Removes an estimator from the ensemble.
    /// </summary>
    public void RemoveEstimator(int index)
    {
        if (index < 0 || index >= _estimators.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        _estimators.RemoveAt(index);
        _weights.RemoveAt(index);
    }

    /// <summary>
    /// Estimates the difficulty of a single sample using all estimators.
    /// </summary>
    public override T EstimateDifficulty(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        if (_estimators.Count == 0)
        {
            throw new InvalidOperationException("Ensemble has no estimators.");
        }

        var difficulties = new T[_estimators.Count];

        for (int i = 0; i < _estimators.Count; i++)
        {
            if (_estimators[i].RequiresModel && model == null)
            {
                throw new ArgumentNullException(nameof(model),
                    $"Estimator '{_estimators[i].Name}' requires a model.");
            }

            difficulties[i] = _estimators[i].EstimateDifficulty(input, expectedOutput, model);
        }

        return CombineDifficulties(difficulties);
    }

    /// <summary>
    /// Estimates difficulty scores for all samples.
    /// </summary>
    public override Vector<T> EstimateDifficulties(
        IDataset<T, TInput, TOutput> dataset,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        if (dataset is null) throw new ArgumentNullException(nameof(dataset));

        if (_estimators.Count == 0)
        {
            throw new InvalidOperationException("Ensemble has no estimators.");
        }

        // Get difficulties from each estimator
        var allDifficulties = new Vector<T>[_estimators.Count];

        for (int i = 0; i < _estimators.Count; i++)
        {
            allDifficulties[i] = _estimators[i].EstimateDifficulties(dataset, model);
        }

        // Combine per-sample
        var combined = new T[dataset.Count];

        for (int sampleIdx = 0; sampleIdx < dataset.Count; sampleIdx++)
        {
            var sampleDifficulties = new T[_estimators.Count];
            for (int estIdx = 0; estIdx < _estimators.Count; estIdx++)
            {
                sampleDifficulties[estIdx] = allDifficulties[estIdx][sampleIdx];
            }
            combined[sampleIdx] = CombineDifficulties(sampleDifficulties);
        }

        var result = new Vector<T>(combined);

        if (_normalize)
        {
            result = NormalizeDifficulties(result);
        }

        return result;
    }

    /// <summary>
    /// Updates all estimators in the ensemble.
    /// </summary>
    public override void Update(int epoch, IFullModel<T, TInput, TOutput> model)
    {
        foreach (var estimator in _estimators)
        {
            estimator.Update(epoch, model);
        }
    }

    /// <summary>
    /// Resets all estimators in the ensemble.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        foreach (var estimator in _estimators)
        {
            estimator.Reset();
        }
    }

    /// <summary>
    /// Combines difficulty values from multiple estimators.
    /// </summary>
    private T CombineDifficulties(T[] difficulties)
    {
        return _combinationMethod switch
        {
            EnsembleCombinationMethod.WeightedAverage => CombineWeightedAverage(difficulties),
            EnsembleCombinationMethod.Maximum => CombineMaximum(difficulties),
            EnsembleCombinationMethod.Minimum => CombineMinimum(difficulties),
            EnsembleCombinationMethod.Median => CombineMedian(difficulties),
            EnsembleCombinationMethod.GeometricMean => CombineGeometricMean(difficulties),
            EnsembleCombinationMethod.HarmonicMean => CombineHarmonicMean(difficulties),
            _ => CombineWeightedAverage(difficulties)
        };
    }

    /// <summary>
    /// Combines using weighted average.
    /// </summary>
    private T CombineWeightedAverage(T[] difficulties)
    {
        var sum = NumOps.Zero;
        var weightSum = NumOps.Zero;

        for (int i = 0; i < difficulties.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(_weights[i], difficulties[i]));
            weightSum = NumOps.Add(weightSum, _weights[i]);
        }

        return NumOps.Compare(weightSum, NumOps.Zero) > 0
            ? NumOps.Divide(sum, weightSum)
            : NumOps.Zero;
    }

    /// <summary>
    /// Combines using maximum.
    /// </summary>
    private T CombineMaximum(T[] difficulties)
    {
        if (difficulties.Length == 0) return NumOps.Zero;

        var max = difficulties[0];
        for (int i = 1; i < difficulties.Length; i++)
        {
            if (NumOps.Compare(difficulties[i], max) > 0)
            {
                max = difficulties[i];
            }
        }
        return max;
    }

    /// <summary>
    /// Combines using minimum.
    /// </summary>
    private T CombineMinimum(T[] difficulties)
    {
        if (difficulties.Length == 0) return NumOps.Zero;

        var min = difficulties[0];
        for (int i = 1; i < difficulties.Length; i++)
        {
            if (NumOps.Compare(difficulties[i], min) < 0)
            {
                min = difficulties[i];
            }
        }
        return min;
    }

    /// <summary>
    /// Combines using median.
    /// </summary>
    private T CombineMedian(T[] difficulties)
    {
        if (difficulties.Length == 0) return NumOps.Zero;

        var sorted = (T[])difficulties.Clone();
        Array.Sort(sorted, (a, b) => NumOps.Compare(a, b));

        int mid = sorted.Length / 2;
        if (sorted.Length % 2 == 0)
        {
            return NumOps.Divide(
                NumOps.Add(sorted[mid - 1], sorted[mid]),
                NumOps.FromDouble(2.0));
        }
        return sorted[mid];
    }

    /// <summary>
    /// Combines using geometric mean.
    /// </summary>
    private T CombineGeometricMean(T[] difficulties)
    {
        if (difficulties.Length == 0) return NumOps.Zero;

        var logSum = NumOps.Zero;
        var epsilon = NumOps.FromDouble(1e-10);

        foreach (var d in difficulties)
        {
            var adjusted = NumOps.Compare(d, epsilon) > 0 ? d : epsilon;
            logSum = NumOps.Add(logSum, NumOps.Log(adjusted));
        }

        var avgLog = NumOps.Divide(logSum, NumOps.FromDouble(difficulties.Length));
        return NumOps.Exp(avgLog);
    }

    /// <summary>
    /// Combines using harmonic mean.
    /// </summary>
    private T CombineHarmonicMean(T[] difficulties)
    {
        if (difficulties.Length == 0) return NumOps.Zero;

        var reciprocalSum = NumOps.Zero;
        var epsilon = NumOps.FromDouble(1e-10);

        foreach (var d in difficulties)
        {
            var adjusted = NumOps.Compare(d, epsilon) > 0 ? d : epsilon;
            reciprocalSum = NumOps.Add(reciprocalSum, NumOps.Divide(NumOps.One, adjusted));
        }

        return NumOps.Divide(NumOps.FromDouble(difficulties.Length), reciprocalSum);
    }
}

/// <summary>
/// Methods for combining multiple difficulty estimates.
/// </summary>
public enum EnsembleCombinationMethod
{
    /// <summary>
    /// Weighted average of difficulty scores.
    /// </summary>
    WeightedAverage,

    /// <summary>
    /// Maximum difficulty from any estimator.
    /// </summary>
    Maximum,

    /// <summary>
    /// Minimum difficulty from any estimator.
    /// </summary>
    Minimum,

    /// <summary>
    /// Median of all difficulty scores.
    /// </summary>
    Median,

    /// <summary>
    /// Geometric mean of difficulty scores.
    /// </summary>
    GeometricMean,

    /// <summary>
    /// Harmonic mean of difficulty scores.
    /// </summary>
    HarmonicMean
}
