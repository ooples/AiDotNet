using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Randomized Search for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Randomized Search evaluates random subsets of features to find good
/// combinations. It's faster than exhaustive search and can handle larger
/// feature spaces while still exploring diverse solutions.
/// </para>
/// <para><b>For Beginners:</b> Instead of trying every combination (too slow) or
/// being systematic (might miss good solutions), this method randomly samples
/// feature combinations. It's like randomly picking lottery numbers - given enough
/// tries, you'll likely find something good.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RandomizedSearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private Tensor<T>? _featureScoreTensor;
    private Vector<T>? _featureScores;
    private double _bestScore;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NIterations => _nIterations;
    public Vector<T>? FeatureScores => _featureScores;
    public double BestScore => _bestScore;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public RandomizedSearch(
        int nFeaturesToSelect = 10,
        int nIterations = 1000,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RandomizedSearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var featureScores = ComputeFeatureScores(data, target, n, p);
        _featureScoreTensor = featureScores;
        _featureScores = featureScores.ToVector();

        _bestScore = double.MinValue;
        _selectedIndices = null;

        // Track seen combinations to avoid duplicates
        var seenCombinations = new HashSet<string>();

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Generate random subset
            var subset = GenerateRandomSubset(p, random);

            // Check if already seen
            string key = string.Join(",", subset.OrderBy(x => x));
            if (seenCombinations.Contains(key))
                continue;
            seenCombinations.Add(key);

            // Evaluate subset
            double score = EvaluateSubset(subset);

            if (score > _bestScore)
            {
                _bestScore = score;
                _selectedIndices = subset.OrderBy(x => x).ToArray();
            }
        }

        // Fallback if no good solution found
        if (_selectedIndices is null)
        {
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(featureScores, Math.Min(_nFeaturesToSelect, p));
        }

        IsFitted = true;
    }

    private int[] GenerateRandomSubset(int p, Random random)
    {
        var allIndices = Enumerable.Range(0, p).ToList();
        var subset = new int[_nFeaturesToSelect];

        for (int i = 0; i < _nFeaturesToSelect; i++)
        {
            int idx = random.Next(allIndices.Count);
            subset[i] = allIndices[idx];
            allIndices.RemoveAt(idx);
        }

        return subset;
    }

    private double EvaluateSubset(int[] subset)
    {
        var featureScores = _featureScoreTensor ?? throw new InvalidOperationException("Feature scores have not been computed.");
        var scores = Engine.TensorIndexSelect(featureScores, new Tensor<int>(subset, new[] { subset.Length }), 0);
        return WrapperScoringOps.ToDouble(Engine.TensorMean(scores));
    }

    private Tensor<T> ComputeFeatureScores(Matrix<T> data, Vector<T> target, int n, int p)
    {
        // |Pearson r| of every feature with the target
        return WrapperScoringOps.AbsTargetCorrelations(data, target, Enumerable.Range(0, p).ToArray());
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomizedSearch has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RandomizedSearch does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomizedSearch has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
