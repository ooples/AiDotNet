using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Stability Selection for robust feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Stability Selection runs feature selection multiple times on random subsamples
/// of the data and selects features that are consistently chosen across iterations.
/// This reduces sensitivity to noise and produces more reliable feature sets.
/// </para>
/// <para><b>For Beginners:</b> A single run of feature selection might pick some features
/// by chance. Stability Selection runs many rounds on different parts of the data and
/// counts how often each feature is selected. Features that are picked consistently
/// (say, 70% of the time) are more trustworthy than those picked rarely.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class StabilitySelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly double _sampleFraction;
    private readonly double _threshold;
    private readonly Func<Matrix<T>, Vector<T>, int, int[]>? _baseSelector;
    private readonly int? _randomState;

    private Vector<T>? _selectionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NIterations => _nIterations;
    public double SampleFraction => _sampleFraction;
    public double Threshold => _threshold;
    public Vector<T>? SelectionProbabilities => _selectionProbabilities;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public StabilitySelection(
        int nFeaturesToSelect = 10,
        int nIterations = 100,
        double sampleFraction = 0.5,
        double threshold = 0.6,
        Func<Matrix<T>, Vector<T>, int, int[]>? baseSelector = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (sampleFraction <= 0 || sampleFraction > 1)
            throw new ArgumentException("Sample fraction must be between 0 and 1.", nameof(sampleFraction));
        if (threshold <= 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _sampleFraction = sampleFraction;
        _threshold = threshold;
        _baseSelector = baseSelector;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StabilitySelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var baseSelector = _baseSelector ?? DefaultSelector;

        var engine = Engine;

        // Count selections across iterations (in int: a Half count stops at 2048 iterations)
        var selectionCounts = new Tensor<int>(new[] { p });
        int sampleSize = Math.Max(1, (int)(n * _sampleFraction));

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Random subsample
            var indices = Enumerable.Range(0, n)
                .OrderBy(_ => random.Next())
                .Take(sampleSize)
                .ToArray();

            // Extract subsample
            var subDataMatrix = PreprocessingTensorOps.SelectRows(data, indices);
            var subTargetVector = PreprocessingTensorOps.SelectElements(target, indices);

            // Run base selector
            int numBase = Math.Min(_nFeaturesToSelect, p);
            var selected = baseSelector(subDataMatrix, subTargetVector, numBase);

            if (selected.Length > 0)
            {
                selectionCounts = engine.TensorAdd(selectionCounts, PreprocessingTensorOps.CategoryCounts<int>(
                    new Tensor<int>((int[])selected.Clone(), new[] { selected.Length }), p));
            }
        }

        // Compute selection probabilities
        var selectionProbabilities = engine.TensorDivideScalar(selectionCounts.Cast<double>(), (double)_nIterations).Cast<T>();
        _selectionProbabilities = selectionProbabilities.ToVector();

        // Select features above threshold, most frequently selected first
        var passing = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            PreprocessingTensorOps.AtLeast(selectionProbabilities, NumOps.FromDouble(_threshold))));
        var candidates = PreprocessingTensorOps.StableOrder(selectionProbabilities, descending: true)
            .Where(j => passing.Contains(j))
            .ToList();

        int numToSelect = Math.Min(_nFeaturesToSelect, candidates.Count);
        if (numToSelect == 0)
        {
            // No features pass threshold, select top by probability
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(selectionProbabilities, Math.Min(_nFeaturesToSelect, p));
        }
        else
        {
            _selectedIndices = candidates
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private int[] DefaultSelector(Matrix<T> data, Vector<T> target, int k)
    {
        // Simple correlation-based selection (absolute Pearson correlation, best first)
        var scores = Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
            Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target)));
        return PreprocessingTensorOps.StableOrder(scores, descending: true).Take(k).ToArray();
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
            throw new InvalidOperationException("StabilitySelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("StabilitySelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StabilitySelection has not been fitted.");

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
