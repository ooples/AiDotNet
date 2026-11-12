using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Relief;

/// <summary>
/// ReliefF algorithm for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// ReliefF is an extension of the Relief algorithm that handles multi-class
/// problems and uses k nearest neighbors. It evaluates features based on their
/// ability to distinguish between instances of different classes.
/// </para>
/// <para><b>For Beginners:</b> ReliefF works by looking at each data point and
/// comparing it to its nearest neighbors. Features are scored based on how well
/// they separate a point from neighbors of different classes while keeping it
/// close to neighbors of the same class. It's like finding features that cluster
/// similar items together.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ReliefF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly int _nSamples;
    private readonly int? _randomState;

    private Vector<T>? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public int NSamples => _nSamples;
    public Vector<T>? FeatureWeights => _featureWeights;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ReliefF(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        int nSamples = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        if (nSamples < 1)
            throw new ArgumentException("Number of samples must be at least 1.", nameof(nSamples));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _nSamples = nSamples;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ReliefF requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Integer class labels and features scaled to [0, 1] by their range (a constant feature never differs)
        var classIndex = FilterStatisticTensorOps.ClassIndex(target, out int classCount);
        var scaled = FeatureSelectionTensorOps.RangeScaled(FilterStatisticTensorOps.AsTensor(data));

        // Sample instances, in the same random order as before
        int sampleCount = Math.Min(_nSamples, n);
        var sampleIndices = Enumerable.Range(0, n)
            .OrderBy(_ => random.Next())
            .Take(sampleCount)
            .ToArray();

        // ReliefF weights: k nearest hits against the k nearest misses of every other class
        var weights = FeatureSelectionTensorOps.ReliefFWeights(scaled, classIndex, classCount, sampleIndices, _nNeighbors);
        _featureWeights = weights.ToVector();

        // Select top features
        _selectedIndices = FilterStatisticTensorOps.TopKIndicesSorted(weights, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
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
            throw new InvalidOperationException("ReliefF has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ReliefF does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ReliefF has not been fitted.");

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
