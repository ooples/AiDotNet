using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Fast Correlation-Based Filter (FCBF) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// FCBF uses Symmetric Uncertainty to evaluate feature-target and feature-feature
/// correlations. It selects features that are highly correlated with the target
/// but not redundant with other selected features, using a fast backwards elimination
/// process.
/// </para>
/// <para><b>For Beginners:</b> Having two features that say the same thing is wasteful.
/// FCBF first picks features that are useful for prediction, then removes features
/// that duplicate what other selected features already tell you. It's like building
/// a team where each member brings unique skills.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FCBF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _threshold;
    private readonly int _nBins;

    private Vector<T>? _suScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Threshold => _threshold;
    public int NBins => _nBins;
    public Vector<T>? SUScores => _suScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FCBF(
        int nFeaturesToSelect = 10,
        double threshold = 0.0,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _threshold = threshold;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FCBF requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var info = DiscreteInformation<T>.ForTarget(
            FeatureScoringTensorOps.EqualWidthBins(FilterStatisticTensorOps.AsTensor(data), _nBins), _nBins, target);
        var suTarget = info.SymmetricUncertainty();
        _suScores = suTarget.ToVector();

        // Step 2: features whose SU with the target reaches the threshold, by decreasing SU (ties by index)
        var below = Engine.TensorLessThan(suTarget, NumOps.FromDouble(_threshold)).Cast<int>().ToArray();
        var candidates = PreprocessingTensorOps.StableOrder(suTarget, descending: true)
            .Where(j => below[j] == 0)
            .ToList();

        if (candidates.Count == 0)
        {
            // If no features pass threshold, take top features by SU
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(suTarget, Math.Min(_nFeaturesToSelect, p));
            IsFitted = true;
            return;
        }

        // Step 3: drop every later candidate fj that fi predominates, SU(fi, fj) >= SU(fj, Y)
        var selected = new List<int>();
        int i = 0;
        while (i < candidates.Count && selected.Count < _nFeaturesToSelect)
        {
            int fi = candidates[i];
            selected.Add(fi);
            var kept = Engine.TensorLessThan(info.FeatureSymmetricUncertainty(fi), suTarget).Cast<int>().ToArray();
            int position = i;
            candidates = candidates
                .Where((fj, idx) => idx <= position || kept[fj] == 1)
                .ToList();
            i++;
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
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
            throw new InvalidOperationException("FCBF has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FCBF does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FCBF has not been fitted.");

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
