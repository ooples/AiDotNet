using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Constraint;

/// <summary>
/// Constraint-based Feature Selection with domain constraints.
/// </summary>
/// <remarks>
/// <para>
/// Selects features while respecting domain-specific constraints such as
/// mandatory features that must be included, forbidden features that must
/// be excluded, and group constraints where features must be selected together.
/// </para>
/// <para><b>For Beginners:</b> Sometimes domain experts know certain features
/// must be included (like patient age in medical data) or must be excluded
/// (like sensitive personal info). This method respects those rules while
/// still finding the best possible feature subset.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ConstraintBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int[]? _mandatoryFeatures;
    private readonly int[]? _forbiddenFeatures;
    private readonly int[][]? _featureGroups;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ConstraintBasedSelector(
        int nFeaturesToSelect = 10,
        int[]? mandatoryFeatures = null,
        int[]? forbiddenFeatures = null,
        int[][]? featureGroups = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        if (mandatoryFeatures is not null && mandatoryFeatures.Length > nFeaturesToSelect)
            throw new ArgumentException(
                $"Mandatory features count ({mandatoryFeatures.Length}) exceeds nFeaturesToSelect ({nFeaturesToSelect}).",
                nameof(mandatoryFeatures));

        if (mandatoryFeatures is not null && forbiddenFeatures is not null)
        {
            var overlap = mandatoryFeatures.Intersect(forbiddenFeatures).ToArray();
            if (overlap.Length > 0)
                throw new ArgumentException(
                    $"Features cannot be both mandatory and forbidden: [{string.Join(", ", overlap)}]");
        }

        _nFeaturesToSelect = nFeaturesToSelect;
        _mandatoryFeatures = mandatoryFeatures;
        _forbiddenFeatures = forbiddenFeatures;
        _featureGroups = featureGroups;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ConstraintBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Validate mandatory feature indices
        if (_mandatoryFeatures is not null)
        {
            var outOfRange = _mandatoryFeatures.Where(idx => idx < 0 || idx >= p).ToArray();
            if (outOfRange.Length > 0)
                throw new ArgumentException(
                    $"Mandatory feature indices out of range [0, {p}): [{string.Join(", ", outOfRange)}]");
        }

        // Validate forbidden feature indices
        if (_forbiddenFeatures is not null)
        {
            var outOfRange = _forbiddenFeatures.Where(idx => idx < 0 || idx >= p).ToArray();
            if (outOfRange.Length > 0)
                throw new ArgumentException(
                    $"Forbidden feature indices out of range [0, {p}): [{string.Join(", ", outOfRange)}]");
        }

        // Compute feature scores (|correlation| with target; 0 for a (near-)constant feature or target)
        var scores = Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
            Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target)));
        _featureScores = scores.ToVector();

        // Initialize selection with mandatory features
        var selected = new HashSet<int>();
        if (_mandatoryFeatures is not null)
        {
            foreach (int idx in _mandatoryFeatures)
            {
                if (idx >= 0 && idx < p)
                    selected.Add(idx);
            }
        }

        // Get candidate features (excluding forbidden and already selected)
        var forbidden = _forbiddenFeatures?.ToHashSet() ?? new HashSet<int>();
        var candidates = PreprocessingTensorOps.StableOrder(scores, descending: true)
            .Where(j => !selected.Contains(j) && !forbidden.Contains(j))
            .ToList();

        // Add features respecting group constraints
        foreach (int candidate in candidates)
        {
            if (selected.Count >= _nFeaturesToSelect)
                break;

            // Check if candidate is part of a group
            var group = _featureGroups?.FirstOrDefault(g => g.Contains(candidate));
            if (group is not null)
            {
                // Add entire group if not already added and space permits (validate bounds)
                var groupToAdd = group
                    .Where(f => f >= 0 && f < p && !selected.Contains(f) && !forbidden.Contains(f))
                    .ToList();
                if (selected.Count + groupToAdd.Count <= _nFeaturesToSelect)
                {
                    foreach (int f in groupToAdd)
                        selected.Add(f);
                }
            }
            else
            {
                selected.Add(candidate);
            }
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
            throw new InvalidOperationException("ConstraintBasedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ConstraintBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ConstraintBasedSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ConstraintBasedSelector has not been fitted.");

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        if (_selectedIndices.Any(i => i >= inputFeatureNames.Length))
            throw new ArgumentException(
                "Input feature names array is shorter than expected based on selected indices.",
                nameof(inputFeatureNames));

        return _selectedIndices.Select(i => inputFeatureNames[i]).ToArray();
    }
}
