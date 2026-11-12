using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Minimal Redundancy feature selection for unsupervised learning.
/// </summary>
/// <remarks>
/// <para>
/// Minimal Redundancy selects a diverse set of features that are not highly
/// correlated with each other. This ensures the selected features provide
/// complementary rather than overlapping information.
/// </para>
/// <para><b>For Beginners:</b> If two features are almost identical (highly
/// correlated), keeping both is wasteful. This method picks features that are
/// as different from each other as possible, giving you maximum diversity.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MinimalRedundancy<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _correlationThreshold;

    private Matrix<T>? _correlationMatrix;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double CorrelationThreshold => _correlationThreshold;
    public Matrix<T>? CorrelationMatrix => _correlationMatrix;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MinimalRedundancy(
        int nFeaturesToSelect = 10,
        double correlationThreshold = 0.9,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _correlationThreshold = correlationThreshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Population variances; the correlation divides by the standard deviation (1 for a constant column)
        var centered = FilterStatisticTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        T count = NumOps.FromDouble(n);
        var variances = Engine.TensorDivideScalar(SelectorTensorOps.ColumnSumOfSquares(centered), count);
        var stds = Engine.TensorSqrt(variances);
        stds = Engine.TensorWhere(Engine.TensorLessThan(stds, NumOps.FromDouble(1e-10)),
            PreprocessingTensorOps.Filled(new[] { p }, NumOps.One), stds);

        // Compute correlation matrix (unit diagonal)
        var covariance = Engine.TensorDivideScalar(Engine.TensorMatMul(Engine.TensorTranspose(centered), centered), count);
        var correlation = Engine.TensorWhere(Engine.TensorEye<T>(p), PreprocessingTensorOps.Filled(new[] { p, p }, NumOps.One),
            Engine.TensorDivide(covariance, Engine.TensorOuter(stds, stds)));
        _correlationMatrix = correlation.ToMatrix();
        var absCorrelation = Engine.TensorAbs(correlation);
        var tooCorrelated = Engine.TensorGreaterThan(absCorrelation, NumOps.FromDouble(_correlationThreshold));

        // Greedy selection: pick features with high variance and low redundancy
        // (score = variance - mean |correlation| with the selected features; the first best wins ties)
        var selected = new List<int>();
        var available = new SortedSet<int>(Enumerable.Range(0, p));
        var redundancySum = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);

        while (selected.Count < _nFeaturesToSelect && available.Count > 0)
        {
            var candidates = available.ToArray();
            var score = selected.Count == 0
                ? variances
                : Engine.TensorSubtract(variances,
                    Engine.TensorDivideScalar(redundancySum, PreprocessingTensorOps.CountAsScalar<T>(selected.Count)));
            int bestFeature = candidates[PreprocessingTensorOps.FirstArgMax(
                Engine.TensorIndexSelect(score, new Tensor<int>(candidates, new[] { candidates.Length }), 0))];

            selected.Add(bestFeature);
            available.Remove(bestFeature);
            redundancySum = Engine.TensorAdd(redundancySum, PreprocessingTensorOps.RowVector(absCorrelation, bestFeature));

            // Remove features too correlated with the newly selected one
            foreach (int j in PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.RowVector(tooCorrelated, bestFeature)))
                available.Remove(j);
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();

        // If not enough selected, add remaining by variance
        if (_selectedIndices.Length < _nFeaturesToSelect)
        {
            var chosen = new HashSet<int>(_selectedIndices);
            var remaining = PreprocessingTensorOps.StableOrder(variances, descending: true)
                .Where(j => !chosen.Contains(j))
                .Take(_nFeaturesToSelect - _selectedIndices.Length);

            _selectedIndices = _selectedIndices.Concat(remaining).OrderBy(x => x).ToArray();
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MinimalRedundancy has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MinimalRedundancy does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MinimalRedundancy has not been fitted.");

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
