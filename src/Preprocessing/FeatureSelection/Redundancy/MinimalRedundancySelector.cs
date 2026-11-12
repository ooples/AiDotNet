using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Redundancy;

/// <summary>
/// Minimal Redundancy Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that minimize redundancy among selected features while
/// maximizing relevance to the target (mRMR-like approach).
/// </para>
/// <para><b>For Beginners:</b> When features are highly correlated with each
/// other, they provide similar information. This selector picks features that
/// are different from each other while still being useful for prediction.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MinimalRedundancySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _mrmrScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? MRMRScores => _mrmrScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MinimalRedundancySelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MinimalRedundancySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // Relevance |corr(feature, target)| and redundancy |corr(feature, feature)|
        var relevance = engine.TensorAbs(engine.Reshape(
            FeatureSelectionTensorOps.ColumnCorrelations(x, engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 })), new[] { p }));
        var correlation = engine.TensorAbs(FeatureSelectionTensorOps.ColumnCorrelations(x, x));

        // Greedy mRMR selection; ties go to the lowest feature index
        var selected = new List<int>();
        var mrmrScores = new Tensor<T>(new[] { p });
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        if (p > 0)
        {
            // First feature: highest relevance
            int firstFeature = FeatureSelectionTensorOps.StableOrder(relevance, descending: true)[0];
            selected.Add(firstFeature);
            var chosen = FeatureSelectionTensorOps.OneHot<T>(firstFeature, p);
            mrmrScores = engine.TensorMultiply(chosen, relevance);
            var remaining = FeatureSelectionTensorOps.OneMinus(chosen);
            var redundancySum = engine.Reshape(engine.TensorSlice(correlation, new[] { 0, firstFeature }, new[] { p, 1 }), new[] { p });

            // Subsequent features: maximize relevance - mean redundancy with the selected features
            while (selected.Count < numToSelect)
            {
                var score = engine.TensorSubtract(relevance, engine.TensorDivideScalar(redundancySum, FeatureSelectionTensorOps.Count<T>(selected.Count)));
                var best = FeatureSelectionTensorOps.TopKWhere(score, remaining, 1);
                if (best.Length == 0)
                    break;

                int bestFeature = best[0];
                selected.Add(bestFeature);
                chosen = FeatureSelectionTensorOps.OneHot<T>(bestFeature, p);
                mrmrScores = engine.TensorAdd(mrmrScores, engine.TensorMultiply(chosen, score));
                remaining = engine.TensorSubtract(remaining, chosen);
                redundancySum = engine.TensorAdd(
                    redundancySum, engine.Reshape(engine.TensorSlice(correlation, new[] { 0, bestFeature }, new[] { p, 1 }), new[] { p }));
            }
        }

        _mrmrScores = mrmrScores.ToVector();
        _selectedIndices = selected.OrderBy(index => index).ToArray();

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
            throw new InvalidOperationException("MinimalRedundancySelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MinimalRedundancySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MinimalRedundancySelector has not been fitted.");

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
