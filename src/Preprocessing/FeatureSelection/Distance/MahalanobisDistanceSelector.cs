using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Distance;

/// <summary>
/// Mahalanobis Distance based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the Mahalanobis distance contribution, which accounts
/// for correlations between features and class-specific covariances.
/// </para>
/// <para><b>For Beginners:</b> Mahalanobis distance is like Euclidean distance but
/// adjusted for how data is spread out. It's useful when features are correlated
/// or have different scales. Features that contribute more to class separation
/// considering the data's shape are selected.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MahalanobisDistanceSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _distanceContributions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? DistanceContributions => _distanceContributions;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MahalanobisDistanceSelector(
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
            "MahalanobisDistanceSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var X = Tensor<T>.FromMatrix(data);
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;

        // Compute class means for each feature
        var membership = PreprocessingTensorOps.ClassOneHot<T>(classIndex, k);
        var classMeans = FeatureSelectionTensorOps.GroupMeans(membership, X, out _);

        // Compute pooled within-class variance for each feature
        var withinClass = engine.TensorSubtract(X, engine.TensorMatMul(membership, classMeans));
        var sumSq = engine.ReduceSum(engine.TensorSquare(withinClass), new[] { 0 }, false);
        var pooledVariance = n > k
            ? engine.TensorDivideScalar(sumSq, NumOps.FromDouble(n - k))
            : PreprocessingTensorOps.Filled(new[] { p }, NumOps.One);

        // Compute Mahalanobis-like contribution for each feature: mean squared class-mean difference over variance
        var variance = engine.TensorMax(pooledVariance, NumOps.FromDouble(1e-10));
        var contributions = engine.TensorDivide(
            FeatureSelectionTensorOps.MeanPairwiseDifference(classMeans, squared: true), variance);
        _distanceContributions = contributions.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(contributions, numToSelect);

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
            throw new InvalidOperationException("MahalanobisDistanceSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MahalanobisDistanceSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MahalanobisDistanceSelector has not been fitted.");

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
