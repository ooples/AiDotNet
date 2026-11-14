using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Maximal Variance feature selection for unsupervised learning.
/// </summary>
/// <remarks>
/// <para>
/// Maximal Variance selects features with the highest variance. Features with
/// higher variance contain more information and are more likely to be useful
/// for distinguishing between samples.
/// </para>
/// <para><b>For Beginners:</b> If a feature has the same value for all samples,
/// it doesn't help distinguish between them. Features that vary a lot are more
/// informative. This method simply picks the features that change the most.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MaximalVariance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _normalize;

    private Vector<T>? _variances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool Normalize => _normalize;
    public Vector<T>? Variances => _variances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MaximalVariance(
        int nFeaturesToSelect = 10,
        bool normalize = false,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _normalize = normalize;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Population mean and variance of every column
        var x = Tensor<T>.FromMatrix(data);
        T count = NumOps.FromDouble(n);
        var mean = Engine.TensorDivideScalar(Engine.ReduceSum(x, new[] { 0 }, false), count);
        var variance = Engine.TensorDivideScalar(
            SelectorTensorOps.ColumnSumOfSquares(FilterStatisticTensorOps.CenterColumns(x)), count);

        var scores = variance;
        if (_normalize)
        {
            // Coefficient of variation (relative variance) where the mean is non-zero
            var nonZeroMean = Engine.TensorNotEquals(mean, NumOps.Zero);
            var absMean = Engine.TensorAbs(mean);
            var safeAbsMean = Engine.TensorWhere(nonZeroMean, absMean, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
            scores = Engine.TensorWhere(nonZeroMean, Engine.TensorDivide(Engine.TensorSqrt(variance), safeAbsMean), variance);
        }

        _variances = scores.ToVector();

        // Select top features by variance
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MaximalVariance has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MaximalVariance does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MaximalVariance has not been fitted.");

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
