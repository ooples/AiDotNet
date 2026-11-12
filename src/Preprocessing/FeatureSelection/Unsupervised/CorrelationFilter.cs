using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Correlation Filter for removing highly correlated features.
/// </summary>
/// <remarks>
/// <para>
/// Identifies pairs of features with correlation above a threshold and removes
/// one from each pair. This reduces multicollinearity and feature redundancy.
/// </para>
/// <para><b>For Beginners:</b> If two features are very similar (highly correlated),
/// keeping both is redundant. This filter finds such pairs and keeps only one,
/// reducing data size without losing much information.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CorrelationFilter<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFeatureSelector<T>
{
    private readonly double _threshold;
    private readonly KeepStrategy _strategy;

    public enum KeepStrategy { KeepFirst, KeepHigherVariance }

    private Matrix<T>? _correlationMatrix;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public Matrix<T>? CorrelationMatrix => _correlationMatrix;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CorrelationFilter(
        double threshold = 0.9,
        KeepStrategy strategy = KeepStrategy.KeepHigherVariance,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _threshold = threshold;
        _strategy = strategy;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute population variances and standard deviations (1 for a constant column)
        var centered = SelectorTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
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

        // Pairs (i < j) whose |correlation| exceeds the threshold, in (i, j) order, and the pairs whose first
        // feature has at least the second's variance
        var pairs = PreprocessingTensorOps.MaskPositions(Engine.TensorMultiply(
            Engine.TensorGreaterThan(Engine.TensorAbs(correlation), NumOps.FromDouble(_threshold)),
            SelectorTensorOps.StrictUpperMask<T>(p)));
        var shape = new[] { p, p };
        var firstHasMoreVariance = new HashSet<int>(PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(
            Engine.TensorLessThan(
                Engine.TensorBroadcastTo(Engine.Reshape(variances, new[] { p, 1 }), shape),
                Engine.TensorBroadcastTo(Engine.Reshape(variances, new[] { 1, p }), shape)))));

        // Find features to drop: walk the pairs in order; a feature that has been dropped no longer removes others
        var toDrop = new HashSet<int>();
        foreach (int position in pairs)
        {
            int i = position / p;
            int j = position % p;
            if (toDrop.Contains(i) || toDrop.Contains(j)) continue;

            // Drop one of the pair
            if (_strategy == KeepStrategy.KeepFirst || firstHasMoreVariance.Contains(position))
                toDrop.Add(j);
            else
                toDrop.Add(i); // Keep higher variance
        }

        _selectedIndices = Enumerable.Range(0, p)
            .Where(i => !toDrop.Contains(i))
            .ToArray();

        if (_selectedIndices.Length == 0)
            _selectedIndices = new[] { 0 };

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        FitCore(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationFilter has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CorrelationFilter does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationFilter has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
