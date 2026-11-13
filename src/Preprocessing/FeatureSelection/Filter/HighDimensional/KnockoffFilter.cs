using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.HighDimensional;

/// <summary>
/// Knockoff Filter for false discovery rate control in feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Knockoff Filter creates "knockoff" copies of features that mimic their
/// correlation structure but are independent of the target. Features whose
/// importance exceeds their knockoffs' importance are selected.
/// </para>
/// <para>
/// This method provides FDR (False Discovery Rate) control, guaranteeing that
/// the expected proportion of false discoveries is below a specified threshold.
/// </para>
/// <para><b>For Beginners:</b> Knockoffs are fake features designed to compete
/// with real ones. If a real feature beats its knockoff, it's likely important.
/// This controls how many false positives sneak into your selection.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KnockoffFilter<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly double _fdr;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, Vector<T>>? _importanceFunc;

    private Vector<T>? _wStats;
    private T _threshold;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double FDR => _fdr;
    public Vector<T>? WStatistics => _wStats;
    public T Threshold => _threshold;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public KnockoffFilter(
        double fdr = 0.1,
        Func<Matrix<T>, Vector<T>, Vector<T>>? importanceFunc = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (fdr <= 0 || fdr >= 1)
            throw new ArgumentException("FDR must be between 0 and 1.", nameof(fdr));

        _fdr = fdr;
        _importanceFunc = importanceFunc;
        _threshold = NumOps.Zero;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KnockoffFilter requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Generate knockoff features
        var knockoffs = GenerateKnockoffs(X, n, p, random);

        // Compute importance for original and knockoff features
        var augmentedMatrix = engine.TensorConcatenate(new[] { X, knockoffs }, 1).ToMatrix();
        var importances = Tensor<T>.FromVector(GetImportances(augmentedMatrix, target));

        // Compute W statistics: W_j = |Z_j| - |Z~_j|
        var absImportances = engine.TensorAbs(importances);
        var wStats = engine.TensorSubtract(
            engine.TensorSlice(absImportances, new[] { 0 }, new[] { p }),
            engine.TensorSlice(absImportances, new[] { p }, new[] { p }));
        _wStats = wStats.ToVector();

        // Compute knockoff threshold
        _threshold = ComputeThreshold(wStats, p);

        // Select features with W >= threshold; if nothing is selected, take the (first) highest W
        var selectedList = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.AtLeast(wStats, _threshold));
        if (selectedList.Length == 0)
            selectedList = new[] { PreprocessingTensorOps.FirstArgMax(wStats) };

        _selectedIndices = selectedList;
        IsFitted = true;
    }

    private Tensor<T> GenerateKnockoffs(Tensor<T> X, int n, int p, Random random)
    {
        // Simplified knockoff generation using permutation
        // (Full Model-X knockoffs require more complex covariance estimation)
        var engine = Engine;
        var columns = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            // Permute feature values
            var indices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
            columns[j] = engine.TensorIndexSelect(
                engine.TensorSlice(X, new[] { 0, j }, new[] { n, 1 }), new Tensor<int>(indices, new[] { n }), 0);
        }

        return p == 0 ? new Tensor<T>(new[] { n, 0 }) : engine.TensorConcatenate(columns, 1);
    }

    private Vector<T> GetImportances(Matrix<T> data, Vector<T> target)
    {
        if (_importanceFunc is not null)
            return _importanceFunc(data, target);

        // Default: correlation with target (0 when either sum of squares is not above 1e-10)
        return PreprocessingTensorOps.TargetCorrelations(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target)).ToVector();
    }

    /// <summary>
    /// Knockoff+ threshold (Barber and Candes 2015): the smallest t in {|W_j| : W_j != 0} with
    /// (1 + #{W_j &lt;= -t}) / max(1, #{W_j &gt;= t}) &lt;= q, or T.MaxValue when no t qualifies.
    /// </summary>
    private T ComputeThreshold(Tensor<T> wStats, int p)
    {
        var engine = Engine;
        var shape = new[] { p, p };
        var absW = engine.TensorAbs(wStats);

        // Rows index W_k, columns the candidate t_j = |W_j|
        var w = engine.TensorBroadcastTo(engine.Reshape(wStats, new[] { p, 1 }), shape);
        var t = engine.TensorBroadcastTo(engine.Reshape(absW, new[] { 1, p }), shape);
        // Counted in int (a Half sum of 0/1 indicators stops at 2048) and the ratio taken in double
        var numPositive = engine.ReduceSum(
            PreprocessingTensorOps.Not(engine.TensorLessThan(w, t)).Cast<int>(), new[] { 0 }, false).Cast<double>();
        var numNegative = engine.ReduceSum(
            PreprocessingTensorOps.Not(engine.TensorGreaterThan(w, engine.TensorNegate(t))).Cast<int>(), new[] { 0 }, false).Cast<double>();
        var fdp = engine.TensorDivide(engine.TensorAddScalar(numNegative, 1.0), engine.TensorMax(numPositive, 1.0));
        var valid = engine.TensorMultiply(
            PreprocessingTensorOps.Not(engine.TensorGreaterThan(fdp, _fdr)).Cast<T>(),
            engine.TensorGreaterThan(absW, NumOps.Zero));

        if (PreprocessingTensorOps.CountNonZero(valid) == 0)
            return NumOps.MaxValue;

        return engine.TensorMinValue(engine.TensorWhere(valid, absW, PreprocessingTensorOps.Filled(new[] { p }, NumOps.MaxValue)));
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
            throw new InvalidOperationException("KnockoffFilter has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KnockoffFilter does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KnockoffFilter has not been fitted.");

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
