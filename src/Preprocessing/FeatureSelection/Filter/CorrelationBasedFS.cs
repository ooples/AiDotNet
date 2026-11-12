using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Correlation-based Feature Selection (CFS).
/// </summary>
/// <remarks>
/// <para>
/// CFS evaluates feature subsets by considering both feature-target correlation
/// (relevance) and feature-feature correlation (redundancy). Good subsets have
/// high correlation with the target but low intercorrelation among features.
/// </para>
/// <para><b>For Beginners:</b> CFS looks for features that are strongly related
/// to what you want to predict but not too similar to each other. Having two
/// features that are almost identical doesn't help much - you want diverse
/// information sources that all point toward the target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CorrelationBasedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

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

    public CorrelationBasedFS(
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
            "CorrelationBasedFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Absolute feature-target and feature-feature correlations (0 when a sum of squares is not above 1e-10)
        var featureTargetCorr = engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(X, Tensor<T>.FromVector(target)));
        var featureCorr = engine.TensorAbs(FeatureSelectionTensorOps.CorrelationMatrix(X));

        // Greedy forward selection using CFS merit k * r_cf / sqrt(k + k(k-1) r_ff). For a candidate c added to the
        // selected set S: k * r_cf = sum_S r_f + r_c and k + k(k-1) r_ff = 1 + sum_{S x S} r_ff + 2 sum_S r_fc.
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        var shape = new[] { p };
        var selectedMask = new Tensor<T>(shape);
        var remainingMask = PreprocessingTensorOps.Filled(shape, NumOps.One);
        var scores = new Tensor<T>(shape);
        var zeros = new Tensor<T>(shape);
        var ones = PreprocessingTensorOps.Filled(shape, NumOps.One);
        var excluded = PreprocessingTensorOps.Filled(shape, NumOps.FromDouble(double.NegativeInfinity));
        T one = NumOps.One;
        T two = NumOps.FromDouble(2);
        T epsilon = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var selected = new List<int>();

        while (selected.Count < numToSelect && selected.Count < p)
        {
            var pairSums = engine.Reshape(engine.TensorMatMul(featureCorr, engine.Reshape(selectedMask, new[] { p, 1 })), shape);
            T selectedTargetSum = engine.TensorSum(engine.TensorMultiply(featureTargetCorr, selectedMask));
            T selectedPairSum = engine.TensorSum(engine.TensorMultiply(pairSums, selectedMask));
            var numerators = engine.TensorAddScalar(featureTargetCorr, selectedTargetSum);
            var denominators = engine.TensorSqrt(engine.TensorAddScalar(engine.TensorAddScalar(
                engine.TensorMultiplyScalar(pairSums, two), selectedPairSum), one));
            var usable = engine.TensorGreaterThan(denominators, epsilon);
            var merits = engine.TensorWhere(usable,
                engine.TensorDivide(numerators, engine.TensorWhere(usable, denominators, ones)), zeros);

            // First remaining candidate with the highest merit
            int bestIdx = PreprocessingTensorOps.FirstArgMax(engine.TensorWhere(remainingMask, merits, excluded));
            var pick = engine.Reshape(engine.TensorOneHot<T>(new Tensor<int>(new[] { bestIdx }, new[] { 1 }), p), shape);
            selected.Add(bestIdx);
            scores = engine.TensorWhere(pick, merits, scores);
            selectedMask = engine.TensorAdd(selectedMask, pick);
            remainingMask = engine.TensorSubtract(remainingMask, pick);
        }

        _featureScores = scores.ToVector();
        _selectedIndices = [.. selected.OrderBy(x => x)];

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
            throw new InvalidOperationException("CorrelationBasedFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CorrelationBasedFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationBasedFS has not been fitted.");

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
