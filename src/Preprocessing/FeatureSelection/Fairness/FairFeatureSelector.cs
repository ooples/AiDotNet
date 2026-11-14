using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Fairness;

/// <summary>
/// Fair Feature Selection that balances predictive power with fairness constraints.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that maximize predictive accuracy while minimizing correlation
/// with protected attributes to promote fair predictions.
/// </para>
/// <para><b>For Beginners:</b> In ML fairness, we want models that don't discriminate
/// based on protected attributes like race or gender. This selector chooses features
/// that are good for prediction but have low correlation with a specified protected
/// attribute. This helps build fairer models by removing proxies for protected groups.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FairFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _fairnessWeight;
    private readonly int _protectedAttributeIndex;

    private Vector<T>? _fairnessScores;
    private Vector<T>? _predictiveScores;
    private Vector<T>? _combinedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double FairnessWeight => _fairnessWeight;
    public int ProtectedAttributeIndex => _protectedAttributeIndex;
    public Vector<T>? FairnessScores => _fairnessScores;
    public Vector<T>? PredictiveScores => _predictiveScores;
    public Vector<T>? CombinedScores => _combinedScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FairFeatureSelector(
        int nFeaturesToSelect = 10,
        int protectedAttributeIndex = 0,
        double fairnessWeight = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (fairnessWeight < 0 || fairnessWeight > 1)
            throw new ArgumentException("Fairness weight must be between 0 and 1.", nameof(fairnessWeight));
        if (protectedAttributeIndex < 0)
            throw new ArgumentException("Protected attribute index must be non-negative.", nameof(protectedAttributeIndex));

        _nFeaturesToSelect = nFeaturesToSelect;
        _fairnessWeight = fairnessWeight;
        _protectedAttributeIndex = protectedAttributeIndex;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FairFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");
        if (_protectedAttributeIndex >= data.Columns)
            throw new ArgumentException("Protected attribute index exceeds number of columns.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var protectedAttr = PreprocessingTensorOps.ColumnVector(X, _protectedAttributeIndex);

        // Protected attribute itself gets zero scores
        // (a [p] one-hot, not a row of a [p, p] identity: 800 MB of doubles for 10,000 features)
        var isProtected = PreprocessingTensorOps.PositionOneHot<T>(_protectedAttributeIndex, p);
        var zeros = new Tensor<T>(new[] { p });

        // Correlation with target (predictive power), 0 when sqrt(sxx * syy) is not above 1e-10
        var predictiveScores = engine.TensorWhere(isProtected, zeros, engine.TensorAbs(
            PreprocessingTensorOps.CorrelationsWithDenominatorGuard(X, Tensor<T>.FromVector(target))));

        // Fairness score: higher when less correlated with protected attribute
        var protectedCorr = engine.TensorAbs(PreprocessingTensorOps.CorrelationsWithDenominatorGuard(X, protectedAttr));
        var fairnessScores = engine.TensorWhere(isProtected, zeros, PreprocessingTensorOps.OneMinus(protectedCorr));

        // Combined score: balance predictive power with fairness
        var combinedScores = engine.TensorAdd(
            engine.TensorMultiplyScalar(predictiveScores, NumOps.FromDouble(1 - _fairnessWeight)),
            engine.TensorMultiplyScalar(fairnessScores, NumOps.FromDouble(_fairnessWeight)));

        _predictiveScores = predictiveScores.ToVector();
        _fairnessScores = fairnessScores.ToVector();
        _combinedScores = combinedScores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p - 1); // Exclude protected attribute
        _selectedIndices = PreprocessingTensorOps.StableOrder(combinedScores, descending: true)
            .Where(j => j != _protectedAttributeIndex)
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

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
            throw new InvalidOperationException("FairFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FairFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FairFeatureSelector has not been fitted.");

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

    /// <summary>
    /// Computes the disparate impact ratio for a specific feature.
    /// </summary>
    public T GetDisparateImpact(Matrix<T> data, int featureIndex)
    {
        if (featureIndex >= data.Columns || featureIndex < 0)
            throw new ArgumentException("Invalid feature index.");

        var engine = Engine;
        int n = data.Rows;
        var X = Tensor<T>.FromMatrix(data);
        var protectedVals = PreprocessingTensorOps.ColumnVector(X, _protectedAttributeIndex);
        var featureVals = PreprocessingTensorOps.ColumnVector(X, featureIndex);

        // Split by protected attribute at its median: the midpoint of the two middle values for an even count, held
        // in double (0.5 for a balanced 0/1 attribute; the upper middle value alone left the upper group empty)
        var protectedColumn = engine.Reshape(protectedVals.Cast<double>(), new[] { n, 1 });
        double lowerMiddle = engine.TensorSum(PreprocessingTensorOps.ColumnOrderStatistic(protectedColumn, (n - 1) / 2));
        double upperMiddle = engine.TensorSum(PreprocessingTensorOps.ColumnOrderStatistic(protectedColumn, n / 2));
        double median = lowerMiddle / 2.0 + upperMiddle / 2.0;
        var upper = engine.TensorGreaterThan(protectedColumn, median);
        var lower = PreprocessingTensorOps.Not(upper);

        // Compute mean feature value for each group in double: group sizes are counted in int (a Half sum of the 0/1
        // mask stops at 2048), the feature sums are taken in double (a Half sum of four 30000s is infinite), and
        // rows outside a group are selected away rather than multiplied by 0 (0 * inf = NaN).
        int count0 = PreprocessingTensorOps.CountNonZero(lower);
        int count1 = PreprocessingTensorOps.CountNonZero(upper);
        var wideFeature = engine.Reshape(featureVals.Cast<double>(), new[] { n, 1 });
        var zeros = new Tensor<double>(new[] { n, 1 });
        double mean0 = count0 > 0 ? engine.TensorSum(engine.TensorWhere(lower, wideFeature, zeros)) / count0 : 0.0;
        double mean1 = count1 > 0 ? engine.TensorSum(engine.TensorWhere(upper, wideFeature, zeros)) / count1 : 0.0;

        // Disparate impact ratio (closer to 1 is fairer)
        bool zero0 = Math.Abs(mean0) < 1e-10;
        bool zero1 = Math.Abs(mean1) < 1e-10;
        if (zero1 && zero0)
            return NumOps.One;
        if (zero1)
            return NumOps.Zero;

        double ratio = mean0 / mean1;
        double inverse = mean1 / mean0;
        return NumOps.FromDouble(inverse < ratio ? inverse : ratio);
    }

}
