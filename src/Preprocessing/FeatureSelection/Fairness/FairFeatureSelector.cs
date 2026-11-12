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
        var isProtected = PreprocessingTensorOps.RowVector(engine.TensorEye<T>(p), _protectedAttributeIndex);
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

        // Split by protected attribute (threshold at the median, the n/2-th smallest value)
        T median = engine.TensorSum(PreprocessingTensorOps.ColumnOrderStatistic(
            engine.Reshape(protectedVals, new[] { n, 1 }), n / 2));
        var upper = engine.TensorGreaterThan(protectedVals, median);
        var lower = PreprocessingTensorOps.Not(upper);

        // Compute mean feature value for each group
        T count0 = engine.TensorSum(lower);
        T count1 = engine.TensorSum(upper);
        T mean0 = NumOps.GreaterThan(count0, NumOps.Zero)
            ? NumOps.Divide(engine.TensorSum(engine.TensorMultiply(featureVals, lower)), count0)
            : NumOps.Zero;
        T mean1 = NumOps.GreaterThan(count1, NumOps.Zero)
            ? NumOps.Divide(engine.TensorSum(engine.TensorMultiply(featureVals, upper)), count1)
            : NumOps.Zero;

        // Disparate impact ratio (closer to 1 is fairer)
        T floor = NumOps.FromDouble(1e-10);
        bool zero0 = NumOps.LessThan(NumOps.Abs(mean0), floor);
        bool zero1 = NumOps.LessThan(NumOps.Abs(mean1), floor);
        if (zero1 && zero0)
            return NumOps.One;
        if (zero1)
            return NumOps.Zero;

        T ratio = NumOps.Divide(mean0, mean1);
        T inverse = NumOps.Divide(mean1, mean0);
        return NumOps.LessThan(inverse, ratio) ? inverse : ratio;
    }

}
