using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Score-CAM based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using Score-CAM, a gradient-free class activation mapping
/// method that uses forward passing scores instead of gradients.
/// </para>
/// <para><b>For Beginners:</b> Unlike GradCAM which uses gradients, Score-CAM
/// measures feature importance by masking each feature and observing how the
/// prediction score changes. Features that cause larger score drops when
/// masked are considered more important. This is more stable than gradient methods.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ScoreCAMSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _scoreWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? ScoreWeights => _scoreWeights;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ScoreCAMSelector(
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
            "ScoreCAMSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        if (p == 0)
            throw new InvalidOperationException("Sequence contains no elements");

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });

        // Normalize every feature to [0, 1] as its activation map (all 0 for a constant feature)
        var min = engine.Reshape(FeatureSelectionTensorOps.ColumnMin(X), new[] { 1, p });
        var range = engine.TensorSubtract(engine.Reshape(FeatureSelectionTensorOps.ColumnMax(X), new[] { 1, p }), min);
        var varying = engine.TensorGreaterThan(range, NumOps.FromDouble(1e-10));
        var safeRange = engine.TensorWhere(varying, range, FeatureSelectionTensorOps.Filled(new[] { 1, p }, NumOps.One));
        var activationMaps = engine.TensorMultiply(
            engine.TensorDivide(engine.TensorSubtract(X, min), safeRange), engine.TensorBroadcastTo(varying, new[] { n, p }));

        // For each feature, the prediction score of the input masked (element-wise) by its activation map
        var maskedScores = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            var activationMap = engine.TensorBroadcastTo(engine.TensorSlice(activationMaps, new[] { 0, j }, new[] { n, 1 }), new[] { n, p });
            maskedScores[j] = ComputePredictionScore(engine.TensorMultiply(X, activationMap), y);
        }

        // Score-CAM weight: softmax of the non-negative masked scores
        var scores = engine.TensorMax(engine.TensorConcatenate(maskedScores, 0), NumOps.Zero);
        var shifted = engine.TensorExp(engine.TensorSubtractScalar(scores, engine.TensorMaxValue(scores)));
        var scoreWeights = engine.TensorDivideScalar(shifted, engine.TensorSum(shifted));
        _scoreWeights = scoreWeights.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scoreWeights, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// R² ([1]) of the sum of the per-feature simple regressions of y on the columns of <paramref name="X"/>
    /// (slope Sxy / Sxx, 0 when Sxx is not above 1e-10); 0 when the target has no variance.
    /// </summary>
    private Tensor<T> ComputePredictionScore(Tensor<T> X, Tensor<T> y)
    {
        var engine = Engine;
        int p = X.Shape[1];
        var centeredY = engine.TensorSubtract(y, engine.ReduceMean(y, new[] { 0 }, true));
        T ssTotal = engine.TensorSum(engine.TensorSquare(centeredY));
        if (NumOps.LessThan(ssTotal, NumOps.FromDouble(1e-10)))
            return new Tensor<T>(new[] { 1 });

        var centeredX = engine.TensorSubtract(X, engine.ReduceMean(X, new[] { 0 }, true));
        var cov = engine.TensorMatMul(engine.TensorTranspose(centeredX), centeredY); // [p, 1]
        var varX = engine.Reshape(engine.ReduceSum(engine.TensorSquare(centeredX), new[] { 0 }, false), new[] { p, 1 });
        var defined = engine.TensorGreaterThan(varX, NumOps.FromDouble(1e-10));
        var safeVarX = engine.TensorWhere(defined, varX, FeatureSelectionTensorOps.Filled(new[] { p, 1 }, NumOps.One));
        var beta = engine.TensorWhere(defined, engine.TensorDivide(cov, safeVarX), new Tensor<T>(new[] { p, 1 }));

        // pred - mean(y) = sum_j beta_j (x_j - mean_j); residual = (y - mean(y)) - that
        var residual = engine.TensorSubtract(centeredY, engine.TensorMatMul(centeredX, beta));
        T ssResid = engine.TensorSum(engine.TensorSquare(residual));
        return FeatureSelectionTensorOps.Filled(new[] { 1 }, NumOps.Subtract(NumOps.One, NumOps.Divide(ssResid, ssTotal)));
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
            throw new InvalidOperationException("ScoreCAMSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ScoreCAMSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ScoreCAMSelector has not been fitted.");

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
