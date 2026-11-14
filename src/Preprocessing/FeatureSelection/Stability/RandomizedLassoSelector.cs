using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Randomized Lasso Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines Lasso with random reweighting and subsampling to provide
/// more stable feature selection that is less sensitive to the choice
/// of regularization parameter.
/// </para>
/// <para><b>For Beginners:</b> Regular Lasso can be unstable - small changes
/// in data can lead to different features being selected. Randomized Lasso
/// runs Lasso many times with random perturbations and selects features
/// that are consistently chosen, giving more reliable results.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RandomizedLassoSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly double _alpha;
    private readonly double _sampleFraction;
    private readonly double _weaknessRange;
    private readonly int? _randomState;

    private Vector<T>? _selectionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? SelectionProbabilities => _selectionProbabilities;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public RandomizedLassoSelector(
        int nFeaturesToSelect = 10,
        int nIterations = 100,
        double alpha = 0.01,
        double sampleFraction = 0.75,
        double weaknessRange = 0.5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _alpha = alpha;
        _sampleFraction = sampleFraction;
        _weaknessRange = weaknessRange;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RandomizedLassoSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Seeded when a random state is given, so subsamples and weights are reproducible
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Normalize features: centre and divide by the sample standard deviation plus 1e-10; centre the target
        var centered = FilterStatisticTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var std = Engine.TensorAddScalar(Engine.TensorSqrt(Engine.TensorDivideScalar(
            SelectorTensorOps.ColumnSumOfSquares(centered), NumOps.FromDouble(n - 1))), PreprocessingTensorOps.Tolerance<T>(1e-10));
        var x = Engine.TensorDivide(centered, Engine.Reshape(std, new[] { 1, p }));
        var y = SelectorTensorOps.CenteredColumn(target);

        var singletons = Enumerable.Range(0, p).ToArray();
        T nonZero = PreprocessingTensorOps.Tolerance<T>(1e-10);
        // (counted in int: a 0/1 sum in T stops counting at 2048 for Half)
        var selectionCounts = new Tensor<int>(new[] { p });
        int sampleSize = (int)(n * _sampleFraction);

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Subsample
            var indices = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).Take(sampleSize).ToArray();
            var rows = new Tensor<int>(indices, new[] { indices.Length });

            // Random weights W_k in [1 - weaknessRange, 1) for features (Meinshausen and Buhlmann, 2010)
            var weights = Engine.TensorUniform<T>(new[] { 1, p }, 1 - _weaknessRange, 1, rand.Next());

            // Randomized lasso: alpha sum_k |beta_k| / W_k, i.e. the lasso on the reweighted columns X_k W_k
            var beta = RunWeightedLasso(Engine.TensorMultiply(Engine.TensorIndexSelect(x, rows, 0),
                Engine.TensorBroadcastTo(weights, new[] { indices.Length, p })), Engine.TensorIndexSelect(y, rows, 0), singletons);

            // Count selected features
            selectionCounts = Engine.TensorAdd(
                selectionCounts, Engine.TensorGreaterThan(Engine.TensorAbs(beta), nonZero).Cast<int>());
        }

        // Compute selection probabilities
        var probabilities = Engine.TensorDivideScalar(selectionCounts.Cast<double>(), (double)_nIterations).Cast<T>();
        _selectionProbabilities = probabilities.ToVector();

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(probabilities, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <summary>
    /// The lasso <c>(1/2m) ||y - X b||^2 + alpha ||b||_1</c> on the subsample; b has the support of the
    /// randomized-lasso coefficients <c>beta = W b</c>.
    /// </summary>
    private Tensor<T> RunWeightedLasso(Tensor<T> x, Tensor<T> y, int[] singletons) =>
        SparseRegressionOps.SparseGroupElasticNet(x, y, singletons, _alpha, 0, 0, 1000, 1e-6);

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
            throw new InvalidOperationException("RandomizedLassoSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RandomizedLassoSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomizedLassoSelector has not been fitted.");

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
