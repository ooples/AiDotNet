using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Density;

/// <summary>
/// Local Density based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their contribution to local density patterns,
/// identifying features that create meaningful local neighborhoods.
/// </para>
/// <para><b>For Beginners:</b> Local density measures how crowded the area around
/// each data point is. Features that create clear differences in local density
/// between classes help separate the data better.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LocalDensitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;

    private Vector<T>? _densityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public Vector<T>? DensityScores => _densityScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LocalDensitySelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LocalDensitySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var classValues = engine.TensorRound(Tensor<T>.FromVector(target));
        int k = Math.Min(_nNeighbors, n - 1);

        // Standardize features (population std; a feature with std not above 1e-10 is left as is)
        var mean = engine.ReduceMean(X, new[] { 0 }, true);
        var centered = engine.TensorSubtract(X, engine.TensorBroadcastTo(mean, new[] { n, p }));
        var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, true));
        var standardized = engine.TensorWhere(
            engine.TensorBroadcastTo(engine.TensorGreaterThan(std, NumOps.FromDouble(1e-10)), new[] { n, p }),
            engine.TensorDivide(centered, engine.TensorBroadcastTo(std, new[] { n, p })), X);

        // Compute local density for each point (1/average distance to k nearest neighbors)
        var pairShape = new[] { n, n };
        var self = engine.TensorEye<T>(n);
        var excluded = PreprocessingTensorOps.Filled(pairShape, NumOps.MaxValue);
        T neighbors = NumOps.FromDouble(k);
        T epsilon = NumOps.FromDouble(1e-10);
        var ones = PreprocessingTensorOps.Filled(new[] { n }, NumOps.One);
        var densities = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            var col = PreprocessingTensorOps.ColumnVector(standardized, j);
            var distances = engine.TensorAbs(engine.TensorSubtract(
                engine.TensorBroadcastTo(engine.Reshape(col, new[] { n, 1 }), pairShape),
                engine.TensorBroadcastTo(engine.Reshape(col, new[] { 1, n }), pairShape)));
            var sorted = engine.TensorSort(engine.TensorWhere(self, excluded, distances), 1, false).Item1;
            var nearest = engine.TensorSlice(sorted, new[] { 0, 0 }, new[] { n, k });
            var avgDist = engine.TensorDivideScalar(engine.ReduceSum(nearest, new[] { 1 }, false), neighbors);
            densities[j] = engine.TensorDivide(ones, engine.TensorAddScalar(avgDist, epsilon));
        }

        // Compute |correlation| between local density and class label
        var scores = p == 0
            ? new Tensor<T>(new[] { 0 })
            : engine.TensorAbs(PreprocessingTensorOps.CorrelationsWithDenominatorGuard(engine.TensorStack(densities, 1), classValues));
        _densityScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

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
            throw new InvalidOperationException("LocalDensitySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LocalDensitySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LocalDensitySelector has not been fitted.");

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
