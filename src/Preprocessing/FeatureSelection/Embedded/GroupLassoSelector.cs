using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Group Lasso feature selection for selecting groups of related features.
/// </summary>
/// <remarks>
/// <para>
/// Extends Lasso to select entire groups of features together rather than individual
/// features. Useful when features naturally belong to groups (e.g., dummy variables
/// for a categorical feature, or related gene pathways).
/// </para>
/// <para><b>For Beginners:</b> Sometimes features come in natural groups. For example,
/// if you have a categorical variable converted to multiple dummy columns, you want
/// to keep or remove ALL columns for that category together. Group Lasso ensures
/// related features are selected or dropped as a unit.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GroupLassoSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _nGroupsToSelect;
    private readonly int[][]? _groups;
    private readonly double _alpha;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private Vector<T>? _groupNorms;
    private int[]? _selectedGroups;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NGroupsToSelect => _nGroupsToSelect;
    public Vector<T>? GroupNorms => _groupNorms;
    public int[]? SelectedGroups => _selectedGroups;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GroupLassoSelector(
        int nGroupsToSelect = 5,
        int[][]? groups = null,
        double alpha = 1.0,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nGroupsToSelect < 1)
            throw new ArgumentException("Number of groups must be at least 1.", nameof(nGroupsToSelect));

        _nGroupsToSelect = nGroupsToSelect;
        _groups = groups;
        _alpha = alpha;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GroupLassoSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Create default groups if not provided (one feature per group)
        var groups = _groups ?? Enumerable.Range(0, p).Select(i => new[] { i }).ToArray();
        int nGroups = groups.Length;

        // Validate groups
        if (nGroups == 0)
            throw new ArgumentException("At least one group must be defined.");

        var seenIndices = new HashSet<int>();
        foreach (var group in groups)
        {
            if (group.Length == 0)
                throw new ArgumentException("Groups cannot be empty.");

            foreach (int idx in group)
            {
                if (idx < 0 || idx >= p)
                    throw new ArgumentException($"Group index {idx} is out of bounds [0, {p}).");
                if (!seenIndices.Add(idx))
                    throw new ArgumentException($"Feature index {idx} appears in multiple groups.");
            }
        }

        var engine = Engine;

        // Standardize data (population std; a near-zero std is replaced by 1) and center the target
        var X = FeatureScoringTensorOps.Standardize(Tensor<T>.FromMatrix(data), out _);
        var residuals = FeatureScoringTensorOps.Center(Tensor<T>.FromVector(target));

        // Per-group column blocks, single columns and penalty thresholds alpha * sqrt(|group|)
        T one = NumOps.One;
        T zero = NumOps.Zero;
        T count = NumOps.FromDouble(n);
        T epsilon = PreprocessingTensorOps.Tolerance<T>(1e-10);
        T alpha = NumOps.FromDouble(_alpha);
        T tol = NumOps.FromDouble(_tolerance);
        var columnShape = new[] { n };
        var groupBlocks = new Tensor<T>[nGroups];
        var groupThresholds = new Tensor<T>[nGroups];
        var columns = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
            columns[j] = PreprocessingTensorOps.ColumnVector(X, j);
        for (int g = 0; g < nGroups; g++)
        {
            groupBlocks[g] = engine.TensorIndexSelect(X, new Tensor<int>(groups[g], new[] { groups[g].Length }), 1);
            groupThresholds[g] = engine.TensorMultiplyScalar(engine.TensorSqrt(engine.ReduceSum(
                PreprocessingTensorOps.Filled(new[] { groups[g].Length }, one), new[] { 0 }, true)), alpha);
        }

        // Block coordinate descent; beta is kept as one [1] tensor per feature
        var beta = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
            beta[j] = new Tensor<T>(new[] { 1 });

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var maxChange = new Tensor<T>(new[] { 1 });

            for (int g = 0; g < nGroups; g++)
            {
                var group = groups[g];
                int groupSize = group.Length;
                var block = groupBlocks[g];

                // Gradient X_G' r for this group, then the group soft-thresholded step
                var gradient = engine.ReduceSum(engine.TensorMultiply(block, engine.TensorBroadcastTo(
                    engine.Reshape(residuals, new[] { n, 1 }), new[] { n, groupSize })), new[] { 0 }, false);
                var old = engine.TensorConcatenate(group.Select(j => beta[j]).ToArray(), 0);
                var step = engine.TensorAdd(old, engine.TensorDivideScalar(gradient, count));
                var gradNorm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(step), new[] { 0 }, true));
                var scale = engine.TensorMax(engine.TensorAddScalar(engine.TensorNegate(
                    engine.TensorDivide(groupThresholds[g], engine.TensorAddScalar(gradNorm, epsilon))), one), zero);
                var updated = engine.TensorMultiply(step, engine.TensorBroadcastTo(scale, new[] { groupSize }));

                // Update coefficients and residuals one feature at a time
                for (int k = 0; k < groupSize; k++)
                {
                    int j = group[k];
                    var newBeta = engine.TensorSlice(updated, new[] { k }, new[] { 1 });
                    var delta = engine.TensorSubtract(newBeta, beta[j]);
                    beta[j] = newBeta;
                    residuals = engine.TensorSubtract(residuals,
                        engine.TensorMultiply(engine.TensorBroadcastTo(delta, columnShape), columns[j]));
                    maxChange = engine.TensorMax(maxChange, engine.TensorAbs(delta));
                }
            }

            if (PreprocessingTensorOps.IsBelow(engine.TensorMaxValue(maxChange), tol))
                break;
        }

        // Compute group norms
        var groupNorms = engine.TensorConcatenate(groups.Select(group => engine.TensorSqrt(engine.ReduceSum(
            engine.TensorSquare(engine.TensorConcatenate(group.Select(j => beta[j]).ToArray(), 0)),
            new[] { 0 }, true))).ToArray(), 0);
        _groupNorms = groupNorms.ToVector();
        // Select top groups
        int nToSelect = Math.Min(_nGroupsToSelect, nGroups);
        _selectedGroups = PreprocessingTensorOps.TopKIndicesSorted(groupNorms, nToSelect);

        // Collect all selected feature indices
        var selectedSet = new HashSet<int>();
        foreach (int g in _selectedGroups)
            foreach (int j in groups[g])
                selectedSet.Add(j);

        _selectedIndices = selectedSet.OrderBy(x => x).ToArray();
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
            throw new InvalidOperationException("GroupLassoSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GroupLassoSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GroupLassoSelector has not been fitted.");

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
