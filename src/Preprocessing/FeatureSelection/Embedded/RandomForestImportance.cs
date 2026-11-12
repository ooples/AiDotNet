using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Random Forest feature importance for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Builds a simplified random forest and measures feature importance based on
/// how much each feature contributes to reducing prediction error (Gini importance
/// or permutation importance).
/// </para>
/// <para><b>For Beginners:</b> Random forests are ensembles of decision trees.
/// When building each tree, features that better split the data are used more
/// often and at higher positions. Feature importance is measured by how much
/// each feature helps reduce errors across all trees.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RandomForestImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int _minSamplesLeaf;
    private readonly int? _randomState;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NTrees => _nTrees;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public RandomForestImportance(
        int nFeaturesToSelect = 10,
        int nTrees = 100,
        int maxDepth = 5,
        int minSamplesLeaf = 5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nTrees < 1)
            throw new ArgumentException("Number of trees must be at least 1.", nameof(nTrees));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _minSamplesLeaf = minSamplesLeaf;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RandomForestImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        T treeCount = NumOps.FromDouble(_nTrees);
        var featureImportances = new Tensor<T>(new[] { p });

        // Build trees and accumulate importances
        for (int t = 0; t < _nTrees; t++)
        {
            // Bootstrap sample
            var sampleIndices = new int[n];
            for (int i = 0; i < n; i++)
                sampleIndices[i] = rand.Next(n);

            // Random feature subset
            int nFeaturesToConsider = (int)Math.Sqrt(p);
            var featureSubset = Enumerable.Range(0, p)
                .OrderBy(_ => rand.NextDouble())
                .Take(nFeaturesToConsider)
                .ToArray();

            // Build tree and get importances
            var splitFeatures = new List<int>();
            var splitGains = new List<Tensor<T>>();
            BuildTree(X, y, sampleIndices, featureSubset, 0, splitFeatures, splitGains);
            var treeImportances = FeatureSelectionTensorOps.ScatterAdd(splitFeatures, splitGains, p);

            featureImportances = engine.TensorAdd(featureImportances, engine.TensorDivideScalar(treeImportances, treeCount));
        }

        // Normalize importances
        T totalImportance = engine.TensorSum(featureImportances);
        if (PreprocessingTensorOps.IsAbove(totalImportance, NumOps.Zero))
            featureImportances = engine.TensorDivideScalar(featureImportances, totalImportance);
        _featureImportances = featureImportances.ToVector();

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(featureImportances, nToSelect);

        IsFitted = true;
    }

    private void BuildTree(Tensor<T> X, Tensor<T> y, int[] indices, int[] features, int depth,
        List<int> splitFeatures, List<Tensor<T>> splitGains)
    {
        if (depth >= _maxDepth || indices.Length < 2 * _minSamplesLeaf)
            return;

        // Compute current variance (or impurity)
        var engine = Engine;
        var rows = new Tensor<int>(indices, new[] { indices.Length });
        var nodeY = engine.TensorIndexSelect(y, rows, 0);
        T currentVar = FeatureSelectionTensorOps.SumSquaredDeviations(nodeY);

        if (PreprocessingTensorOps.IsBelow(currentVar, NumOps.FromDouble(1e-10)))
            return;

        // Find best split (variance reduction)
        var nodeX = engine.TensorIndexSelect(engine.TensorIndexSelect(X, rows, 0),
            new Tensor<int>(features, new[] { features.Length }), 1);
        if (!FeatureSelectionTensorOps.TryBestSplit(nodeX, nodeY, SplitCriterion.SumOfSquares, currentVar, _minSamplesLeaf,
            out int bestColumn, out T bestThreshold, out var bestGain))
            return;

        splitFeatures.Add(features[bestColumn]);
        splitGains.Add(bestGain);

        FeatureSelectionTensorOps.PartitionRows(PreprocessingTensorOps.ColumnVector(nodeX, bestColumn), bestThreshold,
            indices, out var leftIndices, out var rightIndices);

        // Recurse
        BuildTree(X, y, leftIndices, features, depth + 1, splitFeatures, splitGains);
        BuildTree(X, y, rightIndices, features, depth + 1, splitFeatures, splitGains);
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
            throw new InvalidOperationException("RandomForestImportance has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RandomForestImportance does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomForestImportance has not been fitted.");

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
