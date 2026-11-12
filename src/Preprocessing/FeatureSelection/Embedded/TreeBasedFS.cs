using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Tree-Based Feature Selection using decision tree splits.
/// </summary>
/// <remarks>
/// <para>
/// Builds a simple decision tree and measures feature importance by how much
/// each feature improves predictions when used for splitting. Features that
/// create better splits are considered more important.
/// </para>
/// <para><b>For Beginners:</b> Decision trees ask questions like "is feature X
/// greater than value V?" to split data. Features that lead to purer groups
/// (where samples in each group are mostly the same class) are more important.
/// This method finds which features make the best splitting questions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TreeBasedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxDepth;
    private readonly int _minSamplesLeaf;
    private readonly int? _randomState;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxDepth => _maxDepth;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TreeBasedFS(
        int nFeaturesToSelect = 10,
        int maxDepth = 5,
        int minSamplesLeaf = 5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxDepth < 1)
            throw new ArgumentException("Max depth must be at least 1.", nameof(maxDepth));
        if (minSamplesLeaf < 1)
            throw new ArgumentException("Min samples per leaf must be at least 1.", nameof(minSamplesLeaf));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxDepth = maxDepth;
        _minSamplesLeaf = minSamplesLeaf;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TreeBasedFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Class one-hot of the rounded target for the Gini splits
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        var classOneHot = PreprocessingTensorOps.ClassOneHot<T>(classIndex, classes.Length);
        var X = Tensor<T>.FromMatrix(data);

        // Build tree and accumulate importances
        var splitFeatures = new List<int>();
        var splitGains = new List<Tensor<T>>();
        BuildTree(X, classOneHot, Enumerable.Range(0, n).ToArray(), 0, random, splitFeatures, splitGains);
        var featureImportances = FeatureSelectionTensorOps.ScatterAdd(splitFeatures, splitGains, p);

        // Normalize importances
        T totalImportance = Engine.TensorSum(featureImportances);
        if (PreprocessingTensorOps.IsAbove(totalImportance, NumOps.FromDouble(1e-10)))
            featureImportances = Engine.TensorDivideScalar(featureImportances, totalImportance);
        _featureImportances = featureImportances.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(featureImportances, numToSelect);

        IsFitted = true;
    }

    private void BuildTree(Tensor<T> data, Tensor<T> classOneHot, int[] indices, int depth, Random random,
        List<int> splitFeatures, List<Tensor<T>> splitGains)
    {
        if (indices.Length < _minSamplesLeaf * 2 || depth >= _maxDepth)
            return;

        var engine = Engine;
        int p = data.Shape[1];
        var rows = new Tensor<int>(indices, new[] { indices.Length });
        var nodeClasses = engine.TensorIndexSelect(classOneHot, rows, 0);
        T parentImpurity = FeatureSelectionTensorOps.GiniImpurity(nodeClasses);

        // Sample a subset of features
        int nTry = Math.Max(1, (int)Math.Sqrt(p));
        var candidateFeatures = Enumerable.Range(0, p)
            .OrderBy(_ => random.Next())
            .Take(nTry)
            .ToArray();

        // Find best split (Gini decrease) over every boundary between consecutive distinct values
        var nodeX = engine.TensorIndexSelect(engine.TensorIndexSelect(data, rows, 0),
            new Tensor<int>(candidateFeatures, new[] { candidateFeatures.Length }), 1);
        if (!FeatureSelectionTensorOps.TryBestSplit(nodeX, nodeClasses, SplitCriterion.Gini, parentImpurity, _minSamplesLeaf,
            out int bestColumn, out T bestThreshold, out var bestGain))
            return;

        splitFeatures.Add(candidateFeatures[bestColumn]);
        splitGains.Add(engine.TensorMultiplyScalar(bestGain, NumOps.FromDouble(indices.Length)));

        FeatureSelectionTensorOps.PartitionRows(PreprocessingTensorOps.ColumnVector(nodeX, bestColumn), bestThreshold,
            indices, out var left, out var right);

        BuildTree(data, classOneHot, left, depth + 1, random, splitFeatures, splitGains);
        BuildTree(data, classOneHot, right, depth + 1, random, splitFeatures, splitGains);
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
            throw new InvalidOperationException("TreeBasedFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TreeBasedFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TreeBasedFS has not been fitted.");

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
