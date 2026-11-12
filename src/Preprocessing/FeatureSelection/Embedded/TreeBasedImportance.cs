using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Tree-based feature importance using random forest mean decrease impurity.
/// </summary>
/// <remarks>
/// <para>
/// Tree-based importance measures the average reduction in impurity (Gini or entropy)
/// achieved by splits on each feature across all trees in a random forest. Features
/// that are used for impactful splits get higher importance scores.
/// </para>
/// <para><b>For Beginners:</b> Random forests build many decision trees, and each tree
/// splits the data at various points. When a feature is used for a split, it reduces
/// the "messiness" (impurity) of the data. Features that consistently help clean up
/// the mess get high importance scores.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TreeBasedImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int _minSamplesPerLeaf;
    private readonly int? _randomState;

    private Vector<T>? _importances;
    private Vector<T>? _importanceStds;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NTrees => _nTrees;
    public Vector<T>? Importances => _importances;
    public Vector<T>? ImportanceStds => _importanceStds;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TreeBasedImportance(
        int nFeaturesToSelect = 10,
        int nTrees = 100,
        int maxDepth = 5,
        int minSamplesPerLeaf = 5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _minSamplesPerLeaf = minSamplesPerLeaf;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TreeBasedImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        T floor = PreprocessingTensorOps.Tolerance<T>(1e-10);

        // Class one-hot of the rounded target for the Gini splits
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        var classOneHot = PreprocessingTensorOps.ClassOneHot<T>(classIndex, classes.Length);

        // Build forest and collect normalized importances per tree
        var treeImportances = new Tensor<T>[_nTrees];
        for (int t = 0; t < _nTrees; t++)
        {
            // Bootstrap sample
            var sampleIndices = new int[n];
            for (int i = 0; i < n; i++)
                sampleIndices[i] = random.Next(n);

            // Feature importance from this tree
            var splitFeatures = new List<int>();
            var splitGains = new List<Tensor<T>>();
            BuildTree(X, classOneHot, sampleIndices, splitFeatures, splitGains, 0, random);
            var featureImportance = FeatureSelectionTensorOps.ScatterAdd(splitFeatures, splitGains, p);

            // Normalize and store
            T totalImportance = engine.TensorSum(featureImportance);
            treeImportances[t] = engine.Reshape(PreprocessingTensorOps.IsAbove(totalImportance, floor)
                ? engine.TensorDivideScalar(featureImportance, totalImportance)
                : new Tensor<T>(new[] { p }), new[] { 1, p });
        }

        // Compute mean and (population) std of importances across trees
        // (no trees: 0 / 0, i.e. NaN, as the plain sums give)
        Tensor<T> importances;
        if (_nTrees > 0)
        {
            var stacked = engine.TensorConcatenate(treeImportances, 0);
            importances = engine.ReduceMean(stacked, new[] { 0 }, false);
            // Two-pass: E[x^2] - E[x]^2 cancels to a tiny negative value (sqrt gives NaN) when the trees agree
            var deviations = engine.TensorSubtract(stacked,
                engine.TensorBroadcastTo(engine.Reshape(importances, new[] { 1, p }), stacked.Shape.ToArray()));
            _importanceStds = engine.TensorSqrt(
                engine.ReduceMean(engine.TensorSquare(deviations), new[] { 0 }, false)).ToVector();
        }
        else
        {
            importances = PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(double.NaN));
            _importanceStds = importances.ToVector();
        }

        _importances = importances.ToVector();

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importances, numToSelect);

        IsFitted = true;
    }

    private void BuildTree(Tensor<T> data, Tensor<T> classOneHot, int[] sampleIndices,
        List<int> splitFeatures, List<Tensor<T>> splitGains, int depth, Random random)
    {
        if (depth >= _maxDepth || sampleIndices.Length < _minSamplesPerLeaf * 2)
            return;

        var engine = Engine;
        int n = sampleIndices.Length;
        int p = data.Shape[1];

        // Random feature subset (sqrt(p) features)
        int nFeaturesToTry = Math.Max(1, (int)Math.Sqrt(p));
        var featureSubset = Enumerable.Range(0, p)
            .OrderBy(_ => random.Next())
            .Take(nFeaturesToTry)
            .ToArray();

        // Find best split (Gini decrease)
        var rows = new Tensor<int>(sampleIndices, new[] { n });
        var nodeClasses = engine.TensorIndexSelect(classOneHot, rows, 0);
        T parentGini = FeatureSelectionTensorOps.GiniImpurity(nodeClasses);
        var nodeX = engine.TensorIndexSelect(engine.TensorIndexSelect(data, rows, 0),
            new Tensor<int>(featureSubset, new[] { featureSubset.Length }), 1);
        if (!FeatureSelectionTensorOps.TryBestSplit(nodeX, nodeClasses, SplitCriterion.Gini, parentGini, _minSamplesPerLeaf,
            out int bestColumn, out T bestThreshold, out var bestGain))
            return;

        // Record importance as weighted impurity decrease
        splitFeatures.Add(featureSubset[bestColumn]);
        splitGains.Add(engine.TensorMultiplyScalar(bestGain, NumOps.FromDouble(n)));

        // Recursively build subtrees
        FeatureSelectionTensorOps.PartitionRows(PreprocessingTensorOps.ColumnVector(nodeX, bestColumn), bestThreshold,
            sampleIndices, out var leftSplit, out var rightSplit);

        BuildTree(data, classOneHot, leftSplit, splitFeatures, splitGains, depth + 1, random);
        BuildTree(data, classOneHot, rightSplit, splitFeatures, splitGains, depth + 1, random);
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
            throw new InvalidOperationException("TreeBasedImportance has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TreeBasedImportance does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TreeBasedImportance has not been fitted.");

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
