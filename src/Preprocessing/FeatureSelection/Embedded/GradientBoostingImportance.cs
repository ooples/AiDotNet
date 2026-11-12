using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Gradient Boosting-based feature importance using iterative boosting.
/// </summary>
/// <remarks>
/// <para>
/// Gradient Boosting importance measures how often each feature is used for splits
/// and how much those splits contribute to model improvement across all boosting
/// rounds. Features that are frequently used for impactful splits score higher.
/// </para>
/// <para><b>For Beginners:</b> Gradient Boosting builds trees sequentially, where
/// each tree tries to fix the errors of previous trees. The features that help the
/// most in reducing these errors across all trees are considered important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GradientBoostingImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nEstimators;
    private readonly int _maxDepth;
    private readonly double _learningRate;
    private readonly int? _randomState;

    private Vector<T>? _importances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NEstimators => _nEstimators;
    public double LearningRate => _learningRate;
    public Vector<T>? Importances => _importances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GradientBoostingImportance(
        int nFeaturesToSelect = 10,
        int nEstimators = 100,
        int maxDepth = 3,
        double learningRate = 0.1,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nEstimators = nEstimators;
        _maxDepth = maxDepth;
        _learningRate = learningRate;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GradientBoostingImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var y = Tensor<T>.FromVector(target);
        T learningRate = NumOps.FromDouble(_learningRate);
        T floor = NumOps.FromDouble(1e-10);

        // Initialize importances
        var importances = new Tensor<T>(new[] { p });

        // Initialize predictions with mean
        var predictions = PreprocessingTensorOps.Filled(new[] { n }, engine.TensorMean(y));

        // Boosting rounds
        for (int round = 0; round < _nEstimators; round++)
        {
            // Compute residuals (negative gradient for squared loss)
            var residuals = engine.TensorSubtract(y, predictions);

            // Fit tree to residuals and collect importance
            var treePredictions = BuildRegressionTree(X, residuals, n, p, random, out var treeImportance);

            // Update predictions
            predictions = engine.TensorAdd(predictions, engine.TensorMultiplyScalar(treePredictions, learningRate));

            // Accumulate importance
            T totalImp = engine.TensorSum(treeImportance);
            if (PreprocessingTensorOps.IsAbove(totalImp, floor))
                importances = engine.TensorAdd(importances, engine.TensorDivideScalar(treeImportance, totalImp));
        }

        // Normalize final importances
        T total = engine.TensorSum(importances);
        if (PreprocessingTensorOps.IsAbove(total, floor))
            importances = engine.TensorDivideScalar(importances, total);
        _importances = importances.ToVector();

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importances, numToSelect);

        IsFitted = true;
    }

    private Tensor<T> BuildRegressionTree(Tensor<T> data, Tensor<T> residuals, int n, int p,
        Random random, out Tensor<T> importance)
    {
        var leafOf = new int[n];
        var leafMeans = new List<Tensor<T>>();
        var splitFeatures = new List<int>();
        var splitGains = new List<Tensor<T>>();

        BuildNode(data, residuals, Enumerable.Range(0, n).ToArray(), leafOf, leafMeans, splitFeatures, splitGains, 0, random);

        importance = FeatureSelectionTensorOps.ScatterAdd(splitFeatures, splitGains, p);
        if (leafMeans.Count == 0)
            return new Tensor<T>(new[] { n });

        // Every row predicts the mean residual of its leaf
        return Engine.TensorIndexSelect(Engine.TensorConcatenate(leafMeans.ToArray(), 0), new Tensor<int>(leafOf, new[] { n }), 0);
    }

    private void BuildNode(Tensor<T> data, Tensor<T> residuals, int[] indices, int[] leafOf, List<Tensor<T>> leafMeans,
        List<int> splitFeatures, List<Tensor<T>> splitGains, int depth, Random random)
    {
        var engine = Engine;
        var rows = new Tensor<int>(indices, new[] { indices.Length });
        var nodeResiduals = engine.TensorIndexSelect(residuals, rows, 0);

        if (depth >= _maxDepth || indices.Length < 4)
        {
            // Leaf: predict mean residual
            AddLeaf(nodeResiduals, indices, leafOf, leafMeans);
            return;
        }

        int p = data.Shape[1];

        // Random feature subset
        int nFeaturesToTry = Math.Max(1, (int)Math.Sqrt(p));
        var featureSubset = Enumerable.Range(0, p)
            .OrderBy(_ => random.Next())
            .Take(nFeaturesToTry)
            .ToArray();

        // Find best split (variance reduction, at least 2 rows per side)
        var nodeX = engine.TensorIndexSelect(engine.TensorIndexSelect(data, rows, 0),
            new Tensor<int>(featureSubset, new[] { featureSubset.Length }), 1);
        T parentVar = FeatureSelectionTensorOps.SumSquaredDeviations(nodeResiduals);
        if (!FeatureSelectionTensorOps.TryBestSplit(nodeX, nodeResiduals, SplitCriterion.SumOfSquares, parentVar, 2,
            out int bestColumn, out T bestThreshold, out var bestGain))
        {
            AddLeaf(nodeResiduals, indices, leafOf, leafMeans);
            return;
        }

        // Record importance
        splitFeatures.Add(featureSubset[bestColumn]);
        splitGains.Add(bestGain);

        // Split and recurse
        FeatureSelectionTensorOps.PartitionRows(PreprocessingTensorOps.ColumnVector(nodeX, bestColumn), bestThreshold,
            indices, out var left, out var right);

        BuildNode(data, residuals, left, leafOf, leafMeans, splitFeatures, splitGains, depth + 1, random);
        BuildNode(data, residuals, right, leafOf, leafMeans, splitFeatures, splitGains, depth + 1, random);
    }

    private void AddLeaf(Tensor<T> nodeResiduals, int[] indices, int[] leafOf, List<Tensor<T>> leafMeans)
    {
        int leaf = leafMeans.Count;
        leafMeans.Add(Engine.ReduceMean(nodeResiduals, new[] { 0 }, true));
        foreach (int i in indices)
            leafOf[i] = leaf;
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
            throw new InvalidOperationException("GradientBoostingImportance has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GradientBoostingImportance does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GradientBoostingImportance has not been fitted.");

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
