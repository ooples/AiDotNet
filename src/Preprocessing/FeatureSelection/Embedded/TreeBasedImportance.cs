using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class TreeBasedImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int _minSamplesPerLeaf;
    private readonly int? _randomState;

    private double[]? _importances;
    private double[]? _importanceStds;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NTrees => _nTrees;
    public double[]? Importances => _importances;
    public double[]? ImportanceStds => _importanceStds;
    public int[]? SelectedIndices => _selectedIndices;
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

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TreeBasedImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

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

        // Initialize importances
        var treeImportances = new double[_nTrees, p];

        // Build forest and collect importances
        for (int t = 0; t < _nTrees; t++)
        {
            // Bootstrap sample
            var sampleIndices = new int[n];
            for (int i = 0; i < n; i++)
                sampleIndices[i] = random.Next(n);

            // Feature importance from this tree
            var featureImportance = new double[p];
            BuildTree(data, target, sampleIndices, featureImportance, 0, random);

            // Normalize and store
            double totalImportance = featureImportance.Sum();
            for (int j = 0; j < p; j++)
                treeImportances[t, j] = totalImportance > 1e-10 ? featureImportance[j] / totalImportance : 0;
        }

        // Compute mean and std of importances across trees
        _importances = new double[p];
        _importanceStds = new double[p];

        for (int j = 0; j < p; j++)
        {
            double sum = 0, sumSq = 0;
            for (int t = 0; t < _nTrees; t++)
            {
                sum += treeImportances[t, j];
                sumSq += treeImportances[t, j] * treeImportances[t, j];
            }
            _importances[j] = sum / _nTrees;
            _importanceStds[j] = Math.Sqrt(sumSq / _nTrees - _importances[j] * _importances[j]);
        }

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _importances
            .Select((imp, idx) => (Importance: imp, Index: idx))
            .OrderByDescending(x => x.Importance)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private void BuildTree(Matrix<T> data, Vector<T> target, int[] sampleIndices,
        double[] featureImportance, int depth, Random random)
    {
        if (depth >= _maxDepth || sampleIndices.Length < _minSamplesPerLeaf * 2)
            return;

        int n = sampleIndices.Length;
        int p = data.Columns;

        // Random feature subset (sqrt(p) features)
        int nFeaturesToTry = Math.Max(1, (int)Math.Sqrt(p));
        var featureSubset = Enumerable.Range(0, p)
            .OrderBy(_ => random.Next())
            .Take(nFeaturesToTry)
            .ToArray();

        // Find best split
        double bestGain = 0;
        int bestFeature = -1;
        double bestThreshold = 0;

        double parentGini = ComputeGini(target, sampleIndices);

        foreach (int feature in featureSubset)
        {
            // Get unique values for split points
            var values = sampleIndices.Select(i => NumOps.ToDouble(data[i, feature]))
                .Distinct()
                .OrderBy(v => v)
                .ToArray();

            for (int i = 0; i < values.Length - 1; i++)
            {
                double threshold = (values[i] + values[i + 1]) / 2;

                var leftIndices = sampleIndices.Where(idx => NumOps.ToDouble(data[idx, feature]) <= threshold).ToArray();
                var rightIndices = sampleIndices.Where(idx => NumOps.ToDouble(data[idx, feature]) > threshold).ToArray();

                if (leftIndices.Length < _minSamplesPerLeaf || rightIndices.Length < _minSamplesPerLeaf)
                    continue;

                double leftGini = ComputeGini(target, leftIndices);
                double rightGini = ComputeGini(target, rightIndices);

                double weightedGini = (leftIndices.Length * leftGini + rightIndices.Length * rightGini) / n;
                double gain = parentGini - weightedGini;

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature < 0)
            return;

        // Record importance as weighted impurity decrease
        featureImportance[bestFeature] += bestGain * n;

        // Recursively build subtrees
        var leftSplit = sampleIndices.Where(idx => NumOps.ToDouble(data[idx, bestFeature]) <= bestThreshold).ToArray();
        var rightSplit = sampleIndices.Where(idx => NumOps.ToDouble(data[idx, bestFeature]) > bestThreshold).ToArray();

        BuildTree(data, target, leftSplit, featureImportance, depth + 1, random);
        BuildTree(data, target, rightSplit, featureImportance, depth + 1, random);
    }

    private double ComputeGini(Vector<T> target, int[] indices)
    {
        if (indices.Length == 0) return 0;

        var classCounts = new Dictionary<int, int>();
        foreach (int i in indices)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classCounts.ContainsKey(label))
                classCounts[label] = 0;
            classCounts[label]++;
        }

        double gini = 1.0;
        foreach (int count in classCounts.Values)
        {
            double prob = (double)count / indices.Length;
            gini -= prob * prob;
        }

        return gini;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TreeBasedImportance has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TreeBasedImportance does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TreeBasedImportance has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
