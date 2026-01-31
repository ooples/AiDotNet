using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class RandomForestImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int _minSamplesLeaf;
    private readonly int? _randomState;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NTrees => _nTrees;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
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

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RandomForestImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

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

        _featureImportances = new double[p];

        // Convert data to arrays
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
            y[i] = NumOps.ToDouble(target[i]);
        }

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
            var treeImportances = BuildTree(X, y, sampleIndices, featureSubset, 0, rand);

            for (int j = 0; j < p; j++)
                _featureImportances[j] += treeImportances[j] / _nTrees;
        }

        // Normalize importances
        double totalImportance = _featureImportances.Sum();
        if (totalImportance > 0)
            for (int j = 0; j < p; j++)
                _featureImportances[j] /= totalImportance;

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _featureImportances
            .Select((imp, idx) => (Importance: imp, Index: idx))
            .OrderByDescending(x => x.Importance)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] BuildTree(double[,] X, double[] y, int[] indices, int[] features, int depth, Random rand)
    {
        int p = X.GetLength(1);
        var importances = new double[p];

        if (depth >= _maxDepth || indices.Length < 2 * _minSamplesLeaf)
            return importances;

        // Compute current variance (or impurity)
        double currentMean = indices.Average(i => y[i]);
        double currentVar = indices.Sum(i => Math.Pow(y[i] - currentMean, 2));

        if (currentVar < 1e-10)
            return importances;

        // Find best split
        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGain = 0;

        foreach (int f in features)
        {
            var values = indices.Select(i => X[i, f]).Distinct().OrderBy(v => v).ToArray();

            for (int t = 0; t < values.Length - 1; t++)
            {
                double threshold = (values[t] + values[t + 1]) / 2;

                var leftIndices = indices.Where(i => X[i, f] <= threshold).ToArray();
                var rightIndices = indices.Where(i => X[i, f] > threshold).ToArray();

                if (leftIndices.Length < _minSamplesLeaf || rightIndices.Length < _minSamplesLeaf)
                    continue;

                double leftMean = leftIndices.Average(i => y[i]);
                double rightMean = rightIndices.Average(i => y[i]);
                double leftVar = leftIndices.Sum(i => Math.Pow(y[i] - leftMean, 2));
                double rightVar = rightIndices.Sum(i => Math.Pow(y[i] - rightMean, 2));

                double gain = currentVar - leftVar - rightVar;

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = f;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature >= 0)
        {
            importances[bestFeature] += bestGain;

            var leftIndices = indices.Where(i => X[i, bestFeature] <= bestThreshold).ToArray();
            var rightIndices = indices.Where(i => X[i, bestFeature] > bestThreshold).ToArray();

            // Recurse
            var leftImportances = BuildTree(X, y, leftIndices, features, depth + 1, rand);
            var rightImportances = BuildTree(X, y, rightIndices, features, depth + 1, rand);

            for (int j = 0; j < p; j++)
            {
                importances[j] += leftImportances[j];
                importances[j] += rightImportances[j];
            }
        }

        return importances;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomForestImportance has not been fitted.");

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
        throw new NotSupportedException("RandomForestImportance does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomForestImportance has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
