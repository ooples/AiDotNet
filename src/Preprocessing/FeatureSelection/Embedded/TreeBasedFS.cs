using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class TreeBasedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxDepth;
    private readonly int _minSamplesLeaf;
    private readonly int? _randomState;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxDepth => _maxDepth;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
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

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TreeBasedFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _featureImportances = new double[p];
        var sampleIndices = Enumerable.Range(0, n).ToList();

        // Build tree and accumulate importances
        BuildTree(data, target, sampleIndices, 0, random);

        // Normalize importances
        double totalImportance = _featureImportances.Sum();
        if (totalImportance > 1e-10)
        {
            for (int j = 0; j < p; j++)
                _featureImportances[j] /= totalImportance;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureImportances[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private void BuildTree(Matrix<T> data, Vector<T> target, List<int> indices, int depth, Random random)
    {
        if (indices.Count < _minSamplesLeaf * 2 || depth >= _maxDepth)
            return;

        int p = data.Columns;

        // Find best split
        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGain = 0;

        double parentImpurity = ComputeGini(target, indices);

        // Sample a subset of features
        int nTry = Math.Max(1, (int)Math.Sqrt(p));
        var candidateFeatures = Enumerable.Range(0, p)
            .OrderBy(_ => random.Next())
            .Take(nTry)
            .ToList();

        foreach (int j in candidateFeatures)
        {
            var values = indices.Select(i => NumOps.ToDouble(data[i, j])).Distinct().OrderBy(v => v).ToList();
            if (values.Count < 2) continue;

            foreach (double threshold in values.Skip(1).Take(values.Count - 2))
            {
                var leftIndices = indices.Where(i => NumOps.ToDouble(data[i, j]) <= threshold).ToList();
                var rightIndices = indices.Where(i => NumOps.ToDouble(data[i, j]) > threshold).ToList();

                if (leftIndices.Count < _minSamplesLeaf || rightIndices.Count < _minSamplesLeaf)
                    continue;

                double leftImpurity = ComputeGini(target, leftIndices);
                double rightImpurity = ComputeGini(target, rightIndices);

                double weightedImpurity = (leftIndices.Count * leftImpurity + rightIndices.Count * rightImpurity) / indices.Count;
                double gain = parentImpurity - weightedImpurity;

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = j;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature < 0 || _featureImportances is null)
            return;

        _featureImportances[bestFeature] += bestGain * indices.Count;

        var left = indices.Where(i => NumOps.ToDouble(data[i, bestFeature]) <= bestThreshold).ToList();
        var right = indices.Where(i => NumOps.ToDouble(data[i, bestFeature]) > bestThreshold).ToList();

        BuildTree(data, target, left, depth + 1, random);
        BuildTree(data, target, right, depth + 1, random);
    }

    private double ComputeGini(Vector<T> target, List<int> indices)
    {
        if (indices.Count == 0) return 0;

        var classCounts = new Dictionary<int, int>();
        foreach (int i in indices)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            classCounts[label] = classCounts.GetValueOrDefault(label) + 1;
        }

        double gini = 1.0;
        foreach (int count in classCounts.Values)
        {
            double p = (double)count / indices.Count;
            gini -= p * p;
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
            throw new InvalidOperationException("TreeBasedFS has not been fitted.");

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
        throw new NotSupportedException("TreeBasedFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TreeBasedFS has not been fitted.");

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
