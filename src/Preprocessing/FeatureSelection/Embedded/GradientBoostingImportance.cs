using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class GradientBoostingImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nEstimators;
    private readonly int _maxDepth;
    private readonly double _learningRate;
    private readonly int? _randomState;

    private double[]? _importances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NEstimators => _nEstimators;
    public double LearningRate => _learningRate;
    public double[]? Importances => _importances;
    public int[]? SelectedIndices => _selectedIndices;
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

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GradientBoostingImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        _importances = new double[p];

        // Initialize predictions with mean
        var predictions = new double[n];
        double mean = 0;
        for (int i = 0; i < n; i++)
            mean += NumOps.ToDouble(target[i]);
        mean /= n;
        for (int i = 0; i < n; i++)
            predictions[i] = mean;

        // Boosting rounds
        for (int round = 0; round < _nEstimators; round++)
        {
            // Compute residuals (negative gradient for squared loss)
            var residuals = new double[n];
            for (int i = 0; i < n; i++)
                residuals[i] = NumOps.ToDouble(target[i]) - predictions[i];

            // Fit tree to residuals and collect importance
            var treeImportance = new double[p];
            var treePredictions = BuildRegressionTree(data, residuals, n, p, treeImportance, random);

            // Update predictions
            for (int i = 0; i < n; i++)
                predictions[i] += _learningRate * treePredictions[i];

            // Accumulate importance
            double totalImp = treeImportance.Sum();
            if (totalImp > 1e-10)
            {
                for (int j = 0; j < p; j++)
                    _importances[j] += treeImportance[j] / totalImp;
            }
        }

        // Normalize final importances
        double total = _importances.Sum();
        if (total > 1e-10)
        {
            for (int j = 0; j < p; j++)
                _importances[j] /= total;
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

    private double[] BuildRegressionTree(Matrix<T> data, double[] residuals, int n, int p,
        double[] importance, Random random)
    {
        var predictions = new double[n];
        var indices = Enumerable.Range(0, n).ToArray();

        BuildNode(data, residuals, indices, predictions, importance, 0, random);

        return predictions;
    }

    private void BuildNode(Matrix<T> data, double[] residuals, int[] indices,
        double[] predictions, double[] importance, int depth, Random random)
    {
        if (depth >= _maxDepth || indices.Length < 4)
        {
            // Leaf: predict mean residual
            double meanResidual = indices.Select(i => residuals[i]).Average();
            foreach (int i in indices)
                predictions[i] = meanResidual;
            return;
        }

        int n = indices.Length;
        int p = data.Columns;

        // Random feature subset
        int nFeaturesToTry = Math.Max(1, (int)Math.Sqrt(p));
        var featureSubset = Enumerable.Range(0, p)
            .OrderBy(_ => random.Next())
            .Take(nFeaturesToTry)
            .ToArray();

        // Find best split
        double bestGain = 0;
        int bestFeature = -1;
        double bestThreshold = 0;

        double parentMean = indices.Select(i => residuals[i]).Average();
        double parentVar = indices.Select(i => (residuals[i] - parentMean) * (residuals[i] - parentMean)).Sum();

        foreach (int feature in featureSubset)
        {
            var values = indices.Select(i => NumOps.ToDouble(data[i, feature]))
                .Distinct()
                .OrderBy(v => v)
                .ToArray();

            for (int i = 0; i < values.Length - 1; i++)
            {
                double threshold = (values[i] + values[i + 1]) / 2;

                var leftIndices = indices.Where(idx => NumOps.ToDouble(data[idx, feature]) <= threshold).ToArray();
                var rightIndices = indices.Where(idx => NumOps.ToDouble(data[idx, feature]) > threshold).ToArray();

                if (leftIndices.Length < 2 || rightIndices.Length < 2)
                    continue;

                double leftMean = leftIndices.Select(idx => residuals[idx]).Average();
                double rightMean = rightIndices.Select(idx => residuals[idx]).Average();

                double leftVar = leftIndices.Select(idx => (residuals[idx] - leftMean) * (residuals[idx] - leftMean)).Sum();
                double rightVar = rightIndices.Select(idx => (residuals[idx] - rightMean) * (residuals[idx] - rightMean)).Sum();

                double gain = parentVar - leftVar - rightVar;

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature < 0)
        {
            double meanResidual = indices.Select(i => residuals[i]).Average();
            foreach (int i in indices)
                predictions[i] = meanResidual;
            return;
        }

        // Record importance
        importance[bestFeature] += bestGain;

        // Split and recurse
        var left = indices.Where(idx => NumOps.ToDouble(data[idx, bestFeature]) <= bestThreshold).ToArray();
        var right = indices.Where(idx => NumOps.ToDouble(data[idx, bestFeature]) > bestThreshold).ToArray();

        BuildNode(data, residuals, left, predictions, importance, depth + 1, random);
        BuildNode(data, residuals, right, predictions, importance, depth + 1, random);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GradientBoostingImportance has not been fitted.");

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
        throw new NotSupportedException("GradientBoostingImportance does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GradientBoostingImportance has not been fitted.");

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
