using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Boruta feature selection using shadow features and random forest importance.
/// </summary>
/// <remarks>
/// <para>
/// Boruta creates "shadow" features by shuffling original features. A random forest
/// is trained, and features that consistently outperform the best shadow feature
/// are confirmed as important. Features that never beat shadows are rejected.
/// </para>
/// <para><b>For Beginners:</b> Imagine testing each feature against a "random noise"
/// version of itself. If a real feature is important, it should beat its noisy twin
/// repeatedly. Boruta does this systematically, keeping only features that clearly
/// outperform random chance.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BorutaSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nIterations;
    private readonly int _nTrees;
    private readonly double _percentile;
    private readonly int? _randomState;

    private double[]? _featureImportances;
    private bool[]? _confirmedFeatures;
    private bool[]? _tentativeFeatures;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NIterations => _nIterations;
    public double[]? FeatureImportances => _featureImportances;
    public bool[]? ConfirmedFeatures => _confirmedFeatures;
    public bool[]? TentativeFeatures => _tentativeFeatures;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BorutaSelector(
        int nIterations = 100,
        int nTrees = 50,
        double percentile = 100.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _nIterations = nIterations;
        _nTrees = nTrees;
        _percentile = percentile;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BorutaSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Track hits (times feature beats max shadow)
        var hits = new int[p];
        _featureImportances = new double[p];

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Create shadow features by shuffling
            var extendedData = new double[n, p * 2];
            for (int j = 0; j < p; j++)
            {
                for (int i = 0; i < n; i++)
                    extendedData[i, j] = NumOps.ToDouble(data[i, j]);

                // Shuffle for shadow feature
                var shuffled = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
                for (int i = 0; i < n; i++)
                    extendedData[i, p + j] = NumOps.ToDouble(data[shuffled[i], j]);
            }

            // Train simplified random forest and get importance
            var importances = ComputeRandomForestImportance(extendedData, target, n, p * 2, random);

            // Find max shadow importance
            double maxShadow = 0;
            for (int j = p; j < p * 2; j++)
                maxShadow = Math.Max(maxShadow, importances[j]);

            // Count hits
            for (int j = 0; j < p; j++)
            {
                _featureImportances[j] += importances[j];
                if (importances[j] > maxShadow)
                    hits[j]++;
            }
        }

        // Normalize importance
        for (int j = 0; j < p; j++)
            _featureImportances[j] /= _nIterations;

        // Statistical test for confirmation (simplified binomial test)
        _confirmedFeatures = new bool[p];
        _tentativeFeatures = new bool[p];

        double expectedHits = _nIterations * 0.5; // Under null hypothesis
        double stdDev = Math.Sqrt(_nIterations * 0.5 * 0.5);

        for (int j = 0; j < p; j++)
        {
            double zScore = (hits[j] - expectedHits) / stdDev;

            if (zScore > 2.0) // Confirmed
                _confirmedFeatures[j] = true;
            else if (zScore > 0) // Tentative
                _tentativeFeatures[j] = true;
        }

        // Select confirmed features (plus tentative if needed)
        var selected = new List<int>();
        for (int j = 0; j < p; j++)
            if (_confirmedFeatures[j])
                selected.Add(j);

        // If no confirmed, add tentative
        if (selected.Count == 0)
        {
            for (int j = 0; j < p; j++)
                if (_tentativeFeatures[j])
                    selected.Add(j);
        }

        // If still none, select top by importance
        if (selected.Count == 0)
        {
            selected = _featureImportances
                .Select((imp, idx) => (Imp: imp, Index: idx))
                .OrderByDescending(x => x.Imp)
                .Take(Math.Min(10, p))
                .Select(x => x.Index)
                .ToList();
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double[] ComputeRandomForestImportance(double[,] data, Vector<T> target, int n, int p, Random random)
    {
        var importances = new double[p];

        for (int t = 0; t < _nTrees; t++)
        {
            // Bootstrap sample
            var sampleIndices = new int[n];
            for (int i = 0; i < n; i++)
                sampleIndices[i] = random.Next(n);

            // Build simple decision stump and measure importance
            for (int j = 0; j < p; j++)
            {
                // Find best split for this feature
                var values = new List<double>();
                var targets = new List<double>();

                for (int i = 0; i < n; i++)
                {
                    values.Add(data[sampleIndices[i], j]);
                    targets.Add(NumOps.ToDouble(target[sampleIndices[i]]));
                }

                double gini = ComputeGiniReduction(values, targets);
                importances[j] += gini;
            }
        }

        // Normalize
        for (int j = 0; j < p; j++)
            importances[j] /= _nTrees;

        return importances;
    }

    private double ComputeGiniReduction(List<double> values, List<double> targets)
    {
        int n = values.Count;
        if (n < 2) return 0;

        // Find best split point
        var sorted = values.Select((v, i) => (Value: v, Target: targets[i]))
            .OrderBy(x => x.Value)
            .ToList();

        double overallGini = ComputeGini(targets);
        double bestReduction = 0;

        for (int i = 1; i < n; i++)
        {
            if (Math.Abs(sorted[i].Value - sorted[i - 1].Value) < 1e-10)
                continue;

            var leftTargets = sorted.Take(i).Select(x => x.Target).ToList();
            var rightTargets = sorted.Skip(i).Select(x => x.Target).ToList();

            double leftGini = ComputeGini(leftTargets);
            double rightGini = ComputeGini(rightTargets);

            double weightedGini = (leftTargets.Count * leftGini + rightTargets.Count * rightGini) / n;
            double reduction = overallGini - weightedGini;

            bestReduction = Math.Max(bestReduction, reduction);
        }

        return bestReduction;
    }

    private double ComputeGini(List<double> targets)
    {
        if (targets.Count == 0) return 0;

        var classCounts = targets.GroupBy(t => (int)Math.Round(t)).ToDictionary(g => g.Key, g => g.Count());
        double gini = 1.0;

        foreach (var count in classCounts.Values)
        {
            double prob = (double)count / targets.Count;
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
            throw new InvalidOperationException("BorutaSelector has not been fitted.");

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
        throw new NotSupportedException("BorutaSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BorutaSelector has not been fitted.");

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
