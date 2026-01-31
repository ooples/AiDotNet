using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Fuzzy;

/// <summary>
/// Fuzzy Rough Set Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines fuzzy set theory with rough set theory to handle both uncertainty
/// and vagueness in data. Uses fuzzy equivalence relations to compute fuzzy
/// positive regions and find feature reducts.
/// </para>
/// <para><b>For Beginners:</b> Traditional rough sets use sharp boundaries
/// (same/different), but real data often has degrees of similarity. Fuzzy
/// rough sets allow saying "these objects are 80% similar" instead of just
/// "same" or "different". This gives more nuanced feature selection.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FuzzyRoughSetSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _maxFeatures;
    private readonly double _sigma;

    private double[]? _fuzzyDependencies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MaxFeatures => _maxFeatures;
    public double Sigma => _sigma;
    public double[]? FuzzyDependencies => _fuzzyDependencies;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FuzzyRoughSetSelector(
        int maxFeatures = 20,
        double sigma = 0.2,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxFeatures < 1)
            throw new ArgumentException("Max features must be at least 1.", nameof(maxFeatures));
        if (sigma <= 0)
            throw new ArgumentException("Sigma must be positive.", nameof(sigma));

        _maxFeatures = maxFeatures;
        _sigma = sigma;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FuzzyRoughSetSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to arrays
        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Normalize features
        NormalizeFeatures(X, n, p);

        // Compute individual feature fuzzy dependency
        _fuzzyDependencies = new double[p];
        for (int j = 0; j < p; j++)
            _fuzzyDependencies[j] = ComputeFuzzyDependency(X, y, new[] { j }, n, p);

        // Greedy forward selection based on fuzzy dependency
        var selected = new HashSet<int>();
        double fullDependency = ComputeFuzzyDependency(X, y, Enumerable.Range(0, p).ToArray(), n, p);
        double currentDependency = 0;

        while (selected.Count < _maxFeatures && currentDependency < fullDependency - 1e-6)
        {
            int bestFeature = -1;
            double bestDependency = currentDependency;

            for (int j = 0; j < p; j++)
            {
                if (selected.Contains(j)) continue;

                var testSet = selected.Union(new[] { j }).ToArray();
                double testDependency = ComputeFuzzyDependency(X, y, testSet, n, p);

                if (testDependency > bestDependency)
                {
                    bestDependency = testDependency;
                    bestFeature = j;
                }
            }

            if (bestFeature < 0)
                break;

            selected.Add(bestFeature);
            currentDependency = bestDependency;
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();

        // If not enough features selected, add by individual dependency
        if (_selectedIndices.Length < Math.Min(_maxFeatures, p))
        {
            var remaining = Enumerable.Range(0, p)
                .Except(_selectedIndices)
                .OrderByDescending(j => _fuzzyDependencies[j])
                .Take(_maxFeatures - _selectedIndices.Length);
            _selectedIndices = _selectedIndices.Concat(remaining).OrderBy(x => x).ToArray();
        }

        IsFitted = true;
    }

    private void NormalizeFeatures(double[,] X, int n, int p)
    {
        for (int j = 0; j < p; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                min = Math.Min(min, X[i, j]);
                max = Math.Max(max, X[i, j]);
            }

            double range = max - min;
            if (range > 1e-10)
            {
                for (int i = 0; i < n; i++)
                    X[i, j] = (X[i, j] - min) / range;
            }
        }
    }

    private double ComputeFuzzyDependency(double[,] X, int[] y, int[] features, int n, int p)
    {
        // Compute fuzzy similarity matrix for selected features
        var similarity = new double[n, n];
        for (int i1 = 0; i1 < n; i1++)
        {
            similarity[i1, i1] = 1.0;
            for (int i2 = i1 + 1; i2 < n; i2++)
            {
                double sim = ComputeFuzzySimilarity(X, features, i1, i2);
                similarity[i1, i2] = sim;
                similarity[i2, i1] = sim;
            }
        }

        // Compute decision similarity (crisp)
        var decisionSim = new double[n, n];
        for (int i1 = 0; i1 < n; i1++)
        {
            for (int i2 = 0; i2 < n; i2++)
                decisionSim[i1, i2] = y[i1] == y[i2] ? 1.0 : 0.0;
        }

        // Compute fuzzy positive region
        double positiveRegion = 0;
        for (int i = 0; i < n; i++)
        {
            // Lower approximation of decision class for object i
            double lowerApprox = 1.0;
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                double implication = FuzzyImplication(similarity[i, j], decisionSim[i, j]);
                lowerApprox = Math.Min(lowerApprox, implication);
            }
            positiveRegion += lowerApprox;
        }

        return positiveRegion / n;
    }

    private double ComputeFuzzySimilarity(double[,] X, int[] features, int i1, int i2)
    {
        double minSim = 1.0;
        foreach (int j in features)
        {
            double diff = Math.Abs(X[i1, j] - X[i2, j]);
            double featureSim = Math.Exp(-diff * diff / (2 * _sigma * _sigma));
            minSim = Math.Min(minSim, featureSim);
        }
        return minSim;
    }

    private double FuzzyImplication(double a, double b)
    {
        // Łukasiewicz implication: min(1, 1 - a + b)
        return Math.Min(1.0, 1.0 - a + b);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FuzzyRoughSetSelector has not been fitted.");

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
        throw new NotSupportedException("FuzzyRoughSetSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FuzzyRoughSetSelector has not been fitted.");

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
