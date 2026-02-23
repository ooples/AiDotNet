using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Distance;

/// <summary>
/// ReliefF feature selection algorithm.
/// </summary>
/// <remarks>
/// <para>
/// ReliefF is a feature weighting algorithm that estimates feature quality based on
/// how well features distinguish between instances that are near each other.
/// </para>
/// <para>
/// For each randomly sampled instance, ReliefF:
/// 1. Finds k nearest neighbors from the same class (hits)
/// 2. Finds k nearest neighbors from each different class (misses)
/// 3. Updates feature weights based on distance differences
/// </para>
/// <para><b>For Beginners:</b> ReliefF scores each feature by asking:
/// "Does this feature help distinguish different classes?"
///
/// If instances from the same class have similar values but instances from
/// different classes have different values, the feature gets a high score.
///
/// Unlike correlation-based methods, ReliefF can detect feature interactions
/// and works well with non-linear relationships.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class ReliefF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly int _nIterations;
    private readonly int? _randomState;

    // Fitted parameters
    private double[]? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>
    /// Gets the number of nearest neighbors used.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the computed feature weights.
    /// </summary>
    public double[]? FeatureWeights => _featureWeights;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="ReliefF{T}"/>.
    /// </summary>
    /// <param name="nFeaturesToSelect">Number of features to select. Defaults to 10.</param>
    /// <param name="nNeighbors">Number of nearest neighbors (k). Defaults to 10.</param>
    /// <param name="nIterations">Number of sampling iterations. Defaults to all samples.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public ReliefF(
        int nFeaturesToSelect = 10,
        int nNeighbors = 10,
        int nIterations = -1,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
        {
            throw new ArgumentException("Number of features to select must be at least 1.", nameof(nFeaturesToSelect));
        }

        if (nNeighbors < 1)
        {
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        }

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ReliefF requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits ReliefF by computing feature weights.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values (class labels).</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to double arrays
        var X = new double[n, p];
        var y = new double[n];

        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Normalize features to [0, 1] for distance computation
        var minVals = new double[p];
        var maxVals = new double[p];
        for (int j = 0; j < p; j++)
        {
            minVals[j] = double.MaxValue;
            maxVals[j] = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                if (X[i, j] < minVals[j]) minVals[j] = X[i, j];
                if (X[i, j] > maxVals[j]) maxVals[j] = X[i, j];
            }
        }

        var Xnorm = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double range = maxVals[j] - minVals[j];
                Xnorm[i, j] = range > 1e-10 ? (X[i, j] - minVals[j]) / range : 0;
            }
        }

        // Get class information
        var classLabels = y.Distinct().ToArray();
        var classPriors = new Dictionary<double, double>();
        var classIndices = new Dictionary<double, List<int>>();

        foreach (double c in classLabels)
        {
            classIndices[c] = new List<int>();
        }

        for (int i = 0; i < n; i++)
        {
            classIndices[y[i]].Add(i);
        }

        foreach (double c in classLabels)
        {
            classPriors[c] = (double)classIndices[c].Count / n;
        }

        // Initialize weights
        _featureWeights = new double[p];

        // Number of iterations
        int m = _nIterations > 0 ? _nIterations : n;
        int k = Math.Min(_nNeighbors, n - 1);

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // ReliefF algorithm
        for (int iter = 0; iter < m; iter++)
        {
            int idx = random.Next(n);
            double instanceClass = y[idx];

            // Find k nearest hits (same class)
            var hits = FindNearestNeighbors(Xnorm, idx, classIndices[instanceClass], k, p);

            // Find k nearest misses from each other class
            var missesPerClass = new Dictionary<double, int[]>();
            foreach (double c in classLabels)
            {
                if (Math.Abs(c - instanceClass) > 1e-10)
                {
                    missesPerClass[c] = FindNearestNeighbors(Xnorm, idx, classIndices[c], k, p);
                }
            }

            // Update weights
            for (int f = 0; f < p; f++)
            {
                // Contribution from hits (same class - decrease weight if different)
                double hitSum = 0;
                foreach (int h in hits)
                {
                    hitSum += Math.Abs(Xnorm[idx, f] - Xnorm[h, f]);
                }
                hitSum /= (m * k);

                // Contribution from misses (different class - increase weight if different)
                double missSum = 0;
                foreach (double c in classLabels)
                {
                    if (Math.Abs(c - instanceClass) > 1e-10 && missesPerClass.TryGetValue(c, out var misses))
                    {
                        double classMissSum = 0;
                        foreach (int miss in misses)
                        {
                            classMissSum += Math.Abs(Xnorm[idx, f] - Xnorm[miss, f]);
                        }
                        // Weight by prior probability
                        missSum += (classPriors[c] / (1 - classPriors[instanceClass])) * classMissSum / k;
                    }
                }
                missSum /= m;

                _featureWeights[f] += missSum - hitSum;
            }
        }

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _featureWeights
            .Select((w, idx) => (Weight: w, Index: idx))
            .OrderByDescending(x => x.Weight)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] FindNearestNeighbors(double[,] Xnorm, int idx, List<int> candidates, int k, int p)
    {
        var distances = new List<(int Index, double Distance)>();

        foreach (int candIdx in candidates)
        {
            if (candIdx == idx) continue;

            double dist = 0;
            for (int f = 0; f < p; f++)
            {
                double diff = Xnorm[idx, f] - Xnorm[candIdx, f];
                dist += diff * diff;
            }

            distances.Add((candIdx, Math.Sqrt(dist)));
        }

        return distances
            .OrderBy(d => d.Distance)
            .Take(k)
            .Select(d => d.Index)
            .ToArray();
    }

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by selecting ReliefF-ranked features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("ReliefF has not been fitted.");
        }

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                result[i, j] = data[i, _selectedIndices[j]];
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ReliefF does not support inverse transformation.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("ReliefF has not been fitted.");
        }

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
        {
            mask[idx] = true;
        }

        return mask;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
        {
            return Array.Empty<string>();
        }

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
