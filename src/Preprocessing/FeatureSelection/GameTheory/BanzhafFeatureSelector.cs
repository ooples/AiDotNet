using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.GameTheory;

/// <summary>
/// Banzhaf Power Index-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Banzhaf power index from game theory to measure feature importance
/// based on the probability of being a swing voter (decisive feature).
/// </para>
/// <para><b>For Beginners:</b> The Banzhaf index measures how often adding a
/// feature changes the outcome (makes a coalition successful). Unlike Shapley,
/// it treats all coalition sizes equally. Features with high Banzhaf values are
/// "swing voters" that often make the difference.
/// </para>
/// </remarks>
public class BanzhafFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSamples;
    private readonly double _threshold;
    private readonly int? _randomState;

    private double[]? _banzhafIndices;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Threshold => _threshold;
    public double[]? BanzhafIndices => _banzhafIndices;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BanzhafFeatureSelector(
        int nFeaturesToSelect = 10,
        int nSamples = 100,
        double threshold = 0.5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nSamples = nSamples;
        _threshold = threshold;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BanzhafFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        _banzhafIndices = new double[p];
        var swingCounts = new int[p];
        int totalCoalitions = 0;

        // Monte Carlo sampling of coalitions
        for (int sample = 0; sample < _nSamples; sample++)
        {
            // Random coalition (each feature included with 50% probability)
            var coalition = new HashSet<int>();
            for (int j = 0; j < p; j++)
                if (rand.NextDouble() < 0.5)
                    coalition.Add(j);

            double coalitionScore = EvaluateSubset(X, y, coalition, n);
            bool coalitionWins = coalitionScore >= _threshold;

            // Check each feature for swing
            for (int j = 0; j < p; j++)
            {
                var modifiedCoalition = new HashSet<int>(coalition);
                if (coalition.Contains(j))
                    modifiedCoalition.Remove(j);
                else
                    modifiedCoalition.Add(j);

                double modifiedScore = EvaluateSubset(X, y, modifiedCoalition, n);
                bool modifiedWins = modifiedScore >= _threshold;

                // Is this feature a swing voter?
                if (coalitionWins != modifiedWins)
                    swingCounts[j]++;
            }

            totalCoalitions++;
        }

        // Compute Banzhaf indices
        for (int j = 0; j < p; j++)
            _banzhafIndices[j] = (double)swingCounts[j] / totalCoalitions;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _banzhafIndices[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double EvaluateSubset(double[,] X, double[] y, HashSet<int> features, int n)
    {
        if (features.Count == 0) return 0;

        var featureList = features.ToList();
        double yMean = y.Average();

        // Simple correlation-based score
        double totalCorr = 0;
        foreach (int j in featureList)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++) xMean += X[i, j];
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xd = X[i, j] - xMean;
                double yd = y[i] - yMean;
                sxy += xd * yd;
                sxx += xd * xd;
                syy += yd * yd;
            }

            double corr = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
            totalCorr += corr;
        }

        return totalCorr / featureList.Count;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BanzhafFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("BanzhafFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BanzhafFeatureSelector has not been fitted.");

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
