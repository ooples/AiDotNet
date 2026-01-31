using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bayesian;

/// <summary>
/// Bayesian Model Averaging Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their posterior inclusion probabilities across
/// all possible models, weighted by model posterior probabilities.
/// </para>
/// <para><b>For Beginners:</b> Instead of picking one best model, BMA considers
/// all possible models and weights them by how likely each is. Features that
/// appear in many high-probability models are selected.
/// </para>
/// </remarks>
public class BayesianModelAveraging<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nModels;
    private readonly double _priorInclusion;

    private double[]? _inclusionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? InclusionProbabilities => _inclusionProbabilities;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BayesianModelAveraging(
        int nFeaturesToSelect = 10,
        int nModels = 1000,
        double priorInclusion = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nModels = nModels;
        _priorInclusion = priorInclusion;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BayesianModelAveraging requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = RandomHelper.CreateSecureRandom();
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Track inclusion counts and model weights
        var inclusionCounts = new double[p];
        double totalWeight = 0;

        for (int m = 0; m < _nModels; m++)
        {
            // Sample a model (subset of features)
            var included = new bool[p];
            var selectedFeatures = new List<int>();
            for (int j = 0; j < p; j++)
            {
                if (rand.NextDouble() < _priorInclusion)
                {
                    included[j] = true;
                    selectedFeatures.Add(j);
                }
            }

            if (selectedFeatures.Count == 0)
            {
                selectedFeatures.Add(rand.Next(p));
                included[selectedFeatures[0]] = true;
            }

            // Compute model likelihood (using R-squared as proxy)
            double modelWeight = ComputeModelWeight(X, y, selectedFeatures, n);
            totalWeight += modelWeight;

            // Accumulate weighted inclusion
            for (int j = 0; j < p; j++)
                if (included[j])
                    inclusionCounts[j] += modelWeight;
        }

        // Compute posterior inclusion probabilities
        _inclusionProbabilities = new double[p];
        for (int j = 0; j < p; j++)
            _inclusionProbabilities[j] = inclusionCounts[j] / (totalWeight + 1e-10);

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _inclusionProbabilities[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeModelWeight(double[,] X, double[] y, List<int> features, int n)
    {
        if (features.Count == 0) return 1e-10;

        // Simple linear regression R-squared
        double yMean = y.Average();
        double ssTot = y.Sum(yi => (yi - yMean) * (yi - yMean));
        if (ssTot < 1e-10) return 1e-10;

        // Use correlation-based approximation
        double totalCorr = 0;
        foreach (int j in features)
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
            double r = (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
            totalCorr += r * r;
        }

        double rSquared = Math.Min(totalCorr / features.Count, 0.99);

        // BIC-like penalty
        double bic = n * Math.Log(1 - rSquared + 1e-10) + features.Count * Math.Log(n);
        return Math.Exp(-0.5 * bic);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BayesianModelAveraging has not been fitted.");

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
        throw new NotSupportedException("BayesianModelAveraging does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BayesianModelAveraging has not been fitted.");

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
