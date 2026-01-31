using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Randomized Lasso Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines Lasso with random reweighting and subsampling to provide
/// more stable feature selection that is less sensitive to the choice
/// of regularization parameter.
/// </para>
/// <para><b>For Beginners:</b> Regular Lasso can be unstable - small changes
/// in data can lead to different features being selected. Randomized Lasso
/// runs Lasso many times with random perturbations and selects features
/// that are consistently chosen, giving more reliable results.
/// </para>
/// </remarks>
public class RandomizedLassoSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly double _alpha;
    private readonly double _sampleFraction;
    private readonly double _weaknessRange;

    private double[]? _selectionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? SelectionProbabilities => _selectionProbabilities;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public RandomizedLassoSelector(
        int nFeaturesToSelect = 10,
        int nIterations = 100,
        double alpha = 0.01,
        double sampleFraction = 0.75,
        double weaknessRange = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _alpha = alpha;
        _sampleFraction = sampleFraction;
        _weaknessRange = weaknessRange;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RandomizedLassoSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Normalize features
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++) means[j] += X[i, j];
            means[j] /= n;
            for (int i = 0; i < n; i++) stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / (n - 1)) + 1e-10;
            for (int i = 0; i < n; i++) X[i, j] = (X[i, j] - means[j]) / stds[j];
        }

        double yMean = y.Average();
        for (int i = 0; i < n; i++) y[i] -= yMean;

        var selectionCounts = new int[p];
        int sampleSize = (int)(n * _sampleFraction);

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Subsample
            var indices = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).Take(sampleSize).ToList();

            // Random weights for features
            var weights = new double[p];
            for (int j = 0; j < p; j++)
                weights[j] = 1 - rand.NextDouble() * _weaknessRange;

            // Run weighted Lasso
            var beta = RunWeightedLasso(X, y, indices, weights, p, _alpha);

            // Count selected features
            for (int j = 0; j < p; j++)
                if (Math.Abs(beta[j]) > 1e-10)
                    selectionCounts[j]++;
        }

        // Compute selection probabilities
        _selectionProbabilities = new double[p];
        for (int j = 0; j < p; j++)
            _selectionProbabilities[j] = (double)selectionCounts[j] / _nIterations;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _selectionProbabilities[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] RunWeightedLasso(double[,] X, double[] y, List<int> indices, double[] weights, int p, double alpha)
    {
        int n = indices.Count;
        var beta = new double[p];

        // Coordinate descent
        for (int iter = 0; iter < 50; iter++)
        {
            for (int j = 0; j < p; j++)
            {
                double rho = 0;
                for (int idx = 0; idx < n; idx++)
                {
                    int i = indices[idx];
                    double residual = y[i];
                    for (int k = 0; k < p; k++)
                        if (k != j)
                            residual -= beta[k] * X[i, k];
                    rho += weights[j] * X[i, j] * residual;
                }
                rho /= n;

                // Soft thresholding with weighted penalty
                double effectiveAlpha = alpha / (weights[j] + 1e-10);
                beta[j] = SoftThreshold(rho, effectiveAlpha);
            }
        }

        return beta;
    }

    private double SoftThreshold(double x, double lambda)
    {
        if (x > lambda) return x - lambda;
        if (x < -lambda) return x + lambda;
        return 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomizedLassoSelector has not been fitted.");

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
        throw new NotSupportedException("RandomizedLassoSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomizedLassoSelector has not been fitted.");

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
