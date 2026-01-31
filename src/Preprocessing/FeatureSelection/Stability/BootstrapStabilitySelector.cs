using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Bootstrap Stability-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses bootstrap sampling to assess the stability of feature selection,
/// selecting features that are consistently chosen across multiple bootstrap
/// samples.
/// </para>
/// <para><b>For Beginners:</b> Bootstrap means taking many random samples
/// (with replacement) from your data. This method runs feature selection
/// on each sample and counts how often each feature is selected. Features
/// that are consistently selected are more reliable choices.
/// </para>
/// </remarks>
public class BootstrapStabilitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBootstraps;
    private readonly double _threshold;

    private double[]? _selectionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? SelectionProbabilities => _selectionProbabilities;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BootstrapStabilitySelector(
        int nFeaturesToSelect = 10,
        int nBootstraps = 100,
        double threshold = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBootstraps = nBootstraps;
        _threshold = threshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BootstrapStabilitySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var selectionCounts = new int[p];

        for (int b = 0; b < _nBootstraps; b++)
        {
            // Create bootstrap sample
            var indices = new int[n];
            for (int i = 0; i < n; i++)
                indices[i] = rand.Next(n);

            var Xb = new double[n, p];
            var yb = new double[n];
            for (int i = 0; i < n; i++)
            {
                yb[i] = y[indices[i]];
                for (int j = 0; j < p; j++)
                    Xb[i, j] = X[indices[i], j];
            }

            // Run simple correlation-based selection on bootstrap sample
            var selected = SelectFeatures(Xb, yb, n, p);
            foreach (int j in selected)
                selectionCounts[j]++;
        }

        // Compute selection probabilities
        _selectionProbabilities = new double[p];
        for (int j = 0; j < p; j++)
            _selectionProbabilities[j] = (double)selectionCounts[j] / _nBootstraps;

        // Select features above threshold, sorted by probability
        var candidates = Enumerable.Range(0, p)
            .Where(j => _selectionProbabilities[j] >= _threshold)
            .OrderByDescending(j => _selectionProbabilities[j])
            .Take(_nFeaturesToSelect)
            .ToList();

        // If not enough, add more by probability
        if (candidates.Count < _nFeaturesToSelect)
        {
            var additional = Enumerable.Range(0, p)
                .Where(j => !candidates.Contains(j))
                .OrderByDescending(j => _selectionProbabilities[j])
                .Take(_nFeaturesToSelect - candidates.Count);
            candidates.AddRange(additional);
        }

        _selectedIndices = candidates.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private List<int> SelectFeatures(double[,] X, double[] y, int n, int p)
    {
        // Compute correlations
        var correlations = new double[p];
        double yMean = y.Average();

        for (int j = 0; j < p; j++)
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
            correlations[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        // Select top features
        return Enumerable.Range(0, p)
            .OrderByDescending(j => correlations[j])
            .Take(_nFeaturesToSelect)
            .ToList();
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapStabilitySelector has not been fitted.");

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
        throw new NotSupportedException("BootstrapStabilitySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapStabilitySelector has not been fitted.");

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
