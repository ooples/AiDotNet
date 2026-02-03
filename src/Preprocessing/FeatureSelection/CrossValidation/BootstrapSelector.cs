using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.CrossValidation;

/// <summary>
/// Bootstrap based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their importance stability across multiple bootstrap
/// samples, ensuring robust feature selection.
/// </para>
/// <para><b>For Beginners:</b> Bootstrap sampling creates many random subsets of
/// data (with replacement). Features that are consistently important across all
/// these subsets are more reliable than those that only work on specific subsets.
/// </para>
/// </remarks>
public class BootstrapSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBootstraps;
    private readonly double _sampleRatio;
    private readonly int? _randomState;

    private double[]? _stabilityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBootstraps => _nBootstraps;
    public double SampleRatio => _sampleRatio;
    public double[]? StabilityScores => _stabilityScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BootstrapSelector(
        int nFeaturesToSelect = 10,
        int nBootstraps = 100,
        double sampleRatio = 0.8,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBootstraps = nBootstraps;
        _sampleRatio = sampleRatio;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BootstrapSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        int sampleSize = (int)(n * _sampleRatio);
        var selectionCounts = new int[p];

        for (int b = 0; b < _nBootstraps; b++)
        {
            // Create bootstrap sample
            var sampleIndices = new int[sampleSize];
            for (int i = 0; i < sampleSize; i++)
                sampleIndices[i] = rand.Next(n);

            // Compute correlation with target for each feature on bootstrap sample
            var correlations = new double[p];

            double ySum = 0, ySumSq = 0;
            for (int i = 0; i < sampleSize; i++)
            {
                ySum += y[sampleIndices[i]];
                ySumSq += y[sampleIndices[i]] * y[sampleIndices[i]];
            }
            double yMean = ySum / sampleSize;
            double yVar = ySumSq / sampleSize - yMean * yMean;

            for (int j = 0; j < p; j++)
            {
                double xSum = 0, xySumProd = 0, xSumSq = 0;
                for (int i = 0; i < sampleSize; i++)
                {
                    int idx = sampleIndices[i];
                    xSum += X[idx, j];
                    xSumSq += X[idx, j] * X[idx, j];
                    xySumProd += X[idx, j] * y[idx];
                }

                double xMean = xSum / sampleSize;
                double xVar = xSumSq / sampleSize - xMean * xMean;
                double covar = xySumProd / sampleSize - xMean * yMean;

                correlations[j] = (xVar > 1e-10 && yVar > 1e-10)
                    ? Math.Abs(covar / Math.Sqrt(xVar * yVar))
                    : 0;
            }

            // Select top features for this bootstrap
            var topFeatures = Enumerable.Range(0, p)
                .OrderByDescending(j => correlations[j])
                .Take(_nFeaturesToSelect)
                .ToHashSet();

            foreach (int j in topFeatures)
                selectionCounts[j]++;
        }

        // Stability score = proportion of bootstraps where feature was selected
        _stabilityScores = selectionCounts.Select(c => (double)c / _nBootstraps).ToArray();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _stabilityScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapSelector has not been fitted.");

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
        throw new NotSupportedException("BootstrapSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapSelector has not been fitted.");

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
