using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Stability Selection based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their selection stability across multiple subsamples
/// of the data, ensuring robust feature selection.
/// </para>
/// <para><b>For Beginners:</b> Stability selection repeatedly subsamples the data
/// and performs feature selection on each subsample. Features that are consistently
/// selected across many subsamples are more reliable.
/// </para>
/// </remarks>
public class StabilitySelectionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSubsamples;
    private readonly double _subsampleRatio;
    private readonly int? _randomState;

    private double[]? _stabilityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NSubsamples => _nSubsamples;
    public double SubsampleRatio => _subsampleRatio;
    public double[]? StabilityScores => _stabilityScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public StabilitySelectionSelector(
        int nFeaturesToSelect = 10,
        int nSubsamples = 50,
        double subsampleRatio = 0.5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nSubsamples = nSubsamples;
        _subsampleRatio = subsampleRatio;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StabilitySelectionSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var selectionCounts = new int[p];
        int subsampleSize = (int)(n * _subsampleRatio);
        int selectPerSubsample = Math.Max(1, _nFeaturesToSelect / 2);

        for (int s = 0; s < _nSubsamples; s++)
        {
            // Create random subsample
            var indices = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).Take(subsampleSize).ToList();

            // Extract subsample
            var subX = new double[subsampleSize, p];
            var subY = new double[subsampleSize];
            for (int i = 0; i < subsampleSize; i++)
            {
                subY[i] = y[indices[i]];
                for (int j = 0; j < p; j++)
                    subX[i, j] = X[indices[i], j];
            }

            // Compute correlation-based scores on subsample
            var scores = new double[p];
            for (int j = 0; j < p; j++)
            {
                var col = new double[subsampleSize];
                for (int i = 0; i < subsampleSize; i++)
                    col[i] = subX[i, j];
                scores[j] = Math.Abs(ComputeCorrelation(col, subY));
            }

            // Select top features from this subsample
            var topFeatures = Enumerable.Range(0, p)
                .OrderByDescending(j => scores[j])
                .Take(selectPerSubsample);

            foreach (int j in topFeatures)
                selectionCounts[j]++;
        }

        // Stability scores = selection frequency
        _stabilityScores = selectionCounts.Select(c => (double)c / _nSubsamples).ToArray();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _stabilityScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeCorrelation(double[] x, double[] y)
    {
        int n = x.Length;
        if (n < 2) return 0;

        double xMean = x.Average();
        double yMean = y.Average();

        double numerator = 0, xSumSq = 0, ySumSq = 0;
        for (int i = 0; i < n; i++)
        {
            numerator += (x[i] - xMean) * (y[i] - yMean);
            xSumSq += (x[i] - xMean) * (x[i] - xMean);
            ySumSq += (y[i] - yMean) * (y[i] - yMean);
        }

        double denominator = Math.Sqrt(xSumSq * ySumSq);
        return denominator > 1e-10 ? numerator / denominator : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StabilitySelectionSelector has not been fitted.");

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
        throw new NotSupportedException("StabilitySelectionSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StabilitySelectionSelector has not been fitted.");

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
